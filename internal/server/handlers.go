package server

import (
	"bytes"
	"context"
	"io"
	"net/http"

	"github.com/lewangdev/deepseek-v4-flash-vision/internal/convert"
	"github.com/lewangdev/deepseek-v4-flash-vision/internal/ir"
	"github.com/lewangdev/deepseek-v4-flash-vision/internal/router"
	"github.com/lewangdev/deepseek-v4-flash-vision/internal/server/streamconv"
)

const maxRequestBody = 64 << 20 // images raise the ceiling

// ---- endpoint entry points ----

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	s.serve(w, r, famChat, convert.OpenAIRequestToIR)
}

func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	s.serve(w, r, famMessages, convert.AnthropicRequestToIR)
}

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	s.serve(w, r, famResponses, convert.ResponsesRequestToIR)
}

// ---- shared pipeline ----

// serve parses the client request into IR, routes it, and dispatches to a
// same-family raw proxy or a cross-family conversion.
func (s *Server) serve(w http.ResponseWriter, r *http.Request, clientFam family, toIR func([]byte) (*ir.Request, error)) {
	if r.Method != http.MethodPost {
		s.methodError(w, clientFam)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBody))
	if err != nil {
		writeError(w, clientFam, http.StatusBadRequest, "invalid_request_error", "cannot read request body")
		return
	}
	req, err := toIR(body)
	if err != nil {
		writeError(w, clientFam, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	res := s.rt.Resolve(req)
	req.Model = res.Model
	echo := req.ClientModel
	if echo == "" {
		echo = res.Model
	}
	s.log.Info("routed",
		"path", r.URL.Path,
		"client_model", req.ClientModel,
		"has_image", req.HasImage(),
		"model", res.Model,
		"endpoint", res.Endpoint,
		"stream", req.Stream,
	)

	if familyOf(res.Endpoint) == clientFam {
		s.serveSameFamily(w, r, clientFam, body, req, res, echo)
		return
	}
	s.serveCrossFamily(w, r, clientFam, familyOf(res.Endpoint), req, echo)
}

// serveSameFamily proxies the client's request and response verbatim — only the
// model id and auth change — because both sides share a wire format.
func (s *Server) serveSameFamily(w http.ResponseWriter, r *http.Request, fam family, origBody []byte, req *ir.Request, res router.Result, echo string) {
	payload, err := s.buildUpstream(fam, origBody, req)
	if err != nil {
		writeError(w, fam, http.StatusInternalServerError, "invalid_request_error", err.Error())
		return
	}
	upResp, err := s.up.Stream(r.Context(), res.Endpoint, payload)
	if err != nil {
		s.upstreamError(w, fam, err)
		return
	}
	defer upResp.Body.Close()

	if ct := upResp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	} else {
		w.Header().Set("Content-Type", "application/json")
	}
	if req.Stream {
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Accel-Buffering", "no")
	}
	w.WriteHeader(http.StatusOK)
	// Best-effort model echo rewrite; a name split across bytes stays as-is.
	rewrite := func(b []byte) []byte { return replaceModelName(b, res.Model, echo) }
	if req.Stream {
		copyWithFlush(w, upResp.Body, rewrite)
	} else {
		data, _ := io.ReadAll(upResp.Body)
		w.Write(rewrite(data))
	}
}

// serveCrossFamily converts the request into the upstream format and converts
// the upstream response (streaming or not) back into the client's format.
func (s *Server) serveCrossFamily(w http.ResponseWriter, r *http.Request, clientFam, upstreamFam family, req *ir.Request, echo string) {
	var payload []byte
	var err error
	switch upstreamFam {
	case famMessages:
		payload, err = convert.AnthropicRequestFromIR(req, s.cfg.Server.DefaultMaxTokens)
	case famResponses:
		payload, err = convert.ResponsesRequestFromIR(req)
	default:
		payload, err = convert.OpenAIRequestFromIR(req)
	}
	if err != nil {
		writeError(w, clientFam, http.StatusInternalServerError, "invalid_request_error", err.Error())
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	upResp, err := s.up.Stream(ctx, upstreamFam.endpoint(), payload)
	if err != nil {
		s.upstreamError(w, clientFam, err)
		return
	}
	defer upResp.Body.Close()

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		streamconv.Run(string(clientFam), string(upstreamFam), upResp.Body, w, echo)
		return
	}

	raw, err := io.ReadAll(upResp.Body)
	if err != nil {
		writeError(w, clientFam, http.StatusBadGateway, "upstream_error", "failed reading upstream response")
		return
	}
	iresp, err := parseUpstreamResponse(upstreamFam, raw)
	if err != nil {
		s.log.Error("cannot parse upstream response", "error", err)
		writeError(w, clientFam, http.StatusBadGateway, "upstream_error", "unexpected upstream response: "+err.Error())
		return
	}
	out, err := buildDownstreamResponse(clientFam, iresp, echo)
	if err != nil {
		writeError(w, clientFam, http.StatusInternalServerError, "invalid_response_error", err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
}

// buildUpstream serializes the request for the upstream endpoint. When the
// client supplied a body (same-family), it is reused verbatim with the model id
// rewritten, preserving exact tool definitions and extensions.
func (s *Server) buildUpstream(fam family, origBody []byte, req *ir.Request) ([]byte, error) {
	if req.ClientModel != "" {
		return replaceModelName(origBody, req.ClientModel, req.Model), nil
	}
	switch fam {
	case famMessages:
		return convert.AnthropicRequestFromIR(req, s.cfg.Server.DefaultMaxTokens)
	case famResponses:
		return convert.ResponsesRequestFromIR(req)
	default:
		return convert.OpenAIRequestFromIR(req)
	}
}

// copyWithFlush copies the upstream stream to the client, flushing per read.
func copyWithFlush(w http.ResponseWriter, r io.Reader, rewrite func([]byte) []byte) {
	fl, _ := w.(http.Flusher)
	buf := make([]byte, 32*1024)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			chunk := buf[:n]
			if rewrite != nil {
				chunk = rewrite(chunk)
			}
			if _, werr := w.Write(chunk); werr != nil {
				return
			}
			if fl != nil {
				fl.Flush()
			}
		}
		if err == io.EOF {
			return
		}
		if err != nil {
			return
		}
	}
}

func parseUpstreamResponse(fam family, raw []byte) (*ir.Response, error) {
	switch fam {
	case famMessages:
		return convert.AnthropicResponseToIR(raw)
	case famResponses:
		return convert.ResponsesResponseToIR(raw)
	default:
		return convert.OpenAIResponseToIR(raw)
	}
}

func buildDownstreamResponse(fam family, resp *ir.Response, model string) ([]byte, error) {
	switch fam {
	case famMessages:
		return convert.AnthropicResponseFromIR(resp, model)
	case famResponses:
		return convert.ResponsesResponseFromIR(resp, model)
	default:
		return convert.OpenAIResponseFromIR(resp, model)
	}
}

func replaceModelName(b []byte, from, to string) []byte {
	if from == "" || to == "" || from == to {
		return b
	}
	return bytes.ReplaceAll(b, []byte(`"model":"`+from+`"`), []byte(`"model":"`+to+`"`))
}
