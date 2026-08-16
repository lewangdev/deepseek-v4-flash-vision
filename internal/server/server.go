// Package server exposes the gateway's three client-facing endpoints (OpenAI
// Chat Completions, OpenAI Responses, Anthropic Messages) and routes each
// request to the right upstream OpenCode Go model and wire format.
package server

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/lewang/deepseek-v4-flash-vision/internal/config"
	"github.com/lewang/deepseek-v4-flash-vision/internal/router"
	"github.com/lewang/deepseek-v4-flash-vision/internal/upstream"
)

// family identifies one of the three wire formats.
type family string

const (
	famChat      family = "chat"
	famMessages  family = "messages"
	famResponses family = "responses"
)

func familyOf(endpoint string) family {
	switch endpoint {
	case config.EndpointMessages:
		return famMessages
	case config.EndpointResponses:
		return famResponses
	default:
		return famChat
	}
}

func (f family) endpoint() string {
	switch f {
	case famMessages:
		return config.EndpointMessages
	case famResponses:
		return config.EndpointResponses
	default:
		return config.EndpointChat
	}
}

// Server holds dependencies shared by all handlers.
type Server struct {
	cfg config.Config
	rt  *router.Router
	up  *upstream.Client
	log *slog.Logger
	now func() time.Time // overridable for tests
}

func New(cfg config.Config) *Server {
	return &Server{
		cfg: cfg,
		rt:  router.New(cfg),
		up:  upstream.New(cfg.OpenCode.BaseURL, cfg.OpenCode.APIKey, cfg.OpenCode.Headers),
		log: slog.Default(),
		now: time.Now,
	}
}

// Mux returns the fully-wired HTTP handler.
func (s *Server) Mux() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/v1/messages", s.handleMessages)
	mux.HandleFunc("/v1/responses", s.handleResponses)
	return s.auth(mux)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"status": "ok"})
}

// handleModels reports the known models in OpenAI list format, for clients that
// enumerate models on startup.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	data := make([]any, 0)
	for _, id := range s.rt.KnownModels() {
		data = append(data, map[string]any{
			"id": id, "object": "model", "created": s.now().Unix(), "owned_by": "opencode",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}

// auth enforces the configured downstream API key, when non-empty.
// OpenAI-style endpoints expect `Authorization: Bearer <key>`; the Anthropic
// Messages endpoint expects `x-api-key`.
func (s *Server) auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.Server.APIKey != "" {
			key := s.cfg.Server.APIKey
			switch {
			case strings.HasPrefix(r.URL.Path, "/v1/messages"):
				if r.Header.Get("x-api-key") != key && s.bearer(r) != key {
					w.Header().Set("WWW-Authenticate", "Bearer")
					writeError(w, famMessages, http.StatusUnauthorized, "authentication_error", "invalid x-api-key")
					return
				}
			case !strings.HasPrefix(r.URL.Path, "/healthz") &&
				!strings.HasPrefix(r.URL.Path, "/v1/models"):
				if s.bearer(r) != key {
					w.Header().Set("WWW-Authenticate", "Bearer")
					writeError(w, famChat, http.StatusUnauthorized, "invalid_request_error", "invalid API key")
					return
				}
			}
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) bearer(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if strings.HasPrefix(h, "Bearer ") {
		return strings.TrimPrefix(h, "Bearer ")
	}
	return ""
}

// writeError renders an error body in the client family's expected shape.
func writeError(w http.ResponseWriter, fam family, status int, errType, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	switch fam {
	case famMessages:
		json.NewEncoder(w).Encode(map[string]any{
			"type":  "error",
			"error": map[string]any{"type": errType, "message": msg},
		})
	default:
		json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{"type": errType, "message": msg, "code": nil},
		})
	}
}

func (s *Server) upstreamError(w http.ResponseWriter, fam family, err error) {
	s.log.Error("upstream call failed", "error", err)
	// The wrapped err already names the upstream endpoint (e.g. "upstream
	// messages: ..."), so don't re-annotate with the client family here —
	// on cross-family routes they differ and the extra prefix misleads.
	writeError(w, fam, http.StatusBadGateway, "upstream_error", fmt.Sprintf("upstream call failed: %v", err))
}

func (s *Server) methodError(w http.ResponseWriter, fam family) {
	writeError(w, fam, http.StatusMethodNotAllowed, "method_not_allowed", "method not allowed")
}
