// Package streamconv converts between SSE streams of the three wire formats.
// For cross-family routes (e.g. an OpenAI Chat client talking to an Anthropic
// /messages upstream model) an upstream stream is parsed into canonical Deltas
// and re-emitted in the client's format. Text deltas stream live; tool-call
// arguments are accumulated whole (see parse package docs) and delivered as
// complete blocks once the stream ends.
package streamconv

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"

	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

// sse reads Server-Sent Events from an io.Reader. `data:` payloads are joined
// into a single payload per event; the `event:` name is folded into the result.
type sse struct{ sc *bufio.Scanner }

func newSSE(r io.Reader) *sse {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 4096), 8*1024*1024)
	return &sse{sc: sc}
}

// next returns the name ("" if absent) and payload of the next event.
// io.EOF is returned once the stream is exhausted.
func (s *sse) next() (event, data string, err error) {
	var parts []string
	for {
		if !s.sc.Scan() {
			if err := s.sc.Err(); err != nil {
				return "", "", err
			}
			if len(parts) > 0 {
				return event, strings.Join(parts, "\n"), nil
			}
			return "", "", io.EOF
		}
		line := s.sc.Text()
		if line == "" {
			if len(parts) > 0 {
				return event, strings.Join(parts, "\n"), nil
			}
			continue
		}
		switch {
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			parts = append(parts, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
		// comments ("//") and other fields are ignored.
	}
}

// Delta is a canonical chunk emitted by the parsers (one per upstream family).
// Text is streamed live; ToolCalls, FinishReason and Usage arrive on the final
// emit after the stream closes.
type Delta struct {
	Text         string
	ToolCalls    []ir.ToolCall
	FinishReason *string
	Usage        *ir.Usage
}

// upstreamUsage mirrors the union of usage objects across wire formats.
type upstreamUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
}

func (u *upstreamUsage) toIR() *ir.Usage {
	out := &ir.Usage{TotalTokens: u.TotalTokens}
	if u.PromptTokens != 0 || u.InputTokens != 0 {
		out.PromptTokens = u.PromptTokens
		if out.PromptTokens == 0 {
			out.PromptTokens = u.InputTokens
		}
	}
	if u.CompletionTokens != 0 || u.OutputTokens != 0 {
		out.CompletionTokens = u.CompletionTokens
		if out.CompletionTokens == 0 {
			out.CompletionTokens = u.OutputTokens
		}
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.PromptTokens + out.CompletionTokens
	}
	return out
}

type toolAcc struct {
	id, name, args string
}

func buildCalls(order []int, calls map[int]*toolAcc) []ir.ToolCall {
	out := make([]ir.ToolCall, 0, len(order))
	for _, idx := range order {
		a := calls[idx]
		out = append(out, ir.ToolCall{ID: a.id, Name: a.name, Input: json.RawMessage(a.args)})
	}
	return out
}

func buildCallsByID(order []string, calls map[string]*toolAcc) []ir.ToolCall {
	out := make([]ir.ToolCall, 0, len(order))
	for _, id := range order {
		a := calls[id]
		out = append(out, ir.ToolCall{ID: a.id, Name: a.name, Input: json.RawMessage(a.args)})
	}
	return out
}

// normalizeReason maps a wire-format finish reason to a canonical set
// {stop, tool_calls, length, error}. Custom reasons pass through unchanged.
func normalizeReason(reason string) string {
	switch reason {
	case "", "end_turn":
		return "stop"
	case "tool_use", "function_call":
		return "tool_calls"
	case "max_tokens":
		return "length"
	case "content_filter":
		return "error"
	}
	return reason
}

func writeSSE(w io.Writer, event string, payload any) error {
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if event != "" {
		if _, err := w.Write([]byte("event: " + event + "\n")); err != nil {
			return err
		}
	}
	_, err = w.Write(append(append([]byte("data: "), b...), '\n', '\n'))
	return err
}