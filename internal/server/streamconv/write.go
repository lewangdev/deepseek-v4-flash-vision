package streamconv

import (
	"crypto/rand"
	"encoding/hex"
	"io"
	"time"

	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

func nowUnix() int64 { return time.Now().Unix() }

// Writer emits canonical deltas as SSE in one downstream wire format.
type Writer interface {
	Text(chunk string) error
	ToolCall(tc ir.ToolCall) error
	Done(reason string, usage *ir.Usage) error
}

func randIDSuffix() string {
	var b [6]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// ---- OpenAI Chat Completions SSE ----

type chatWriter struct {
	w         io.Writer
	model     string
	id        string
	created   int64
	toolIndex int
	done      bool
}

// NewChatWriter emits an OpenAI chat.completion.chunk stream. It is also the
// fallback when the client family is unknown.
func NewChatWriter(w io.Writer, model string) Writer {
	return &chatWriter{w: w, model: model, id: "chatcmpl-" + randIDSuffix(), created: nowUnix()}
}

func (c *chatWriter) Text(chunk string) error {
	return writeSSE(c.w, "", map[string]any{
		"id": c.id, "object": "chat.completion.chunk", "created": c.created, "model": c.model,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": chunk}, "finish_reason": nil}},
	})
}

func (c *chatWriter) ToolCall(tc ir.ToolCall) error {
	idx := c.toolIndex
	c.toolIndex++
	return writeSSE(c.w, "", map[string]any{
		"id": c.id, "object": "chat.completion.chunk", "created": c.created, "model": c.model,
		"choices": []any{map[string]any{
			"index": 0,
			"delta": map[string]any{"tool_calls": []any{map[string]any{
				"index": idx, "id": tc.ID, "type": "function",
				"function": map[string]any{"name": tc.Name, "arguments": string(tc.Input)},
			}}},
			"finish_reason": nil,
		}},
	})
}

func (c *chatWriter) Done(reason string, usage *ir.Usage) error {
	if c.done {
		return nil
	}
	c.done = true
	if err := writeSSE(c.w, "", map[string]any{
		"id": c.id, "object": "chat.completion.chunk", "created": c.created, "model": c.model,
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": reason}},
	}); err != nil {
		return err
	}
	_, err := io.WriteString(c.w, "data: [DONE]\n\n")
	return err
}

// ---- Anthropic Messages SSE ----

type messagesWriter struct {
	w            io.Writer
	model        string
	id           string
	nextBlockIdx int
	textBlockIdx int
	textOpen     bool
	started      bool
	done         bool
}

// NewMessagesWriter emits an Anthropic Messages SSE stream.
func NewMessagesWriter(w io.Writer, model string) Writer {
	return &messagesWriter{w: w, model: model, id: "msg_" + randIDSuffix()}
}

func (m *messagesWriter) start() error {
	if m.started {
		return nil
	}
	m.started = true
	return writeSSE(m.w, "message_start", map[string]any{
		"type":    "message_start",
		"message": map[string]any{
			"id": m.id, "type": "message", "role": "assistant", "model": m.model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	})
}

func (m *messagesWriter) startText() error {
	if m.textOpen {
		return nil
	}
	if err := m.start(); err != nil {
		return err
	}
	m.textBlockIdx = m.nextBlockIdx
	m.nextBlockIdx++
	m.textOpen = true
	return writeSSE(m.w, "content_block_start", map[string]any{
		"type":         "content_block_start",
		"index":        m.textBlockIdx,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
}

func (m *messagesWriter) closeText() error {
	if !m.textOpen {
		return nil
	}
	m.textOpen = false
	return writeSSE(m.w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": m.textBlockIdx})
}

func (m *messagesWriter) Text(chunk string) error {
	if err := m.startText(); err != nil {
		return err
	}
	return writeSSE(m.w, "content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": m.textBlockIdx,
		"delta": map[string]any{"type": "text_delta", "text": chunk},
	})
}

func (m *messagesWriter) ToolCall(tc ir.ToolCall) error {
	if err := m.start(); err != nil {
		return err
	}
	if err := m.closeText(); err != nil {
		return err
	}
	idx := m.nextBlockIdx
	m.nextBlockIdx++
	if err := writeSSE(m.w, "content_block_start", map[string]any{
		"type":  "content_block_start",
		"index": idx,
		"content_block": map[string]any{
			"type": "tool_use", "id": tc.ID, "name": tc.Name, "input": map[string]any{},
		},
	}); err != nil {
		return err
	}
	if err := writeSSE(m.w, "content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": idx,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": string(tc.Input)},
	}); err != nil {
		return err
	}
	return writeSSE(m.w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
}

func (m *messagesWriter) Done(reason string, usage *ir.Usage) error {
	if m.done {
		return nil
	}
	m.done = true
	if !m.started {
		if err := m.start(); err != nil {
			return err
		}
	}
	if err := m.closeText(); err != nil {
		return err
	}
	out := 0
	if usage != nil {
		out = usage.CompletionTokens
	}
	if err := writeSSE(m.w, "message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": toAnthropicReason(reason)},
		"usage": map[string]any{"output_tokens": out},
	}); err != nil {
		return err
	}
	return writeSSE(m.w, "message_stop", map[string]any{"type": "message_stop"})
}

// toAnthropicReason maps the canonical reason back to Anthropic's stop_reason.
func toAnthropicReason(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	case "error":
		return "stop"
	}
	return reason
}

// ---- OpenAI Responses SSE ----

type responsesWriter struct {
	w           io.Writer
	model       string
	id          string
	created     int64
	assistantID string
	textOpen    bool
	accText     string
	started     bool
	done        bool
	output      []any
}

// NewResponsesWriter emits an OpenAI Responses SSE stream.
func NewResponsesWriter(w io.Writer, model string) Writer {
	return &responsesWriter{w: w, model: model, id: "resp_" + randIDSuffix(), created: nowUnix()}
}

func (r *responsesWriter) start() error {
	if r.started {
		return nil
	}
	r.started = true
	return writeSSE(r.w, "response.created", map[string]any{
		"type": "response.created",
		"response": map[string]any{
			"id": r.id, "object": "response", "created": r.created, "model": r.model,
			"status": "in_progress", "output": []any{},
			"usage": map[string]any{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
		},
	})
}

func (r *responsesWriter) openMessage() error {
	if r.textOpen {
		return nil
	}
	if err := r.start(); err != nil {
		return err
	}
	r.assistantID = "msg_" + randIDSuffix()
	r.textOpen = true
	return writeSSE(r.w, "response.output_item.added", map[string]any{
		"type": "response.output_item.added",
		"output_index": 0,
		"item": map[string]any{
			"id": r.assistantID, "type": "message", "status": "in_progress", "role": "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": "", "annotations": []any{}}},
		},
	})
}

func (r *responsesWriter) closeMessage() error {
	if !r.textOpen {
		return nil
	}
	r.textOpen = false
	r.output = append(r.output, map[string]any{
		"id": r.assistantID, "type": "message", "status": "completed", "role": "assistant",
		"content": []any{map[string]any{"type": "output_text", "text": r.accText, "annotations": []any{}}},
	})
	return writeSSE(r.w, "response.output_item.done", map[string]any{
		"type": "response.output_item.done",
		"output_index": 0,
		"item": map[string]any{
			"id": r.assistantID, "type": "message", "status": "completed", "role": "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": r.accText, "annotations": []any{}}},
		},
	})
}

func (r *responsesWriter) Text(chunk string) error {
	if err := r.openMessage(); err != nil {
		return err
	}
	r.accText += chunk
	return writeSSE(r.w, "response.output_text.delta", map[string]any{
		"type": "response.output_text.delta",
		"item_id": r.assistantID,
		"output_index": 0,
		"content_index": 0,
		"delta": chunk,
	})
}

func (r *responsesWriter) ToolCall(tc ir.ToolCall) error {
	if err := r.start(); err != nil {
		return err
	}
	if err := r.closeMessage(); err != nil {
		return err
	}
	fcID := "fc_" + randIDSuffix()
	inProgress := map[string]any{
		"id": fcID, "type": "function_call", "status": "in_progress",
		"call_id": tc.ID, "name": tc.Name, "arguments": "",
	}
	if err := writeSSE(r.w, "response.output_item.added", map[string]any{
		"type": "response.output_item.added", "output_index": len(r.output), "item": inProgress,
	}); err != nil {
		return err
	}
	if string(tc.Input) != "" {
		if err := writeSSE(r.w, "response.function_call_arguments.delta", map[string]any{
			"type": "response.function_call_arguments.delta", "item_id": fcID, "output_index": len(r.output),
			"delta": string(tc.Input),
		}); err != nil {
			return err
		}
	}
	done := map[string]any{
		"id": fcID, "type": "function_call", "status": "completed",
		"call_id": tc.ID, "name": tc.Name, "arguments": string(tc.Input),
	}
	r.output = append(r.output, done)
	return writeSSE(r.w, "response.output_item.done", map[string]any{
		"type": "response.output_item.done", "output_index": len(r.output) - 1, "item": done,
	})
}

func (r *responsesWriter) Done(reason string, usage *ir.Usage) error {
	if r.done {
		return nil
	}
	r.done = true
	if err := r.start(); err != nil {
		return err
	}
	if err := r.closeMessage(); err != nil {
		return err
	}
	status := "completed"
	switch reason {
	case "length":
		status = "incomplete"
	case "error":
		status = "failed"
	}
	u := map[string]any{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
	if usage != nil {
		u = map[string]any{
			"input_tokens": usage.PromptTokens, "output_tokens": usage.CompletionTokens, "total_tokens": usage.TotalTokens,
		}
	}
	return writeSSE(r.w, "response.completed", map[string]any{
		"type": "response.completed",
		"response": map[string]any{
			"id": r.id, "object": "response", "created": r.created, "model": r.model,
			"status": status, "output": r.output, "usage": u,
		},
	})
}