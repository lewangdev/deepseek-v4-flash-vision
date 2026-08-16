package convert

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

// Anthropic Messages <-> IR.

type anthropicMsgT struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthropicBlock struct {
	Type      string           `json:"type"`
	Text      string           `json:"text"`
	Source    *anthropicSource `json:"source,omitempty"`
	ID        string           `json:"id,omitempty"`
	Name      string           `json:"name,omitempty"`
	Input     json.RawMessage  `json:"input,omitempty"`
	ToolUseID string           `json:"tool_use_id,omitempty"`
	Content   string           `json:"content,omitempty"`
}

const anthropicVersion = "2023-06-01"

func anthropicToolsToOpenAI(tools []json.RawMessage) []json.RawMessage {
	var out []json.RawMessage
	for _, t := range tools {
		var a struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			InputSchema json.RawMessage `json:"input_schema"`
		}
		if err := json.Unmarshal(t, &a); err != nil {
			continue
		}
		def := newOpenAITool(a.Name, a.Description, a.InputSchema)
		b, _ := json.Marshal(def)
		out = append(out, b)
	}
	return out
}

// AnthropicRequestToIR parses an Anthropic Messages request.
func AnthropicRequestToIR(raw []byte) (*ir.Request, error) {
	var req struct {
		Model         string            `json:"model"`
		System        json.RawMessage   `json:"system"`
		Messages      []anthropicMsgT   `json:"messages"`
		MaxTokens     *int              `json:"max_tokens"`
		Temperature   *float64          `json:"temperature"`
		TopP          *float64          `json:"top_p"`
		StopSequences []string          `json:"stop_sequences"`
		Stream        bool              `json:"stream"`
		Tools         []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("parse Anthropic request: %w", err)
	}
	out := &ir.Request{
		ClientModel: req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stop:        req.StopSequences,
		Stream:      req.Stream,
		Tools:       anthropicToolsToOpenAI(req.Tools),
	}
	if sys := anthropicSystemToParts(req.System); sys != nil {
		out.Messages = append(out.Messages, ir.Message{Role: ir.RoleSystem, Parts: sys})
	}
	for _, m := range req.Messages {
		im, err := anthropicMsgToIR(m)
		if err != nil {
			return nil, err
		}
		out.Messages = append(out.Messages, im)
	}
	return out, nil
}

// anthropicSystemToParts parses the top-level `system` field (string or array
// of text blocks) into content parts, nil when absent.
func anthropicSystemToParts(raw json.RawMessage) []ir.ContentPart {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var parts []ir.ContentPart
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single != "" {
			parts = append(parts, ir.TextPart{Text: single})
		}
		return parts
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err == nil {
		for _, b := range blocks {
			parts = append(parts, ir.TextPart{Text: b.Text})
		}
	}
	return parts
}

func anthropicMsgToIR(m anthropicMsgT) (ir.Message, error) {
	msg := ir.Message{Role: ir.Role(m.Role)}
	raw := bytes.TrimSpace(m.Content)
	if len(raw) == 0 || string(raw) == "null" {
		return msg, nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return msg, err
		}
		msg.Parts = append(msg.Parts, ir.TextPart{Text: s})
		return msg, nil
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return msg, fmt.Errorf("invalid Anthropic content: %w", err)
	}
	for _, b := range blocks {
		switch b.Type {
		case "text":
			msg.Parts = append(msg.Parts, ir.TextPart{Text: b.Text})
		case "image":
			if b.Source == nil {
				continue
			}
			switch b.Source.Type {
			case "base64":
				data, _ := base64.StdEncoding.DecodeString(b.Source.Data)
				msg.Parts = append(msg.Parts, ir.ImagePart{MediaType: b.Source.MediaType, Data: data})
			case "url":
				msg.Parts = append(msg.Parts, ir.ImagePart{MediaType: b.Source.MediaType, URL: b.Source.URL})
			}
		case "tool_use":
			msg.Parts = append(msg.Parts, ir.ToolUsePart{ID: b.ID, Name: b.Name, Input: b.Input})
		case "tool_result":
			msg.Parts = append(msg.Parts, ir.ToolResultPart{ToolUseID: b.ToolUseID, Content: b.Content})
		}
	}
	return msg, nil
}

// AnthropicRequestFromIR serializes an IR request as an Anthropic Messages body.
// max_tokens is mandatory on this wire format, so a missing value from the client
// is filled from defaultMaxTokens.
func AnthropicRequestFromIR(r *ir.Request, defaultMaxTokens int) ([]byte, error) {
	mt := defaultMaxTokens
	if r.MaxTokens != nil && *r.MaxTokens > 0 {
		mt = *r.MaxTokens
	}

	var systemParts []ir.ContentPart
	var msgs []ir.Message
	for _, m := range r.Messages {
		if m.Role == ir.RoleSystem {
			systemParts = append(systemParts, m.Parts...)
		} else {
			msgs = append(msgs, m)
		}
	}
	msgs = mergeConsecutiveSameRole(msgs)

	body := map[string]any{
		"model":      r.Model,
		"max_tokens": mt,
		"messages":   anthropicMessagesFromIR(msgs),
	}
	if len(systemParts) > 0 {
		body["system"] = anthropicSystemFromParts(systemParts)
	}
	// Anthropic rejects temperature and top_p set together; prefer temperature.
	if r.Temperature != nil {
		body["temperature"] = *r.Temperature
	} else if r.TopP != nil {
		body["top_p"] = *r.TopP
	}
	if len(r.Stop) > 0 {
		body["stop_sequences"] = r.Stop
	}
	if r.Stream {
		body["stream"] = true
	}
	if tools := anthropicToolsFromOpenAI(r.Tools); len(tools) > 0 {
		body["tools"] = tools
	}
	return json.Marshal(body)
}

func anthropicSystemFromParts(parts []ir.ContentPart) any {
	var b strings.Builder
	for _, p := range parts {
		if t, ok := p.(ir.TextPart); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}

// anthropicMessagesFromIR renders IR messages as Anthropic content arrays.
// Tool results are wrapped in a user message per the Anthropic protocol.
func anthropicMessagesFromIR(msgs []ir.Message) []any {
	out := make([]any, 0, len(msgs))
	for _, m := range msgs {
		role := string(m.Role)
		content := make([]map[string]any, 0, len(m.Parts))
		for _, p := range m.Parts {
			switch v := p.(type) {
			case ir.TextPart:
				content = append(content, map[string]any{"type": "text", "text": v.Text})
			case ir.ImagePart:
				if v.URL != "" {
					content = append(content, map[string]any{
						"type": "image",
						"source": map[string]any{"type": "url", "url": v.URL},
					})
				} else {
					data := base64.StdEncoding.EncodeToString(v.Data)
					content = append(content, map[string]any{
						"type": "image",
						"source": map[string]any{
							"type": "base64", "media_type": v.MediaType, "data": data,
						},
					})
				}
			case ir.ToolUsePart:
				input := v.Input
				if len(input) == 0 || string(input) == "null" {
					input = json.RawMessage(`{}`)
				}
				content = append(content, map[string]any{"type": "tool_use", "id": v.ID, "name": v.Name, "input": input})
			case ir.ToolResultPart:
				content = append(content, map[string]any{"type": "tool_result", "tool_use_id": v.ToolUseID, "content": v.Content})
			}
		}
		if role == string(ir.RoleTool) {
			role = string(ir.RoleUser)
		}
		out = append(out, map[string]any{"role": role, "content": content})
	}
	return out
}

// mergeConsecutiveSameRole merges adjacent same-role messages (Anthropic
// requires strict user/assistant alternation), except the leading system role
// which has already been extracted. Text-only messages are concatenated into a
// single text block for compactness.
func mergeConsecutiveSameRole(msgs []ir.Message) []ir.Message {
	var out []ir.Message
	for _, m := range msgs {
		if n := len(out); n > 0 && out[n-1].Role == m.Role {
			if textOnly(out[n-1]) && textOnly(m) {
				out[n-1].Parts[0] = ir.TextPart{Text: out[n-1].Text() + m.Text()}
			} else {
				out[n-1].Parts = append(out[n-1].Parts, m.Parts...)
			}
			continue
		}
		out = append(out, m)
	}
	return out
}

func textOnly(m ir.Message) bool {
	if len(m.Parts) == 0 {
		return false
	}
	for _, p := range m.Parts {
		if _, ok := p.(ir.TextPart); !ok {
			return false
		}
	}
	return true
}

// anthropicToolsFromOpenAI renders canonical OpenAI tool definitions as
// Anthropic tool definitions.
func anthropicToolsFromOpenAI(tools []json.RawMessage) []any {
	var out []any
	for _, t := range tools {
		var def openAIToolDef
		if err := json.Unmarshal(t, &def); err != nil || def.Function.Name == "" {
			continue
		}
		schema := def.Function.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object"}`)
		}
		out = append(out, map[string]any{
			"name":        def.Function.Name,
			"description": def.Function.Description,
			"input_schema": schema,
		})
	}
	return out
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// AnthropicResponseToIR parses a (non-streaming) Anthropic Messages response.
func AnthropicResponseToIR(raw []byte) (*ir.Response, error) {
	var resp struct {
		Content    []anthropicBlock `json:"content"`
		StopReason string           `json:"stop_reason"`
		Usage      *anthropicUsage  `json:"usage"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("parse Anthropic response: %w", err)
	}
	out := &ir.Response{FinishReason: "stop"}
	for _, b := range resp.Content {
		switch b.Type {
		case "text":
			out.Text += b.Text
		case "tool_use":
			out.ToolCalls = append(out.ToolCalls, ir.ToolCall{ID: b.ID, Name: b.Name, Input: b.Input})
		}
	}
	if resp.StopReason != "" {
		out.FinishReason = anthropicStopReasonToOpenAI(resp.StopReason)
	}
	if resp.Usage != nil {
		out.Usage = &ir.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		}
	}
	return out, nil
}

func anthropicStopReasonToOpenAI(s string) string {
	switch s {
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	default:
		return "stop"
	}
}

// AnthropicResponseFromIR serializes an IR response as an Anthropic Messages
// (non-streaming) object.
func AnthropicResponseFromIR(resp *ir.Response, model string) ([]byte, error) {
	content := make([]map[string]any, 0, len(resp.ToolCalls)+1)
	if resp.Text != "" {
		content = append(content, map[string]any{"type": "text", "text": resp.Text})
	}
	for _, tc := range resp.ToolCalls {
		content = append(content, map[string]any{"type": "tool_use", "id": tc.ID, "name": tc.Name, "input": tc.Input})
	}
	body := map[string]any{
		"id":              newID("msg"),
		"type":            "message",
		"role":            "assistant",
		"model":           model,
		"content":         content,
		"stop_reason":     resp.FinishReason,
		"stop_sequence":   nil,
		"usage":           map[string]any{"input_tokens": 0, "output_tokens": 0},
	}
	if resp.FinishReason == "" {
		body["stop_reason"] = "end_turn"
	}
	if resp.Usage != nil {
		body["usage"] = map[string]any{
			"input_tokens":  resp.Usage.PromptTokens,
			"output_tokens": resp.Usage.CompletionTokens,
		}
	}
	return json.Marshal(body)
}

// AnthropicVersion is the anthropic-version header value used on upstream calls.
func AnthropicVersion() string { return anthropicVersion }