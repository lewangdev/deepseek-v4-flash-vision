package convert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lewangdev/deepseek-v4-flash-vision/internal/ir"
)

// OpenAI Chat Completions <-> IR.

type openAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openAIMsg struct {
	Role       string           `json:"role"`
	Content    json.RawMessage  `json:"content"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// OpenAIRequestToIR parses an OpenAI Chat Completions request.
func OpenAIRequestToIR(raw []byte) (*ir.Request, error) {
	var req struct {
		Model               string            `json:"model"`
		Messages            []openAIMsg       `json:"messages"`
		MaxTokens           *int              `json:"max_tokens"`
		MaxCompletionTokens *int              `json:"max_completion_tokens"`
		Temperature         *float64          `json:"temperature"`
		TopP                *float64          `json:"top_p"`
		Stop                json.RawMessage   `json:"stop"`
		Stream              bool              `json:"stream"`
		Tools               []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("parse OpenAI request: %w", err)
	}
	out := &ir.Request{
		ClientModel: req.Model,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Tools:       req.Tools,
	}
	if out.MaxTokens == nil {
		out.MaxTokens = req.MaxCompletionTokens
	}
	out.Stop = parseStopNormalizes(req.Stop)
	for _, m := range req.Messages {
		im, err := openAIMsgToIR(m)
		if err != nil {
			return nil, err
		}
		out.Messages = append(out.Messages, im)
	}
	return out, nil
}

func openAIMsgToIR(m openAIMsg) (ir.Message, error) {
	role := ir.Role(m.Role)
	msg := ir.Message{Role: role}

	if role == ir.RoleTool {
		var text string
		_ = json.Unmarshal(m.Content, &text)
		msg.Parts = append(msg.Parts, ir.ToolResultPart{ToolUseID: m.ToolCallID, Content: text})
		return msg, nil
	}

	if err := appendOpenAIContent(&msg, m.Content); err != nil {
		return msg, err
	}
	for _, tc := range m.ToolCalls {
		msg.Parts = append(msg.Parts, ir.ToolUsePart{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		})
	}
	return msg, nil
}

func appendOpenAIContent(msg *ir.Message, raw json.RawMessage) error {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return err
		}
		if s != "" {
			msg.Parts = append(msg.Parts, ir.TextPart{Text: s})
		}
		return nil
	}
	var items []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(raw, &items); err != nil {
		return fmt.Errorf("invalid content array: %w", err)
	}
	for _, it := range items {
		switch it.Type {
		case "text":
			msg.Parts = append(msg.Parts, ir.TextPart{Text: it.Text})
		case "image_url", "input_image", "input_image_url":
			if it.ImageURL.URL == "" {
				continue
			}
			if img, ok := imageFromURL(it.ImageURL.URL); ok {
				msg.Parts = append(msg.Parts, img)
			}
		}
	}
	return nil
}

// OpenAIRequestFromIR serializes an IR request as an OpenAI Chat Completions body.
// The upstream model must already be resolved on req.Model.
func OpenAIRequestFromIR(r *ir.Request) ([]byte, error) {
	msgs := make([]any, 0, len(r.Messages))
	for _, m := range r.Messages {
		msg, err := openAIMsgFromIR(m)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, msg)
	}
	body := map[string]any{
		"model":    r.Model,
		"messages": msgs,
	}
	if r.MaxTokens != nil {
		body["max_tokens"] = *r.MaxTokens
	}
	if r.Temperature != nil {
		body["temperature"] = *r.Temperature
	}
	if r.TopP != nil {
		body["top_p"] = *r.TopP
	}
	if len(r.Stop) > 0 {
		body["stop"] = r.Stop
	}
	if r.Stream {
		body["stream"] = true
	}
	if len(r.Tools) > 0 {
		body["tools"] = r.Tools
	}
	return json.Marshal(body)
}

func openAIMsgFromIR(m ir.Message) (any, error) {
	if m.Role == ir.RoleTool {
		text := ""
		toolID := ""
		for _, p := range m.Parts {
			if t, ok := p.(ir.ToolResultPart); ok {
				text += t.Content
				if toolID == "" {
					toolID = t.ToolUseID
				}
			}
		}
		return map[string]any{"role": "tool", "tool_call_id": toolID, "content": text}, nil
	}

	var textParts []string
	var imageParts []map[string]any
	var toolCalls []any
	for _, p := range m.Parts {
		switch v := p.(type) {
		case ir.TextPart:
			textParts = append(textParts, v.Text)
		case ir.ImagePart:
			url := v.URL
			if url == "" {
				url = dataURI(v.MediaType, v.Data)
			}
			imageParts = append(imageParts, map[string]any{
				"type": "image_url", "image_url": map[string]any{"url": url},
			})
		case ir.ToolUsePart:
			toolCalls = append(toolCalls, map[string]any{
				"id": v.ID, "type": "function",
				"function": map[string]any{"name": v.Name, "arguments": string(v.Input)},
			})
		}
	}

	msg := map[string]any{"role": string(m.Role)}
	if len(imageParts) > 0 {
		content := make([]any, 0, len(textParts)+len(imageParts))
		for _, t := range textParts {
			content = append(content, map[string]any{"type": "text", "text": t})
		}
		for _, im := range imageParts {
			content = append(content, im)
		}
		msg["content"] = content
	} else if len(textParts) > 0 {
		msg["content"] = strings.Join(textParts, "")
	} else {
		msg["content"] = ""
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	return msg, nil
}

// OpenAIResponseToIR parses a (non-streaming) OpenAI Chat Completions response.
func OpenAIResponseToIR(raw []byte) (*ir.Response, error) {
	var resp struct {
		Choices []struct {
			Message struct {
				Content   string           `json:"content"`
				ToolCalls []openAIToolCall `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *usageJSON `json:"usage"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("parse OpenAI response: %w", err)
	}
	out := &ir.Response{FinishReason: "stop"}
	if len(resp.Choices) > 0 {
		m := resp.Choices[0].Message
		out.Text = m.Content
		for _, tc := range m.ToolCalls {
			out.ToolCalls = append(out.ToolCalls, ir.ToolCall{
				ID: tc.ID, Name: tc.Function.Name, Input: json.RawMessage(tc.Function.Arguments),
			})
		}
		if resp.Choices[0].FinishReason != "" {
			out.FinishReason = resp.Choices[0].FinishReason
		}
	}
	if resp.Usage != nil {
		u := resp.Usage.toIR()
		out.Usage = &ir.Usage{
			PromptTokens:     u.PromptTokens,
			CompletionTokens: u.CompletionTokens,
			TotalTokens:      u.TotalTokens,
		}
	}
	return out, nil
}

// OpenAIResponseFromIR serializes an IR response as an OpenAI Chat Completions
// (non-streaming) object. model is the client-facing model name to echo.
func OpenAIResponseFromIR(resp *ir.Response, model string) ([]byte, error) {
	msg := map[string]any{"role": "assistant"}
	if resp.Text != "" {
		msg["content"] = resp.Text
	} else {
		msg["content"] = nil
	}
	var toolCalls []any
	for _, tc := range resp.ToolCalls {
		toolCalls = append(toolCalls, map[string]any{
			"id": tc.ID, "type": "function",
			"function": map[string]any{"name": tc.Name, "arguments": string(tc.Input)},
		})
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	fr := resp.FinishReason
	if fr == "" {
		fr = "stop"
	}
	body := map[string]any{
		"id": newID("chatcmpl"), "object": "chat.completion", "created": nowUnix(),
		"model": model,
		"choices": []any{map[string]any{
			"index": 0, "message": msg, "finish_reason": fr,
		}},
	}
	if resp.Usage != nil {
		body["usage"] = map[string]any{
			"prompt_tokens":     resp.Usage.PromptTokens,
			"completion_tokens": resp.Usage.CompletionTokens,
			"total_tokens":      resp.Usage.TotalTokens,
		}
	}
	return json.Marshal(body)
}
