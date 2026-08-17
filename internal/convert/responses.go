package convert

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/lewangdev/deepseek-v4-flash-vision/internal/ir"
)

// OpenAI Responses API <-> IR.

type responsesContent struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL string `json:"image_url"`
}

type responsesInputItem struct {
	Type      string             `json:"type"`
	Role      string             `json:"role"`
	Content   []responsesContent `json:"content,omitempty"`
	CallID    string             `json:"call_id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Arguments string             `json:"arguments,omitempty"`
	Output    string             `json:"output,omitempty"`
}

func responsesToolsToOpenAI(tools []json.RawMessage) []json.RawMessage {
	var out []json.RawMessage
	for _, t := range tools {
		var a struct {
			Type        string          `json:"type"`
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		}
		if err := json.Unmarshal(t, &a); err != nil || a.Name == "" {
			continue
		}
		def := newOpenAITool(a.Name, a.Description, a.Parameters)
		b, _ := json.Marshal(def)
		out = append(out, b)
	}
	return out
}

// ResponsesRequestToIR parses an OpenAI Responses API request.
func ResponsesRequestToIR(raw []byte) (*ir.Request, error) {
	var req struct {
		Model           string            `json:"model"`
		Instructions    json.RawMessage   `json:"instructions"`
		Input           json.RawMessage   `json:"input"`
		MaxOutputTokens *int              `json:"max_output_tokens"`
		Temperature     *float64          `json:"temperature"`
		TopP            *float64          `json:"top_p"`
		Stream          bool              `json:"stream"`
		Tools           []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("parse Responses request: %w", err)
	}
	out := &ir.Request{
		ClientModel: req.Model,
		MaxTokens:   req.MaxOutputTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Tools:       responsesToolsToOpenAI(req.Tools),
	}
	if ins := responsesInstructionsToParts(req.Instructions); ins != nil {
		out.Messages = append(out.Messages, ir.Message{Role: ir.RoleSystem, Parts: ins})
	}
	if err := appendResponsesInput(out, req.Input); err != nil {
		return nil, err
	}
	return out, nil
}

func responsesInstructionsToParts(raw json.RawMessage) []ir.ContentPart {
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
	var blocks []responsesContent
	if err := json.Unmarshal(raw, &blocks); err == nil {
		for _, b := range blocks {
			parts = append(parts, ir.TextPart{Text: b.Text})
		}
	}
	return parts
}

// appendResponsesInput handles `input` being a plain string or an item array.
func appendResponsesInput(out *ir.Request, raw json.RawMessage) error {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single != "" {
			out.Messages = append(out.Messages, ir.Message{
				Role:  ir.RoleUser,
				Parts: []ir.ContentPart{ir.TextPart{Text: single}},
			})
		}
		return nil
	}
	var items []responsesInputItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return fmt.Errorf("invalid input array: %w", err)
	}
	for _, it := range items {
		switch it.Type {
		case "message":
			var msg ir.Message
			switch it.Role {
			case "user":
				msg.Role = ir.RoleUser
			case "assistant":
				msg.Role = ir.RoleAssistant
			default:
				msg.Role = ir.RoleUser
			}
			for _, c := range it.Content {
				switch c.Type {
				case "input_text":
					msg.Parts = append(msg.Parts, ir.TextPart{Text: c.Text})
				case "input_image":
					if img, ok := imageFromURL(c.ImageURL); ok {
						msg.Parts = append(msg.Parts, img)
					}
				}
			}
			out.Messages = append(out.Messages, msg)
		case "function_call":
			out.Messages = append(out.Messages, ir.Message{
				Role: ir.RoleAssistant,
				Parts: []ir.ContentPart{ir.ToolUsePart{
					ID: it.CallID, Name: it.Name, Input: json.RawMessage(it.Arguments),
				}},
			})
		case "function_call_output":
			out.Messages = append(out.Messages, ir.Message{
				Role: ir.RoleTool,
				Parts: []ir.ContentPart{ir.ToolResultPart{
					ToolUseID: it.CallID, Content: it.Output,
				}},
			})
		}
	}
	return nil
}

// ResponsesRequestFromIR serializes an IR request as an OpenAI Responses body.
func ResponsesRequestFromIR(r *ir.Request) ([]byte, error) {
	body := map[string]any{"model": r.Model}

	var instructions []string
	input := make([]any, 0, len(r.Messages))
	for _, m := range r.Messages {
		if m.Role == ir.RoleSystem {
			instructions = append(instructions, m.Text())
			continue
		}
		switch m.Role {
		case ir.RoleTool:
			for _, p := range m.Parts {
				if tr, ok := p.(ir.ToolResultPart); ok {
					input = append(input, map[string]any{
						"type": "function_call_output", "call_id": tr.ToolUseID, "output": tr.Content,
					})
				}
			}
		default:
			item := map[string]any{"type": "message", "role": string(m.Role)}
			content := make([]any, 0, len(m.Parts))
			for _, p := range m.Parts {
				switch v := p.(type) {
				case ir.TextPart:
					content = append(content, map[string]any{"type": "input_text", "text": v.Text})
				case ir.ImagePart:
					url := v.URL
					if url == "" {
						url = dataURI(v.MediaType, v.Data)
					}
					content = append(content, map[string]any{"type": "input_image", "image_url": url})
				case ir.ToolUsePart:
					input = append(input, map[string]any{
						"type": "function_call", "call_id": v.ID, "name": v.Name,
						"arguments": string(v.Input),
					})
				}
			}
			item["content"] = content
			input = append(input, item)
		}
	}
	if len(instructions) > 0 {
		if len(instructions) == 1 {
			body["instructions"] = instructions[0]
		} else {
			body["instructions"] = instructions
		}
	}
	if len(input) > 0 {
		body["input"] = input
	}
	if r.MaxTokens != nil {
		body["max_output_tokens"] = *r.MaxTokens
	}
	if r.Temperature != nil {
		body["temperature"] = *r.Temperature
	}
	if r.TopP != nil {
		body["top_p"] = *r.TopP
	}
	if r.Stream {
		body["stream"] = true
	}
	if tools := responsesToolsFromOpenAI(r.Tools); len(tools) > 0 {
		body["tools"] = tools
	}
	return json.Marshal(body)
}

// responsesToolsFromOpenAI renders canonical tool definitions in Responses form.
func responsesToolsFromOpenAI(tools []json.RawMessage) []any {
	var out []any
	for _, t := range tools {
		var def openAIToolDef
		if err := json.Unmarshal(t, &def); err != nil || def.Function.Name == "" {
			continue
		}
		params := def.Function.Parameters
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object"}`)
		}
		out = append(out, map[string]any{
			"type": "function", "name": def.Function.Name,
			"description": def.Function.Description, "parameters": params, "strict": false,
		})
	}
	return out
}

type responsesUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// ResponsesResponseToIR parses a (non-streaming) OpenAI Responses response.
func ResponsesResponseToIR(raw []byte) (*ir.Response, error) {
	var resp struct {
		Model  string `json:"model"`
		Status string `json:"status"`
		Output []struct {
			Type      string             `json:"type"`
			Content   []responsesContent `json:"content"`
			CallID    string             `json:"call_id"`
			Name      string             `json:"name"`
			Arguments string             `json:"arguments"`
		} `json:"output"`
		Usage *responsesUsage `json:"usage"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, fmt.Errorf("parse Responses response: %w", err)
	}
	out := &ir.Response{FinishReason: "stop"}
	for _, item := range resp.Output {
		switch item.Type {
		case "message", "reasoning":
			for _, c := range item.Content {
				if c.Type == "output_text" || c.Type == "text" {
					out.Text += c.Text
				}
			}
		case "function_call":
			id := item.CallID
			if id == "" {
				id = item.Name
			}
			out.ToolCalls = append(out.ToolCalls, ir.ToolCall{
				ID: id, Name: item.Name, Input: json.RawMessage(item.Arguments),
			})
		}
	}
	switch resp.Status {
	case "completed":
		out.FinishReason = "stop"
	case "incomplete":
		out.FinishReason = "length"
	case "failed":
		out.FinishReason = "error"
	}
	if resp.Usage != nil {
		out.Usage = &ir.Usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}
	return out, nil
}

// ResponsesResponseFromIR serializes an IR response as an OpenAI Responses
// (non-streaming) object.
func ResponsesResponseFromIR(resp *ir.Response, model string) ([]byte, error) {
	output := make([]any, 0, len(resp.ToolCalls)+1)
	if resp.Text != "" {
		output = append(output, map[string]any{
			"id": newID("msg"), "type": "message", "status": "completed", "role": "assistant",
			"content": []any{map[string]any{"type": "output_text", "text": resp.Text, "annotations": []any{}}},
		})
	}
	for _, tc := range resp.ToolCalls {
		output = append(output, map[string]any{
			"id": newID("fc"), "type": "function_call", "status": "completed",
			"call_id": tc.ID, "name": tc.Name, "arguments": string(tc.Input),
		})
	}
	body := map[string]any{
		"id": newID("resp"), "object": "response", "created": nowUnix(), "model": model,
		"status": "completed", "output": output,
	}
	if resp.Usage != nil {
		body["usage"] = map[string]any{
			"input_tokens":  resp.Usage.PromptTokens,
			"output_tokens": resp.Usage.CompletionTokens,
			"total_tokens":  resp.Usage.TotalTokens,
		}
	}
	return json.Marshal(body)
}
