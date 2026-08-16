package streamconv

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

// All three parsers emit live text deltas through fn, and exactly one final
// Delta (after the stream ends) carrying the accumulated complete ToolCalls,
// the normalized FinishReason and any Usage. Tool-call arguments are buffered
// whole rather than forwarded incrementally across families (see package doc).

// ParseChat parses an OpenAI chat.completion.chunk SSE stream.
func ParseChat(r io.Reader, fn func(Delta)) error {
	s := newSSE(r)
	calls := map[int]*toolAcc{}
	var order []int
	var finish *string
	var usage *ir.Usage
	for {
		event, data, err := s.next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if event != "" {
			continue
		}
		if strings.TrimSpace(data) == "[DONE]" {
			break
		}
		var p struct {
			Choices []struct {
				Delta struct {
					Content   string `json:"content"`
					Reasoning string `json:"reasoning_content"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Usage *upstreamUsage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &p); err != nil {
			continue
		}
		if len(p.Choices) > 0 {
			ch := p.Choices[0]
			text := ch.Delta.Content
			if text == "" {
				text = ch.Delta.Reasoning
			}
			for _, tc := range ch.Delta.ToolCalls {
				a, ok := calls[tc.Index]
				if !ok {
					a = &toolAcc{}
					calls[tc.Index] = a
					order = append(order, tc.Index)
				}
				if tc.ID != "" {
					a.id = tc.ID
				}
				if tc.Function.Name != "" {
					a.name = tc.Function.Name
				}
				a.args += tc.Function.Arguments
			}
			if text != "" {
				fn(Delta{Text: text})
			}
			if ch.FinishReason != nil {
				r := normalizeReason(*ch.FinishReason)
				finish = &r
			}
		}
		if p.Usage != nil {
			usage = p.Usage.toIR()
		}
	}
	fn(Delta{ToolCalls: buildCalls(order, calls), FinishReason: finish, Usage: usage})
	return nil
}

// ParseMessages parses an Anthropic Messages SSE stream.
func ParseMessages(r io.Reader, fn func(Delta)) error {
	s := newSSE(r)
	calls := map[int]*toolAcc{}
	var order []int
	var finish *string
	var usage *ir.Usage
	for {
		event, data, err := s.next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		switch event {
		case "message_start":
			var p struct {
				Message struct {
					Usage *upstreamUsage `json:"usage"`
				} `json:"message"`
			}
			if json.Unmarshal([]byte(data), &p) == nil && p.Message.Usage != nil {
				usage = p.Message.Usage.toIR()
			}
		case "content_block_start":
			var p struct {
				Index int `json:"index"`
				ContentBlock struct {
					Type string `json:"type"`
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"content_block"`
			}
			if json.Unmarshal([]byte(data), &p) == nil && p.ContentBlock.Type == "tool_use" {
				if _, ok := calls[p.Index]; !ok {
					calls[p.Index] = &toolAcc{id: p.ContentBlock.ID, name: p.ContentBlock.Name}
					order = append(order, p.Index)
				}
			}
		case "content_block_delta":
			var p struct {
				Index int `json:"index"`
				Delta struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
			}
			if json.Unmarshal([]byte(data), &p) != nil {
				continue
			}
			switch p.Delta.Type {
			case "text_delta":
				if p.Delta.Text != "" {
					fn(Delta{Text: p.Delta.Text})
				}
			case "input_json_delta":
				if a := calls[p.Index]; a != nil {
					a.args += p.Delta.PartialJSON
				}
			}
		case "message_delta":
			var p struct {
				Delta struct {
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
				Usage *upstreamUsage `json:"usage"`
			}
			if json.Unmarshal([]byte(data), &p) != nil {
				continue
			}
			if p.Delta.StopReason != "" {
				r := normalizeReason(p.Delta.StopReason)
				finish = &r
			}
			if p.Usage != nil {
				if usage == nil {
					usage = &ir.Usage{}
				}
				u := p.Usage.toIR()
				usage.CompletionTokens = u.CompletionTokens
				usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
			}
		}
	}
	fn(Delta{ToolCalls: buildCalls(order, calls), FinishReason: finish, Usage: usage})
	return nil
}

// ParseResponses parses an OpenAI Responses SSE stream.
func ParseResponses(r io.Reader, fn func(Delta)) error {
	s := newSSE(r)
	calls := map[string]*toolAcc{}
	var order []string
	var finish *string
	var usage *ir.Usage
	for {
		event, data, err := s.next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		switch event {
		case "response.output_text.delta":
			var p struct {
				Delta string `json:"delta"`
			}
			if json.Unmarshal([]byte(data), &p) == nil && p.Delta != "" {
				fn(Delta{Text: p.Delta})
			}
		case "response.output_item.added", "response.output_item.done":
			var p struct {
				Item struct {
					Type      string `json:"type"`
					CallID    string `json:"call_id"`
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"item"`
			}
			if json.Unmarshal([]byte(data), &p) != nil {
				continue
			}
			it := p.Item
			if it.Type == "function_call" && it.CallID != "" {
				a, ok := calls[it.CallID]
				if !ok {
					a = &toolAcc{id: it.CallID}
					calls[it.CallID] = a
					order = append(order, it.CallID)
				}
				if it.Name != "" {
					a.name = it.Name
				}
				if it.Arguments != "" {
					a.args = it.Arguments
				}
			}
		case "response.function_call_arguments.delta":
			var p struct {
				CallID string `json:"call_id"`
				Delta  string `json:"delta"`
			}
			if json.Unmarshal([]byte(data), &p) == nil {
				if a := calls[p.CallID]; a != nil {
					a.args += p.Delta
				}
			}
		case "response.completed":
			var p struct {
				Response struct {
					Status string         `json:"status"`
					Usage  *upstreamUsage `json:"usage"`
				} `json:"response"`
			}
			if json.Unmarshal([]byte(data), &p) != nil {
				continue
			}
			r := normalizeReason(responsesStatusToReason(p.Response.Status))
			finish = &r
			if p.Response.Usage != nil {
				usage = p.Response.Usage.toIR()
			}
		}
	}
	fn(Delta{ToolCalls: buildCallsByID(order, calls), FinishReason: finish, Usage: usage})
	return nil
}

func responsesStatusToReason(status string) string {
	switch status {
	case "incomplete":
		return "length"
	case "failed":
		return "error"
	default:
		return "stop"
	}
}