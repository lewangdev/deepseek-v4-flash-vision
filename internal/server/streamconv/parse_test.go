package streamconv

import (
	"strings"
	"testing"

	"github.com/lewangdev/deepseek-v4-flash-vision/internal/ir"
)

func collect(t *testing.T, fn func(func(Delta)) error) (text string, last Delta) {
	t.Helper()
	texts := []string{}
	var final Delta
	fn(func(d Delta) {
		if d.Text != "" {
			texts = append(texts, d.Text)
		}
		if len(d.ToolCalls) > 0 || d.FinishReason != nil || d.Usage != nil {
			final = d
		}
	})
	return strings.Join(texts, ""), final
}

func TestParseChatTextAndToolCall(t *testing.T) {
	src := `data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"f","arguments":"{\"x\":"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"!"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}

data: [DONE]

`
	text, final := collect(t, func(fn func(Delta)) error { return ParseChat(strings.NewReader(src), fn) })
	if text != "hello!" {
		t.Fatalf("text = %q", text)
	}
	if len(final.ToolCalls) != 1 {
		t.Fatalf("toolcalls = %+v", final.ToolCalls)
	}
	tc := final.ToolCalls[0]
	if tc.ID != "c1" || tc.Name != "f" || string(tc.Input) != `{"x":1}` {
		t.Fatalf("toolcall = %+v", tc)
	}
	if final.FinishReason == nil || *final.FinishReason != "tool_calls" {
		t.Fatalf("finish = %v", final.FinishReason)
	}
	if final.Usage == nil || final.Usage.TotalTokens != 5 {
		t.Fatalf("usage = %+v", final.Usage)
	}
}

func TestParseMessages(t *testing.T) {
	src := `event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"f","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`
	text, final := collect(t, func(fn func(Delta)) error { return ParseMessages(strings.NewReader(src), fn) })
	if text != "hi" {
		t.Fatalf("text = %q", text)
	}
	if len(final.ToolCalls) != 1 || final.ToolCalls[0].Name != "f" || string(final.ToolCalls[0].Input) != `{"a":1}` {
		t.Fatalf("toolcalls = %+v", final.ToolCalls)
	}
	if final.FinishReason == nil || *final.FinishReason != "stop" {
		t.Fatalf("finish = %v", final.FinishReason)
	}
	if final.Usage == nil || final.Usage.CompletionTokens != 2 {
		t.Fatalf("usage = %+v", final.Usage)
	}
}

func TestParseResponses(t *testing.T) {
	src := `event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"hello"}

event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"fc1","type":"function_call","call_id":"c1","name":"f","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","call_id":"c1","delta":"{\"x\":1}"}

event: response.completed
data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":4,"output_tokens":1,"total_tokens":5}}}

`
	text, final := collect(t, func(fn func(Delta)) error { return ParseResponses(strings.NewReader(src), fn) })
	if text != "hello" {
		t.Fatalf("text = %q", text)
	}
	want := []ir.ToolCall{{ID: "c1", Name: "f", Input: []byte(`{"x":1}`)}}
	if len(final.ToolCalls) != 1 || final.ToolCalls[0].ID != want[0].ID || string(final.ToolCalls[0].Input) != `{"x":1}` {
		t.Fatalf("toolcalls = %+v", final.ToolCalls)
	}
	if final.FinishReason == nil || *final.FinishReason != "stop" {
		t.Fatalf("finish = %v", final.FinishReason)
	}
}

func TestChatWriterDoneOnce(t *testing.T) {
	var sb strings.Builder
	w := NewChatWriter(&sb, "deepseek-v4-flash")
	if err := w.Text("hi"); err != nil {
		t.Fatal(err)
	}
	if err := w.Done("stop", nil); err != nil {
		t.Fatal(err)
	}
	if err := w.Done("stop", nil); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sb.String(), "[DONE]") {
		t.Fatalf("missing [DONE]: %q", sb.String())
	}
	if strings.Count(sb.String(), "[DONE]") != 1 {
		t.Fatalf("[DONE] appears %d times", strings.Count(sb.String(), "[DONE]"))
	}
}

func TestMessagesWriterStream(t *testing.T) {
	var sb strings.Builder
	w := NewMessagesWriter(&sb, "qwen3.7-max")
	_ = w.Text("hi")
	_ = w.ToolCall(ir.ToolCall{ID: "t1", Name: "f", Input: []byte(`{"x":1}`)})
	_ = w.Done("tool_calls", &ir.Usage{CompletionTokens: 3})
	for _, want := range []string{"message_start", "content_block_delta", "tool_use", "input_json_delta", "tool_use", "message_delta", `stop_reason":"tool_use"`, "message_stop"} {
		if !strings.Contains(sb.String(), want) {
			t.Fatalf("output missing %q: %q", want, sb.String())
		}
	}
}
