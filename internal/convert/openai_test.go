package convert

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

func TestOpenAIRequestImageDataURI(t *testing.T) {
	png := base64.StdEncoding.EncodeToString([]byte("fake-png-bytes"))
	body := `{
		"model":"deepseek-v4-flash",
		"messages":[
			{"role":"user","content":[
				{"type":"text","text":"what is this?"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,` + png + `"}}
			]}
		],
		"max_tokens":100
	}`
	req, err := OpenAIRequestToIR([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.HasImage() {
		t.Fatal("expected image detected")
	}
	if req.MaxTokens == nil || *req.MaxTokens != 100 {
		t.Fatalf("max_tokens = %v", req.MaxTokens)
	}
	msg := req.Messages[0]
	if len(msg.Parts) != 2 {
		t.Fatalf("want 2 parts, got %d", len(msg.Parts))
	}
	img, ok := msg.Parts[1].(ir.ImagePart)
	if !ok {
		t.Fatalf("want ImagePart, got %T", msg.Parts[1])
	}
	if img.MediaType != "image/png" {
		t.Errorf("media type = %q", img.MediaType)
	}
	if string(img.Data) != "fake-png-bytes" {
		t.Errorf("payload = %q", img.Data)
	}
}

func TestOpenAIToolCallRoundTrip(t *testing.T) {
	body := `{
		"model":"deepseek-v4-flash",
		"messages":[
			{"role":"assistant","content":"","tool_calls":[
				{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"sh\"}"}}
			]},
			{"role":"tool","tool_call_id":"call_1","content":"sunny"}
		]
	}`
	req, err := OpenAIRequestToIR([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("want 2 messages, got %d", len(req.Messages))
	}
	if req.Messages[0].Role != ir.RoleAssistant {
		t.Fatalf("role = %q", req.Messages[0].Role)
	}
	if _, ok := req.Messages[0].Parts[0].(ir.ToolUsePart); !ok {
		t.Fatalf("want ToolUsePart, got %T", req.Messages[0].Parts[0])
	}
	if _, ok := req.Messages[1].Parts[0].(ir.ToolResultPart); !ok {
		t.Fatalf("want ToolResultPart, got %T", req.Messages[1].Parts[0])
	}

	req.Model = "deepseek-v4-flash"
	out, err := OpenAIRequestFromIR(req)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	var decoded struct {
		Messages []struct {
			Role       string `json:"role"`
			ToolCalls  []any  `json:"tool_calls"`
			ToolCallID string `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("unmarshal built body: %v", err)
	}
	if len(decoded.Messages) != 2 {
		t.Fatalf("built messages = %d", len(decoded.Messages))
	}
	if len(decoded.Messages[0].ToolCalls) != 1 {
		t.Fatalf("built tool_calls = %d", len(decoded.Messages[0].ToolCalls))
	}
	if decoded.Messages[1].ToolCallID != "call_1" {
		t.Errorf("tool_call_id = %q", decoded.Messages[1].ToolCallID)
	}
}

func TestOpenAIResponseRoundTrip(t *testing.T) {
	raw := `{
		"id":"chatcmpl-1","object":"chat.completion","model":"deepseek-v4-flash",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
	}`
	resp, err := OpenAIResponseToIR([]byte(raw))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Text != "hi" || resp.FinishReason != "stop" {
		t.Fatalf("resp = %+v", resp)
	}
	out, err := OpenAIResponseFromIR(resp, "deepseek-v4-flash")
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if !json.Valid(out) {
		t.Fatal("built body is not valid JSON")
	}
}