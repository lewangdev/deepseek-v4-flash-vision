package convert

import (
	"encoding/json"
	"testing"

	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

func TestAnthropicRequestImageAndSystem(t *testing.T) {
	body := `{
		"model":"qwen3.7-max",
		"system":"你是助手",
		"messages":[
			{"role":"user","content":[
				{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"AAAA"}},
				{"type":"text","text":"描述这张图"}
			]}
		]
	}`
	req, err := AnthropicRequestToIR([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.HasImage() {
		t.Fatal("expected image detected")
	}
	if len(req.Messages) != 2 {
		t.Fatalf("want system + user, got %d messages", len(req.Messages))
	}
	if req.Messages[0].Role != ir.RoleSystem {
		t.Fatalf("first message role = %q, want system", req.Messages[0].Role)
	}
	img, ok := req.Messages[1].Parts[0].(ir.ImagePart)
	if !ok {
		t.Fatalf("want ImagePart, got %T", req.Messages[1].Parts[0])
	}
	if img.MediaType != "image/jpeg" {
		t.Errorf("media type = %q", img.MediaType)
	}
}

func TestAnthropicBuildDefaultsAndMerges(t *testing.T) {
	toolJSON := `{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object"}}}`
	req := &ir.Request{
		Model: "qwen3.7-max",
		Messages: []ir.Message{
			{Role: ir.RoleSystem, Parts: []ir.ContentPart{ir.TextPart{Text: "你是助手"}}},
			{Role: ir.RoleUser, Parts: []ir.ContentPart{ir.TextPart{Text: "a"}}},
			{Role: ir.RoleUser, Parts: []ir.ContentPart{ir.TextPart{Text: "b"}}}, // consecutive same role
			{Role: ir.RoleAssistant, Parts: []ir.ContentPart{ir.TextPart{Text: "hi"}, ir.ToolUsePart{ID: "t1", Name: "get_weather", Input: json.RawMessage(`{"city":"sh"}`)}}},
			{Role: ir.RoleTool, Parts: []ir.ContentPart{ir.ToolResultPart{ToolUseID: "t1", Content: "sunny"}}},
		},
		Tools: []json.RawMessage{json.RawMessage(toolJSON)},
	}
	out, err := AnthropicRequestFromIR(req, 2048)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	var decoded struct {
		Model      string `json:"model"`
		MaxTokens  int    `json:"max_tokens"`
		System     string `json:"system"`
		Messages   []struct {
			Role    string `json:"role"`
			Content []struct {
				Type       string `json:"type"`
				Text       string `json:"text"`
				ToolUseID  string `json:"tool_use_id,omitempty"`
				Input      any    `json:"input,omitempty"`
			} `json:"content"`
		} `json:"messages"`
		Tools []struct {
			Name        string `json:"name"`
			InputSchema any    `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("unmarshal built body: %v", err)
	}
	if decoded.Model != "qwen3.7-max" {
		t.Errorf("model = %q", decoded.Model)
	}
	if decoded.MaxTokens != 2048 {
		t.Errorf("max_tokens = %d, want default 2048", decoded.MaxTokens)
	}
	if decoded.System != "你是助手" {
		t.Errorf("system = %q", decoded.System)
	}
	if len(decoded.Messages) != 3 {
		t.Fatalf("messages = %d, want 3 (merged a+b, assistant, tool_result)", len(decoded.Messages))
	}
	// messages[0] user content should merge "ab".
	if len(decoded.Messages[0].Content) != 1 || decoded.Messages[0].Content[0].Text != "ab" {
		t.Errorf("merged user content = %+v", decoded.Messages[0].Content)
	}
	if len(decoded.Messages[1].Content) != 2 {
		t.Errorf("assistant content blocks = %d", len(decoded.Messages[1].Content))
	}
	if decoded.Messages[2].Role != "user" || decoded.Messages[2].Content[0].ToolUseID != "t1" {
		t.Errorf("tool_result block = %+v", decoded.Messages[2])
	}
	if len(decoded.Tools) != 1 || decoded.Tools[0].Name != "get_weather" {
		t.Errorf("tools = %+v", decoded.Tools)
	}
}

func TestAnthropicResponseRoundTrip(t *testing.T) {
	raw := `{
		"id":"msg_1","type":"message","role":"assistant","model":"qwen3.7-max",
		"content":[{"type":"text","text":"42"}],
		"stop_reason":"end_turn",
		"usage":{"input_tokens":5,"output_tokens":2}
	}`
	resp, err := AnthropicResponseToIR([]byte(raw))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Text != "42" || resp.FinishReason != "stop" {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 5 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	out, err := AnthropicResponseFromIR(resp, "qwen3.7-max")
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if !json.Valid(out) {
		t.Fatal("built body is not valid JSON")
	}
}