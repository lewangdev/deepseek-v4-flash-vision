package convert

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/lewangdev/deepseek-v4-flash-vision/internal/ir"
)

func TestResponsesRequestImageAndTools(t *testing.T) {
	png := base64.StdEncoding.EncodeToString([]byte("img"))
	body := `{
		"model":"gpt-5.6-luna",
		"instructions":"be brief",
		"input":[
			{"type":"message","role":"user","content":[
				{"type":"input_text","text":"看图"},
				{"type":"input_image","image_url":"data:image/png;base64,` + png + `"}
			]},
			{"type":"function_call","call_id":"fc1","name":"f","arguments":"{\"x\":1}"},
			{"type":"function_call_output","call_id":"fc1","output":"ok"}
		]
	}`
	req, err := ResponsesRequestToIR([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !req.HasImage() {
		t.Fatal("expected image detected")
	}
	if req.Messages[0].Role != ir.RoleSystem || req.Messages[0].Text() != "be brief" {
		t.Fatalf("instructions not parsed: %+v", req.Messages[0])
	}
	if req.Messages[1].Role != ir.RoleUser {
		t.Fatalf("role = %q", req.Messages[1].Role)
	}
	if req.Messages[2].Role != ir.RoleAssistant {
		t.Fatalf("function_call role = %q", req.Messages[2].Role)
	}
	if _, ok := req.Messages[3].Parts[0].(ir.ToolResultPart); !ok {
		t.Fatalf("function_call_output not parsed: %+v", req.Messages[3])
	}
}

func TestResponsesBuild(t *testing.T) {
	req := &ir.Request{
		Model: "gpt-5.6-luna",
		Messages: []ir.Message{
			{Role: ir.RoleSystem, Parts: []ir.ContentPart{ir.TextPart{Text: "sys"}}},
			{Role: ir.RoleUser, Parts: []ir.ContentPart{
				ir.TextPart{Text: "what"},
				ir.ImagePart{MediaType: "image/png", Data: []byte("img")},
			}},
			{Role: ir.RoleAssistant, Parts: []ir.ContentPart{ir.ToolUsePart{ID: "fc1", Name: "f", Input: json.RawMessage(`{"x":1}`)}}},
			{Role: ir.RoleTool, Parts: []ir.ContentPart{ir.ToolResultPart{ToolUseID: "fc1", Content: "ok"}}},
		},
	}
	out, err := ResponsesRequestFromIR(req)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	var decoded struct {
		Model        string `json:"model"`
		Instructions string `json:"instructions"`
		Input        []any  `json:"input"`
	}
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("unmarshal built body: %v", err)
	}
	if decoded.Instructions != "sys" {
		t.Errorf("instructions = %q", decoded.Instructions)
	}
	if len(decoded.Input) != 4 {
		t.Fatalf("input = %d items, want 4", len(decoded.Input))
	}
}

func TestResponsesResponseRoundTrip(t *testing.T) {
	raw := `{
		"id":"resp_1","object":"response","model":"gpt-5.6-luna","status":"completed",
		"output":[
			{"id":"m1","type":"message","status":"completed","role":"assistant",
			 "content":[{"type":"output_text","text":"hi"}]},
			{"id":"f1","type":"function_call","status":"completed","call_id":"fc1","name":"f","arguments":"{\"a\":2}"}
		],
		"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}
	}`
	resp, err := ResponsesResponseToIR([]byte(raw))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Text != "hi" {
		t.Errorf("text = %q", resp.Text)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "f" {
		t.Fatalf("toolcalls = %+v", resp.ToolCalls)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	out, err := ResponsesResponseFromIR(resp, "gpt-5.6-luna")
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if !json.Valid(out) {
		t.Fatal("built body is not valid JSON")
	}
}
