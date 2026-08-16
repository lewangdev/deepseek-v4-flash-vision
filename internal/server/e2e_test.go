package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lewang/deepseek-v4-flash-vision/internal/config"
)

// newTestGateway wires a minimal config against a fake upstream, returning the
// gateway URL and the fake upstream server.
func newTestGateway(t *testing.T, up http.HandlerFunc) (*httptest.Server, *httptest.Server) {
	t.Helper()
	upSrv := httptest.NewServer(up)
	cfg := config.Default()
	cfg.OpenCode.BaseURL = upSrv.URL
	cfg.OpenCode.APIKey = "sk-test"
	gw := httptest.NewServer(New(cfg).Mux())
	t.Cleanup(func() {
		upSrv.Close()
		gw.Close()
	})
	return gw, upSrv
}

func post(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post %s: %v", url, err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func readBody(t *testing.T, resp *http.Response) string {
	t.Helper()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	return string(b)
}

func TestE2E_ChatTextPassthrough(t *testing.T) {
	var got []byte
	gw, _ := newTestGateway(t, func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		got = b
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"x","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"hello-chat"},"finish_reason":"stop"}]}`))
	})

	resp := post(t, gw.URL+"/v1/chat/completions",
		`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d body=%s", resp.StatusCode, readBody(t, resp))
	}
	// Upstream must have received the primary model on the chat endpoint.
	if !bytes.Contains(got, []byte(`"deepseek-v4-flash"`)) {
		t.Fatalf("upstream body missing model: %s", got)
	}
	if !strings.Contains(readBody(t, resp), "hello-chat") {
		t.Fatalf("response missing content: %s", readBody(t, resp))
	}
}

func TestE2E_ChatImageRoutesToVision(t *testing.T) {
	var gotMsgBody []byte
	png := base64.StdEncoding.EncodeToString([]byte("fake"))
	gw, _ := newTestGateway(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages" {
			t.Errorf("expected upstream /messages, got %s", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		gotMsgBody = b
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"qwen3.7-max","content":[{"type":"text","text":"hello-anthropic"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})

	resp := post(t, gw.URL+"/v1/chat/completions",
		`{"model":"deepseek-v4-flash","messages":[{"role":"user","content":[
			{"type":"text","text":"what"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,`+png+`"}}
		]}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d body=%s", resp.StatusCode, readBody(t, resp))
	}

	// The upstream call must target qwen3.7-max over Anthropic messages and
	// carry the image as a base64 block plus a max_tokens default.
	var upstream struct {
		Model     string `json:"model"`
		MaxTokens int    `json:"max_tokens"`
		Messages  []struct {
			Role    string `json:"role"`
			Content []struct {
				Type   string `json:"type"`
				Source map[string]any `json:"source,omitempty"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(gotMsgBody, &upstream); err != nil {
		t.Fatalf("unmarshal upstream body: %v", err)
	}
	if upstream.Model != "qwen3.7-max" {
		t.Errorf("upstream model = %q", upstream.Model)
	}
	if upstream.MaxTokens != config.Default().Server.DefaultMaxTokens {
		t.Errorf("upstream max_tokens = %d", upstream.MaxTokens)
	}
	found := false
	for _, m := range upstream.Messages {
		for _, c := range m.Content {
			if c.Type == "image" && c.Source["type"] == "base64" && c.Source["data"] == png {
				found = true
			}
		}
	}
	if !found {
		t.Errorf("image block missing in upstream body: %s", gotMsgBody)
	}

	// And the gateway converts the Anthropic response back to a chat completion.
	out := readBody(t, resp)
	if !strings.Contains(out, "hello-anthropic") {
		t.Fatalf("response missing text: %s", out)
	}
}

func TestE2E_MessagesClientTextToDeepSeek(t *testing.T) {
	var gotChatBody []byte
	gw, _ := newTestGateway(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected upstream /chat/completions, got %s", r.URL.Path)
		}
		b, _ := io.ReadAll(r.Body)
		gotChatBody = b
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"x","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"hello-deepseek"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`))
	})

	// An Anthropic-style client pointing at an unknown model name falls back to
	// the text primary, which is converted to the OpenAI chat wire format.
	resp := post(t, gw.URL+"/v1/messages",
		`{"model":"some-unknown-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d body=%s", resp.StatusCode, readBody(t, resp))
	}
	if !bytes.Contains(gotChatBody, []byte(`"deepseek-v4-flash"`)) {
		t.Fatalf("upstream chat body: %s", gotChatBody)
	}
	out := readBody(t, resp)
	// Downstream must be a valid Anthropic message object.
	var message struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(out), &message); err != nil {
		t.Fatalf("response is not valid Anthropic JSON: %v\n%s", err, out)
	}
	if message.Type != "message" || message.Role != "assistant" {
		t.Errorf("message header = %+v", message)
	}
	if len(message.Content) != 1 || message.Content[0].Text != "hello-deepseek" {
		t.Errorf("content = %+v", message.Content)
	}
}

func TestE2E_StreamChatToVision(t *testing.T) {
	gw, _ := newTestGateway(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/messages" {
			t.Errorf("expected /messages, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"streamed"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}

event: message_stop
data: {"type":"message_stop"}

`)
	})

	resp := post(t, gw.URL+"/v1/chat/completions",
		`{"model":"deepseek-v4-flash","stream":true,"messages":[{"role":"user","content":[
			{"type":"text","text":"what"},
			{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
		]}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d body=%s", resp.StatusCode, readBody(t, resp))
	}
	out := readBody(t, resp)
	if !strings.Contains(out, `"object":"chat.completion.chunk"`) {
		t.Fatalf("not a chat stream: %s", out)
	}
	if !strings.Contains(out, "streamed") {
		t.Fatalf("missing streamed text: %s", out)
	}
	if !strings.Contains(out, "[DONE]") {
		t.Fatalf("missing [DONE]: %s", out)
	}
}