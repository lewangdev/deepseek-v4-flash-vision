// Package ir defines the canonical intermediate representation used between the
// three client wire formats (OpenAI Chat Completions, OpenAI Responses,
// Anthropic Messages) and the three upstream OpenCode Go endpoints.
package ir

import (
	"encoding/json"
	"strings"
)

// Role is a normalized message role.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool" // only meaningful in the OpenAI-style representations
)

// ContentPart is one block inside a message's content.
type ContentPart interface{ isContentPart() }

// TextPart is a plain text block.
type TextPart struct{ Text string }

// ImagePart is an image block. Exactly one of URL / Data is meaningful:
// Data is the decoded bytes for an inline (base64) image, URL is a remote
// http(s) reference.
type ImagePart struct {
	MediaType string
	URL       string
	Data      []byte
}

// ToolUsePart is an assistant tool invocation.
type ToolUsePart struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResultPart is the outcome of a tool call.
type ToolResultPart struct {
	ToolUseID string
	Content   string
}

func (TextPart) isContentPart()       {}
func (ImagePart) isContentPart()      {}
func (ToolUsePart) isContentPart()    {}
func (ToolResultPart) isContentPart() {}

// Message is one normalized message.
type Message struct {
	Role  Role
	Parts []ContentPart
}

// HasImage reports whether the message carries at least one image block.
func (m Message) HasImage() bool {
	for _, p := range m.Parts {
		if _, ok := p.(ImagePart); ok {
			return true
		}
	}
	return false
}

// Text concatenates the textual content of the message (text parts plus any
// tool result body), used for system prompts and tool results.
func (m Message) Text() string {
	var sb strings.Builder
	for _, p := range m.Parts {
		if t, ok := p.(TextPart); ok {
			sb.WriteString(t.Text)
		}
		if tr, ok := p.(ToolResultPart); ok {
			sb.WriteString(tr.Content)
		}
	}
	return sb.String()
}

// Request is a normalized chat request.
//
// ClientModel is the model id as sent by the client (used for override routing
// and echoed back in responses); Model is the resolved upstream model id the
// router fills in before the request is serialized upstream.
type Request struct {
	ClientModel string
	Model       string
	Messages    []Message
	MaxTokens   *int
	Temperature *float64
	TopP        *float64
	Stop        []string
	Stream      bool

	// Tools holds OpenAI-style function tool definitions
	// ([{type:"function",function:{name,description,parameters}}]), normalized
	// from whatever the client sent. They are converted per upstream format.
	Tools []json.RawMessage
}

// HasImage reports whether any message carries an image block.
func (r *Request) HasImage() bool {
	for _, m := range r.Messages {
		if m.HasImage() {
			return true
		}
	}
	return false
}

// ToolCall is a complete assistant tool invocation in a response.
type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// Usage reports token counts.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// Response is a normalized completion response.
type Response struct {
	Text         string
	ToolCalls    []ToolCall
	FinishReason string
	Usage        *Usage
}
