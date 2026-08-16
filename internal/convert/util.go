package convert

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"strings"
	"time"

	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

// This file holds helpers shared by the three wire-format converters.

// usageJSON mirrors the OpenAI-style usage object shared by all endpoints.
type usageJSON struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
	// Anthropic Messages uses these names.
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

func (u *usageJSON) toIR() *usageJSON {
	if u.TotalTokens == 0 && (u.PromptTokens != 0 || u.CompletionTokens != 0) {
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
	}
	return u
}

// OpenAIToolDef is the canonical (OpenAI-style) tool definition stored in
// ir.Request.Tools and translated into each upstream format.
type openAIToolDef struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

func newOpenAITool(name, description string, params json.RawMessage) openAIToolDef {
	t := openAIToolDef{Type: "function"}
	t.Function.Name = name
	t.Function.Description = description
	t.Function.Parameters = params
	return t
}

// splitDataURI parses an RFC 2397 data URI and decodes base64 payloads.
func splitDataURI(uri string) (mediaType string, raw []byte, ok bool) {
	if !strings.HasPrefix(uri, "data:") {
		return "", nil, false
	}
	comma := strings.IndexByte(uri, ',')
	if comma < 0 {
		return "", nil, false
	}
	meta := uri[len("data:"):comma]
	payload := uri[comma+1:]
	mt, base64enc := "text/plain", false
	if meta != "" {
		parts := strings.Split(meta, ";")
		mt = parts[0]
		for _, p := range parts[1:] {
			if p == "base64" {
				base64enc = true
			}
		}
	}
	if base64enc {
		raw, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return mt, nil, false
		}
		return mt, raw, true
	}
	return mt, []byte(payload), true
}

// dataURI renders inline image bytes as a base64 data URI.
func dataURI(mt string, data []byte) string {
	return "data:" + mt + ";" + "base64," + base64.StdEncoding.EncodeToString(data)
}

// imageFromURL builds an ir.ImagePart from either a data URI or a remote URL.
func imageFromURL(uri string) (ir.ImagePart, bool) {
	if mt, data, ok := splitDataURI(uri); ok {
		return ir.ImagePart{MediaType: mt, Data: data}, true
	}
	if strings.HasPrefix(uri, "http://") || strings.HasPrefix(uri, "https://") {
		return ir.ImagePart{URL: uri}, true
	}
	return ir.ImagePart{}, false
}

// newID produces a short random id with the given prefix.
func newID(prefix string) string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return prefix + "-000000000000"
	}
	return prefix + "-" + hex.EncodeToString(b[:])
}

// nowUnix returns the current unix timestamp.
func nowUnix() int64 { return time.Now().Unix() }

// replaceModelName performs a best-effort substitution of the JSON field
// "model":"<from>" with "model":"<to>". It is applied to raw (stream) bytes so
// a model name split across chunk boundaries is not an error — just left as-is.
func replaceModelName(b []byte, from, to string) []byte {
	if from == "" || to == "" || from == to {
		return b
	}
	return []byte(strings.ReplaceAll(string(b), `"model":"`+from+`"`, `"model":"`+to+`"`))
}

// parseStopNormalizes an OpenAI stop field that may be a string or an array.
func parseStopNormalizes(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		return []string{single}
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		return many
	}
	return nil
}