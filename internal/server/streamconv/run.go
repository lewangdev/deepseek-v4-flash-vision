package streamconv

import (
	"io"
	"net/http"

	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

// Run bridges a cross-family stream: it parses an upstream SSE stream written
// in one wire format and re-emits it to dst in the downstream client's format.
// Text deltas are forwarded live; tool calls and the terminal event land once
// the upstream stream closes. model is the client-facing name echoed in events.
func Run(outFamily, inUpstreamFam string, src io.Reader, dst io.Writer, model string) {
	var sw Writer
	switch outFamily {
	case "messages":
		sw = NewMessagesWriter(dst, model)
	case "responses":
		sw = NewResponsesWriter(dst, model)
	default:
		sw = NewChatWriter(dst, model)
	}
	var parse func(io.Reader, func(Delta)) error
	switch inUpstreamFam {
	case "messages":
		parse = ParseMessages
	case "responses":
		parse = ParseResponses
	default:
		parse = ParseChat
	}

	fl, _ := dst.(http.Flusher)
	finish := "stop"
	var usage *ir.Usage
	_ = parse(src, func(d Delta) {
		if d.Text != "" {
			_ = sw.Text(d.Text)
		}
		for i := range d.ToolCalls {
			_ = sw.ToolCall(d.ToolCalls[i])
		}
		if d.FinishReason != nil {
			finish = *d.FinishReason
		}
		if d.Usage != nil {
			usage = d.Usage
		}
		if fl != nil {
			fl.Flush()
		}
	})
	_ = sw.Done(finish, usage)
	if fl != nil {
		fl.Flush()
	}
}
