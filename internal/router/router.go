// Package router decides which upstream model and endpoint format handles a
// given request. Text traffic goes to the primary model (DeepSeek V4 Flash);
// requests carrying image content go to the vision model. An explicit
// client-chosen model listed in overrides always wins over auto-routing.
package router

import (
	"github.com/lewang/deepseek-v4-flash-vision/internal/config"
	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
)

// Result is the routing decision: the upstream model id and endpoint format.
type Result struct {
	Model    string
	Endpoint string
}

// Router evaluates ir.Requests against the resolved config.
type Router struct {
	cfg config.Config
}

func New(cfg config.Config) *Router { return &Router{cfg: cfg} }

// Resolve returns the target upstream model + endpoint for req.
//
// Precedence: an explicit client-chosen model wins when it names a model other
// than the auto-routed primary/vision targets (e.g. gpt-5.6-luna). Requests
// naming the primary or vision model (or an unknown model) fall through to
// auto-routing: image content goes to the vision model, everything else to the
// primary text model.
func (rt *Router) Resolve(req *ir.Request) Result {
	// A client explicitly choosing a non-default model (e.g. gpt-5.6-luna)
	// always wins over auto-routing.
	if req.ClientModel != "" &&
		req.ClientModel != rt.cfg.Router.Primary &&
		req.ClientModel != rt.cfg.Router.Vision {
		if m, ok := rt.cfg.Router.Overrides[req.ClientModel]; ok {
			return Result{Model: or(m.ID, req.ClientModel), Endpoint: m.Endpoint}
		}
	}

	if rt.cfg.Router.AutoVision && req.HasImage() {
		v := rt.cfg.Router.Vision
		if m, ok := rt.cfg.Router.Overrides[v]; ok {
			return Result{Model: or(m.ID, v), Endpoint: m.Endpoint}
		}
		return Result{Model: v, Endpoint: config.EndpointMessages}
	}

	p := rt.cfg.Router.Primary
	if m, ok := rt.cfg.Router.Overrides[p]; ok {
		return Result{Model: or(m.ID, p), Endpoint: m.Endpoint}
	}
	return Result{Model: p, Endpoint: config.EndpointChat}
}

// KnownModels returns the deduplicated set of model ids exposed via /v1/models.
func (rt *Router) KnownModels() []string {
	seen := map[string]bool{}
	var out []string
	add := func(m string) {
		if m == "" || seen[m] {
			return
		}
		seen[m] = true
		out = append(out, m)
	}
	for name := range rt.cfg.Router.Overrides {
		if id := rt.cfg.UpstreamModelID(name); id != "" {
			add(id)
		} else {
			add(name)
		}
	}
	add(rt.cfg.Router.Primary)
	add(rt.cfg.Router.Vision)
	return out
}

func or(a, b string) string {
	if a != "" {
		return a
	}
	return b
}