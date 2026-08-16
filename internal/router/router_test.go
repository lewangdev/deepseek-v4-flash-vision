package router_test

import (
	"encoding/json"
	"testing"

	"github.com/lewang/deepseek-v4-flash-vision/internal/config"
	"github.com/lewang/deepseek-v4-flash-vision/internal/ir"
	"github.com/lewang/deepseek-v4-flash-vision/internal/router"
)

func textReq() *ir.Request {
	return &ir.Request{Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.ContentPart{ir.TextPart{Text: "hi"}}}}}
}

func imageReq() *ir.Request {
	return &ir.Request{Messages: []ir.Message{{Role: ir.RoleUser, Parts: []ir.ContentPart{
		ir.TextPart{Text: "what"},
		ir.ImagePart{MediaType: "image/png", Data: []byte("x")},
	}}}}
}

func TestResolveTextUsesPrimary(t *testing.T) {
	rt := router.New(config.Default())
	res := rt.Resolve(textReq())
	if res.Model != "deepseek-v4-flash" || res.Endpoint != config.EndpointChat {
		t.Fatalf("got %+v", res)
	}
}

func TestResolveImageUsesVision(t *testing.T) {
	rt := router.New(config.Default())
	res := rt.Resolve(imageReq())
	if res.Model != "qwen3.7-max" || res.Endpoint != config.EndpointMessages {
		t.Fatalf("got %+v", res)
	}
}

func TestResolvePrimaryNameWithImageGoesVision(t *testing.T) {
	// Tools pointed at the gateway typically name the primary model explicitly,
	// e.g. model:"deepseek-v4-flash". An image in that request must still route
	// to the vision model (that is the whole point of the proxy).
	rt := router.New(config.Default())
	req := imageReq()
	req.ClientModel = "deepseek-v4-flash"
	res := rt.Resolve(req)
	if res.Model != "qwen3.7-max" || res.Endpoint != config.EndpointMessages {
		t.Fatalf("got %+v", res)
	}
}

func TestResolvePrimaryNameTextStaysPrimary(t *testing.T) {
	rt := router.New(config.Default())
	req := textReq()
	req.ClientModel = "deepseek-v4-flash"
	res := rt.Resolve(req)
	if res.Model != "deepseek-v4-flash" || res.Endpoint != config.EndpointChat {
		t.Fatalf("got %+v", res)
	}
}

func TestResolveVisionNameWithImageStaysVision(t *testing.T) {
	rt := router.New(config.Default())
	req := imageReq()
	req.ClientModel = "qwen3.7-max"
	res := rt.Resolve(req)
	if res.Model != "qwen3.7-max" || res.Endpoint != config.EndpointMessages {
		t.Fatalf("got %+v", res)
	}
}

func TestResolveExplicitOverrideWins(t *testing.T) {
	rt := router.New(config.Default())
	// A non-primary, non-vision model chosen explicitly always wins.
	res := rt.Resolve(&ir.Request{ClientModel: "gpt-5.6-luna", Messages: imageReq().Messages})
	if res.Model != "gpt-5.6-luna" || res.Endpoint != config.EndpointResponses {
		t.Fatalf("got %+v", res)
	}
}

func TestResolveUnknownClientModelFallsBackToAuto(t *testing.T) {
	rt := router.New(config.Default())
	// Unknown name + no image -> primary.
	if res := rt.Resolve(&ir.Request{ClientModel: "mystery", Messages: textReq().Messages}); res.Model != "deepseek-v4-flash" {
		t.Fatalf("got %+v", res)
	}
	// Unknown name + image -> vision.
	if res := rt.Resolve(&ir.Request{ClientModel: "mystery", Messages: imageReq().Messages}); res.Model != "qwen3.7-max" {
		t.Fatalf("got %+v", res)
	}
}

func TestAutoVisionDisabled(t *testing.T) {
	cfg := config.Default()
	cfg.Router.AutoVision = false
	rt := router.New(cfg)
	if res := rt.Resolve(imageReq()); res.Model != "deepseek-v4-flash" {
		t.Fatalf("got %+v", res)
	}
}

func TestKnownModelsDedupes(t *testing.T) {
	rt := router.New(config.Default())
	seen := map[string]bool{}
	for _, m := range rt.KnownModels() {
		if seen[m] {
			t.Fatalf("duplicate model %q", m)
		}
		seen[m] = true
	}
	if !seen["deepseek-v4-flash"] || !seen["qwen3.7-max"] || !seen["gpt-5.6-luna"] {
		t.Fatalf("missing expected models: %v", rt.KnownModels())
	}
}

// Ensure router Result is trivially serializable (used in logs).
func TestResultJSON(t *testing.T) {
	if _, err := json.Marshal(router.Result{Model: "a", Endpoint: "b"}); err != nil {
		t.Fatal(err)
	}
}