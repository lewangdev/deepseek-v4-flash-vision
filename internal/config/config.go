// Package config loads and validates the single YAML configuration file.
package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Upstream endpoint names, as exposed by the OpenCode Go subscription.
// Each model on OpenCode Go maps to exactly one of these wire formats.
const (
	EndpointChat      = "chat/completions"
	EndpointMessages  = "messages"
	EndpointResponses = "responses"
)

// Config is the root of the configuration file.
type Config struct {
	Version  int      `yaml:"version"`
	Server   Server   `yaml:"server"`
	OpenCode OpenCode `yaml:"opencode"`
	Router   Router   `yaml:"router"`
}

// Server holds the gateway's own listening / auth settings.
type Server struct {
	Address          string `yaml:"address"`
	APIKey           string `yaml:"api_key"` // non-empty => downstream clients must present it
	DefaultMaxTokens int    `yaml:"default_max_tokens"`
}

// OpenCode holds the upstream subscription credentials.
type OpenCode struct {
	BaseURL string            `yaml:"base_url"`
	APIKey  string            `yaml:"api_key"`
	Headers map[string]string `yaml:"headers"` // extra headers sent to every upstream call
}

// Router decides which model handles which traffic.
type Router struct {
	Primary    string                   `yaml:"primary"`     // text default (usually deepseek-v4-flash)
	Vision     string                   `yaml:"vision"`      // used when the request contains images
	AutoVision bool                     `yaml:"auto_vision"` // route image requests to Vision
	Overrides  map[string]ModelEndpoint `yaml:"overrides"`   // explicit client model name -> upstream model + endpoint
}

// ModelEndpoint resolves a client-facing model name to an upstream model and endpoint.
type ModelEndpoint struct {
	ID       string `yaml:"id"`       // upstream model id on OpenCode Go
	Endpoint string `yaml:"endpoint"` // one of EndpointChat / EndpointMessages / EndpointResponses
}

// Default returns the built-in configuration. A minimal or missing config file
// layers on top of these defaults, so the gateway runs out of the box.
func Default() Config {
	return Config{
		Version: 1,
		Server: Server{
			Address:          "127.0.0.1:8787",
			DefaultMaxTokens: 2048,
		},
		OpenCode: OpenCode{
			BaseURL: "https://opencode.ai/zen/go/v1",
		},
		Router: Router{
			Primary:    "deepseek-v4-flash",
			Vision:     "mimo-v2.5",
			AutoVision: true,
			Overrides: map[string]ModelEndpoint{
				"deepseek-v4-flash": {ID: "deepseek-v4-flash", Endpoint: EndpointChat},
				// mimo-v2.5 is the verified image-capable default: it shares the
				// chat/completions wire format with the primary, so default text
				// and vision traffic never crosses families. The other entries
				// remain reachable when a client names them explicitly.
				"mimo-v2.5":    {ID: "mimo-v2.5", Endpoint: EndpointChat},
				"qwen3.7-max":  {ID: "qwen3.7-max", Endpoint: EndpointMessages},
				"gpt-5.6-luna": {ID: "gpt-5.6-luna", Endpoint: EndpointResponses},
				"grok-4.5":     {ID: "grok-4.5", Endpoint: EndpointResponses},
			},
		},
	}
}

// Load reads the config at path and overlays it onto Default(). If the file does
// not exist, the built-in defaults are returned together with a warning. Missing
// api_key is reported as a warning so the process still starts for local tooling.
func Load(path string) (Config, []string, error) {
	var warnings []string
	if path == "" {
		path = "config.yaml"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			warnings = append(warnings, fmt.Sprintf("config file %q not found; using built-in defaults. Copy config.example.yaml to %q.", path, path))
			return Default(), warnings, nil
		}
		return Config{}, warnings, err
	}
	cfg := Default()
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, warnings, fmt.Errorf("parse %s: %w", path, err)
	}
	if err := cfg.validate(); err != nil {
		return cfg, warnings, err
	}
	if cfg.OpenCode.APIKey == "" {
		warnings = append(warnings, "opencode.api_key is empty — upstream calls will fail. Get your key at https://opencode.ai/auth")
	}
	return cfg, warnings, nil
}

func (c Config) validate() error {
	valid := map[string]bool{EndpointChat: true, EndpointMessages: true, EndpointResponses: true}
	for name, m := range c.Router.Overrides {
		if !valid[m.Endpoint] {
			return fmt.Errorf("router.overrides.%s.endpoint %q invalid (want one of %s, %s, %s)",
				name, m.Endpoint, EndpointChat, EndpointMessages, EndpointResponses)
		}
	}
	if c.Router.Primary == "" && c.Router.Vision == "" {
		return fmt.Errorf("router.primary and router.vision are both empty")
	}
	return nil
}

// UpstreamModelID returns the upstream model id for a client-facing name.
func (c Config) UpstreamModelID(model string) string {
	if m, ok := c.Router.Overrides[model]; ok && m.ID != "" {
		return m.ID
	}
	return model
}

// EndpointFor returns the upstream endpoint format for a model, falling back to
// `fallback` when the model has no explicit override.
func (c Config) EndpointFor(model, fallback string) string {
	if m, ok := c.Router.Overrides[model]; ok && m.Endpoint != "" {
		return m.Endpoint
	}
	return fallback
}
