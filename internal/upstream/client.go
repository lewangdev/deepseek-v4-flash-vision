// Package upstream implements the HTTP client and per-endpoint authentication
// for the OpenCode Go subscription endpoints.
package upstream

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/lewang/deepseek-v4-flash-vision/internal/config"
	"github.com/lewang/deepseek-v4-flash-vision/internal/convert"
)

// Client talks to one OpenCode Go base URL.
type Client struct {
	baseURL string
	apiKey  string
	headers map[string]string
	http    *http.Client
}

func New(baseURL, apiKey string, headers map[string]string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		headers: headers,
		// No global Timeout: streaming completions can legitimately run long.
		// The request context from the caller bounds execution.
		http: &http.Client{},
	}
}

// Do POSTs body to the named endpoint and returns the raw response. The caller
// owns resp.Body and must Close it. All non-2xx responses are surfaced as errors.
func (c *Client) Do(ctx context.Context, endpoint string, body []byte) (*http.Response, error) {
	url := c.baseURL + "/" + endpoint
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	// Each OpenCode Go endpoint family authenticates differently. Anthropic
	// Messages uses x-api-key (+ anthropic-version); the OpenAI-style endpoints
	// use a Bearer token. We send both forms on /messages for compatibility.
	switch endpoint {
	case config.EndpointMessages:
		req.Header.Set("x-api-key", c.apiKey)
		req.Header.Set("anthropic-version", convert.AnthropicVersion())
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	default:
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upstream %s: %w", endpoint, err)
	}
	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		return nil, fmt.Errorf("upstream %s: %s: %s", endpoint, resp.Status, strings.TrimSpace(string(b)))
	}
	return resp, nil
}

// Call sends a non-streaming request and returns the full response body.
func (c *Client) Call(ctx context.Context, endpoint string, body []byte) ([]byte, error) {
	resp, err := c.Do(ctx, endpoint, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// Stream sends a request (usually with stream=true) and returns the open
// response for incremental reading.
func (c *Client) Stream(ctx context.Context, endpoint string, body []byte) (*http.Response, error) {
	return c.Do(ctx, endpoint, body)
}
