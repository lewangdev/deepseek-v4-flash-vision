// Command deepseek-v4-flash-vision is a local vision gateway for the
// DeepSeek V4 Flash model served by the OpenCode Go subscription. It exposes
// OpenAI Chat Completions, OpenAI Responses and Anthropic Messages endpoints,
// routing text traffic to DeepSeek V4 Flash and image traffic to a configurable
// vision model (default mimo-v2.5). All configuration lives in one YAML file.
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/lewang/deepseek-v4-flash-vision/internal/config"
	"github.com/lewang/deepseek-v4-flash-vision/internal/server"
)

func main() {
	cfgPath := flag.String("config", "config.yaml", "path to the YAML configuration file")
	addr := flag.String("address", "", "override server.address")
	flag.Parse()

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil)))

	cfg, warnings, err := config.Load(*cfgPath)
	if err != nil {
		slog.Error("failed to load config", "path", *cfgPath, "error", err)
		os.Exit(1)
	}
	for _, w := range warnings {
		slog.Warn(w)
	}
	if *addr != "" {
		cfg.Server.Address = *addr
	}

	httpServer := &http.Server{
		Addr:    cfg.Server.Address,
		Handler: server.New(cfg).Mux(),
	}

	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		slog.Info("shutting down")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(ctx)
	}()

	slog.Info("gateway listening",
		"address", cfg.Server.Address,
		"primary", cfg.Router.Primary,
		"vision", cfg.Router.Vision,
		"auto_vision", cfg.Router.AutoVision,
	)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		slog.Error("server error", "error", err)
		os.Exit(1)
	}
}
