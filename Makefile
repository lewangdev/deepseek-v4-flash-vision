BIN := deepseek-v4-flash-vision

# CGO_ENABLED=0 is required: it keeps the binary fully static / portable, and
# it is also the only way to build in macOS sandboxes where cgo-linked
# net/http binaries fail to exec (dyld "missing LC_UUID load command").
build:
	CGO_ENABLED=0 go build -o $(BIN) .

test:
	CGO_ENABLED=0 go test -count=1 ./...

vet:
	go vet ./...

run: build
	./$(BIN) -config config.yaml

clean:
	rm -f $(BIN)

.PHONY: build test vet run clean