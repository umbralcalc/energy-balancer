#!/usr/bin/env bash
# Generated build script for energy.
# Compiles cmd/energy/register_step to src/main.wasm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$SCRIPT_DIR/src"
cd "$PROJECT_ROOT"
GOOS=js GOARCH=wasm go build -o "$SCRIPT_DIR/src/main.wasm" ./cmd/energy/register_step
echo "Built $SCRIPT_DIR/src/main.wasm"
