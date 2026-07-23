#!/usr/bin/env bash

set -e
set -x

cd "$(dirname "$0")"

rm -rf build
mkdir -p build

ARCH="$(uname -m)"

if [[ "$ARCH" == "arm64" ]]; then
  SUFFIX="arm64"
elif [[ "$ARCH" == "x86_64" ]]; then
  SUFFIX="x86_64"
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

swiftc \
  -O \
  -whole-module-optimization \
  -emit-library \
  EnnMetalShaders.swift \
  EnnMetalBackend.swift \
  -o "build/libeneural_metal_${SUFFIX}.dylib" \
  -framework Foundation \
  -framework Metal \
  -framework MetalPerformanceShaders
