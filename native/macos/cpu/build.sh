#!/usr/bin/env bash
#
# Manual/debug build of the macOS Accelerate (CPU) backend into build/.
#
# The primary build path is `hook/build.dart`, which the Dart SDK runs
# automatically on `dart run`/`dart test`/`dart build` and which mirrors the
# swiftc invocation below. This script stays useful for building outside a Dart
# command; its output is picked up by the loader's `DynamicLibrary.open`
# fallback (see `lib/src/native/native_accelerator_io.dart`).

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
  EnnCpuBackend.swift \
  -o "build/libeneural_cpu_${SUFFIX}.dylib" \
  -framework Accelerate \
  -framework Foundation
