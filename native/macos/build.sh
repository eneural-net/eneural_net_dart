#!/usr/bin/env bash

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "----------------------------------------------------------"
echo "Building CPU backend..."
(
  cd "$ROOT_DIR/cpu"
  ./build.sh
)

echo "----------------------------------------------------------"
echo "Building Metal backend..."
(
  cd "$ROOT_DIR/metal"
  ./build.sh
)

echo "----------------------------------------------------------"
echo
echo "Built libraries:"
find "$ROOT_DIR" -name "*.dylib"
