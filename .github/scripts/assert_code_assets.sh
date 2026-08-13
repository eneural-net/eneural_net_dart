#!/usr/bin/env bash
#
# Asserts that `hook/build.dart` emitted the named code assets.
#
# Usage: assert_code_assets.sh eneural_cpu eneural_metal
#
# Without this, the acceleration jobs are a false green: the differential tests
# self-skip when no GPU is present (hosted runners have none), and the build hook
# stays silent when a toolchain is missing, so a hook that built nothing looks
# exactly like a hook that worked. This checks the hook's own output.

set -uo pipefail

outputs=$(find .dart_tool/hooks_runner -name output.json 2>/dev/null)

if [ -z "$outputs" ]; then
  echo "::error::No build hook output found under .dart_tool/hooks_runner —" \
       "the hook did not run at all."
  exit 1
fi

status=0

for name in "$@"; do
  id="package:eneural_net/$name"
  if grep -q "$id" $outputs; then
    echo "OK: $id"
  else
    echo "::error::The build hook did not emit '$id'."
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "--- build hook output ---"
  for f in $outputs; do
    echo "== $f"
    cat "$f"
  done
  echo "--- build hook stderr ---"
  find .dart_tool/hooks_runner -name stderr.txt -print -exec cat {} \;
fi

exit "$status"
