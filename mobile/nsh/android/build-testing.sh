#!/usr/bin/env bash
# Test-track build: Android cannot tell test tracks apart at runtime, so test
# mode (developer reports, the one-time price question) is set at compile time.
set -euo pipefail
SMUDE_TESTING=1 exec "$(dirname -- "${BASH_SOURCE[0]}")/build.sh" "$@"
