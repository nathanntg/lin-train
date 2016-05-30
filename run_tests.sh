#!/usr/bin/env bash

# set -o xtrace # debug
set -o pipefail # get errors even when piped
set -o nounset # prevent using undeclared variables
set -o errexit # exit on command fail; allow failure: || true

python -m lintrain.tests.test
