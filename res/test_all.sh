#!/usr/bin/env bash
set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

(
    cd "$CWD/.."

    mkn build -p cpu run test -W 9
    mkn build -x res/mkn/hipcc test -p rocm -W 9
    mkn build -x res/mkn/clang_cuda test -p cuda -W 9

)
