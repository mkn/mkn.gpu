#!/usr/bin/env bash
set -e

CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

(
    cd "$CWD/.."

    mkn build -p cpu run test -OW 9
    mkn build -x res/mkn/hipcc test -p rocm -OW 9
    mkn build -x res/mkn/clang_cuda test -p cuda -OW 9
    mkn build -x res/mkn/hipcc test -p rocm_trace -OW 9
    mkn build -x res/mkn/clang_cuda test -p cuda_trace -OW 9

)
