#!/usr/bin/env bash
# The demo's solver.js and solver-threads.js: cuwave's scalar kernels as cpu.RawModule
# compiles them, built with em++, or with `native` the solver<order>.so tests/web_test.py loads
# (SPACE_ORDER, PYTHON, EMXX and CXX override; OUT names a native build's folder)
set -euo pipefail
cd "$(dirname "$0")"
if [ "${1:-}" = native ]; then OUT=${OUT:-.}; else OUT=.; fi
mkdir -p "$OUT"

SPACE_ORDER=${SPACE_ORDER:-4}
POOL="Math.min(8, Math.max(1, (navigator.hardwareConcurrency || 2) >> 1))" # demo.js asks the same
FLAGS="-std=c++17 -O3 -ffp-contract=off -w -DNDIM=2 -DUSE_FLOAT -DUSE_DAMPING -DUSE_DOMAIN -DDIRICHLET_FACES=0"
EXPORTS="_configure,_boundary,_rebuild,_repaint,_origin,_resize,_stride,_extent,_field,_walls"
EXPORTS="$EXPORTS,_codes,_dampings,_coefficient,_radius,_burst,_advance,_plan,_clear"

${PYTHON:-python3} - "$SPACE_ORDER" > "$OUT/module.cpp" <<'EOF'
import sys
from pathlib import Path

sys.path.insert(0, "..")
from cuwave.stencils import preamble

kernels = Path("../cuwave/kernels")
order = int(sys.argv[1])
print((kernels / "cpu.hpp").read_text() + preamble(order, "optimized"), end="")
print((kernels / "common.cuh").read_text() + (kernels / "scalar.cu").read_text())
EOF

THREADS="-DCUWAVE_THREADS -std=c++20"
if [ "${1:-}" = native ]; then
    # beside its module.cpp, which the include finds first
    [ "$OUT" -ef . ] || cp solver.cpp "$OUT/"
    ${CXX:-c++} $FLAGS $THREADS -pthread -shared -fPIC "$OUT/solver.cpp" -o "$OUT/solver$SPACE_ORDER.so"
else
    WEB="-sMODULARIZE -sEXPORT_NAME=Solver -sSINGLE_FILE -sALLOW_MEMORY_GROWTH \
        -sEXPORTED_FUNCTIONS=$EXPORTS,_threads -sEXPORTED_RUNTIME_METHODS=HEAPF32,HEAPU8"
    ${EMXX:-em++} $FLAGS solver.cpp -o solver.js -sENVIRONMENT=web $WEB
    # the pool's workers spawned at load, as a page's main thread cannot wait for one
    ${EMXX:-em++} $FLAGS $THREADS -pthread solver.cpp -o solver-threads.js \
        -sENVIRONMENT=web,worker -sPTHREAD_POOL_SIZE="$POOL" $WEB
    rm module.cpp
fi
