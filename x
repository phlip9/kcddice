#!/bin/bash

set -e

function usage() {
  echo "x [option ...] subcommand"
  echo ""
  echo "· x build-wasm - build the wasm sub-project"
  echo "· x run - run the cli"
  echo "· x run-perf - run the cli inside perf"
  echo "· x repl-rs - open an evcxr repl with some helpers"
  echo "· x repl-py - open a python3 repl with some helpers"
}

case "$1" in
  build-wasm)
    wasm-pack build \
      --no-typescript \
      --release \
      --mode normal \
      --target bundler \
      kcddice-wasm
    ;;

  build-www)
    cd kcddice-www
    npm exec -- webpack \
      --config webpack.config.js \
      --mode production
    ;;

  run)
    shift
    RUSTFLAGS="-C target-cpu=native" cargo run --release -p kcddice -- $@
    ;;

  run-perf)
    shift

    RUSTFLAGS="-C target-cpu=native" cargo build --release -p kcddice

    # perf record --call-graph=dwarf -F 2000 ./target/release/kcddice $@
    perf record --call-graph=lbr -F 5000 ./target/release/kcddice $@

    perf script -F +pid | rustfilt > perf.data.perf
    # perf script -F +pid --no-inline | rustfilt > perf.data.perf

    cp perf.data.perf /mnt/c/Users/phlip9/Desktop/perf.data.perf
    ;;

  repl-rs)
    evcxr --edit-mode vi --opt 0
    ;;

  repl-py)
    python3 -i repl.py
    ;;

  *)
    echo "error: unrecognized subcommand '$1'"
    usage
    ;;
esac