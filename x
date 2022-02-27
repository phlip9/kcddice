#!/bin/bash

set -e

function usage() {
  echo "x [option ...] subcommand"
  echo ""
  # echo "· x build-wasm - build the wasm sub-project"
  echo "· x build-www - build the www sub-project"
  echo "· x run - run the cli"
  echo "· x run-perf - run the cli inside perf"
  echo "· x repl-rs - open an evcxr repl with some helpers"
  echo "· x repl-py - open a python3 repl with some helpers"
}

case "$1" in
  # build-wasm)
  #   wasm-pack build \
  #     --no-typescript \
  #     --release \
  #     --mode normal \
  #     --target bundler \
  #     kcddice-wasm
  #   ;;

  build-www)
    trunk build --release -- kcddice-www/index.html

    cp kcddice-www/dist/index.html kcddice-www/dist/index.pre.html 
    cargo run --bin render < kcddice-www/dist/index.pre.html > kcddice-www/dist/index.html
    ;;

  watch-www)
    # for some reason, this http crate has problems serving assets. like they're
    # flaky idk...
    # cargo watch --ignore kcddice-www/dist -s "./x build-www && http kcddice-www/dist"
    cargo watch --ignore kcddice-www/dist -s "./x build-www && python3 -m http.server --bind 0.0.0.0 --directory kcddice-www/dist"
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
