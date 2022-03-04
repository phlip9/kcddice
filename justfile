#!/usr/bin/env -S just --justfile

set shell := ["bash", "-uc"]
set positional-arguments

alias t := test
alias c := clippy
alias w := watch-www

default:
  @just --list

install-dev:
  # need recent trunk version with web worker support
  RUSTFLAGS="-C target-cpu=native" cargo install \
    --git="https://github.com/thedodd/trunk.git" \
    --rev="b989bc9bf" \
    trunk

  RUSTFLAGS="-C target-cpu=native" cargo install \
    evcxr_repl \
    cargo-watch \
    rustfilt

test:
  RUST_BACKTRACE=1 cargo test

clippy:
  cargo clippy --all-targets

build-www:
  trunk build --release -- kcddice-www/index.html

serve-www: build-www
  python3 -m http.server \
    --bind 0.0.0.0 \
    --directory kcddice-www/dist

watch-www:
  cargo watch \
    --ignore kcddice-www/dist \
    -s "just serve-www"

# run cli tool
run *args='':
  RUSTFLAGS="-C target-cpu=native" cargo run --release -p kcddice -- $@

# run cli tool under perf
run-perf *args='':
  RUSTFLAGS="-C target-cpu=native" cargo build --release -p kcddice

  # perf record --call-graph=dwarf -F 2000 ./target/release/kcddice $@
  perf record --call-graph=lbr -F 5000 ./target/release/kcddice $@

  perf script -F +pid | rustfilt > perf.data.perf
  # perf script -F +pid --no-inline | rustfilt > perf.data.perf

repl-rs:
  evcxr --edit-mode vi --opt 0

repl-py:
  python3 -i repl.py
