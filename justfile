#!/usr/bin/env -S just --justfile

set shell := ["bash", "-uc"]
set positional-arguments

alias bw := build-www
alias c := clippy
alias dw := deploy-www
alias sw := serve-www
alias t := test
alias ww := watch-www

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

build-www +PUBLIC_URL='/':
  trunk build \
    --release \
    --public-url "{{PUBLIC_URL}}" \
    -- kcddice-www/index.html

serve-www:
  http kcddice-www/dist

watch-www:
  cargo watch \
    --ignore kcddice-www/dist \
    -s "just build-www"

deploy-www: (build-www '/kcddice/')
  #!/usr/bin/env bash
  set -euxo pipefail

  DIST_DIR="$(pwd)/kcddice-www/dist"

  HEAD=$(git rev-parse --verify HEAD)
  ORIGIN=$(git config --get remote.origin.url)
  TMP_DIR=$(mktemp -d -t kcddice-XXXXXXXX)

  cd $TMP_DIR

  git init .
  git remote add origin $ORIGIN
  git checkout -b gh-pages

  cp -r $DIST_DIR/* ./
  touch .nojekyll
  git add .
  git commit -m "deploy kcddice-www: $HEAD"

  git push -f origin gh-pages

  cd ..
  rm -rf $TMP_DIR

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
