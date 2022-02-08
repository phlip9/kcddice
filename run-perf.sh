#!/bin/bash

set -e

RUSTFLAGS="-C target-cpu=native" cargo build --release

perf record --call-graph dwarf -F 1000 ./target/release/kcddice $@

perf script -F +pid | rustfilt > perf.data.perf
