#!/bin/bash

set -e

RUSTFLAGS="-C target-cpu=native" cargo build --release

# perf record --call-graph=dwarf -F 2000 ./target/release/kcddice $@
perf record --call-graph=lbr -F 5000 ./target/release/kcddice $@

perf script -F +pid | rustfilt > perf.data.perf
# perf script -F +pid --no-inline | rustfilt > perf.data.perf

cp perf.data.perf /mnt/c/Users/phlip9/Desktop/perf.data.perf
