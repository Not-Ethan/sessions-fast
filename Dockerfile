# Shopping Sessions: native Rust pipeline in an empty (scratch) image.
# The scorer builds this natively on linux/amd64; it also builds on linux/arm64
# for local testing on Apple silicon.
ARG RUST=1.100.0
FROM rust:${RUST}-slim-bookworm AS build
ARG TARGETARCH
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# Static binaries (glibc, crt-static), no runtime dependencies.
# /sessions: x86-64-v2 baseline (SSE4.2/POPCNT, every x86 server since 2009);
# its CSV parsing loop is also compiled for AVX2+BMI1/2 and selected at run
# time only when CPUID reports those features. /sessions-v3: the whole program
# for x86-64-v3 (AVX2, BMI1/2, FMA, F16C, LZCNT, MOVBE). At start the baseline
# checks CPUID (and OS support for the YMM state) and execs /sessions-v3 only
# when every v3 feature is present (SS_NOEXEC=1: never), so there is no
# illegal-instruction risk on any CPU. The two builds run in parallel.
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
      amd64) \
        T=x86_64-unknown-linux-gnu; \
        RUSTFLAGS="-C target-cpu=x86-64-v2 -C target-feature=+crt-static" cargo build --release --locked --target "$T" --target-dir /tmp/t2 & P2=$!; \
        RUSTFLAGS="-C target-cpu=x86-64-v3 -C target-feature=+crt-static" cargo build --release --locked --target "$T" --target-dir /tmp/t3 & P3=$!; \
        wait $P2; wait $P3; \
        cp "/tmp/t2/$T/release/sessions" /sessions; \
        SS_NOEXEC=1 /sessions --self-test; \
        cp "/tmp/t3/$T/release/sessions" /sessions-v3; \
        if SS_NOEXEC=1 /sessions --has-v3; then /sessions-v3 --self-test; /sessions --self-test; fi ;; \
      arm64) \
        T=aarch64-unknown-linux-gnu; \
        RUSTFLAGS="-C target-feature=+crt-static" cargo build --release --locked --target "$T"; \
        cp "target/$T/release/sessions" /sessions; \
        cp /sessions /sessions-v3; \
        /sessions --self-test ;; \
      *) echo "unsupported arch ${TARGETARCH}"; exit 1 ;; \
    esac

FROM scratch
COPY --from=build /sessions /sessions-v3 /
USER 1000:1000
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/sessions"]
