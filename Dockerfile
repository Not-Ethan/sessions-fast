# Shopping Sessions: native Rust pipeline in an empty (scratch) image.
# The scorer builds this natively on linux/amd64; it also builds on linux/arm64
# for local testing on Apple silicon.
FROM rust:1.85-slim-bookworm AS build
ARG TARGETARCH
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# Static binary (glibc, crt-static), no runtime dependencies. x86-64-v2 is the
# baseline (SSE4.2/POPCNT, every x86 server since 2009). The CSV parsing loop
# is also compiled for AVX2+BMI1/2 and selected at run time only when CPUID
# reports those features, so there is no illegal-instruction risk on any CPU.
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in \
      amd64) T=x86_64-unknown-linux-gnu; F="-C target-cpu=x86-64-v2" ;; \
      arm64) T=aarch64-unknown-linux-gnu; F="" ;; \
      *) echo "unsupported arch ${TARGETARCH}"; exit 1 ;; \
    esac; \
    RUSTFLAGS="$F -C target-feature=+crt-static" cargo build --release --locked --target "$T"; \
    cp "target/$T/release/sessions" /sessions; \
    /sessions --self-test

FROM scratch
COPY --from=build /sessions /sessions
USER 1000:1000
ENV DATA_DIR=/data OUTPUT_DIR=/output SCRATCH_DIR=/scratch
CMD ["/sessions"]
