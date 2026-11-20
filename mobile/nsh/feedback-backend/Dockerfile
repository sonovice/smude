# Offizielles PocketBase-Release-Binary, Version und Prüfsummen fest verdrahtet.
# Update: PB_VERSION und beide SHA256 aus checksums.txt des Releases ersetzen.
FROM alpine:3.22

ARG PB_VERSION=0.40.6
ARG PB_SHA256_ARM64=d22c66ceb67ca92b7d767f36d826e848a276bcfa3c197ead20bf87c42208984f
ARG PB_SHA256_AMD64=0e1f609158ed945f809f8a02ae012aa0427129c66ebb63c1aee76f2c0309373f
ARG TARGETARCH

RUN apk add --no-cache ca-certificates unzip wget \
 && case "${TARGETARCH:-$(uname -m)}" in \
      arm64|aarch64) arch=arm64; sum="$PB_SHA256_ARM64" ;; \
      amd64|x86_64)  arch=amd64; sum="$PB_SHA256_AMD64" ;; \
      *) echo "unsupported arch: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
 && wget -qO /tmp/pb.zip "https://github.com/pocketbase/pocketbase/releases/download/v${PB_VERSION}/pocketbase_${PB_VERSION}_linux_${arch}.zip" \
 && echo "${sum}  /tmp/pb.zip" | sha256sum -c - \
 && unzip -q /tmp/pb.zip pocketbase -d /usr/local/bin \
 && rm /tmp/pb.zip \
 && apk del unzip

WORKDIR /pb
EXPOSE 8090
ENTRYPOINT ["/usr/local/bin/pocketbase"]
CMD ["serve", "--http=0.0.0.0:8090", "--dir=/pb/pb_data", "--migrationsDir=/pb/pb_migrations"]
