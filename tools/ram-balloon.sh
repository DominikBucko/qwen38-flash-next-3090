#!/usr/bin/env bash
# Make a bigger machine behave like a smaller one: lock (MemTotal - TARGET_GIB) of RAM in a helper container.
#   tools/ram-balloon.sh start 64 | stop | status
# The benchmarks in this repository ran with a 64 GiB target on a 128 GB host, started after the server's
# expert arena was populated (a freshly booted 64 GB machine has unfragmented memory for huge pages).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
image=${IMAGE:-qwen38-flash-next-3090:local}
case "${1:-status}" in
  start)
    docker run -d --rm --name qwen38-ram-balloon --ulimit memlock=-1 -e TARGET_GIB="${2:-64}" \
      -v "$here:/w:ro" --entrypoint python3 "$image" /w/ram_balloon.py >/dev/null
    until docker logs qwen38-ram-balloon 2>&1 | grep -q "balloon:\|failed\|not above"; do sleep 2; done
    docker logs qwen38-ram-balloon 2>&1 | tail -1 ;;
  stop) docker stop qwen38-ram-balloon >/dev/null 2>&1 || true; echo stopped ;;
  status) docker logs qwen38-ram-balloon 2>&1 | tail -1 || echo "no balloon"; grep -E "Mlocked|MemAvailable" /proc/meminfo ;;
  *) echo "usage: $0 start [TARGET_GIB] | stop | status" >&2; exit 2 ;;
esac
