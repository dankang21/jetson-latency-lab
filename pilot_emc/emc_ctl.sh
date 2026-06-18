#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# EMC (memory clock) control via BPMP debugfs — Orin Nano, L4T R36.
#
# Part 2 deliberately left EMC dynamic (set_domain.sh only observes it).
# This pilot needs to PIN it, which goes through the BPMP debug interface:
#   /sys/kernel/debug/bpmp/debug/clk/emc/{rate,mrq_rate_locked,min_rate,max_rate}
# Writing `rate` while mrq_rate_locked=1 forces the BPMP to hold EMC there;
# the BPMP rounds to the nearest entry in its frequency table, so always read
# the rate back instead of trusting the value you wrote.
#
# CRITICAL (R36): mrq_rate_locked alone is NOT enough. The BPMP bandwidth
# manager keeps arbitrating EMC from live bandwidth votes and will override the
# lock back to the demand-driven rate (e.g. the 204 MHz floor under light load),
# so the requested rate is silently ignored. You must also halt the bandwidth
# manager: /sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt = 1. jetson_clocks sets
# this implicitly, which is why a board that has run jetson_clocks locks fine
# while a fresh one does not. We set/clear bwmgr_halt here so locking is
# self-contained and does not depend on ambient jetson_clocks state.
#
# Root only (debugfs). Always `unlock` when done — a locked-low EMC under GPU
# load is the failure mode the Part 2 comment warned about.
#
# Usage: sudo ./emc_ctl.sh {status|points|lock <rate_hz>|unlock}
set -euo pipefail

EMC=/sys/kernel/debug/bpmp/debug/clk/emc
BWMGR=/sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt

die() { echo "emc_ctl: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || die "must run as root (debugfs)"
[ -d "$EMC" ] || die "$EMC not found — check 'mount | grep debugfs' and L4T version (expected R36.x)"
for f in rate min_rate max_rate mrq_rate_locked; do
  [ -e "$EMC/$f" ] || die "$EMC/$f missing — BPMP debugfs layout differs on this L4T"
done

status() {
  echo "rate=$(cat $EMC/rate) locked=$(cat $EMC/mrq_rate_locked) \
min=$(cat $EMC/min_rate) max=$(cat $EMC/max_rate)"
}

case "${1:-}" in
  status) status ;;
  points)
    # 3-point sweep targets: min / arithmetic-mid / max. BPMP rounds the mid
    # to a real table entry on lock; the actual rate is what gets logged.
    MIN=$(cat $EMC/min_rate); MAX=$(cat $EMC/max_rate)
    echo "$MIN $(( (MIN + MAX) / 2 )) $MAX"
    ;;
  lock)
    [ -n "${2:-}" ] || die "lock needs a rate in Hz"
    # Halt the bandwidth manager first, or it overrides the lock (see header).
    [ -e "$BWMGR" ] && echo 1 > "$BWMGR" || echo "emc_ctl: WARN no bwmgr_halt node — lock may be overridden by bandwidth manager" >&2
    echo 1  > $EMC/mrq_rate_locked
    echo "$2" > $EMC/rate
    sleep 1
    ACTUAL=$(cat $EMC/rate)
    echo "requested=$2 actual=$ACTUAL"
    ;;
  unlock)
    echo 0 > $EMC/mrq_rate_locked
    [ -e "$BWMGR" ] && echo 0 > "$BWMGR"   # restore dynamic EMC bandwidth management
    status
    ;;
  *) echo "usage: $0 {status|points|lock <rate_hz>|unlock}"; exit 1 ;;
esac
