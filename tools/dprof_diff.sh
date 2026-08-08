#!/usr/bin/env bash
# dprof_diff.sh <control.log> <test.log> [mark...] — pair two DSV4_DPROF=1 DSV4_KSWEEP=1 runs by
# (K, sub-op) and print the delta.
#
# Why a script: every cycle since F70 has hand-read these tables, and F73's whole lesson was that
# the decisive evidence is a mark inside the SAME run that the change must not have moved (trap 12).
# Reading one number out of a 17 KB log by eye is how a control gets skipped.
set -eu
CTL="$1"; TST="$2"; shift 2
MARKS="${*:-o:wo_a o:wo_b q:wq_a q:wq_b moe:w1w3 moe:w2 i:score TOTAL}"

pull(){  # pull <log> <mark> -> "K1 K2 K3 K4 K5"
  awk -v m="$2" '
    /^\[dprof\] K=/ { split($0,a,"K="); k=a[2]+0; next }
    k>0 && index($0, m" ")>0 {
      for(i=1;i<=NF;i++) if($i==m){ v[k]=$(i+1); break }
      if(m=="TOTAL" && $2=="TOTAL") v[k]=$3
    }
    END{ for(i=1;i<=5;i++) printf "%s ", (i in v ? v[i] : "-") }' "$1"
}

printf "%-10s %-8s %8s %8s %8s\n" "mark" "K" "control" "test" "delta"
for m in $MARKS; do
  c=$(pull "$CTL" "$m"); t=$(pull "$TST" "$m")
  set -- $c; C1=$1 C2=$2 C3=$3 C4=$4 C5=$5
  set -- $t; T1=$1 T2=$2 T3=$3 T4=$4 T5=$5
  for k in 1 2 3 4 5; do
    eval "cv=\$C$k; tv=\$T$k"
    [ "$cv" = "-" ] || [ "$tv" = "-" ] && continue
    printf "%-10s %-8s %8s %8s %7s%%\n" "$m" "K=$k" "$cv" "$tv" \
      "$(awk -v a="$cv" -v b="$tv" 'BEGIN{printf "%+.1f", (b-a)/a*100}')"
  done
done

echo
printf "%-10s %-8s %8s %8s %8s\n" "ksweep" "K" "control" "test" "delta"
for k in 1 2 3 4 5; do
  cv=$(awk -v k="$k" '$1=="[ksweep]" && $2==k && $3=="|" {print $4}' "$CTL" | head -1)
  tv=$(awk -v k="$k" '$1=="[ksweep]" && $2==k && $3=="|" {print $4}' "$TST" | head -1)
  [ -n "$cv" ] && [ -n "$tv" ] && printf "%-10s %-8s %8s %8s %7s%%\n" "verify" "K=$k" "$cv" "$tv" \
    "$(awk -v a="$cv" -v b="$tv" 'BEGIN{printf "%+.1f", (b-a)/a*100}')"
done
