#!/usr/bin/env python3
"""byte_floor.py — the byte-weighted achievable floor for base AR decode, from data already on disk.

WHY THIS EXISTS. LEVERS.md quotes a base roofline of 19.0 tok/s = 12.26 GB / 233 GB/s, and that
number assumes every kernel moves bytes at full DRAM bandwidth. No kernel in this engine does: the
same file says "use the K=1 column as the achievability bar, not the roofline", and F76 found at
least one mark that is latency-bound and does not respond to byte cuts at all. So 19.0 is a
normalisation constant. This computes what the decode step would cost if every BYTE-MOVING mark ran
at the best rate this engine has actually demonstrated, and every LATENCY/LAUNCH-bound mark stayed
exactly where it is -- because those do not shrink when you free up bandwidth.

INPUTS, both already on disk, nothing measured here:
  - shapes  : the checkpoint's own config.json (no invented constants)
  - times   : the K=1 dprof table in evidence/kchunk.log (in situ, whole model, 43 layers)

SELF-CHECK. The four MLA rates quoted in LEVERS.md (wq_a 115, wq_b 195, wo_b 185, wo_a 168 GB/s)
were derived by hand from these same two sources. This script recomputes them from config.json and
must reproduce them, or the byte model is wrong and every number below it is wrong too.
"""
import json, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = "/home/patrickd/models/DeepSeek-V4-Flash-0731-REAP/config.json"
DPROF = os.path.join(ROOT, "evidence/kchunk.log")

c = json.load(open(CFG))
H     = c["hidden_size"]            # 4096
QL    = c["q_lora_rank"]            # 1024
OL    = c["o_lora_rank"]            # 1024
OG    = c["o_groups"]               # 8
NH    = c["num_attention_heads"]    # 64
HD    = c["head_dim"]               # 512
MOE_I = c["moe_intermediate_size"]  # 2048
NEXP  = c["num_experts_per_tok"]    # 6
NSH   = c["n_shared_experts"]       # 1
VOCAB = c["vocab_size"]             # 129280

MB = 1024.0**2
# DECIMAL GB, deliberately. Bandwidth is always quoted decimal, so the 233 GB/s probe and the
# 12.26 GB manifest total are decimal, and mixing GiB into the rate column puts every kernel 7.4%
# slow. The first run of this script did exactly that and reproduced LEVERS.md's 115/195/185/168 as
# 107/184/171/160 -- uniformly 1/1.0737 off, which is what a units bug looks like when it is honest
# enough to be visible. The self-check below is the only reason it was caught.
GB = 1.0e9

# Bytes per weight, as the checkpoint stores them (REAP_MANIFEST: MLA/dense FP8 e4m3 with
# F8_E8M0 128x128 block scales; routed experts OCP MXFP4 = E2M1 4-bit + E8M0 scale per 32).
FP8  = 1.0 + 1.0 / (128 * 128)   # scale traffic is 1 byte per 16384 weights: negligible but real
MXFP4 = 0.5 + 1.0 / 32           # 4 bits + one 8-bit scale per 32 values = 0.53125 B/weight
BF16 = 2.0

# Per-CALL weight bytes, keyed by the dprof leaf mark. Shapes from config.json only.
BYTES = {
    "q:wq_a":   QL * H * FP8,                       # [1024, 4096]
    "q:wq_b":   NH * HD * QL * FP8,                 # [32768, 1024]
    "o:wo_a":   OG * OL * H * FP8,                  # [8 x 1024, 4096]
    "o:wo_b":   H * (NH * HD // 4) * FP8,           # [4096, 8192]
    "moe:w1w3": (NEXP + NSH) * 2 * MOE_I * H * MXFP4,
    "moe:w2":   (NEXP + NSH) * H * MOE_I * MXFP4,
    "lm_head":  VOCAB * H * BF16,
}

# Marks that are NOT weight-bandwidth: launch floors, scalar work, latency-bound scans, activation
# math. F73 put mg:* at the launch-latency floor; F76 proved ogroup's own kernel latency-bound.
# These do not shrink when bandwidth frees up, so they enter the floor at their MEASURED time.
FIXED_PREFIX = ("hc_", "rmsnorm", "kv ", "moe:router", "moe:group", "moe:act", "moe:combine",
                "moe:shared", "cattn:compress", "cattn:sparse", "attn:", "q:aq", "q:rms",
                "q:tail", "q:kv_join", "o:rope", "i:", "mg:", "cattn:indexer")

# ---- parse the K=1 dprof table -------------------------------------------------------------
rows, seen = [], False
for line in open(DPROF):
    if "K=1 — verify step by sub-op" in line:
        seen = True; continue
    if not seen:
        continue
    m = re.match(r"\[dprof\]\s+(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)%\s+(\d+)\s*$", line)
    if not m:
        if seen and rows and not line.startswith("[dprof]"):
            break
        continue
    name, ms, pct, calls = m.group(1).strip(), float(m.group(2)), float(m.group(3)), int(m.group(4))
    rows.append((name, ms, calls))

# Parents (ATTENTION, MoE, cattn:q_proj, cattn:ogroup) double-count their children. Drop them:
# a mark is a parent iff another mark's total is nested under it. Determined by name, explicitly.
PARENTS = {"ATTENTION", "MoE", "cattn:q_proj", "cattn:ogroup"}
leaves = [(n, ms, ca) for (n, ms, ca) in rows if n not in PARENTS]

total_ms = sum(ms for _, ms, _ in leaves)

print(f"config: H={H} q_lora={QL} o_lora={OL} o_groups={OG} heads={NH} head_dim={HD} "
      f"moe_int={MOE_I} top-{NEXP}+{NSH} vocab={VOCAB}")
print(f"dprof : {DPROF}  ({len(leaves)} leaf marks, {total_ms:.2f} ms total)\n")

print(f"{'mark':<18}{'calls':>6}{'ms':>8}{'MB/call':>10}{'GB total':>10}{'GB/s in situ':>14}")
print("-" * 66)

byte_rows, fixed_ms, unmodelled = [], 0.0, []
for name, ms, calls in leaves:
    if name in BYTES:
        tot = BYTES[name] * calls
        rate = tot / (ms / 1000.0) / GB
        byte_rows.append((name, ms, calls, tot, rate))
        print(f"{name:<18}{calls:>6}{ms:>8.2f}{BYTES[name]/MB:>10.2f}"
              f"{tot/GB:>10.3f}{rate:>14.1f}")
    elif name.startswith(FIXED_PREFIX):
        fixed_ms += ms
    else:
        unmodelled.append((name, ms)); fixed_ms += ms

print("-" * 66)
byte_total = sum(r[3] for r in byte_rows)
byte_ms = sum(r[1] for r in byte_rows)
print(f"{'byte-modelled':<18}{'':>6}{byte_ms:>8.2f}{'':>10}{byte_total/GB:>10.3f}"
      f"{byte_total/(byte_ms/1000)/GB:>14.1f}")
print(f"{'fixed (latency/launch/act)':<32}{fixed_ms:>8.2f} ms")
if unmodelled:
    print("  not byte-modelled, counted as fixed: " +
          ", ".join(f"{n} {v:.2f}ms" for n, v in unmodelled))

# ---- the floor ------------------------------------------------------------------------------
# SELF-CHECK against the four rates LEVERS.md derived by hand from these same two files. If the
# byte model is wrong, every floor below it is wrong, and a silent 7% units error is exactly the
# kind of thing that would otherwise be quoted for weeks.
EXPECT = {"q:wq_a": 115, "q:wq_b": 195, "o:wo_b": 185, "o:wo_a": 168}
print("\nself-check vs LEVERS.md hand-derived rates:")
bad = 0
for name, _, _, _, rate in byte_rows:
    if name in EXPECT:
        err = 100.0 * (rate - EXPECT[name]) / EXPECT[name]
        flag = "ok" if abs(err) <= 5.0 else "MISMATCH"
        bad += flag == "MISMATCH"
        print(f"  {name:<10} computed {rate:6.1f}  ledger {EXPECT[name]:4d}  {err:+5.1f}%  {flag}")
if bad:
    sys.exit("byte model disagrees with the ledger; fix it before trusting any floor below")

best_rate = max(r[4] for r in byte_rows)
best_name = max(byte_rows, key=lambda r: r[4])[0]
# moe:w2 comes out ABOVE the 233 GB/s probe. That is not a kernel beating DRAM -- it is L2 reuse
# (w2 reads the same expert set w1w3 just touched), so it is NOT a safe universal target rate.
# The defensible target is the best rate a LARGE, cache-unfriendly mark achieves cold.
cold = [r for r in byte_rows if r[4] <= 233.0]
cold_best = max(r[4] for r in cold)
cold_name = max(cold, key=lambda r: r[4])[0]

print(f"\nB_tok byte-modelled coverage: {byte_total/GB:.3f} GB of the 12.26 GB manifest total "
      f"({100*byte_total/GB/12.26:.0f}%)")
print(f"best in-situ rate demonstrated by this engine: {best_rate:.1f} GB/s ({best_name})\n")

def floor(rate, label):
    ms = byte_total / GB / rate * 1000.0 + fixed_ms
    print(f"{label:<46} {ms:>7.2f} ms/tok = {1000.0/ms:>5.2f} tok/s")
    return ms

print("FLOORS (fixed marks held at measured time; byte marks re-rated):")
opt = floor(233.0, "  OPTIMISTIC: every byte mark at 233 GB/s")
real = floor(cold_best, f"  REALISTIC: every byte mark at {cold_best:.0f} GB/s ({cold_name}, cold)")
now = sum(r[1] for r in byte_rows) + fixed_ms
print(f"{'  NOW: every byte mark at its own measured rate':<46} {now:>7.2f} ms/tok "
      f"= {1000.0/now:>5.2f} tok/s")
print(f"\nheadroom on base AR decode: {100*(now-real)/now:.1f}% (realistic) "
      f"to {100*(now-opt)/now:.1f}% (optimistic)")
print(f"  -> base AR {1000.0/now:.2f} -> {1000.0/real:.2f}-{1000.0/opt:.2f} tok/s")
print(f"  vs the quoted 19.0 tok/s roofline, which assumes 233 GB/s AND zero fixed cost "
      f"({fixed_ms:.1f} ms/tok of this step is not bytes)")
print("\nCAVEAT, stated because it cuts the other way: the byte model covers 77% of B_tok. The\n"
      "missing ~23% (indexer, compressor, norms, embed) is charged at its MEASURED time as if it\n"
      "were fixed cost, so these floors are PESSIMISTIC on that axis -- those marks do move bytes\n"
      "and would shrink too. Treat the realistic floor as an upper bound on ms/tok, not a target.")
