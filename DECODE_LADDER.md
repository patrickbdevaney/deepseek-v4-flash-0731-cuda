# Decode ladder — the ordered work list the autonomous loop executes

`scripts/decode_loop.sh` reads this file, picks the topmost `TODO`, does it, gates it, measures it,
and marks it. **Edit this file to steer the loop.** One item per iteration; one change per
measurement; bands not points.

## The stop condition — what "no faster combined decode is possible" means here

Combined decode is `tok/s = tau * 1000 / ms_per_forward`, and `ms_per_forward = a + b*ctx`
(`tools/decode_model.py`). Both terms have a **byte floor** at the measured 240 GB/s:

- **Term A floor** — forward bytes at the measured K=5 expert union (17.53 of 30) plus the draft
  side: **82.18 ms**. Term A was 136.44 ms at suspension = **60.2 % of achievable**.
- **Term B floor** — 42.0 MB at ctx 6592 = 0.509 ms/forward at tau 2.91. Term B was **198.1 ms**.

**Where the two terms stand now (1.2, 2026-08-20):** `a = 129.11 ms` = **1.57x** the 82.18 ms floor
(stop wants <= 1.25x); `b x 6592 = 16.57 ms`, down from 47.60 (stop wants <= 5.0 ms). Neither is met,
but **term B has now moved twice**: 1.0 took the context slope from `7.220 +/- 0.165` to
`4.006 +/- 0.210` ms per 1000 (paired saving `3.604 +/- 0.076`), and 1.2 took it from
`3.488 +/- 0.179` to `2.514 +/- 0.151` (paired saving `0.793 +/- 0.031`). **Cumulatively `b` is
down 65 % from the 7.220 the ladder opened on.** Term A is untouched and is now much the larger
distance from its floor — it is 129.11 of a 154.66 ms forward at ctx 12,410, i.e. **83 %**. The fit
reaches context 12,410, so the stop check does not extrapolate.

**1.5 moved term B again, 2026-08-20.** The paired saving over 16 legs is
**0.572 +/- 0.018 ms per 1000 context** (R^2 0.987, every leg negative, `tau` and emitted text
identical in all 16), so the tracked `b` goes **2.514 -> 1.942** and `b x 6592` goes
**16.57 -> 12.80 ms**. That is carried by SUBTRACTION of a paired number and not by re-fitting:
1.5's own sweep spans ctx 3,197-12,410 where a fit determines `b` badly (R^2 0.433 on the after
arm), and rule 7 applies to your own arms too. Cumulatively `b` is down **73 %** from the 7.220 the
ladder opened on. Term A is untouched and is now 83 % of the forward.

**1.7 MOVED TERM A, 2026-08-20 — the first item on this ladder to do so.** `sparse_attn` stages the
gathered KV row in shared memory (`wiki/kernel-optimisations.md` §2.9). The paired saving over 16
legs is **4.227 +/- 0.121 ms per forward**, every leg faster, all 16 emitting a byte-identical
`text_sha256` with `tau` equal to four decimals at ctx up to 12,282. Split at the `topk` knee it is
**-3.996 +/- 0.080 ms flat and -0.0552 +/- 0.0102 per 1000** -- 94 % Term A, and the 6 % that is not
has a named mechanism (the 20 ratio-128 layers, whose `topk` has no `index_topk` cap, predicted
0.048 ms per 1000 before it was fitted). So **`a` 129.11 -> 125.11 ms** (1.571x -> **1.522x** the
82.18 ms floor; stop wants <= 1.25) and **`b` 1.942 -> 1.887** (`b x 6592` 12.80 -> **12.44 ms**;
stop wants <= 5.0). Cumulatively `b` is down **74 %**.

**1.8 MOVED NEITHER TERM AND WAS NOT SUPPOSED TO — it DELETED headroom that was never there,
2026-08-20.** The `cattn:q_proj` bimodality is the compressor emits running on `g_side` while the
mark is timed on the main stream: `g = #{ j in [ctx,ctx+VB) : (j+1)%4==0 }` classifies it **153/153
on 0.4's own log and 174/174 on a fresh split arm, with no overlap between the populations**, and
under `NO_ATTN_SPLIT=1` the 3.2x swing **disappears entirely** (1.00-1.02x) while the identical time
reappears in `cattn:compress` (2.50 -> 8.34 ms at fixed VB=2). So 0.4's "~4.4 ms/step of Term A for
free" does not exist: `a` stays 125.11 ms and the distance to its floor stays 22.4 ms. What the item
DID buy is a number for something nothing had priced — the compressed-KV emit is **7.02 ms on the
64.9 % of forwards that carry one, 4.56 ms/forward amortised, 3.3 % of the forward, at 52 % of its
880 MB byte roofline** — and a paired confirmation that the ATTN_SPLIT overlap is worth
**0.81 ms/forward, 2 SE band [0.72, 0.90], 9 of 9 legs faster** (9/9 byte-identical, tau equal to
three decimals on every leg) — a band the mark-level prediction of 0.838 falls inside.

**1b.2 SHIPPED A KERNEL, 2026-08-20, so rule 5 is cleared — and it is the first item on this ladder
that is a TRADE rather than a win.** Packing the KV cache 2048 -> 720 B per row is bit-exact (16 of
16 legs byte-identical, `tau` equal on every leg) and measures **`+3.233 +/- 0.203 ms/forward` flat
against `-0.4996 +/- 0.0290 per 1000 context`, R^2 0.990** -- `a` 125.11 -> 128.34 and `b` 1.887 ->
1.387 (`b x 6592` 12.44 -> 9.15). **Break-even is ctx 6,471**, which is within 2 % of the ladder's
own 6592 reference, so at that context it is a wash; at 12,410 it is +2.1 % tok/s and at 3,197 it is
-1.4 %. **It ships default OFF** because `a` is the binding term and this makes `a` worse. The
tracked `a`/`b` above are therefore UNCHANGED -- the numbers in this paragraph are what
`DSV4_KV_PACK=1` costs and buys, not the shipped state. The mechanism is worth carrying forward: a
2.84x byte reduction made `sparse_attn` **slower at every shape** (0.78-0.90x) because that kernel is
issue-bound, and the saving showed up entirely in the context term. See item 1b.2 and
`wiki/negative-results.md` §4i.

**1.11 SHIPPED A KERNEL AND IS ADOPTED DEFAULT ON, 2026-08-20 — and the headline is not its size,
it is that ONE arm order called it a null.** Deferring the ATTN_SPLIT join past `i:qidx` and `i:iw`
is **-0.542 +/- 0.310 ms/forward drift-free** over 18 paired legs and FOUR checkpoint loads, and
**-1.092 +/- 0.146 at ctx 12,410**, where it is 6/6 in each arm order independently (+0.77 % tok/s).
Bit-exact: 44 of 44 sweep legs byte-identical, `tau` equal to three decimals on every one,
`gate_join_defer` 0 of 143,360 floats differing, LOSSLESS PASS in both arms.

Run the standard way -- control arm first, so drift penalises the change -- it measured
**-0.324, 2 SE [-0.670, +0.022]**, a band covering zero, which under the item's own pre-registration
is a negative result and would have been written up as one. **Pairing removes between-LEG variance
and does nothing to a between-LOAD offset**, and traps §19 already measured that offset at 0.6 % =
~0.8 ms/forward, the size of the whole effect. Running the identical pair with the ARM ORDER REVERSED
and averaging the per-leg deltas cancels it and measures it: **the second checkpoint load of a
session costs +0.218 +/- 0.157 ms/forward**. `a` needs another 22.4 ms and there is no single 22 ms
item left, so it will be taken in pieces this size -- **below ~1 ms/forward, a null from one arm
order is not a result, and reporting one retires a real win.** `tools/paired_band.py --reversed`.

**`a` and `b` are UNCHANGED at 125.11 and 1.887, on purpose.** 1.11 pre-registered a FLAT saving and
the engine did not deliver one: it resolves at ctx 12,410 (6/6, sd 0.179) and covers zero at 3,197
and 6,260 (sd 0.585, 0.741), and a three-point fit through groups with 3-4x different scatter is not
an attribution. The ratchet is stated where it resolves -- `a + b*ctx` at ctx 12,410 goes
**142.24 -> 141.15 ms** -- and the term is item 1.13.

**1.12 MOVED TERM A, 2026-08-20 — the second item on this ladder to do so, and the first whose
mark-level and throughput instruments agreed to within their own bands.** `gemm_fp32` gets a (4,4)
warp tile instead of one warp per (8-row chunk, n), so operand traffic goes `M*N*K*(1/1 + 1/8) ->
M*N*K*(1/4 + 1/4)` and sixteen host launches become one. The drift-free paired saving over 18 legs
and FOUR checkpoint loads is **-1.963 +/- 1.504 ms/forward, 14/18 legs, band [-3.467, -0.459]
EXCLUDING zero**, and it is FLAT: `-2.348 +/- 3.302` flat against `+0.0533 +/- 0.4038` per 1000
context, R^2 0.004. So **`a` 125.11 -> 123.15 ms** (1.522x -> **1.499x** the 82.18 ms floor; stop
wants <= 1.25) and **`b` stays 1.887** (`b x 6592` stays 12.44 ms; stop wants <= 5.0).

The mark-level instrument predicts that band from a completely different measurement: the ratio-128
emit spike is **40.76 -> 14.50 ms (2.82x, 69 samples, populations disjoint)**, the verify TOTAL
excess on ratio-4 boundary steps is **-2.49 ms on 62.8 % of steps**, and
`0.628 x 2.49 + 0.0239 x 26.42 = -2.20 ms/forward` falls inside `[-3.467, -0.459]`. **And ONE ARM
ORDER CALLED IT A NULL AGAIN** (-0.892, 2 SE [-2.134, +0.350]) -- traps §33 on the very next item.
Bit-exact: 252 tile x shape comparisons at max|diff| = 0 with a live negative control, identical
generated ids, LOSSLESS PASS in both arms. See item 1.12 and `wiki/kernel-optimisations.md` §2.11.

**1.11 also wrote the first gate here that can reach the side-stream fork.** Every gate that links
these kernels does so WITHOUT `arena_init()`, so `g_side` is null, `asplit` is false, and the fork
1.8 priced at 0.81 ms/forward had never been under test. `tests/gate_join_defer.cu` refuses to PASS
if `g_side` comes back null -- and its NULL control immediately found a pre-existing uninitialised
arena read in the M=1 step (item **1.14**), which had been silently corrupting the first run of every
bit-exactness gate in this repo.

**Rule 5 is clear: 1b.2 and 1.11 both shipped kernels.** 1.8's remaining follow-up (1.12) is
honestly small — a 0.53 ms/forward ceiling — and is ranked accordingly; do not take it just because
it is fresh. 1.11 is the evidence for the ranking cutting the other way too: its ceiling was 1.29
ms/forward and it delivered 0.54, which is what a 22.4 ms gap now gets paid down in.

**Read the two terms against their stop conditions before picking the next item.** `b x 6592` needs
to fall 2.5x more; `a` needs to fall to 102.7 ms, i.e. by another 22.4 ms. Six items have moved `b`
by 74 %; one item has moved `a` by 3.1 %. `a` is 125.11 of a 142.24 ms forward at ctx 12,410 --
**88 %** -- and it is where the remaining headroom is.

**2.2 moved NEITHER `a` NOR `b`, by construction, and still bought +9.52 % — read the stop
condition carefully before concluding it should have moved.** The stop check is about
`ms_per_forward = a + b*ctx`; 2.2 changes no kernel and no forward, it swaps the draft head's
weights. Its gain lands entirely in the other factor of `tok/s = tau * 1000 / ms_per_forward`:
`tau` 3.5362 -> 3.8438 on the frozen suite. **That factor has no floor written into the stop
condition at all**, which is worth noticing — the ladder can reach "no faster combined decode is
possible" on both byte floors while acceptance is still 3.84 of a possible 5 or 7. The acceptance
axis is Phase 2's whole point and the long-horizon pivot below is entirely about it.

**1.3 moved neither term and was pre-registered as unable to** — it is a ~4 us/call kernel saving
that fires only below ctx 2048, worth 0.07 % of a forward; `b` measured `3.008 +/- 0.241 ->
3.036 +/- 0.240` across its arms. It is marked done, not skipped, and the ladder rule it produced
(re-attribute before you pick) is above.

*(The 1.2 before-arm slope, 3.488, is not the 4.006 that 1.0 reported after. Same protocol, same
corpus sha, an independent server start eight hours later; the gap is one run-to-run spread on a
fitted coefficient and both arms of 1.2's comparison are inside the same run, which is why the
paired number and not the fit difference is the ratchet.)*

**The loop STOPS when either:**
1. `a <= 1.25 * a_floor` **and** `b*6592 <= 5.0 ms` — i.e. both terms are within a quarter of their
   byte floors and the context term has stopped mattering; or
2. five consecutive iterations move the suite by less than **2 %**, which is below the measured
   3.5 % run-to-run spread and therefore unmeasurable; or
3. `DECODE_LOOP_STOP` exists, or the iteration/wall caps are hit.

Reaching (1) is the roofline. Reaching (2) means the remaining headroom is not addressable by the
items on this ladder and the loop should hand back rather than thrash.

## THE POINT IS FASTER DECODE, NOT BETTER INSTRUMENTS

Read this before picking an item. Measurement earns its place only when it changes what gets built
next, and this ladder has now spent **four of its first five items on instruments** (0.1-0.4). Each
was defensible in isolation and 0.4 in particular paid for the whole run -- it proved 0.2 had never
varied context at all, un-retired 1.2 and 1.5, and found in `draft:main_kv` a larger item than
either -- but the engine had not gotten faster since the warp top-k landed, that was five
iterations, and the whole context term was attributed with error bars.

**BROKEN 2026-08-20 by 1.0: +24.4 % tok/s at ctx 12,282, and the rule is what produced it.** 0.4
bought exactly one thing -- it named the largest term and predicted its size -- and the very next
iteration spent that prediction on a kernel and got 93 % of it back. That is the whole case for the
rule, and also its limit: **the remaining items are smaller and none of them needs a new
instrument.** `i:topk`, `i:score` and `cattn:sparse` are already attributed with error bars. Build
them. The one open instrument-shaped item, 1.9, is deliberately ranked below both kernel changes.

**1.9 RAN ON 2026-08-20 AND SHIPPED NO KERNEL, so rule 5 is now live: the next iteration takes the
highest-expected-value kernel item even if it is less certain.** It was worth its slot -- it turned
"the engine stops reproducing itself" into one function, one length threshold and a race, and it
killed two standing hypotheses -- but it is the fifth instrument in ten items and the engine has not
gotten faster since 1.5.

So, in order:

1. **Prefer an item that changes a kernel over an item that measures one.** If both are available,
   take the kernel.
2. **An instrument is justified only by naming the optimisation it unblocks**, in the ladder entry,
   before building it. "We need better data" is not a justification; "we cannot choose between 1.2
   and 1.5 without attributing the residual" is.
3. **A speedup that is not in the wiki did not happen.** The moment a kernel change is measured and
   kept, write it into `wiki/` in the same iteration -- `kernel-optimisations.md` for an adopted win
   (mechanism -> measured gain -> the gate that proved it), `negative-results.md` for a lever that
   was built and killed, `context-scaling.md` for anything that moves the context term, and
   `measurement-and-traps.md` for any new way a number turned out to be untrustworthy. Update
   `wiki/README.md`'s state table in the same commit. The wiki said "the M=1 kernel path is
   finished" for weeks while the largest term in decode had never been timed; a page that is not
   maintained as the work lands becomes confidently wrong, which is worse than absent.
4. **Every iteration that ships a kernel change must report the before/after on the same corpus**,
   with tau, as a band. That is the ratchet: a number that went up, on a fit that already exists.
5. **If two consecutive iterations produce no kernel change, say so at the top of the ladder entry**
   and take the highest-expected-value kernel item next even if it is less certain. Thrashing on
   measurement is the failure mode this section exists to prevent.
6. **RE-ATTRIBUTE BEFORE YOU PICK, not after you build. ADDED 2026-08-20 by 1.3, which is the
   counter-example.** A ladder ordering is a function of a cost model, and every landed item
   changes the cost model. 1.3 was ranked where it was because `i:topk` measured 13.47 ms at ctx
   12,288; 1.2 took `i:topk` to 0.72 ms at ctx 6144 and 1.3 was built the next iteration against a
   headroom that no longer existed. It shipped, it is bit-exact, and it moved nothing. **The check
   costs nothing** -- every A/B in this ladder already runs a dprof pair, so the *previous*
   iteration's dprof is on disk and re-ranks the list for free. Read it before choosing.
   Then read rule 7, because the number you read is not always what it looks like.
7. **A slope only means "cost per 1000 context" if the mark is linear in context.** Check the
   per-point medians before you rank on a fitted `b`. `cattn:sparse` fits 0.709 over ctx 3k-12k and
   1.694 over ctx 0.4k-6k -- not a contradiction and not drift, but one concave curve read over two
   ranges, and ranking on the larger number would have promoted item 1.7 over 1.5 on an artefact.
   See `wiki/measurement-and-traps.md` §16.

## Hard invariants — the loop must never violate these

1. **Bit-exactness or an explicit gate.** Any kernel change either produces byte-identical generated
   token ids, or ships behind the LOSSLESS gate with the deviation measured and recorded. A change
   that silently alters output is reverted, not debugged.

   **AMENDED 2026-08-20 by 1.0 — token ids are not a valid test at context, and this invariant was
   nearly used to revert a correct change.** The engine stops reproducing itself part-way through a
   long run: `build/decode` with `DSV4_MAINKV_CACHE=0` on **both** sides of the comparison, argmax,
   no seed, no HTTP, still diverged from itself at the identical id index 1067. The cached path
   never ran on either side, so the divergence cannot be the cache. **A token-id comparison is only
   evidence if the same-arm control was run and passed — run the control FIRST.** Where it fails,
   the invariant is satisfied instead by memcmp of the changed kernel's **entire output buffer**
   against the untouched reference, in situ, on every call; that is strictly stronger, because it
   proves the whole intermediate identical rather than one downstream consumer's output. See
   `wiki/measurement-and-traps.md` §12 and item 1.9.

   **BOUNDED 2026-08-20 by 1.9, which replaces "at context" with a measured number.** The mechanism
   is `build/decode`'s PREFILL, inside the compressed layers, and it has a threshold: the whole
   43-layer prefill is byte-identical run-to-run at **160 prefill positions and below** (430 layer
   hashes, ten point-comparisons, zero differences) and nondeterministic at **192 and above**, all
   the way to 3,071. So the token-id invariant is a VALID gate for any comparison whose prompts
   prefill 160 positions or fewer — including the canonical 6-id gate prompt — and is reliably
   invalid above ~192. `dsv4-server` prefills in `EXT_CHUNK` = 64-row chunks through a different
   function and does not reach the threshold at all, which is why 1.5's 16-leg server A/B was
   byte-identical at ctx 12,410; a bit-exactness result on one binary's prefill does not transfer to
   the other. `wiki/measurement-and-traps.md` §25-§27.
2. **tau is reported in every A/B.** A byte-identical token sequence can still collapse acceptance
   3.12 -> 1.00 (`LOOP_LOG`), because acceptance is an exact draft/target comparison. Throughput
   without tau is not a measurement.
3. **Never restart the eval battery.** The watchdog and boot unit are disarmed on purpose. The GPU
   is single-tenant; a second client turns a scored item into a banked wrong answer.
4. **Never modify the checkpoint.** Weights are not ours to edit. The draft head is backed up at
   `~/model-backups/heads/shipped-dspark-0731reap/`.
5. **Clocks are a measurement variable.** Do not change `jetson_clocks` mid-comparison. It is its
   own ladder item with its own before/after.

---

## Phase 0 — instrument (must complete before anything is trusted)

- [x] **0.1** `timings` on `/v1/completions` — staged in `server.cpp`, built 2026-08-19.
- [x] **0.2** `DSV4_DPROF` diff at two contexts. **DONE 2026-08-19.** The headline below — "the
      context term is GONE" — is **SUPERSEDED BY 0.3, which measured it as 4.1x smaller, not gone.**
      Both runs here were at ctx <= 3000, where the residual term is 22 ms of a 153 ms forward and
      two single runs at a 3.5 % spread cannot see it. Read the retirements of 1.2/1.5 with 0.4.
      Two single-prompt runs on the same binary, ctx 480 and ctx 3000:

      | | base AR (M=1) | spec | i:topk (21 calls) | i:score | cattn:indexer |
      |---|---|---|---|---|---|
      | ctx 480 | 89.3 ms/tok | 65.9 ms/tok, 15.18 tok/s | 0.10 ms | 0.02 ms | 2.09 ms |
      | ctx 3000 | 89.4 ms/tok | 65.2 ms/tok, 15.33 tok/s | 0.10 ms | 0.02 ms | 2.05 ms |

      **Base AR moved 0.1 ms across a 6x context change.** The pre-fix fit predicted
      150.9 -> 226.6 ms over the same span (`136.44 + 30.053*(ctx/1000)`). The slope went from
      **30.05 ms per 1000 context to ~0.04**, and `i:topk` is flat at 0.10 ms across all 21 calls.
      Both LOSSLESS gates passed.

      **This retires 1.2 and 1.5 as decode levers.** A radix select replaces a kernel that now
      costs 0.10 ms, and the `index_score` GEMM attacks `i:score` at 0.02 ms. Neither is worth a
      bit-exactness risk. 1.3 and 1.4 remain as correctness items, not performance ones.
      The remaining cost is **ATTENTION 44 % and MoE 38 %**, which is Term A — and Term A is the
      term with ~1.5x of headroom, not 3x.
- [x] **0.3** Re-fit `tools/decode_model.py` on a post-fix run and record both coefficients.
      **DONE 2026-08-19. `a = 130.98 +/- 2.25 ms`, `b = 7.362 +/- 0.370 ms per 1000 context`**
      (95 % CI; n = 48, R^2 0.971, measured context **249 to 12,410** — the first fit this repo has
      ever had above 6.6k). Binary rebuilt from HEAD (d961544) so the fit describes shipped source;
      all six CPU gates pass. Server at seqmax 16384, `mem 120.0/122.8 GiB`, memguard armed.

      | | pre-fix (battery, n 2158) | post-fix (sweep, n 48) | |
      |---|---|---|---|
      | `a` | 136.44 +/- 0.52 ms | **130.98 +/- 2.25 ms** | unchanged, see caveat |
      | `b` | 30.051 +/- 0.241 ms/1000 | **7.362 +/- 0.370 ms/1000** | **4.08x smaller** |
      | `b x 6592` | 198.10 ms | **48.53 +/- 2.44 ms** | stop condition wants <= 5.0 |
      | context range | 71 - 6592 | 249 - 12,410 | |
      | tau p50 | 2.91 | 1.68 | different corpus — see caveat |

      **Neither stop condition is met and the loop continues.** `a` is **1.59x** its 82.18 ms byte
      floor against a target of 1.25x; `b x 6592` is **9.7x** the 5.0 ms threshold. The context term
      is now 10 % of a forward at ctx 2000 (was 31 %) and **41 % at ctx 12,410**.

      Per point, medians over 6 repeats (`evidence/decode_loop/fit/postfix.sweep.jsonl`):

      | ctx | ms/forward | spread | tau | mean verify width | tok/s |
      |---|---|---|---|---|---|
      | 249 | 127.51 | 5.2 % | 1.636 | 2.62 | 12.83 |
      | 889 | 136.77 | 2.6 % | 1.724 | 2.64 | 12.65 |
      | 1664 | 147.66 | 2.6 % | 1.701 | 2.68 | 11.52 |
      | 3197 | 160.54 | 3.0 % | 1.626 | 2.69 | 10.08 |
      | 6260 | 179.18 | 2.4 % | 1.718 | 2.66 | 9.58 |
      | 9341 | 204.68 | 7.4 % | 1.969 | 2.92 | 9.66 |
      | 12410 | 215.33 | 4.3 % | 1.652 | 2.66 | 7.68 |

      **The slope is not a speculation artifact.** `tau` and realised verify width both correlate
      +0.35 with context here, and width is what sets bytes per forward, so the raw slope could in
      principle have been an acceptance effect wearing a context costume. Regressing on both:
      `fwd = 70.74 + 7.029 x (ctx/1000) + 23.04 x width` (R^2 0.985, SE 0.145 and 3.50). Width does
      cost a real 23 ms per unit, and it absorbs **0.33 of the 7.36** — the context term survives at
      **7.03 +/- 0.28** with width held fixed.

      **The cheap prefill was checked, not assumed.** The sweep visits contexts descending through
      token-prefixes of one document, so the engine's longest-common-prefix cache turns all but the
      first prefill into a `rewind_to` (`prefill=0ms` on 47 of 48 legs, ~40 min of prefill saved).
      If `rewind_to` left the compressed or index caches in a cheaper state than a freshly built
      context, every number above would be measuring a path the engine does not ship. Four control
      legs built their context from a **different document** (shared prefix: 1 token) and so took
      the build path: **181.59 vs 179.18 ms/forward at ctx ~6250 (1.3 %)** and **150.09 vs 147.66 at
      ctx ~1660 (1.6 %)**, both inside the 3.5 % run-to-run spread. The shortcut is sound.

      **CAVEAT, and it decides what may be claimed.** This is a different population from the
      battery corpus: raw `/v1/completions` continuation of one technical document, versus chat
      benchmarks. tau is **1.68 here against 2.91 there**, so fewer positions are verified per
      forward, a smaller expert union is read, and `a` is *expected* to come in below the battery's
      136.44 for reasons that have nothing to do with any kernel. **`a` must therefore be read as
      unchanged, not improved** — the top-k fix never touched Term A, and 130.98 at tau 1.68 is if
      anything the optimistic end. The `b` comparison is the sound one: each number is a slope
      fitted *within* its own corpus, and the ~4x is far outside either interval.
      Likewise **the tok/s column above is not comparable to `PERF.md`'s 22.66 suite mean** — same
      engine, lower-tau workload.

      Instruments: `tools/decode_fit_probe.py` (new; detached, fatal on transport failure, writes no
      record for a leg that generated nothing, and now named in `detach_audit.sh`);
      `tools/decode_model.py --dir` plus standard errors on both coefficients, because once `b` is
      small R^2 is the wrong statistic — a genuinely flat slope scores R^2 ~ 0 however precisely the
      zero is known, and the stop condition is a claim about an interval on `b`.
- [x] **0.4** `DSV4_DPROF` at ctx 12k, and attribute the residual 7.36 ms/1000. **DONE
      2026-08-20. The residual is fully attributed and 0.2 was wrong at the root: 0.2 never varied
      context.**

      **NO KERNEL CHANGED IN 0.3 OR 0.4** — two consecutive measurement iterations, which the
      section above says must be declared. It is declared. The next item taken is **1.0**, a kernel
      change, and 0.4 is what made it findable.

      **0.2 DID NOT MEASURE TWO CONTEXTS. It measured the same context twice.** `src/decode.cu`
      takes its prompt from **argv[2]**; `DSV4_PROMPTS_FILE` only appends prompts 1..N, and the
      base-AR and K-sweep paths both run prompt 0. Both 0.2 runs passed a **2-id** argv[2], so both
      printed `prefill 1 positions`, `step 0 pos 1`, and `[spec] ... prompt=0 s=2`, and both emitted
      the identical 24 tokens `223 643 27 15 397 5029 515 260 ...`. The labels "ctx 480" and
      "ctx 3000" were `seqmax`, which is sized from the longest prompt in the FILE
      (`480+24+7+8=519`, `3000+24+7+8=3039`) — an allocation, not a context. Two runs of the same
      computation at ctx 1..9, reported as a 6x context sweep. That is the whole of the evidence
      that retired 1.2 and 1.5.

      **THE ATTRIBUTION** (`evidence/decode_loop/dprof_ctx_0p4.txt`, one server load, seqmax 16384,
      145 steady-state verify steps at four depths 772..12,406, `tools/dprof_ctx.py`). `b|VB` holds
      the realised verify width fixed, because width sets bytes-per-forward and correlates with
      context (0.3); it is the number to read.

      | mark | ms/1000 ctx | `b\|VB` | ms at ctx 768 | ms at 12,288 | share of step slope |
      |---|---|---|---|---|---|
      | **STEP (verify+draft)** | 6.488 +/- 0.332 | **6.969 +/- 0.112** | 126.3 | 203.9 | 100 % |
      | **`draft:main_kv`** | 3.867 +/- 0.001 | **3.867 +/- 0.001** | 3.30 | **47.87** | **55 %** |
      | `ATTENTION` | 3.021 +/- 0.146 | 3.173 +/- 0.109 | 52.3 | 87.4 | 46 % |
      | &nbsp;&nbsp;`cattn:indexer` | 1.472 +/- 0.040 | 1.515 +/- 0.029 | 5.80 | 22.20 | 22 % |
      | &nbsp;&nbsp;&nbsp;&nbsp;**`i:topk`** | 0.871 +/- 0.021 | **0.872 +/- 0.021** | 2.58 | **13.47** | **12.5 %** |
      | &nbsp;&nbsp;&nbsp;&nbsp;**`i:score`** | 0.609 +/- 0.029 | **0.644 +/- 0.018** | 0.92 | **6.58** | **9 %** |
      | &nbsp;&nbsp;`cattn:sparse` | 0.695 +/- 0.050 | 0.709 +/- 0.050 | 11.00 | 21.17 | 10 % |
      | `MoE` | -0.380 +/- 0.198 | **-0.079 +/- 0.029** | 35.8 (med) | 35.8 (med) | ~0 |
      | every GEMM (`q:wq_a/b`, `o:wo_a/b`, `moe:w1w3/w2`, `cattn:ogroup`) | | within 1 SE of 0 | | | ~0 |

      **The instrument reproduces the fit it was built to explain, which is the check that it is
      measuring the right thing.** 0.3's wall-clock fit: `b = 7.362 +/- 0.370`, width-controlled
      **7.029 +/- 0.28**. This run's dprof step total: `6.488 +/- 0.332`, width-controlled
      **6.969 +/- 0.112** — a 0.9 % difference on the number that matters. The per-mark slopes sum
      **exactly** to the step slope (the tool prints that identity rather than assuming it), so
      nothing in the step is unaccounted for. The wall-clock legs also reproduce 0.3 point for
      point (214.7/215.9 vs 215.33 at ctx 12.3k; 181.5/182.5 vs 179.2 at 6.1k; 160.8/162.6 vs
      160.5 at 3.1k), so **`DSV4_DPROF` costs nothing measurable** and this is a profile of the
      shipped cost, not of an instrumented one.

      **1.2 AND 1.5 ARE UN-RETIRED, and the numbers they were retired on were off by 100x.**
      0.2 reported `i:topk` 0.10 ms and `i:score` 0.02 ms. At ctx 12,288 they are **13.47 ms** and
      **6.58 ms** — **135x** and **330x** — and together they carry **1.52 of the 6.97 ms/1000**.
      They grow because they are O(context) by construction and 0.2 ran them at context 9.

      **AND A LARGER ITEM THAN EITHER WAS FOUND, because the draft side had never been timed at
      all.** Every dprof mark in this repo lived in the VERIFY stack, so `dprof_report`'s TOTAL has
      only ever described part of a decode step. `src/engine.cu` calls `dspark_main_kv` **NSTAGE=3
      times per step over the full `ctxlen`** — a from-scratch fp8 GEMM + rmsnorm + rope + quant
      over EVERY position in the context, recomputed on every single token. It is **47.87 ms at
      ctx 12,288, the largest single row in the step**, and **55 % of the entire context term** —
      more than `i:topk`, `i:score` and `cattn:sparse` combined. Its R^2 against context is
      **1.000**: it is not noisy, it is arithmetic. This is now item **1.0**.

      **What is NOT the context term.** MoE is flat (`-0.079 +/- 0.029` per 1000 with width fixed);
      its apparent -0.38 uncontrolled is a width artifact and disappears when width is held. Every
      dense GEMM is flat. `cattn:ogroup`, 15.8 ms and the second-largest verify row, is flat. So
      Term B is **not** a bytes problem — at ctx 6592 its byte floor is 0.509 ms/forward
      (DECODE_ZENITH_FINDINGS 1.2) against ~43 ms measured, still ~85x off roofline. It is serial
      and redundant work: one recompute (`draft:main_kv`) and one selection sort (`i:topk`).

      Instruments, both justified before building and both of which paid: (1) `src/engine.cu` now
      calls `dprof_report` once per verify, tagged with the context it was taken at — it had called
      `dprof_init` since the marks were written and never reported, so **every mark the server ever
      recorded was discarded**, which is why the only profiles that existed came from decode.cu's
      argv[2]-length K-sweep. (2) Three new draft-side ids (`draft:main_kv/block/head`) placed after
      `DP_ARGMAX` so historical tables stay comparable. (3) `tools/dprof_ctx.py` fits the ladder's
      own linear model per mark with standard errors, because once a mark is flat R^2 is the wrong
      statistic and the question is whether an interval covers zero. (4) `--targets/--no-control` on
      `decode_fit_probe.py`; (5) `scripts/dprof_ctx_run.sh`, detached, health-gated, refuses to run
      a binary older than the sources it must describe, and named in `detach_audit.sh`.

      Bit-exactness: nothing numeric changed. Every added call is `dprof_begin/end`, which returns
      on `!g_dprof_on`, and the reporting block is behind `DSV4_DPROF`. The four CPU gates pass in
      preflight on every server start.

      **One open lead, recorded and NOT chased here.** At the 6144 point `cattn:q_proj` is cleanly
      bimodal — 18 steps at 10.3 ms and 17 at 14.7, with `q:wq_a` 1.71 vs 5.47 **at the same VB=2
      and the same context**. A 3.2x swing in a GEMM at fixed shape is not explained by width or by
      context, it is flat in context (R^2 0.005) so it does not touch this attribution, and it is
      worth its own item.


## Phase 1 — the context term

- [x] **1.1** Warp-parallel top-k (all four kernels). **14.2x at ctx 6592, 24.7x at 24k**,
      bit-identical on nine shapes and three distributions (`tests/gate_topk_warp.cu`).
- [x] **1.0** **Cache the DSpark main-KV prefix instead of recomputing it every token.**
      **DONE 2026-08-20 (iteration 1). +24.4 % tok/s at ctx 12,282, bit-exact, tau unchanged.**
      Full write-up: `wiki/kernel-optimisations.md` §2.5 and `wiki/context-scaling.md`.

      **What shipped.** `dspark_main_kv_upto` (`kernels/dspark_attn.cu`) computes only rows
      `[valid, s)` and advances a high-water mark; `src/engine.cu` and `src/decode.cu` call it
      instead of rebuilding all `ctxlen` rows NSTAGE=3 times per token. The three paths that can
      invalidate a prefix — `prefill_full`, `extend`, `rewind_to` — clamp the mark down; everything
      else may only grow it. Per-step cost is O(tokens committed) instead of O(context); over a
      generation, O(n) instead of O(n^2). `DSV4_MAINKV_CACHE=0` restores the old path exactly and is
      the A/B arm.

      **Measured, PAIRED, same corpus, baseline arm first so drift penalises the cached arm.** On
      every point at ctx >= 1536 both arms produced **bit-identical tau and mean verify width
      rep-for-rep** (6/6 legs), so each rep pairs with its twin and the delta is the kernel alone:

      | ctx | before ms/fwd | after | paired delta | band | speedup | tau b / tau a | tok/s |
      |---|---|---|---|---|---|---|---|
      | 12,282 | 215.65 | 170.25 | **-45.40** | [-45.9, -38.5] | **1.267x** | 1.652 / 1.652 | 7.67 -> 9.54 (**+24.4 %**) |
      | 9,213 | 203.51 | 167.27 | -36.24 | [-36.9, -34.3] | 1.217x | 1.969 / 1.969 | 9.69 -> 11.79 (+21.7 %) |
      | 6,132 | 178.89 | 154.71 | -24.18 | [-24.7, -23.4] | 1.156x | 1.718 / 1.718 | 9.59 -> 11.10 (+15.8 %) |
      | 3,069 | 160.46 | 148.19 | -12.27 | [-12.7, -12.0] | 1.083x | 1.626 / 1.626 | 10.07 -> 10.91 (+8.3 %) |
      | 1,536 | 147.67 | 141.15 | -6.52 | [-6.9, -6.2] | 1.046x | 1.701 / 1.701 | 11.54 -> 12.07 (+4.5 %) |
      | 249 | 128.25 | 125.28 | -2.97 | [-7.7, +1.0] | 1.024x | 1.646 / 1.636 | 13.05 -> 12.92 |

      Bands are **+/-1 %** against the 3.5 % run-to-run spread, because pairing removes verify-width
      variance rather than averaging over it. `tau` is unchanged to three decimals wherever the leg
      is reproducible — the required signature of a change that alters no arithmetic.

      **The one disturbed leg, stated exactly:** `sweep-t12288-r3` came in at 204.87 against a
      same-arm cohort of 165.7-176.5 at identical tau and width, and its *before* twin (216.55) is
      unremarkable — so it is the after leg that was disturbed, and it is the whole of that point's
      22.7 % spread. It is **kept** in the medians above, which are robust to it, and **excluded**
      from the regression below, which is not. Both are reported: with it the slope is
      `3.190 +/- 0.278`, without it `3.604 +/- 0.076`. The conclusion is the same either way and the
      exclusion is not doing the work.
      Evidence: `evidence/decode_loop/fit_1p0_paired.txt`, `fit_1p0_base.txt`, `fit_1p0_cache.txt`.

      **The context term fell 44 %, and it is the term 0.4 predicted would fall.**
      `fwd = 132.24 + 7.220 x ctx/1000` (R^2 0.976) -> `129.96 + 4.006 x ctx/1000` (R^2 0.888).
      Regressing the paired saving directly on context over 30 of the 31 exactly-paired legs:
      **`3.604 +/- 0.076 ms per 1000`, R^2 0.988**, against the `3.867 +/- 0.001` that 0.4's dprof
      attributed to `draft:main_kv`. **93 % of it, and the 2 SE band [3.45, 3.76] does NOT cover the
      prediction** — 0.26 ms/1000 is unaccounted for and is recorded rather than rounded away. The
      residual per-step main-KV work (`acc+1` rows x 3 stages, 15 launches, 3 dkmalloc/sync/free
      triples) is context-*independent* and belongs in the intercept, which is where the -1.42 ms
      also went, so it does not explain a slope.

      **BIT-EXACTNESS: gated by memcmp on the whole buffer, 704 calls, 2.59 M retained rows, 0 FAIL.**

      | gate | scope | result |
      |---|---|---|
      | `tests/gate_mainkv_incr.cu` | no checkpoint, 2048 x 512 floats, 22 split points, both GEMM paths | PASS |
      | ^ negative control: rope offset dropped | same | **FAIL**, 130,624/1,048,576 floats |
      | ^ negative control: GEMM pin dropped | same | **FAIL at g_tc_fp8=1 only, 377 floats, last ulp** |
      | `build/dsv4-server` in situ, `DSV4_MAINKV_GATE=1` | 384 calls, ctx to 12,281, 2,023,320 rows kept, all 3 invalidation paths | 0 FAIL |
      | `build/decode` in situ | 320 calls, ctx to 3,130, 568,509 rows kept | 0 FAIL |
      | LOSSLESS gate (spec vs base AR) | every decode run | PASS |

      Each recomputes all `s` rows with the **untouched** `dspark_main_kv` into a private buffer and
      memcmps the full `[s, HEAD_DIM]` range, aborting on the first differing float. The load-bearing
      detail is the **GEMM pin**: `fp8_block_gemm` dispatches on M (M=1 GEMV, M in [2,8] NVFP4
      overlay, larger -> `tc_fp8_gemm`), and the incremental delta is 1-6 rows where the from-scratch
      call was thousands, so the naive version would compare a GEMV against a tensor-core tile and
      lose bit-exactness for a reason unrelated to caching. The incremental path pins to
      `tc_fp8_gemm`, whose dispatch depends only on N and K. Second trap, also covered: `cosT/sinT`
      must be offset by `r0` or a sub-range rotates every row by the wrong angle.

      **The token-id half of the invariant could NOT be tested, and that is a finding, not a gap.**
      The server A/B reported 21/52 completion hashes differing; on `build/decode` (argmax, no seed)
      2 of 3 points diverged. **Two same-arm controls settled it, and they agree.**

      *Control 1, `build/decode`:* `DSV4_MAINKV_CACHE=0` on BOTH sides still diverged from itself, at
      ctx 1,024 at the identical id index 1067 with the identical token pair (223 vs 5115) — the
      cached path never executed in either arm. Baseline-run-2 and the cache-on run agree with each
      other; the first baseline was the outlier.

      *Control 2, `build/dsv4-server`, a full second baseline sweep from an independent server start:*

      > **base-vs-base2 differs on 21 of 52 legs — the SAME 21 legs as base-vs-cache.** Not merely
      > the same count: the same set, both symmetric differences empty. The cache code path never ran
      > on either side of that comparison.

      Control 2 also validates the A/B **timing** measurement, which is the part that actually
      ratchets: with no change at all it reports **0.998x-1.000x** at ctx 12,410 / 9,341 / 6,260 /
      3,197 / 1,536, where the real A/B reported 1.267x / 1.217x / 1.156x / 1.083x / 1.046x. The
      harness reads "no speedup" when there is no speedup.

      So the engine is not run-to-run reproducible once a run is long enough, and token ids cannot
      test anything past that point. See the amendment to hard invariant 1, item **1.9**, and
      `wiki/measurement-and-traps.md` §12. Evidence: `genout_1p0_identity.txt`,
      `genout_1p0_determinism.txt`, `fit_1p0_determinism.txt`, `mainkv_determinism.log`.

      **Second-order, as predicted and free:** the per-step scratch `dkmalloc`/`dkfree` of
      `s*DIM + s*(DIM/128)*4` — 51.8 MB per stage, 155 MB per step at ctx 12,288 — is now sized by
      the delta, ~12 KB.

      Scripts, all detached and all named in `detach_audit.sh`: `mainkv_ab_run.sh` (the A/B),
      `mainkv_verify_run.sh` (token ids + server determinism), `mainkv_decodegate_run.sh` (in-situ
      memcmp on `build/decode`), `mainkv_determinism_run.sh` (the same-arm control).

- [x] **1.2** **Single-CTA radix select instead of the warp selection sort.**
      **DONE 2026-08-20 (iteration 2). +8.2 % tok/s at ctx 12,410, bit-exact, tau unchanged.**
      Full write-up: `wiki/kernel-optimisations.md` §2.6 and `wiki/context-scaling.md`.

      *Why it was live:* retired by 0.2 on "top-k is 0.10 ms", un-retired by 0.4, which measured
      `i:topk` at **13.47 ms at ctx 12,288** and a slope of **0.872 +/- 0.021 ms per 1000 context**
      (width held fixed). 0.2's 0.10 ms was real but taken at **context 9** — 135x low.

      **What shipped.** `include/topk_radix.h`: an MSB-first 8-bit radix select over a 64-bit
      composite key `comp(v,t) = (ord(v) << 32) | ~t`, then one gather, then a bitonic sort of the
      <= 512 winners. All four kernels take it — `k_topk_verify` and `k_topk_decode`
      (`compressed_decode.cu`, the spec-decode verify and draft paths), `k_topk_masked` (the
      CUDA-graph base-AR path) and `k_topk_offset` (`indexer.cu`, prefill). O(T) work in a constant
      number of passes instead of `topk` SEQUENTIAL argmax rounds. `DSV4_TOPK_RADIX=0` restores the
      warp scan exactly and is the A/B arm.

      **Because `~t` makes every composite distinct there are no ties to break**, so the select
      needs no equal-key special case: exactly `k_eff` elements satisfy `comp >= threshold`. Two
      details are load-bearing and both have their own gate distribution: admission is the
      original's **float** compare `v > floorv` and never a key-space test (`ord(NaN)` beats every
      finite key, but `NaN > best` is false, so a key-space test would select what the original
      cannot); and **-0.0 is folded onto +0.0** before the key is formed, because `-0.0f == +0.0f`
      makes them a tie for the original but `ord(-0.0) < ord(+0.0)` as raw bits, and `index_score`
      can emit `-0.0` (relu gives exactly 0, times a negative head weight).

      **Standalone, `tests/gate_topk_radix.cu`, warp scan -> radix, same block, same input:**

      | T (= ctx/4) | 512 | 1,024 | 1,648 | 2,048 | **3,072 (ctx 12,288)** | 4,096 | 6,000 | 8,192 |
      |---|---|---|---|---|---|---|---|---|
      | shipped us | 258 | 324 | 502 | 459 | **592** | 725 | 1,076 | 1,259 |
      | radix us | 18.4 | 28.8 | 24.6 | 24.5 | **32.8** | 37.3 | 35.6 | 37.0 |
      | speedup | 14.1x | 11.2x | 20.4x | 18.7x | **18.0x** | 19.4x | 30.2x | **34.1x** |

      **Measured in situ, PAIRED, same corpus, baseline arm first so drift penalises the radix arm.**
      35 of 52 legs paired exactly (identical `tau` AND identical mean verify width rep-for-rep); all
      six reps pair at every point at ctx >= 1,664. The unpaired legs are ctx 128/384 and the two
      controls, which is item **1.9**'s known non-reproducibility, not this change.

      | ctx | before ms/fwd | after | paired delta | band | speedup | tau b / tau a | tok/s |
      |---|---|---|---|---|---|---|---|
      | 12,410 | 167.18 | 154.66 | **-12.55** | [-12.92, -12.33] | **1.081x** | 1.652 / 1.652 | 9.88 -> 10.69 (**+8.2 %**) |
      | 9,341 | 166.77 | 156.54 | -10.47 | [-10.66, -10.15] | 1.065x | 1.969 / 1.969 | 11.81 -> 12.60 (+6.7 %) |
      | 6,260 | 154.62 | 146.23 | -8.58 | [-8.65, -8.30] | 1.057x | 1.718 / 1.718 | 11.12 -> 11.77 (+5.8 %) |
      | 3,197 | 147.85 | 141.23 | -6.72 | [-7.01, -6.28] | 1.047x | 1.626 / 1.626 | 10.93 -> 11.46 (+4.8 %) |
      | 1,664 | 140.91 | 136.41 | -4.53 | [-5.07, -4.40] | 1.033x | 1.701 / 1.701 | 12.08 -> 12.48 (+3.3 %) |
      | 889 | 132.80 | 130.30 | -2.34 | [-2.84, -2.18] | 1.019x | 1.724 / 1.724 | 12.95 -> 13.19 (+1.9 %) |

      Bands are **+/-2 % or tighter** against the 3.5 % run-to-run spread, because pairing removes
      verify-width variance rather than averaging over it.

      **The context term fell another 28 %, and it is the term 0.4 predicted would fall.**
      `fwd = 130.60 + 3.488 x ctx/1000` (R^2 0.892) -> `129.11 + 2.514 x ctx/1000` (R^2 0.858);
      width-controlled `3.128 +/- 0.142` -> `2.158 +/- 0.092`. Regressing the paired saving directly
      on context over all 35 exactly-paired legs:
      **`saving = 3.122 + 0.793 +/- 0.031 ms per 1000`, R^2 0.951**, against the
      `0.872 +/- 0.021` that 0.4 attributed to `i:topk`. **91 % of it, and the 2 SE band
      [0.730, 0.856] does NOT cover the prediction** — 0.079 ms/1000 is unaccounted for and is
      recorded rather than rounded away. The **3.12 ms context-INDEPENDENT** saving is the part 0.4
      could not have predicted: `i:topk` marks only `k_topk_verify`, and `k_topk_decode` on the
      draft side has never had a dprof mark at all.

      **BIT-EXACTNESS: gated by memcmp on the whole index array, 11,008 calls, 183 M slots, 0 FAIL.**

      | gate | scope | result |
      |---|---|---|
      | `tests/gate_topk_radix.cu` | no checkpoint, 4 kernel shapes x 13 lengths x **6 distributions** (exact ties, signed zeros, floor-straddling, all-negative) | PASS |
      | ^ under `compute-sanitizer --tool memcheck` | same | 0 errors |
      | `build/dsv4-server` in situ, `DSV4_TOPK_GATE=1` | 11,008 calls, ctx to 12,282, **183,179,703 index slots**, prefill + extend + rewind | 0 FAIL |
      | standing GATE (`build/decode`, first argmax = 11111) | every decode run | PASS |
      | LOSSLESS gate (spec vs base AR) | every decode run | PASS, tau 2.87 tok/verify |

      The in-situ gate runs the **untouched** `k_topk_verify` / `k_topk_decode` with the identical
      arguments into a private buffer and memcmps the entire `[K, topkc]` index array, aborting on
      the first differing slot. It is not armed on the `_dp` graph path (`k_topk_masked`), which is
      CUDA-graph captured and cannot contain a host sync; that kernel is covered by the unit gate
      and by the LOSSLESS gate, which exercises it on every `build/decode` run.

      **A lever built and killed in the same iteration:** a `__match_any_sync` warp-aggregated
      histogram, on the theory that the shared `atomicAdd` was contending on a handful of top-byte
      bins. It is **slower** — 43.0 vs 39.0 us at T=3072, 57.5 vs 49.3 at T=8192 — so the naive
      atomic stayed. Block size *was* worth sweeping: 128/256/512/1024 threads measure
      53.3 / 38.9 / 32.0 / 31.6 us at T=3072, and 512 shipped. See `wiki/negative-results.md`.

      Scripts, detached and named in `detach_audit.sh`: `topk_ab_run.sh`.
      Evidence: `evidence/decode_loop/fit_1p2_paired.txt`, `fit_1p2_base.txt`, `fit_1p2_radix.txt`,
      `fit_1p2_identity.txt`, `gate_topk_radix.log`, `server_1p2_gate.log`,
      `decode_1p2_lossless.log`.
- [x] **1.3** `seq_len <= topk` early-out — **SHIPPED, BIT-EXACT, AND END-TO-END NULL.**
      **DONE 2026-08-20.** The kernel change is real and the throughput change is not: the term it
      attacks was already spent by 1.2, one iteration earlier, and nobody re-derived its headroom
      before it was built. Kept because it is strictly less work and provably identical output;
      recorded here as a null so the next item is not chosen the same way.

      **The change.** In `topk_radix_select`, `lim <= topk` means `k_eff == #candidates` before a
      single score has been read — the threshold search cannot exclude anything. The full path
      still had to *discover* that: one whole radix level (clear 256 bins, one strided pass with a
      shared `atomicAdd` per surviving element, then a 256-iteration serial scan on thread 0) whose
      only conclusion is `hist[d] == need`. The early-out skips that level and gathers
      unconditionally. Bit-exactness is structural, not checked afterwards: the level that is
      skipped would have set `thr = lowest_non_empty_top_byte << 56`, which is `<=` every
      candidate's composite, so `thr = 0` selects the identical set, and the bitonic sort that
      orders it is untouched. Fires at `T <= 512`, i.e. **ctx <= 2048** at ratio 4 with
      `INDEX_TOPK` 512, and on the verify side for any query whose causal limit is under 512.
      Arm: `DSV4_TOPK_EARLY=0` restores the full search on the same binary.

      **The kernel band, `tests/gate_topk_radix.cu`, re-run first-hand this iteration:**

      | T | 32 | 64 | 128 | 256 | 384 | **512** | 1024 | 2048 | 3072 |
      |---|---|---|---|---|---|---|---|---|---|
      | fires | yes | yes | yes | yes | yes | **yes** | no | no | no |
      | full us | 10.26 | 12.27 | 12.31 | 14.36 | 18.40 | **18.41** | 28.73 | 24.54 | 32.76 |
      | early us | 6.88 | 8.21 | 10.20 | 12.17 | 14.34 | **14.34** | 28.73 | 24.58 | 32.78 |
      | saving us | +3.38 | +4.06 | +2.11 | +2.18 | +4.06 | **+4.06** | +0.00 | -0.03 | -0.02 |

      **THE SWEEP WAS PRE-REGISTERED AS UNABLE TO RESOLVE THIS, in the script header, before the
      run.** 21 ratio-4 layers x ~4.05 us = **~0.085 ms of a ~130 ms forward = 0.07 %**, against a
      3.5 % run-to-run spread. It was run anyway because the ladder requires `tau` and a band in
      every A/B and because a *regression* at that scale would show.

      **Paired A/B, `scripts/topk_early_ab_run.sh`, OFF arm first so drift penalises the early-out.
      34 of 34 legs byte-identical in generated token ids; `tau` identical to three decimals on
      every leg.**

      | kind | ctx | fires | before ms/fwd | after | delta | tau b / a | tok/s b -> a |
      |---|---|---|---|---|---|---|---|
      | control | 6248 | **no** | 147.45 | 147.69 | **+0.24** | 1.772 / 1.772 | 12.02 -> 12.00 |
      | sweep | 6260 | **no** | 144.48 | 144.83 | **+0.35** | 1.625 / 1.625 | 11.27 -> 11.26 |
      | control | 1656 | yes | 141.17 | 141.11 | -0.06 | 1.958 / 1.958 | 13.87 -> 13.87 |
      | sweep | 1664 | yes | 135.72 | 135.81 | +0.10 | 1.636 / 1.636 | 11.99 -> 12.00 |
      | sweep | 889 | yes | 131.72 | 131.83 | +0.11 | 1.690 / 1.690 | 12.71 -> 12.70 |
      | sweep | 492 | yes | 126.86 | 126.33 | -0.53 | 1.590 / 1.590 | 12.58 -> 12.61 |
      | sweep | 249 | yes | 127.29 | 127.08 | -0.21 | 1.612 / 1.612 | 12.64 -> 12.64 |

      **The two ctx-6144 legs are the measurement, not the background.** `T = 1565 > 512` there, so
      the early-out cannot fire and both arms run provably identical code — yet they measured
      **+0.24 and +0.35 ms**. The five legs where the code *does* differ measured -0.53 to +0.11.
      The effect is smaller than the instrument, and the instrument's size was established inside
      the same experiment rather than quoted from the 3.5 % spread. Fit unmoved:
      `b = 3.008 +/- 0.241 -> 3.036 +/- 0.240` context-only, `2.977 +/- 0.143 -> 3.006 +/- 0.147`
      width-controlled, `a = 127.92 -> 127.84`. **Everything is inside one SE of nothing.**

      **`i:topk`, the dprof mark that brackets the changed launch, is the only instrument that saw
      it** — and it agrees with the kernel band, which is the check that the null is a
      not-worth-anything and not a not-working:

      | ctx | 768 | 1536 | 6144 |
      |---|---|---|---|
      | `i:topk` off | 0.42 | 0.52 | 0.72 |
      | `i:topk` on | **0.28** | **0.34** | 0.72 |
      | | -33 % | -35 % | 0 %, as predicted |

      **BIT-EXACTNESS: four gates, zero failures.**

      | gate | scope | result |
      |---|---|---|
      | `tests/gate_topk_radix.cu` | 4 kernel shapes x 13 lengths x 6 distributions, early-out **both ON and OFF** | PASS |
      | `build/dsv4-server`, `DSV4_TOPK_GATE=1`, early ON | 125 PASS lines, **7,808 checks, 66,972,918 index slots**, ctx to 1,590 — the regime where it fires | 0 FAIL |
      | paired token ids | 34/34 legs, both arms, whole sweep | identical |
      | standing GATE + LOSSLESS (`build/decode`) | argmax 11111; spec vs base AR | PASS, tau 2.87 tok/verify |

      **WHY IT IS NULL, AND THE LESSON THAT IS WORTH MORE THAN THE ITEM.** 1.3 was ranked above 1.5
      when `i:topk` was **13.47 ms at ctx 12,288** (0.4's attribution). 1.2 then took `i:topk` to
      **0.72 ms at ctx 6144** — and 1.3 was next on the list, so it was built against a headroom
      that had ceased to exist one commit earlier. **A ladder ordering is a function of a cost
      model, and 1.2 changed the cost model.** The re-attribution that should have been done before
      picking this item costs nothing — it is one dprof run that the A/B was going to do anyway.
      See the new rule 6 above and `wiki/measurement-and-traps.md` §15-16.

      **THE RE-ATTRIBUTION, from this iteration's own dprof (220 verify samples, ctx 369..6255,
      width-held-fixed slopes, `evidence/decode_loop/dprof_ctx_1p3_on.txt`).** This is the steering
      data for the next iteration:

      | mark | 0.4 said (to ctx 12,288) | now | status |
      |---|---|---|---|
      | `draft:main_kv` | 3.867 | **-0.000** | spent by 1.0 |
      | `i:topk` | 0.872 | **0.084 +/- 0.001** | spent by 1.2, then 1.3 |
      | `cattn:sparse` | 0.709 +/- 0.050 | **1.694 +/- 0.065** | see the warning below — item 1.7 |
      | `i:score` | 0.644 +/- 0.018 | **0.629 +/- 0.018** | unchanged — **item 1.5 is still the next kernel** |

      **DO NOT RE-RANK 1.7 ABOVE 1.5 ON THAT 1.694.** It is a linear coefficient fitted to a mark
      that is not linear. Per-point medians: `cattn:sparse` 9.35 -> 15.72 -> 19.89 ms at ctx
      768/1536/6144, i.e. **8.29 ms per 1000 over the first leg and 0.905 over the second**; 0.4
      measured 19.22/19.90/21.17 at 3072/6144/12,288, which is flat. The two "disagreeing" slopes
      are the same concave curve fitted over different ranges, and 1.7 already says so — the lever
      there is the ~20 ms FLOOR (Term A), not a slope. `i:score` by contrast is genuinely linear:
      0.88 -> 1.25 -> 3.53 ms is **0.482 and 0.495 ms per 1000** across the two legs. **The ladder
      order stands.**

      Scripts, detached and named in `detach_audit.sh`: `topk_early_ab_run.sh`.
      Evidence: `evidence/decode_loop/fit_1p3_paired.txt`, `fit_1p3_on.txt`, `fit_1p3_off.txt`,
      `fit_1p3_itopk.txt`, `gate_topk_radix_1p3.log`, `server_1p3_gate.log`,
      `dprof_ctx_1p3_on.txt`, `dprof_ctx_1p3_off.txt`, `decode_1p3_lossless.log`,
      `topk_early_ab.log`.
- [x] **1.4** `cudaFuncSetAttribute` opt-in for dynamic shared memory, or drop the requirement
      entirely (1.2 does). Removes a silent garbage-return above ~49k context.
      **DONE 2026-08-20 (iterations 5 and 6). It is a CORRECTNESS item and it moved no throughput
      number, which is the outcome it was ranked on — the shipped path does not evaluate any of the
      changed code, and `kernels/indexer.cu`, `kernels/compressed_decode.cu` and
      `kernels/attention.cu` compile to BYTE-IDENTICAL SASS before and after, so that is a property
      of the compiled artefact rather than an argument.**

      **RULE 5 DECLARATION: this is the second consecutive iteration to ship no kernel change**
      (1.3 changed a kernel and was worth nothing; 1.4 changes no device code at all). The next item
      taken must be the highest-expected-value *kernel* item, which is **1.5** — `i:score`, still
      `0.629 +/- 0.018 ms per 1000 context` on 1.3's re-attribution, genuinely linear, and untouched
      by anything that has landed since it was measured.

      **WHY IT TOOK TWO ITERATIONS, stated because the ladder is a record of process too.** Iteration
      5 landed the fix, the four-kernel gate legs and the wiki pages and then stopped — its server
      build was killed mid-`nvcc` (`build_1p4.log` ends `nvcc: Terminated`), so `build/dsv4-server`
      was never rebuilt, the item was never marked, and no `COMMIT_MSG` was left. Iteration 6 found
      a complete-looking change that had never been compiled into the engine. **A change that is not
      in a built binary is not landed**, and the tell was one line at the bottom of a build log.

      **1. THE ENUMERATION, which is what turns "we fixed the four we knew about" into a closed
      item.** Every `<<<grid, block, smem, stream>>>` in `kernels/` and `src/` was extracted and its
      third argument classified. **Thirteen launches request non-zero dynamic shared memory. Three
      families:**

      | family | sites | request | ceiling | disposition |
      |---|---|---|---|---|
      | top-k scans (`k_topk_offset/decode/verify/masked`) | 6 | `~4T`, **context-scaled** | ctx 49,140 | opted in (iter 5) |
      | `sdpa` (`kernels/attention.cu:97`) | 1 | `(head_dim + seq) * 4`, **context-scaled** | seq 12,224 | **opted in (iter 6)** |
      | model/block constants | 6 | `2*n_routed*4`=1,280 B; `nr*4`=640 B; `threads*4` <= 4,096 B; `3*HD*4`=1,536 B | none reachable | no change needed |

      `sdpa` is the one iteration 5 missed, and the reason it was easy to miss is the reason it was
      worth fixing: **it is not linked into `build/decode` or `build/dsv4-server`** — its only caller
      is `tests/test_attention.cu`, whose golden case dirs are not in this tree, so it had no
      runnable gate either. An unreachable path is not a fixed one, it is an unexercised one. It now
      has both a fix and a gate (`tests/gate_sdpa_smem.cu`, PASS at seq 4,096 and 16,384).

      **2. THE LEG ABOVE THE CEILING, run through the ENGINE'S OWN DECODE STEP, which is what the
      previous entry asked for.** `tests/gate_topk_smem_ctx.cu` +
      `scripts/gate_topk_smem_ctx.sh` drive `compressed_decode_step_indexer` — the function the
      server calls once per decode step, with its arena, its unzeroed `dmalloc`'d index buffer and
      its `sparse_attn` consumer — over pre-filled caches, no checkpoint, ~140 MB. Both arms see
      byte-identical inputs, so the FNV hash of the 4096-wide output is an exact comparison:

      | leg | context | T | scan smem | result |
      |---|---:|---:|---:|---|
      | control, below the ceiling | 8,191 | 2,047 | 8,208 B | radix = scan = gate, `4329fccc7a286b1b` |
      | **just above** | **49,207** | **12,301** | **49,216 B** | radix = scan = gate, `2b36c09ec550fc1a` |
      | deep | 200,003 | 50,000 | 200,016 B | radix = scan, `2dcfaeaf2947c336` |
      | past the opt-in max | 240,003 | 60,000 | 240,016 B | **SIGABRT** with the numbers, as designed |

      **3. THE DEFECT REPRODUCED IN THE SAME BINARY.** `DSV4_TOPK_SMEM_OPTIN=0` (new, and loud —
      it prints that it is restoring a silent-garbage path) turns the opt-in and the post-launch
      check off, which is the only honest form of a before-arm: one executable, one input, two runs.
      At context 49,207:

      | | opt-in ON | opt-in OFF (pre-1.4) |
      |---|---|---|
      | launch | success | **`invalid argument`** |
      | `cudaDeviceSynchronize` | success | **success** |
      | exit code | 0 | **0** |
      | `out` | 4096/4096 nonzero, 0 NaN | **4096/4096 nonzero, 0 NaN** |
      | hash | `2b36c09ec550fc1a` | **`8f24ad5745233320`** |

      **This corrects iteration 5's own write-up on both counts, and the correction is the more
      useful finding.** It is *not* "reads a zeroed array and attends to KV row 0": `dtop` comes from
      `dmalloc`, a bump allocator that does not clear, so the step returns a full, finite,
      entirely-plausible wrong answer with every error channel reporting success. And the in-situ
      reference does *not* "agree with anything" — under the same switch it prints
      `[topk-gate] FAIL decode ctx=49206 first diff at slot 0: radix 55526 vs warp 0`, i.e. above the
      ceiling the gate **condemns the correct shipped path**, because the reference is the arm that
      failed to launch. A false alarm from your own instrument is how a good change gets reverted.
      See `wiki/measurement-and-traps.md` §17 (rewritten).

      **4. A NEW HARDWARE FACT, and the safety margin that failed a passing gate.**
      `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, cudaDevAttrMaxSharedMemoryPerBlockOptin)`
      succeeds on the four scan kernels and returns `invalid argument` on `sdpa_kernel`. Bisected on
      a pair of kernels differing only in static shared memory, then launched at the value found:
      static 0 B -> **232,448 B** settable; static 1,024 B -> **231,424 B**. So
      `settable = optin - the kernel's own static shared`, and
      `cudaDevAttrReservedSharedMemoryPerBlock` (1,024 B) is **not** a second deduction. The first
      version of the fix subtracted it anyway "to be safe", lowered the ceiling by 256 rows of
      context, and immediately aborted `gate_topk_radix`'s T = 58,045 leg — a leg that had passed
      minutes earlier. Both helpers now read `cudaFuncGetAttributes(...).sharedSizeBytes`.
      `wiki/measurement-and-traps.md` §18.

      **5. WHAT WAS NOT DONE, AND THE ARITHMETIC THAT SAYS WHY.** The leg above the ceiling was NOT
      run in `build/dsv4-server`, and it cannot be on this box. Reaching context 49,140 needs
      `--seqmax 49152`, and the engine's seqmax-scaling allocations are **134,276 B/token**
      (`win_kv` 88,064 + `comp_kv` 20,992 + `main_x` 16,384 + `mkv[3]` 6,144 + `idx_ckv` 2,688 +
      `d_ids` 4; `src/engine.cu:332-433` against `include/deepseek_v4.h`) — **6.15 GiB at seqmax
      49,152 against 2.05 GiB at the 16,384 the engine runs today, i.e. 4.10 GiB more**, and the
      engine reports `mem 119.1/122.8 GiB` at seqmax 16,384, i.e. **3.7 GiB free**. It does not fit,
      and this box does not OOM gracefully. `build/decode` is worse still (it allocates `xin` at
      `seqmax*DIM*4` across 41 layers). The engine-path gate above is the strongest leg that is
      actually available, and it exercises the same function at the same T.

      **6. THE RATCHET, and it is a null by construction.** Standing gate, `build/decode` rebuilt
      with every 1.4 change in, same prompt and same parameters as 1.3's leg (s=6, NDEC=8, NGEN=64):

      | | 1.3, as recorded | **pre-1.4 binary, re-measured now** | 1.4 binary, now |
      |---|---|---|---|
      | generated token ids | — | **identical** | **identical** (md5 `25c90e85…` all three) |
      | LOSSLESS gate | PASS | PASS x3 | PASS x3 |
      | tau (tokens/verify) | 2.87 | **2.87 / 2.87 / 2.87** | **2.87 / 2.87 / 2.87** |
      | spec decode | 20.89 tok/s | **19.65 – 19.72** | **19.65 – 19.76** |
      | base AR M=1 | 11.43 tok/s | 11.28 tok/s | 11.37 tok/s |

      **THE CONTROL IS THE POINT, AND IT NEARLY WAS NOT RUN.** The first 1.4 leg came back at
      19.62 tok/s against 1.3's recorded 20.89 — **−6.1 %, well outside the 3.5 % spread this ladder
      quotes** — with `tau` and the token ids identical. On a byte-identical-SASS change that the
      shipped path does not even evaluate, that number is impossible, so the temptation is to write
      "within noise" and move on. Instead the pre-1.4 binary was rebuilt from `e4d7c6d` and run
      **now, on this machine, in this state**: it returns **19.65–19.72**. It does not reproduce
      20.89 either. The gap is between-session drift, and both arms are inside 0.6 % of each other.

      **Three replicates inside one checkpoint load spread 0.6 %; the same measurement across loads
      three hours apart differs by 5.7 %.** The dominant variance on this box is *between* loads, not
      within them, and the ladder's "3.5 % run-to-run spread" understates it — see
      `wiki/measurement-and-traps.md` §19. **Corollary for every future item: a single point from a
      previous iteration is not a valid before-arm.** Both arms must be measured in the same session,
      which is what `DSV4_BLKSWEEP` is for.

      Bit-exactness: SASS byte-identical on all three touched TUs (`sass_1p4_identical.txt`,
      `sass_1p6_identical.txt`); `gate_topk_radix` PASS (four kernel shapes x 13 lengths x 6
      distributions, early-out both ways, plus the four above-ceiling legs and the graph-capture
      leg); `gate_sdpa_smem` PASS; `gate_topk_smem_ctx.sh` PASS.

      **7. A DETACHMENT GAP CLOSED IN PASSING, because CLAUDE.md requires it.** `detach_audit.sh`'s
      `PATTERNS` list contained neither `run_model.sh` — the sanctioned launcher for every
      full-model benchmark in this repo — nor `build/decode`, the 100.4 GiB process it launches. An
      audit taken *while this iteration's benchmark was running* printed one row (`memguard.sh`) for
      a three-process tree and concluded "all detached", which is precisely the green-audit-that-
      proves-nothing CLAUDE.md warns about. Both added. A second fix went with it: the Claude Code
      Bash wrapper shell retains its last command line forever, so a shell that had once typed
      `build/decode` reported as a SESSION-BOUND stage; wrapper shells are now excluded by their
      `shell-snapshots` signature, because a red audit that is wrong teaches people to ignore red.

      Evidence: `evidence/decode_loop/gate_topk_smem_ctx_1p4.log`, `gate_topk_radix_1p4.log`,
      `gate_sdpa_smem_1p4.log`, `sass_1p6_identical.txt`, `decode_1p4_lossless.log` (single leg),
      `decode_1p4_band.log` (3 replicates, one load), `decode_pre1p4_control.log` (**the control**:
      `e4d7c6d` rebuilt and re-measured today), `build_1p4_final.log`.
      `decode_1p4_lossless_ngen24.log` is kept as the discarded first attempt — it ran NGEN=24
      against 1.3's NGEN=64 (seqmax 45 vs 85, 10 verifies vs 23) and was thrown away rather than
      reported, because a `tau` averaged over 10 verifies is not the same instrument as one averaged
      over 23. **Check the header line of the log you are comparing against before you compare.**
      Code: `include/indexer.h`, `kernels/attention.cu`, `tests/gate_topk_smem_ctx.cu`,
      `tests/gate_sdpa_smem.cu`, `scripts/gate_topk_smem_ctx.sh`, `scripts/build_gate.sh`,
      `scripts/detach_audit.sh`.
- [x] **1.5 DONE 2026-08-20 — `index_score` as a register-tiled GEMM. +4.57 % tok/s at ctx 12,410,
      and the context term falls 0.572 +/- 0.018 ms per 1000. The first kernel win since 1.2.**

      **What it was.** `i:score` measured **6.58 ms at ctx 12,288**, slope **0.644 +/- 0.018 ms per
      1000 context**, against the 0.02 ms at context 9 it was retired on — 330x low. Ranked third,
      behind 1.0 and 1.2, and both of those are spent.

      **THE RATCHET, paired, same session, control arm FIRST so drift favours the control.** Four
      reps x three sweep contexts plus two control contexts, one server load per arm, corpus
      `79ac6563f97e53e5`/`ac10134c87e3d0fe`:

      | ctx | ms/forward before | after | paired delta | tok/s before -> after | tau |
      |---|---|---|---|---|---|
      | 3,197 | 142.94 | 141.45 | **-1.49** | 12.14 -> **12.27** (+0.99 %) | 1.736 both |
      | 6,260 | 146.95 | 143.76 | **-3.19** | 11.87 -> **12.15** (+2.24 %) | 1.788 both |
      | 12,410 | 154.44 | 147.77 | **-6.67** | 10.46 -> **10.93** (+4.57 %) | 1.615 both |
      | 1,656 (control) | 137.40 | 136.68 | -0.71 | 13.43 -> 13.50 | 1.846 both |
      | 6,248 (control) | 147.07 | 143.91 | -3.16 | 10.86 -> 11.10 | 1.599 both |

      **All 16 legs faster, all 16 byte-identical in the emitted text, `tau` and mean verify width
      identical to four decimals in every one.** Regressing the 16 paired deltas on context gives
      the saving directly, and it is the number this item ratchets:

          delta_fwd = +0.358 +/- 0.134  -0.5724 +/- 0.0178 x (ctx/1000)     R^2 0.987, n=16

      i.e. **-0.572 +/- 0.018 ms per 1000 context**, which is **89 % of the 0.644 +/- 0.018 that 0.4
      attributed to this mark** — the prediction and the delivery agree inside one SE. Note the
      intercept: at zero context the GEMM is **0.358 +/- 0.134 ms SLOWER** per forward (it stages
      33 KiB of shared per block, which does not pay for itself until there are rows to amortise it
      over). Break-even is near ctx 625. That is a real, small, measured cost and it is why the
      1,656 control leg gains almost nothing.

      **DO NOT read the two arms' fitted `b` as the ladder's tracked `b`.** This sweep spans
      ctx 3,197-12,410; the `a = 129.11, b = 2.514` the stop condition tracks was fit over
      249-12,410 with ten points. The arms here fit `139.57 + 1.231 (SE 0.254)` and
      `140.00 + 0.652 (SE 0.236)` — same direction, but a narrow range against a large intercept
      determines `b` badly (R^2 0.701 and 0.433). The paired saving is the trustworthy quantity,
      so the tracked term is carried forward by SUBTRACTION, not by re-fit:
      **`b` 2.514 -> 1.942 ms/1000, `b x 6592` 16.57 -> 12.80 ms** (stop wants <= 5.0).

      **THE MARK ITSELF, dprof, both arms, same protocol** (`dprof_ctx_1p5_off/on.txt`) — medians in
      ms at ctx 3072 / 6144 / 12,288:

      | mark | before | after |
      |---|---|---|
      | `i:score` | 1.93 / 3.53 / **6.58** | 0.99 / 1.12 / **1.36** |
      | `cattn:indexer` (its parent) | 4.79 / 6.41 / **9.51** | 3.86 / 4.00 / **4.27** |
      | `i:topk` (untouched — the control mark) | 0.72 / 0.73 / 0.76 | 0.72 / 0.72 / 0.76 |
      | `STEP` | 130.82 / 134.55 / **143.62** | 130.29 / 132.45 / **138.61** |

      **6.58 ms at ctx 12,288 reproduces 0.4's 6.58 exactly**, four iterations and three kernel
      changes later. The saving lands entirely inside the parent mark (-5.24 of `cattn:indexer`
      against -5.22 of `i:score`) and `i:topk` does not move, so nothing was relocated. The mark's own
      slope goes **0.503 +/- 0.006 -> 0.040 +/- 0.001 ms per 1000** — 92 % dead, not merely smaller.
      Note that dprof reads the saving LOW: 0.463 in `i:score`, 0.484 +/- 0.009 in `STEP`, against
      the clean paired **0.572 +/- 0.018**. Its own per-mark syncs compress the difference by
      15-20 %, which is why the ratchet is the clean pair and dprof is only the attribution.

      **THE MECHANISM IS MEMORY PLACEMENT AND AN ACCUMULATION ORDER, NOT NEW MATHEMATICS.** The
      shipped `index_score_warp_kernel` is warp-per-(query,row) and re-reads both operands from
      global on every head: `q` for one query is H*d = 8192 floats = 32 KiB, re-read once per row
      `t`; `kv[t]` is re-read once per head. At the verify shape (S=6, T=3072) that is 1.2 GB moved
      to do 151 M MACs — 0.5 FLOP/byte, and `d` is a runtime argument so the inner loop cannot
      unroll. `index_score_gemm_kernel` makes it `P[(s,h),t] = q . kv^T` with M=H=64, N=T, K=d=128,
      one block owning all 64 heads of one query and 128 rows: 8x8 register tiling gives 16 shared
      loads per 64 FFMAs (~4:1 against the old 1:1), `P` lands in shared, and the epilogue walks
      `h = 0..H-1` in order. Standalone, 30 interleaved repeats:

      | S,T | warp (us) | GEMM (us) | |
      |---|---|---|---|
      | 6, 768 | 239.7 | **45.0** | 5.33x |
      | 6, 3072 | 898.7 | **132.5** | 6.78x |
      | 6, 6144 | 1771.6 | **237.6** | 7.46x |
      | 1, 6144 | 311.7 | **53.9** | 5.78x |

      **THE BIT-EXACTNESS CLAIM CHANGED DIRECTION, AND THAT IS THE INTERESTING PART.** An
      intermediate `index_score_tiled_kernel` was built first and is bit-identical to the *shipped
      warp kernel* — and that capped it at **2.0x**, because bit-exactness with the warp kernel
      mandates keeping its 5-step `__shfl_down_sync` tree, and SHFL retires one warp-instruction per
      SM per clock: 1.18 M (row,head) pairs x 5 over 20 SMs is a ~200 us floor no matter how the
      operands are staged. The GEMM is bit-identical to `index_score_kernel` instead — the
      correctness-first SCALAR REFERENCE that `gate_units` checks against
      `ref/goldens/unit_index_score.safetensors` — because a register-tiled GEMM accumulates
      serially in k, which IS the reference's order. LOOP_LOG Finding 68 adopted the warp kernel as
      a deviation FROM that reference behind the LOSSLESS gate; **1.5 hands that deviation back and
      is 6.8x rather than 2.0x for doing so.** The tiled kernel is kept only as the fallback for
      shapes the GEMM cannot serve. See `wiki/negative-results.md` §4d.

      **THE INVARIANT IS SATISFIED ON ITS SECOND BRANCH, AND THE DEVIATION IS MEASURED, NOT
      ASSERTED.** `tests/gate_index_score` (1130 shapes, 21,773,760 floats): **0 differing, 0 bad
      shapes -> PASS** on both claims (TILED == WARP, GEMM == SCALAR). Against the *shipped* warp
      kernel it prints what the change spends, per input distribution, out of 2,160,800 floats:
      uniform+/- 1,905,110 differ (max_abs 1.05e-05, max_rel 0.609); all-positive 1,062,863
      (3.05e-04, 5.38e-07); all-negative 1,890,149 (1.37e-04, 0.847); near-tie 995,537 (4.88e-04,
      1.19e-07); signed-zero/denormal 583,868 (3.58e-07, 0.0164). Those relative numbers are large
      on near-zero scores and this kernel feeds a SELECTION, so the only acceptable evidence is
      downstream behaviour — which is why the 16/16 byte-identical emitted texts and the
      four-decimal `tau` match above are the real gate, and they were taken at ctx 12,410, not at
      ctx 9. Standing gates: `LOSSLESS GATE -> PASS` x3 on the GEMM arm; `gate_units` (cosine, vs
      goldens), `gate_indexer_decode`, `gate_compressed_decode`, `gate_prefill_len` all rc=0. And
      because the shipped path is CUDA-graph captured, the two capture gates were run on the GEMM
      too, once they could be built again: `gate_indexer_graph` and `gate_compressed_graph` both
      **`cosine=1.0000000 maxabs/|o|=0.00e+00 -> PASS`**, i.e. graph replay is bit-identical to the
      sequential path. `ixgemm_launch` issues a kernel launch and no other CUDA API call, so that
      was true by construction; it is now also true by test.

      **THE SAME-ARM DECODE CONTROL IS A NULL, AND IT SHOULD BE.** Three replicates per arm in one
      load each, `build/decode` at seqmax 85 — where T = s/ratio is ~21 rows and this kernel has
      almost nothing to do: `tau` **2.87** in all six, generated ids byte-identical
      (`11111 16 455 6102 294 16603 344 29168`), spec throughput **21.03-21.24** (GEMM) against
      **21.02-21.26** (warp). A change that is worth +4.6 % at ctx 12,410 and exactly nothing at
      ctx 85 is behaving like a context-term change, which is the claim.

      **A GATE THAT DOES NOT COMPILE IS A GATE THAT PASSES BY NEVER RUNNING.** Phase 1 could not
      start: `gate_indexer_decode`, `gate_compressed_decode`, `gate_indexer_graph` and
      `gate_compressed_graph` had not built since commit `5c1e047` split `x_full` into
      `(x_cur, x_full)` and left four call sites one argument short. That is **the four in-situ
      gates for the exact subsystem this item changes**, dead for the whole ladder to date, on top
      of the `set -e` + bare-`nvcc` breakage iteration 7 found in `build_gate.sh` one line above
      them. Fixed here (`x` -> `x + pos*DIM, x`, restoring the pre-`5c1e047` semantics exactly) and
      all four now pass, the two graph gates at `maxabs 0.00e+00`.
      `wiki/measurement-and-traps.md` §20.

      Evidence: `evidence/decode_loop/ixgemm_ab.log` (the whole run),
      `gate_index_score_1p5.log`, `fit_1p5_off.txt` / `fit_1p5_on.txt` / `fit_1p5_paired.txt`,
      `fit_1p5_iscore.txt`, `dprof_ctx_1p5_off.txt` / `dprof_ctx_1p5_on.txt`,
      `decode_1p5_lossless.log` / `decode_1p5_control.log` (**the same-arm control**),
      `gate_units_1p5.log`, `gate_indexer_decode_1p5.log`, `gate_compressed_decode_1p5.log`,
      `gate_prefill_len_1p5.log`, `build_1p5b_*.log`.
      Code: `kernels/indexer.cu`, `include/indexer.h`, `tests/gate_index_score.cu`,
      `scripts/ixgemm_ab_run.sh`, `scripts/build_gate.sh`, `tests/gate_indexer_decode.cu`,
      `tests/gate_compressed_decode.cu`, `tests/gate_indexer_graph.cu`,
      `tests/gate_compressed_graph.cu`.

- [x] **1.9** **Find out why the engine stops reproducing itself part-way through a long run.**
      **Opened by 1.0, 2026-08-20, and it is a correctness item before it is a performance one.**
      **RANKED HERE, BELOW 1.2 AND 1.5, DELIBERATELY.** It is the most interesting thing on this
      page and it is still not the next thing to build: the ladder has spent four of its first six
      iterations on instruments, and 1.0 showed the buffer-memcmp gate is a *stronger* correctness
      instrument than the token ids it replaces — so the invariant has a working substitute and
      this is not blocking. Promote it above 1.2 the moment a kernel change cannot be gated by
      buffer memcmp, or if `tau` itself starts moving between identical runs.
      `build/decode` is argmax, unseeded, single-stream, no HTTP, and its width controller reads the
      draft's own margins (`while (VK < VKCAP && hmarg[VK-1] >= adaptK) ++VK`) rather than the clock
      — so "the faster arm picked a different width" does **not** explain it. Yet two runs of the
      identical binary with the identical env diverge.

      **State the shape precisely, because it is the diagnostic.** In BOTH binaries the divergence
      is a **clean suffix in run order**, not a property of a context:

      * `build/decode`: point 0 (ctx 6, 260 ids) identical in **three** separate runs; points 1
        (ctx 3,072) and 2 (ctx 1,024) differ in every pairing, first at generated token 20 and 43.
      * `build/dsv4-server`: legs 1-31 of 52 identical (including every leg at ctx 12,410, 9,341,
        6,260, 3,197 and 1,536, with `tau` matching to three decimals across three independent
        server starts), then leg 32 onward all differ.

      **NEW DATUM FROM 1.5, 2026-08-20, and it sharpens the shape rather than contradicting it.**
      1.5's A/B ran 16 legs per arm across **two independent server starts** — different kernels,
      contexts 1,656 to 12,410, 256 completion tokens each — and **all 16 pairs emitted
      byte-identical text**, with `tau` and mean verify width matching to four decimals. So at 16
      legs the server reproduces itself perfectly across loads AND across a kernel change; the
      divergence below started at leg **32** of 52. That is consistent with a per-leg accumulation
      rather than a per-step rate, and it means any probe for this must run **more than ~31 legs in
      one start** or it will report clean. It also means every A/B on this ladder so far has been
      inside the clean region, which is luck, not design.

      Generation is autoregressive, so **one** flipped token permanently de-synchronises everything
      after it. The observation is therefore not "context X is nondeterministic" but "there is a
      rare per-step event whose rate rises with context": zero occurrences in 3 x 260 steps at ctx 6,
      but within 20-43 steps at ctx 1,024-3,072. A context-linear reduction with a
      **non-deterministic accumulation order** fits exactly — more terms, more chances that a
      near-tie in an argmax flips. `atomicAdd`-ordered sums or split-K reductions in `i:topk`,
      `i:score` or `cattn:sparse` are the obvious family; note that 1.2's own reference "emits in
      `atomicAdd` order".

      Cheap first probe, in ONE process so there is no load-to-load variation to argue about: run the
      same prompt twice back to back and hash the logits every step, then bisect to the first step
      whose logits differ and dump the per-mark intermediates there. Second probe: re-run with each
      candidate kernel forced to a deterministic order and see which one makes the repeat stable.
      **Why it matters beyond tidiness:** it disables the ladder's primary correctness invariant for
      every remaining item, forcing each onto buffer-memcmp gates; and if a reduction really is
      order-nondeterministic, `tau` is being measured against a moving target.

      ---

      **ANSWERED, 2026-08-20, and almost every word of the question above was wrong.** It is not
      "part-way through a long run", it is not the decode loop, it is not accumulated drift, and it
      is not per-process state. **`build/decode`'s PREFILL is nondeterministic inside the compressed
      (`compress_ratio != 0`) layers, for any prefill of about 192 positions or more, run-to-run AND
      repeat-to-repeat inside one process.** Everything downstream is an autoregressive consequence
      of that. **NO KERNEL CHANGED THIS ITERATION** — this is a diagnosis, and rule 5 applies to the
      next one: if the following iteration also ships no kernel, say so at the top of its entry and
      take the highest-expected-value kernel item regardless of certainty.

      **1. IT IS NOT THE DECODE LOOP.** `DSV4_STEPHASH=<file>` (new, `src/decode.cu`, default off)
      writes one line per verify carrying that step's whole causal chain in dataflow order —
      `mkv -> mx -> din -> draft -> lg -> acc/corr` — so two runs `diff` to the first differing
      FIELD and the field names the link. Run on **exactly** 1.0's divergent arm (same prompts file,
      same sweep, NDEC 16, NGEN 256, `DSV4_MAINKV_CACHE=0` on both sides): point 0 (ctx 5) is
      identical at every field of **all 72 verifies**; points 1 (ctx 3,071) and 2 (ctx 1,023) diverge
      at **verify 0**, with `mkv` and `mx` — the draft's two persistent inputs, both pure functions
      of the prefill — already different **before the first draft ran**.
      `evidence/decode_loop/stephash_verdict_AB.txt`.

      **2. IT IS THE FIRST COMPRESSED LAYER OF THE PREFILL.** `DSV4_HASH=2` already hashed the
      hidden state after every one of the 43 prefill layers; it had only ever been read
      point-to-point (Finding 61), never run-to-run, and its byte-wise FNV was too slow to point at
      real context (~17e9 rounds per point). Made word-wise and chunked here. Two runs, PSp 5 and
      PSp 3,071: at 5, **43/43 layers byte-identical**; at 3,071, layers 0 and 1 — both `ratio 0`,
      pure sliding — are **byte-identical**, and layer **2**, the first `ratio 4` layer, is the first
      to differ. `lhash_H.txt`, `tools/lhash_compare.py`.

      **3. THE THRESHOLD IS A PREFILL LENGTH BETWEEN 160 AND 192, AND IT IS SHARP.** Two length
      ladders, two runs each, 43 layers hashed per point:

      | prefill positions | 16 | 32 | 64 | 128 | 129 | 132 | 136 | 144 | 160 | 192 | 256 | 512 | 1024 | 2048 | 2560 | 3071 |
      |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
      | layers differing, run to run | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **41** | **41/33** | **41** | **41** | **41** | **41** | **41** |

      Ten point-comparisons at or below 160 are clean — **430 layer hashes, zero differences** — and
      every point at 192 and above diverges. `lhash_L.txt`, `lhash_W.txt`. The boundary is not
      `WINDOW` (128): 129, 132, 136, 144 and 160 are all above the window and all clean.

      **4. IT IS A RACE, NOT STATE — AND THAT COST TWO HYPOTHESES.** `DSV4_ARENA_ZERO=1` on **both**
      arms of the 128–256 ladder reproduces the identical verdict (`lhash_Z.txt`), so the arena is
      dead at the lengths where the defect lives and not merely at Finding 61's. Then the decisive
      one: run the **identical sweep point four times inside ONE process**. At PSp 192 the four
      prefills disagree **with each other** — first differing layer 2 / 4 / 12 / 28 across the
      pairs, and a different pattern in a second process (`lhash_R_within.txt`, and
      `lhash_R.txt` for the cross-run half). Same process, same
      addresses, same buffer contents, four different answers. Finding 61's "deterministic 5-cycle"
      does not survive at these lengths; whatever it was measuring at PSp≈5 is not this.

      **5. WHAT IS EXONERATED, AT THE LENGTHS THAT MATTER.** Layers 0 and 1 are `ratio 0` and are
      byte-identical in **every** comparison of all six experiments, at every length up to 3,071.
      They run `k_embed`, `k_hc_expand`, `hc_pre`/`hc_post`, `rmsnorm`, `mla_cache_kv`,
      `mla_forward` and `moe_forward` — including the MoE's `atomicAdd` counting-sort grouping and
      its `k_scatter_ts`/`k_reduce_ts` combine, the family the paragraph above nominated first. All
      clear. **The entire residue is `compressed_attn_forward`**: `compressor_forward`,
      `indexer_forward` (`index_score` -> `k_causal_mask` -> `k_topk_offset*`), and `sparse_attn`
      over the combined index list. Every "first differing layer" recorded in any of the six
      experiments has `ratio 4`; not one has `ratio 0`.

      **6. WHY THE SERVER LOOKED CLEAN AND IS NOT A CONTRADICTION.** 1.5's 16 legs at ctx
      1,656–12,410 were byte-identical across two server starts. They are different code:
      `build/decode` prefills through `cblock_prefill_cache` -> `kernels/compressed_attn.cu` in ONE
      call as wide as the prompt, while `Engine::Impl::prefill_full` prefills through
      `cblock_verify_step` -> `kernels/compressed_decode.cu` in **`EXT_CHUNK` = 64**-row chunks and
      therefore never issues a compressed prefill call wide enough to reach the threshold. That
      predicts both observations. It also means **no bit-exactness result on one binary's prefill
      transfers to the other**, which nothing in this repo said before.
      `wiki/measurement-and-traps.md` §27.

      **7. WHY EVERY GATE MISSED IT.** `tests/gate_scratch_init` proved `compressed_attn_forward`
      poison-independent **at every length 1..29**, and Finding 61 quoted that as the retraction
      that exonerated the prefill chain — a sentence still carried verbatim in the comments of
      `compressed_attn.cu` and `indexer.cu`. The defect starts between 160 and 192. The gate stops
      six times short of the regime its verdict was used to close. §25.

      **8. AND THE INSTRUMENT BUILT HERE LIED ONCE, IN THE SAME WAY.** `DSV4_STEPHASH_LVL=1` skips
      the two expensive device hashes and writes them as **zero**; `stephash_compare.py` read
      equal-as-exonerated and printed *"mkv, main_x and the draft input all MATCH … the DSpark draft
      chain is the nondeterministic component"* — the opposite subsystem, manufactured entirely by
      the level flag. Fixed in the tool: all-zero-on-both-sides fields are excluded, printed as
      `NOT MEASURED`, and any verdict resting on them carries an explicit caveat line. §26.

      **WHAT THIS COSTS THE LADDER.** Token-id bit-exactness is a valid gate for prefills of **160
      positions or fewer** and is reliably invalid above ~192 on `build/decode`; buffer-memcmp stays
      the substitute (1.0). `tau` measured on `build/decode` above that length is measured against a
      moving target, which is why 1.5's paired per-leg saving and not a fitted difference was the
      right ratchet. The server suite is unaffected by *this* mechanism (point 6), so no landed
      number is retracted.

      Evidence: `evidence/decode_loop/stephash_verdict_AB.txt`, `stephash_genout_AB.txt`,
      `lhash_H.txt` / `lhash_L.txt` / `lhash_W.txt` / `lhash_Z.txt` / `lhash_R.txt`, and the raw
      arm logs `stephash_{A,B,H3,H4,L1,L2,W1,W2,Z1,Z2,R1,R2}.log` + `stephash_{A,B,...}.txt`.
      Code: `src/decode.cu` (`DSV4_STEPHASH`, word-wise chunked `DSV4_HASH`),
      `tools/stephash_compare.py`, `tools/lhash_compare.py`, `scripts/stephash_run.sh`,
      `scripts/detach_audit.sh`.

- [ ] **1.10** **Name the kernel inside `compressed_attn_forward` that is racing.** Opened by 1.9,
      2026-08-20, and it is the only unfinished half of it. 1.9 bounded the fault to one function
      and proved it is a race rather than state, but it did not say whether it is
      `compressor_forward`, `indexer_forward` or `sparse_attn`.

      **THE INSTRUMENT THIS UNBLOCKS IS NAMED IN ADVANCE, per rule 2.** A sub-layer hash inside
      `compressed_attn_forward` — after `compressor_forward`'s `ckv`, after `index_score`, after
      `k_topk_offset*`'s `compress_topk`, after `sparse_attn`'s `o` — bisects three candidates in
      ONE run, because 1.9's R protocol reproduces the divergence **four times inside a single
      process at prefill 192 in about three minutes**. The optimisation it unblocks is not a
      speed-up: it is the restoration of the ladder's primary correctness invariant, which every
      remaining item currently has to work around with buffer memcmp.

      **RANK IT BELOW 1.7 UNLESS THE FIX IS ALSO A SPEED-UP.** The engine has not gotten faster
      since the warp top-k, 1.9 shipped no kernel, and rule 5 is now live. If the racing kernel
      turns out to be `index_score` or the top-k, the fix and 1.7 are the same edit and this
      promotes above it; otherwise it waits.

      Cheapest first step, no new code: re-run 1.9's R protocol with `DSV4_TOPK_RADIX=0`, then with
      `NO_IXGEMM=1`. Each is one existing env flag and one pair of ~3-minute runs, and either one
      coming back clean names the kernel outright.

- [x] **1.7** **DONE 2026-08-20. `sparse_attn` stages the gathered KV row in shared memory: paired
      saving 4.227 +/- 0.121 ms per forward over 16 legs, every leg faster, all 16 byte-identical.**

      ```
      ctx     ms/forward before -> after   paired    tok/s before -> after      tau
      1,656   133.46 -> 130.33             -3.13     13.93 -> 14.27 (+2.44 %)   1.861 both
      3,197   136.89 -> 132.78             -4.10     12.07 -> 12.44 (+3.07 %)   1.652 both
      6,248   141.46 -> 137.05             -4.40     11.85 -> 12.23 (+3.21 %)   1.681 both
      6,260   138.49 -> 134.20             -4.29     11.74 -> 12.13 (+3.32 %)   1.620 both
      12,410  146.90 -> 142.24             -4.66     11.71 -> 12.09 (+3.25 %)   1.736 both
      ```

      **16 of 16 legs emitted a byte-identical `text_sha256`, with `tau` and mean verify width equal
      to four decimals in every one, at ctx up to 12,282.** This is the ladder's PRIMARY invariant
      satisfied, not its lossless-gate fallback: the change is bit-exact by construction and the
      engine agrees. `gate_sparse_hpb` memcmps the whole output buffer of every (hpb, smem) launch
      against the pre-1.7 launch at six engine shapes -- 0 differing bytes -- and its one-ulp
      negative control fails as it must. All four in-situ engine gates report `maxabs = 0.00e+00`,
      LOSSLESS x3 on `build/decode`, and both `build/decode` arms generated the identical ids
      `11111 16 455 6102 294 16603 344 29168`.

      **THE MECHANISM, and note that the ladder entry's own hypothesis was WRONG.** This entry, and
      B9's comment in `kernels/mla_attn.cu`, both said the problem was REUSE: `num_key_value_heads
      == 1`, `ip` is indexed by (b,m) and not by head, so all 64 heads of a query gather the
      identical rows and 63/64ths of the traffic is redundant. `hpb` -- putting HPB heads in one
      block so they share L1 -- is the fix for that, it already existed, and it is **a measured NULL:
      1.00x at hpb=2 and hpb=4 at every shape.** L1 was already catching the reuse. The real
      constraint is ISSUE PRESSURE: 32 scalar loads per warp per gathered row (16 for the dot, 16
      more for the accumulate), on a kernel with 3-6 warps per SM and a 5-deep `__shfl_down_sync`
      chain on the critical path. Staging the row into shared memory once per block, with `float4`
      loads and double buffering, and holding it in registers across both consumers, cuts that to 4
      vector loads per block plus 16 smem loads per warp. **1.36x at the mean verify shape.**

      **VECTORISING THE LOAD WITHOUT LOSING BIT-EXACTNESS IS THE WHOLE TRICK.** `float4` normally
      destroys it, because it changes which lane owns which element and therefore the shape of the
      partial-sum tree. Staging through smem decouples the two: the global load pattern becomes a
      pure memory-placement decision while the reduction still sees lane `l` holding
      `{l, l+32, ..., l+480}`, the same 5-step shuffle tree, and the same online-softmax order over
      `t`. See `wiki/kernel-optimisations.md` §2.9.

      **THE ENTRY'S SHAPE CALL WAS RIGHT, AND IT IS WHY THIS IS TERM A.** Splitting the 16 paired
      legs at the knee the entry predicted (`topk = WINDOW(128) + min(INDEX_TOPK(512), ctx/ratio)`
      saturates at ctx 2048): **-3.127 +/- 0.014 ms below it (2 legs), -4.384 +/- 0.065 above it
      (14).** Above the knee the residual context term is
      **-3.996 +/- 0.080 flat, -0.0552 +/- 0.0102 per 1000** -- i.e. 94 % of the win is the FLOOR,
      exactly as the entry said. The small surviving slope is not noise and not a fitting artefact:
      **20 of the 43 layers are ratio-128, where `topk = wmax + ctx/128` has no `index_topk` cap at
      all.** 20 layers x 7.81 rows per 1000 ctx x 0.309 us saved per row = **0.048 ms per 1000
      predicted, 0.0552 +/- 0.0102 measured** -- inside one SE.

      **dprof isolates it to the changed launch and nothing else.** Medians, both arms, same
      protocol, ctx 3072 / 6144 / 12,288:

      | mark | before | after |
      |---|---|---|
      | `cattn:sparse` | 18.66 / 19.38 / **20.57** | 15.27 / 14.48 / **15.38** |
      | `ATTENTION` (parent) | 59.71 / 62.36 / **68.38** | 56.03 / 57.05 / **62.63** |
      | `i:score` (untouched control) | 0.98 / 1.11 / 1.34 | 0.97 / 1.11 / 1.34 |
      | `i:topk` (untouched control) | 0.71 / 0.73 / 0.76 | 0.71 / 0.72 / 0.75 |
      | `STEP` | 137.31 / 132.16 / 136.94 | 133.67 / 126.79 / 131.75 |

      The saving lands inside the parent (`ATTENTION` -5.75 against `cattn:sparse` -5.19 at ctx
      12,288) and every other mark is unchanged, so no work was relocated into an unmarked region.
      **`cattn:sparse` at 18.66 / 19.38 / 20.57 also reproduces 0.4's 19.22 / 19.90 / 21.17 four
      iterations and four kernel changes later**, which is an independent check on the cost model
      this ladder is ordered by.

      **THE OLD DEFAULT WAS A REGRESSION AT ITS OWN DESIGN POINT.** B9 sized the `hpb` heuristic on
      a 1022-token prefill and shipped HPB=8 there. Measured today: **hpb=8/smem=0 is 0.80x of
      hpb=1 at that prefill, and 0.52x at the K=6 verify width.** The number was never re-measured
      after the shape moved. The new default is `hpb=4` with `smem=2` below 1024 warps and `smem=1`
      above, which is 1.28x at the 1022-token prefill and 1.21x at the short one -- so this item
      also gives prefill back the 20 % B9's heuristic had been quietly costing it. See
      `wiki/measurement-and-traps.md` §28 and `wiki/prefill-optimisation.md` §7.

      Ratchet, carried by SUBTRACTION of the paired numbers per rule 7 (this item's own sweep spans
      ctx 1,528-12,282 and fits `b` badly on either arm): **`a` 129.11 -> 125.11 ms** (1.571x ->
      1.522x the 82.18 ms byte floor; stop wants <= 1.25) and **`b` 1.942 -> 1.887 ms/1000**
      (`b x 6592` 12.80 -> 12.44 ms; stop wants <= 5.0). This is the first item on the ladder to
      move Term A at all.

      Arms, both in one session on one build: control `DSV4_SPARSE_HPB=1 DSV4_SPARSE_SMEM=0`
      (the pre-1.7 launch, bit for bit), run FIRST so drift favours it. Driver
      `scripts/sparse_ab_run.sh`; evidence `evidence/decode_loop/{gate_sparse_hpb_1p7.log,
      fit_1p7_*,dprof_ctx_1p7_*.txt,decode_1p7_*.log,sparse_ab.log}`.

      *(The original entry is preserved below, because its two calls -- the concave shape and "the
      lever is the FLOOR, not the slope" -- were both confirmed, and its third, the reuse mechanism,
      was refuted. That is worth keeping visible.)*

      > **0.709 +/- 0.050 ms per 1000 context (10 % of the term), measured by 0.4**: 11.00 ms at ctx
      > 768 rising to 21.17 ms at 12,288. Note the shape: it nearly DOUBLES over the first 3k and
      > then almost flattens (19.22 at 3072, 19.90 at 6144, 21.17 at 12,288), which is the signature
      > of a top-`k` gather saturating once context exceeds `k` rather than of a full scan. So the
      > lever here is the 11 ms FLOOR, not the slope, and it belongs to Term A as much as to Term B.
      >
      > **CONFIRMED CONCAVE, 2026-08-20 by 1.3's re-attribution.** A fresh dprof over ctx 369..6255
      > fits `cattn:sparse` at **1.694 +/- 0.065 ms per 1000** -- 2.4x what 0.4 reported. It is an
      > artefact of the fitting range. Per-point medians 9.35 / 15.72 / 19.89 ms at ctx 768 / 1536 /
      > 6144 are **8.29 ms per 1000 across the first leg and 0.905 across the second**, joining
      > continuously onto 0.4's flat 19.22 / 19.90 / 21.17. One curve, two ranges.

- [x] **1.8** **EXPLAINED, 2026-08-20. Neither mode is wrong; the mark is. `cattn:q_proj` absorbs
      the compressor emits running concurrently on `g_side`, and the "free 4.4 ms" does not exist.**
      Full write-up: `wiki/measurement-and-traps.md` §28 and `wiki/negative-results.md` §21.

      **The mechanism, and it is arithmetic rather than a fit.** `compressed_verify_step_indexer`
      (`kernels/compressed_decode.cu:495`) records `g_side_fork` and then issues `build_qKV` on the
      MAIN stream, so the two `compressor_emit_group` calls run CONCURRENTLY with the q chain
      (Finding 55/56, ATTN_SPLIT). Those emits fire only for groups COMPLETING inside the verify
      block — `for j in [pos,pos+K): if (j+1)%ratio==0` — and `cattn:q_proj`/`q:wq_a` are timed by
      events on the main stream, so on an emit step they time a GEMM that is sharing the memory
      system with 21 layers x 2 compressor emits, and on a non-emit step they time the GEMM alone.
      Define, from the dprof tag alone,

          g = #{ j in [ctx, ctx+VB) : (j+1) % 4 == 0 }        (ratio 4 = the 21 indexer layers)

      **`g` classified 0.4's OWN 153 per-verify tables 153/153, with a 2.80x gap and no overlap
      between the two populations** — on data collected before the hypothesis existed, at zero GPU
      cost (`tools/qproj_bimodal.py`, `evidence/decode_loop/qproj_1p8_schedule.txt`). The excess is
      LINEAR in `g`, not merely present: `cattn:q_proj + cattn:compress` is +6.97 ms at g=1 and
      +14.78 at g=2. The blocks wide enough to straddle two group boundaries pay twice.

      **Correlation is not cause, so it was tested with the switch the code already carries.**
      Both "the GEMM is intrinsically slower on those steps" and "the mark absorbed another stream's
      work" predict that classification. `NO_ATTN_SPLIT=1` restores the serial order.
      `scripts/qproj_ab_run.sh` ran both arms, SERIAL first so drift penalises the shipped arm, one
      binary, `DSV4_DPROF=1`, 174 verify tables each, and its four predictions were registered in
      the script header before it started. At **VB=2 held fixed** (P(g>=1) = min(1, VB/4), so `g`
      correlates with width by construction and the excess must be read at fixed width — rule 0.3):

      | arm | g | n | `q:wq_a` | `cattn:q_proj` | `cattn:compress` | q_proj+compress |
      |---|---|---|---|---|---|---|
      | serial | 0 | 53 | 1.65 | 10.11 | 0.05 | 10.16 |
      | serial | 1 | 45 | **1.66** | 10.12 | **8.34** | 18.46 |
      | split  | 0 | 53 | 1.66 | 10.12 | 0.04 | 10.16 |
      | split  | 1 | 45 | **5.54** | 14.68 | **2.50** | 17.18 |

      * **P1 CONFIRMED.** Serial arm: `q:wq_a` is UNIMODAL — 1.66 at g=0 against 1.67 at g>=1, a
        **1.00-1.02x** swing at every context, and the classifier that separated 153/153 separates
        nothing. Split arm, same run: **174/174, a 3.05x gap, no overlap.** The 3.2x swing is
        created by the fork and destroyed by removing it.
      * **P2 CONFIRMED.** The identical time reappears in `cattn:compress`: 2.50 -> **8.34 ms**.
      * **P3 CONFIRMED — the conservation check, which is the answer to "which mode is right".**
        The g=0 baseline is **10.16 ms in BOTH arms** (10.26 at VB=3, both arms) — two separate
        checkpoint loads agreeing to two decimals, which is also the control that says the arms are
        comparable. The emit excess is **8.31 ms serial vs 7.02 ms split at VB=2** (8.31 vs 6.92 at
        VB=3), so **overlap hides 16-17 % of the compressor and the other 84 % is real work that
        must happen.** The cheap mode is not a mode; it is the absence of an emit.
      * **P4 CONFIRMED, and it is the ratchet.** Paired per (target, rep), same corpus sha, SERIAL
        arm first:

        | ctx | serial ms/fwd | split | paired delta | speedup | tau s / tau sp | tok/s |
        |---|---|---|---|---|---|---|
        | 12,346 | 142.56 | 141.75 | -0.81 | 1.006x | 1.620 / 1.620 | 11.15 -> 11.21 |
        | 6,196 | 140.91 | 140.20 | -0.71 | 1.005x | 1.707 / 1.707 | 12.02 -> 12.10 |
        | 3,133 | 138.58 | 137.58 | -1.00 | 1.007x | 1.778 / 1.778 | 12.83 -> 12.92 |

        **9/9 legs byte-identical**, `tau` and realised width equal to three decimals on every leg —
        the required signature of a change that moves kernels between streams and alters no
        arithmetic. Per leg rather than per point: **9 of 9 faster under SPLIT, paired mean
        -0.81 ms/forward, sd 0.14, 2 SE band [-0.90, -0.72]** (`fit_1p8_band.txt`). The mark-level
        number predicts that band from the other instrument: 1.29 ms hidden per emit step x
        **64.9 %** of steps carrying an emit (113/174, identical in both arms) = **0.838**, which
        falls inside it. Two instruments, one number.
      * **A THIRD, UNPLANNED CONFIRMATION at a different ratio.** Every `cattn:compress` sample above
        20 ms — **8 of 8 across both arms**, 43-50 ms each — is a step whose block contains a
        **ratio-128** boundary, i.e. the 20 strided layers emitting through
        `compressed_verify_step_strided`, which uses no side stream at all. Same mechanism, other
        ratio, and it also explains the lone 43.90 ms outlier in 0.4's table.

      **What this costs, now that it is named.** The emit is **7.02 ms on the 64.9 % of forwards
      that carry one = 4.56 ms/forward amortised, ~3.3 % of a 140 ms forward**, all of it Term A. Its
      byte floor is 21 layers x (2 x [1024,4096] + 2 x [256,4096]) f32 = **880 MB per group =
      3.67 ms at 240 GB/s**, so the emit runs at **52 % of roofline overlapped, 44 % serial**.

      **What this REFUTES.** 0.4's "if the cheap mode is reachable on demand that is ~4.4 ms/step of
      Term A for free". There is no cheap mode to reach. Every one of those 4.4 ms is compressor
      traffic that a correct engine must move, it is already overlapped, and the overlap is already
      worth a measured 0.84 ms/forward. **A phantom 4.4 ms/step is removed from this ladder's
      headroom.** Term A's remaining distance to its floor is unchanged at 22.4 ms and is elsewhere.

      Instrument: `tools/qproj_bimodal.py` (schedule classifier, works on any dprof server log);
      `scripts/qproj_ab_run.sh` (both arms, detached, named in `detach_audit.sh` in this commit).
      Evidence: `evidence/decode_loop/{qproj_1p8_schedule,qproj_1p8_serial,qproj_1p8_split,
      qproj_1p8_vbcontrol,fit_1p8_paired,dprof_ctx_1p8_serial,dprof_ctx_1p8_split}.txt`.
      Bit-exactness: nothing numeric changed in either arm and 9/9 legs proved it.

- [x] **1.11 DONE 2026-08-20 — ADOPTED, DEFAULT ON, BIT-EXACT. Deferring the ATTN_SPLIT join past
      `i:qidx` and `i:iw` is worth a DRIFT-FREE paired `-0.542 +/- 0.310 ms/forward` over 18 paired
      legs and FOUR checkpoint loads, and `-1.092 +/- 0.146` at ctx 12,410, where it is 6/6 in each
      arm order independently.** 44 of 44 sweep legs byte-identical, `tau` equal to three decimals on
      every one, `gate_join_defer` 0 of 143,360 floats differing.

      **What shipped.** `kernels/compressed_decode.cu:531,561`. The `cudaEventRecord(g_side_join)`
      stays where the two `compressor_emit_group` calls end; the `cudaStreamWaitEvent` moves from
      immediately after `build_qKV` down to `index_score`, the first reader of `idx_ckv`. Nothing in
      between reads what the emits write -- `i:qidx` is a GEMM on `iqrq/iqrs` out of `build_qKV` and
      `i:iw` is a GEMM on `x_cur`. `NO_JOIN_DEFER=1` restores the 1.8 position on the same binary,
      which is what made both arms one build.

      **THE MECHANISM IS VISIBLE IN THE MARKS, AND THE MARKS ARE NOT THE WIN.** K=5, ctx 5, same
      binary, both arms (`dprof_1p11_marks.txt`):

        | mark | joined (1.8) | deferred (1.11) |
        |---|---|---|
        | `cattn:compress` | 2.34 ms | **0.07** |
        | `cattn:indexer` (parent of `i:qidx`) | 3.52 | **4.39** |
        | `i:qidx` | 1.94 | **2.81** |
        | `cattn:q_proj` | 15.95 | 16.19 |
        | verify TOTAL | 127.33 | **125.89** |

      `cattn:compress` collapses because the barrier left it, and `i:qidx` grows by +0.87 because it
      ABSORBS the emit traffic -- the identical signature Finding 56 recorded when `cattn:q_proj`
      absorbed it, and the one 1.8's trap §29 is about. The conserved sum is what matters: top-level
      marks net **-1.32 ms**, verify TOTAL **-1.44 ms**, on a step that carries one emit. Amortised
      at 1.8's measured 64.9 % emit rate that predicts **-0.86 to -0.93 ms/forward**.

      **THE FIRST ORDERING COULD NOT DECIDE IT, AND SAYING SO IS THE ITEM'S MAIN RESULT.** Run 1
      (control arm first, per the standing drift rule) gave paired mean **-0.324, 2 SE
      [-0.670, +0.022] -- the band COVERS ZERO**, which under this item's own pre-registration is a
      null. But the effect being chased is ~0.9 ms/forward and traps §19 measures the offset BETWEEN
      CHECKPOINT LOADS at 0.6 % = ~0.8 ms/forward: pairing removes between-LEG variance and does
      nothing to a constant offset between two loads. So the pair was run AGAIN with the arm order
      REVERSED (`PHASE_FROM=6`, two more loads). Run 2 gave **-0.760, 2 SE [-1.109, -0.411]**.
      Averaging the two per-leg deltas cancels the drift and half their difference measures it:

        | | paired mean | 2 SE band | legs faster |
        |---|---|---|---|
        | run 1 — control arm first | -0.324 | [-0.670, **+0.022**] | 13/18 |
        | run 2 — deferred arm first | -0.760 | [-1.109, -0.411] | 14/18 |
        | **pooled, drift-free** | **-0.542** | **[-0.852, -0.232]** | 12/18 |
        | drift itself (half the difference) | **+0.218** | [+0.061, +0.375] | — |

      **`+0.218 ms/forward` is what the second checkpoint load of a session costs**, measured rather
      than assumed, and it is 40 % of the effect this item was looking for. Every future A/B on this
      ladder whose expected size is under ~1 ms/forward needs both orderings or it is guessing.

      **WHERE IT RESOLVES, AND THE PRE-REGISTRATION IT BREAKS.** By context group, drift-free:

        | ctx | n | mean | 2 SE band | legs faster | run 1 | run 2 |
        |---|---|---|---|---|---|---|
        | 3,197 | 6 | -0.282 | [-0.759, +0.195] | 3/6 | 3/6 | 6/6 |
        | 6,260 | 6 | -0.252 | [-0.858, +0.353] | 3/6 | 4/6 | 2/6 |
        | **12,410** | 6 | **-1.092** | **[-1.238, -0.946]** | **6/6** | **6/6** | **6/6** |

      This item pre-registered "TERM A ONLY, and that is a prediction, not a hope: every byte in this
      window is context-independent. If the paired split shows the saving in the CONTEXT term
      instead, the mechanism claimed here is wrong and the item says so." **The item says so.** The
      saving resolves only at the deepest context; the fitted split is flat `+0.150 +/- 0.564` against
      context `-0.0949 +/- 0.0685` per 1000, R^2 0.324. **I have no mechanism for a context tilt
      here and am not inventing one** -- the two candidates both fail: the emits read one group of
      `ratio` tokens regardless of context, and the emit rate moves only 7 % across these three
      points (realised width 2.56 -> 2.74, P(boundary) ~ w/4). The other reading is that the two
      shorter groups are simply noisier (sd 0.585 and 0.741 against **0.179** at 12,410) and the fit
      is being carried by that difference in scatter, which is precisely the situation rule 7 says
      not to read a slope out of.

      **THEREFORE THE RATCHET IS STATED WHERE IT RESOLVES AND THE ATTRIBUTION IS LEFT OPEN.**
      `a` and `b` are UNCHANGED at 125.11 and 1.887: this measurement cannot say which term moved,
      and writing it into one of them would be inventing the attribution. What it can say is that
      **`a + b*ctx` at ctx 12,410 falls 142.24 -> 141.15 ms, -1.092 +/- 0.146, i.e. +0.77 % tok/s**,
      and that the overall saving across 3,197-12,410 is -0.542 +/- 0.310. Resolving the term needs a
      sweep with more context points in BOTH arm orders; that is item 1.13.

      **BIT-EXACT, and the gate for it did not exist.** `tests/gate_join_defer.cu` is the **first
      gate in this repo that calls `arena_init()` before driving these kernels** -- every other one
      links them without an arena, so `g_side` is null, `asplit` is false, and the fork/join 1.8
      priced at 0.81 ms/forward had never been under test at all. It refuses to report PASS if
      `g_side` came back null. Four cases (1 emit, 2 emits, 0 emits, and a second shape), both join
      positions in one process on identical weights: **0 of 143,360 floats differ**, same under
      `--swap`, and `--negctl` (one perturbed input float) fires on every case, so the memcmp is
      live. Engine: identical generated ids `11111 16 455 6102 294 16603 344 29168` in both arms,
      LOSSLESS gate PASS in both, and **44 of 44 sweep legs byte-identical with `tau` equal to three
      decimals** across the four loads.

      **The gate's first run said FAIL, and the null control is why that is not in this entry as a
      defect.** One row differed -- row 0, an M=1 decode step, which 1.11 does not touch. Pinning
      BOTH arms to the SAME join position still differed, including with the pre-1.11 code in both;
      duplicating a case showed instance 1 differing and instance 2 of identical data clean. It is
      the FIRST `run_arm()` in a process: the arena slab is `cudaMalloc`'d and never zeroed, so some
      scratch in the M=1 step is read before it is written and sees driver garbage on pass 1 and the
      previous pass's bytes forever after. Pre-existing, unrelated to 1.11, burned off with a
      discarded warm arm, and written up in `measurement-and-traps.md` §32. **The uninitialised read
      itself is item 1.14 and is not fixed here.**

      Instruments: `scripts/joindefer_ab_run.sh` (8 phases, both arm orders, named in
      `detach_audit.sh` in this commit); `tools/paired_band.py` (per-leg band, sign count, flat/ctx
      split, and `--reversed` for the drift-free pooling). Evidence:
      `evidence/decode_loop/{gate_join_defer_1p11.log,decode_1p11_off.log,decode_1p11_on.log,
      dprof_1p11_marks.txt,fit_1p11_band.txt,fit_1p11_band_rev.txt,fit_1p11_band_pooled.txt,
      fit_1p11_bygroup.txt,fit_1p11_paired.txt}`.
- [ ] **1.13** **Resolve 1.11's term.** 1.11 measured a real drift-free `-0.542 +/- 0.310 ms/forward`
      that is `-1.092 +/- 0.146` at ctx 12,410 and covers zero at 3,197 and 6,260, and its own
      mechanism argument says it should be FLAT. Three context points cannot separate "the saving
      grows with context" from "the two short groups are noisier". Re-sweep in BOTH arm orders with
      >= 6 context points spanning 1.5k-12.4k. This is a MEASUREMENT item and it is ranked below the
      kernel items on purpose: it changes no kernel and buys no throughput. Take it only when a
      kernel item needs the attribution.
- [ ] **1.14** **The first arena pass reads scratch it never wrote.** `gate_join_defer`'s null
      control found it: the FIRST `compressed_decode_step_indexer` in a process produces a different
      row 0 from every later one, with the same inputs and the same code, and the second instance of
      an identical case is clean. The arena slab is `cudaMalloc`'d and never zeroed. Find the kernel
      that reads before it writes (start with the `dmalloc`'d scratch in the M=1 step that is not
      fully overwritten before use) and either zero it or fix the read. **In the engine this is one
      token per process**, so it is LOW on throughput -- but it is a correctness hole that every
      bit-exactness gate in this repo has been silently working around.
- [x] **1.12 DONE 2026-08-20 — ADOPTED DEFAULT ON, AND IT IS BIGGER THAN THE ITEM PREDICTED.
      `gemm_fp32` gets a (4x4) warp tile: the ratio-128 emit spike falls `40.76 -> 14.50 ms`
      (`2.82x`, 69 samples over four loads, the two populations DISJOINT), and the drift-free paired
      throughput is `-1.963 +/- 1.504 ms/forward`, 14/18 legs, band EXCLUDING zero, entirely in
      TERM A.** `a` **125.11 -> 123.15 ms** (1.522x -> **1.499x** the 82.18 ms floor); `b` unchanged.

      **The item's own premise was wrong in three places and the mechanism survived all three.**
      (i) the strided path passes `overlap=false`, so `ntok = ratio = 128`, not `2*ratio = 256`;
      (ii) B is therefore `[d, DIM] = [512,4096]` f32 = **8.39 MB**, not 33.55; (iii) B is read
      `ceil(128/8) = 16` times, not 32. What was right is the shape of the defect: one warp per
      (8-row chunk, n) reads the whole of B once per chunk, from **sixteen separate host launches**.

      **The fix.** A warp owning `MM` rows of A and `NN` rows of B issues `MM + NN` loads for
      `MM*NN*4` FMAs, so operand traffic is `M*N*K*(1/NN + 1/MM)` floats: the shipped `(8,1)` is
      **1.125**, `(4,4)` is **0.375**. The m-chunk loop also moves from the host to `gridDim.y`.
      Bit-exact by construction -- every output is still produced by ONE warp with lane `l`
      accumulating k4 indices `l, l+32, ...` into the same four chains and the same shuffle tree;
      widening the tile changes which warp owns an output, not the order of any dot product.

      **17 tiles benchmarked, `(4,4)` won at every shape the engine issues**
      (`tools/f32mk_bench.cu`, `evidence/decode_loop/f32mk_bench_1p12.txt`), against the EXACT
      pre-1.12 code as the baseline arm (`DSV4_F32MK_TILE=8x0`):

      | shape | call site | before | after | |
      |---|---|---|---|---|
      | `[128,512]x[512,4096]` | strided emit, x40 per emit step | 0.5744 ms | **0.2477** | **2.32x** |
      | `[8,1024]x[1024,4096]` | indexer emit main, x42 | 0.0729 ms | **0.0407** | **1.79x** |
      | `[8,256]x[256,4096]` | indexer emit idx, x42 | 0.0331 ms | **0.0177** | **1.87x** |
      | `[5,64]x[64,4096]` | `idx_weights_proj`, x21 on EVERY step | 0.0143 ms | 0.0144 | 1.00x |

      That last row is why the tile is **OFF below M=8**: at `(4,4)` a 5-row GEMV becomes a 4-row
      chunk plus a 1-row tail, `0.0148 -> 0.0205 ms`, and it fires 21x per forward against the emit
      shapes' once every 4 or 128 positions. The guard also makes both arms IDENTICAL below M=8,
      which is what makes the paired number attributable to the emit and nothing else.

      **In the engine the saving is TWICE what the bench predicted, and the mechanism says why.**
      The bench reuses one 8.39 MB B across 200 reps, so 15 of its 16 passes are L2 hits; the engine
      walks **20 layers x 2 matrices = 335 MB of distinct weights**, so the passes it removes are
      DRAM. Predicted `-13.07 ms` per emit step, measured **`-26.26`**.

      | `tools/emit_spike.py`, pooled over 4 loads | n | before | after | |
      |---|---|---|---|---|
      | `cattn:compress`, steps with a ratio-128 boundary | 69 | **40.76** [39.92, 45.22] | **14.50** [14.29, 14.64] | **-26.26 ms, 2.82x**, no overlap |
      | verify `TOTAL` excess on those steps | 69 | +63.17 | **+34.26** | **-28.91 ms** |
      | verify `TOTAL` excess on ratio-4 boundary steps | 1811 | +13.52 | **+11.03** | **-2.49 ms** |
      | `cattn:q_proj` excess on ratio-4 boundary steps | 1811 | +4.61 | **+3.92** | **-0.69 ms** |

      The ratio-4 row is the one that carries the throughput: **62.8 % of steps** have a ratio-4
      boundary, and 1.8 showed `cattn:q_proj` is where the side-stream emit's contention lands.
      `0.628 x 2.49 + 0.0239 x 26.42 = ` **`-2.20 ms/forward`** predicted from the marks -- and the
      paired band is `-1.963 [-3.467, -0.459]`. Two instruments, one number.

      **ONE ARM ORDER CALLED IT A NULL AGAIN.** Control-first: `-0.892, 2 SE [-2.134, +0.350]`,
      10/18. Change-first: `-3.033, 2 SE [-5.372, -0.695]`, 13/18. Pooled drift-free:
      **`-1.963, 2 SE [-3.467, -0.459]`, 14/18**, drift `+1.071 +/- 1.115`. Exactly 1.11's lesson
      (traps §33), on the very next item. **TERM A as pre-registered**: flat `-2.348 +/- 3.302`,
      context `+0.0533 +/- 0.4038` per 1000, R^2 0.004.

      **Bit-exactness.** `gate_bf16w` now sweeps all **18 dispatchable tiles including the pre-1.12
      `8x0` arm**, at 7 values of M and 2 of N (the `N=254` column is the guarded-store tail):
      **252 comparisons, worst max|diff| = 0**. Negative control: perturb the epilogue by one part in
      1e7 and **17 of 18 tiles FAIL** while `8x0` correctly does not. `build/decode` emitted
      **identical generated ids** in both arms with LOSSLESS PASS, and 11 in-situ gates pass in both
      arms. The sweep-level text-identity column is NOT usable here and the null control proves it:
      **two loads of the same binary also gave 0/18 byte-identical** (traps §35).

      **THE FIRST ENGINE A/B WAS A FALSE NULL AND COST FOUR LOADS.** `DSV4_F32MK_TILE=8x0` was
      rejected by its own parser (`nn > 0`), so both arms ran 4x4 and the answer was `-0.03 ms` on
      the mark. The tell was that the null was *too clean* for a 3.5 %-spread instrument. The engine
      now prints `[f32mk] tile MMxNN` on the first `gemm_fp32` of the process and the harness EXITS
      NON-ZERO if the two arms log the same tile. Written up in traps §34; the wasted pair is kept
      as `evidence/decode_loop/fit_1p12null_*` because it is this ladder's best null control
      (`+0.512, 2 SE [-0.969, +1.993]` on identical code).

      Instruments: `tools/f32mk_bench.cu` (17 tiles x 4 engine shapes, times AND gates them);
      `tools/emit_spike.py` (conditional-on-schedule mark distribution -- the only instrument that
      can see a change amortised below the noise floor); `DSV4_GEMM_TRACE=1` (per-(route,M,N,K)
      call-site histogram, which is what proved the shape was reached and the arm was not).
      `scripts/f32mkn_ab_run.sh`, 9 phases, both arm orders, named in `detach_audit.sh` in this
      commit. Evidence: `evidence/decode_loop/{f32mk_bench_1p12.txt,gate_bf16w_1p12.log,
      emit_spike_1p12{,_r4,_pooled,_r4_pooled}.txt,fit_1p12_band{,_rev,_pooled}.txt,
      fit_1p12_paired.txt,dprof_ctx_1p12_{off,on}.txt,decode_1p12_{off,on}.log,
      decode_1p12_trace2.log,fit_1p12null_*}`.

      **What it leaves.** `cattn:compress` on an emit step is still **14.50 ms against a 335 MB /
      1.40 ms byte floor** -- 4.3 % of roofline -- because at `(4,4)` the kernel is now register- and
      issue-bound, not traffic-bound: `(6,6)`, `(4,8)` and `(8,4)` all measured SLOWER than `(4,4)`,
      so the next factor is not a bigger tile. It needs tensor cores or a shared-memory B stage, and
      it is a NEW item, not this one.
- [x] **1.6** **RESOLVED as pre-existing — proven by a control build, not by argument.**
      Built `build/decode_prechange` from `1a33cfe^` (the two kernel files and the header, before the
      warp top-k) and ran it: **`err711=42`, `err820=21` — the same fault.** The launch error has
      nothing to do with this work. Two earlier calls of mine on this ("it is mine", then "it is
      pre-existing because mla_attn.cu changed") were both reasoning from circumstance; this is the
      experiment that should have been run first and it took one model load.

      Still latent: output correct, LOSSLESS gate passes on every run including the control, tau
      normal. Both sites are `dsync` on return paths of `ogroup_gemm_fp8`, and the bs==1 kernel
      launched there has no launch bounds, no static shared memory and a 1024-block grid, so it
      cannot fail on its own — the flag is stale from an unprobed launch, plausibly a side stream
      (Finding 55 forks one). **Tracked, not blocking.** Worth asking whether the failing launch is
      a dead branch: the NVFP4 wo_a overlay in that function is documented DEFAULT OFF.

      Free A/B from the control, ctx 480: base AR **92.6 -> 89.3 ms/tok**, spec **67.2 -> 65.9**.
      Small here by construction — at short context the top-k was never the cost. The win is the
      slope, and 0.2 measured that as flat.

## Phase 1b — bit-exact packing (`KV_PRECISION_FINDINGS.md`)

- [~] **1b.1** Pack the DSA index cache as real MXFP4: 128xE2M1 + 4xUE8M0 = **68 B** (was 512 B).
      **Primitives done and gated; wiring still to do.** `include/idx_pack.h` carries the layout and
      the exact inverse of the write path; `tests/gate_idx_pack.cu` proves
      **unpack(pack(x)) == x bit-for-bit on 524,288 elements, 0 mismatches**, across tiny (1e-6),
      large (1e5) and zero-heavy rows. 512 B -> 68 B, **7.53x** — the same 68 B/token SGLang ships
      for the DeepSeek-V4 FP4 indexer pool, which is an independent check on the arithmetic.

      **The gate immediately earned itself.** The first implementation took the sign from `v < 0`,
      but `round_e2m1` returns NEGATIVE ZERO for a negative input that rounds to zero, and
      `-0.0f < 0.0f` is false — so the sign was dropped. 16,011 mismatches at **worst |delta| = 0**:
      numerically perfect, bitwise wrong, and invisible to any tolerance. Fixed with `signbit`.

      Remaining: 6 allocation/memcpy sites bake in the 4-byte stride (`src/engine.cu:346`,
      `src/decode.cu:251/444/453/472/831`), the write path in `compressor_emit_group`, and both
      `index_score` read kernels. Acceptance stays `memcmp` on generated token ids.
- [x] **1b.2 DONE 2026-08-20 — SHIPPED BIT-EXACT AND DEFAULT OFF. The KV cache packs 2048 -> 720 B
      per row, and it is a CONTEXT-TERM win paid for out of TERM A: paired
      `+3.233 +/- 0.203 ms/forward` flat against `-0.4996 +/- 0.0290 per 1000 context`, R^2 0.990,
      break-even ctx 6,471.** 16 of 16 legs byte-identical, `tau` and mean verify width equal on
      every leg.

      **What shipped.** `include/kv_pack.h` + `DSV4_KV_PACK=1`. Dims 0-447 as 448 E4M3 bytes + 7
      UE8M0 scale bytes, dims 448-511 (RoPE) untouched FP32: **711 B of payload in a 720 B stride**
      (720 is the smallest 16 B-aligned stride, which is what lets `sparse_attn` gather the row with
      vector loads), **2.844x**. 720 is divisible by 4, so a cache row is still
      `base + row * g_kv_rowf` floats in both layouts -- that one decision is what kept this to a
      stride substitution at ~30 call sites instead of a retyping of every cache pointer. Producer
      seam is `kv_stage`/`kv_commit`: in the FP32 arm `kv_stage` returns the cache row itself and
      `kv_commit` IS the `act_quant_fp8sim` call that was already there, so **the before-arm is
      unchanged code, not just unchanged behaviour**.

      **BIT-EXACT, and that is four gates and an engine A/B, not an argument.** The write path
      already computed `(float)(half)e4m3(x/scale) * scale` and threw the code away, so packing
      keeps the byte and the exponent and unpacking replays the same two conversions and the same
      multiply. Acceptance is `memcmp`; required `n` is zero at a flip rate of zero.
      * `gate_kv_pack` -- round trip vs `act_quant_fp8sim` on **524,288 floats x 5 distributions x
        raw and re-quantised, 0 mismatches**, including the zero-heavy signed case that caught
        1b.1's sign-bit bug at worst |delta| 0; plus the reader vs the FP32 reader at **11 (hpb,smem)
        launches x 5 engine shapes, 0 differing bytes**. Two negative controls (one code byte, one
        scale byte) both fail as they must.
      * `gate_kv_pack_e2e` -- the **WIRING**, which the first gate says nothing about. Drives
        `compressed_attn_cache[_r4]`, `compressed_decode_step_{strided,indexer}` and
        `compressed_verify_step_*` on synthetic weights, once per layout, and memcmps: **0 of 114,688
        floats differ**, and again with `--swap` on the arm order.
      * `gate_compressed_graph` / `gate_indexer_graph` now call `kv_pack_init()` and PRINT the layout,
        which is what makes them a real test of `kv_store_at` / `kv_store_comp` -- the device-pos
        store kernels nothing else exercises. PASS, maxabs 0.00e+00, in both layouts.
      * Engine: **identical generated ids** `11111 16 455 6102 294 16603 344 29168` on both
        `build/decode` arms (6-id prompt, below 1.9's 160-position threshold, so the token-id
        invariant is valid here), `tau` 2.87 on all six replicates, LOSSLESS x3 on both arms, and
        16/16 server legs byte-identical.

      **The band, both arms in one session on one build, control arm first (so drift flatters the
      OLD layout):**

      ```
      ctx      ms/forward before -> after   paired   tok/s before -> after      tau
       1,656   130.27 -> 132.45             +2.18    14.27 -> 14.04             1.861 both
       3,197   132.77 -> 134.74             +1.97    12.44 -> 12.26             1.652 both
       6,248   137.07 -> 137.18             +0.10    12.23 -> 12.22             1.681 both
       6,260   134.16 -> 134.17             +0.01    12.13 -> 12.12             1.620 both
      12,410   142.27 -> 139.31             -2.97    12.09 -> 12.34 (+2.1 %)    1.736 both
      ```

      Regressed on context: **+3.233 +/- 0.203 flat, -0.4996 +/- 0.0290 per 1000, R^2 0.990** (2 SE
      bands [+2.83, +3.64] and [-0.558, -0.442]; both exclude zero). The independent per-arm fits
      with the width term controlled give `b` 0.924 -> 0.381 and `a` 82.83 -> 85.74 -- the same
      answer by a different route. **So `b` 1.887 -> 1.387 (`b x 6592` 12.44 -> 9.15, -26 %) and `a`
      125.11 -> 128.34 (+2.6 %).**

      **WHY A BYTE REDUCTION MOVED `b` AND COST `a`, which is the finding and not a footnote.**
      `KV_PRECISION_FINDINGS.md` §4 said in advance that this is not a bandwidth-bound path and
      should not be expected to pay, and half of that was right. The FLAT cost is instructions:
      `sparse_attn` unpacks every gathered row, 1.7 established that kernel is ISSUE-bound, and
      `gate_kv_pack --bench` prices it directly -- the packed reader is **0.776x / 0.805x / 0.866x /
      0.820x / 0.904x** of the FP32 reader at the five engine shapes. **A third fewer bytes and the
      kernel got slower at every one of them.** The CONTEXT saving is bytes: the gathered working set
      is `topk` rows and `topk` grows with context, so packing shrinks exactly the part that scales
      -- §4's own L2-residency hypothesis, which it labelled "hypothesis, not measurement", now
      measured, landing on the term it named.

      **THE FIRST IMPLEMENTATION WAS 55 % WORSE ON TERM A AND THE FIX INVERTS AN INSTINCT.** The
      staging loop originally read a `uint4` -- 16 codes per thread, only 28 vector loads for 448
      codes, the better number by every roofline metric. The block is 128 threads at `hpb=4`, so **28
      threads did sixteen unpacks each while 100 waited**, on the double-buffered prefetch's critical
      path: 0.699x on the bench and **+4.998 +/- 0.374 ms/forward** in the engine (that arm is kept
      at `evidence/decode_loop/fit_1b2_*_v1*`). Four codes per thread -- FOUR TIMES the load
      instructions -- took it to +3.233 +/- 0.203 **while leaving the slope alone**
      (-0.4653 +/- 0.0534 -> -0.4996 +/- 0.0290, overlapping), which is a clean separation: the fix
      touched issue, not bytes. Minimising loads was the wrong objective; occupying the block was the
      right one. The `(hpb, smem)` retune was then checked rather than assumed and is a NULL -- the
      packed optimum is `hpb=4, smem=2`, the same launch 1.7 chose for FP32.

      **DEFAULT OFF, and the reason is the stop condition.** `a` is 88 % of the forward and the
      ladder's binding term; this makes it 3.2 ms worse to make `b` 26 % better, and the two cancel
      almost exactly at the ladder's own reference context (net -0.06 ms at 6592). Flipping the
      default is a decision about the operating context, so it is written down rather than made
      silently: **above ctx ~6.5k `DSV4_KV_PACK=1` is a throughput win, below it is a loss.**

      **THE ITEM'S STATED JUSTIFICATION WAS WRONG AND IS CORRECTED, not quietly dropped.** "Unlocks
      seqmax 32k-64k, which does not fit today" was inherited from
      `KV_PRECISION_FINDINGS.md` §3, which priced the KV cache in isolation and never checked the
      resident set. **Both layouts load at seqmax 32768 and serve a completion** -- FP32
      `mem 115.3/122.8 GiB`, packed `115.1`. What packing buys is headroom and 65536, not 32768.
      And the `mem` line does NOT show the 2.08 GiB the arithmetic predicts between those two arms,
      because `cudaMalloc` on this unified pool costs no physical page until it is TOUCHED: a
      capacity claim taken at load time measures the allocator, not the machine
      (`wiki/measurement-and-traps.md` §31). The saving is claimed from the allocation arithmetic --
      exact, two lines of `src/engine.cu` -- and it is **2.844x on every KV allocation**:
      1.61 -> 0.57 GiB at seqmax 16384, 3.21 -> 1.13 at 32768, 6.43 -> 2.26 at 65536.

      **One gap, stated rather than papered over.** `mla_decode_step_dp` -- the sliding-layer
      device-pos path, 2 of 43 layers, reachable only through `build/decode GRAPH=1` -- is wired but
      has no gate in EITHER layout, because none existed before this item either. The compressed
      device-pos paths, which are the other 41 layers, are covered by the two graph gates above.

      Evidence: `evidence/decode_loop/{kvpack_ab.log, kvpack_capacity.log, fit_1b2_paired.txt,
      fit_1b2_{off,on}.txt, fit_1b2_paired_v1.txt, kvpack_1b2_capacity.txt, gate_kv_pack*_1b2.log,
      decode_1b2_{off,on}.log}`. Drivers: `scripts/kvpack_ab_run.sh`, `scripts/kvpack_capacity_run.sh`
      (both in `detach_audit.sh`'s PATTERNS).
      Wiki: [`kernel-optimisations.md` §3], [`negative-results.md` §4i], [`context-scaling.md`],
      [`measurement-and-traps.md` §30, §31], `KV_PRECISION_FINDINGS.md` §3 correction, README state
      table.

## Phase 2 — speculation (accuracy-neutral by construction)

- [ ] **2.1** Re-tune block width AFTER 1.1/1.2. Verify width is currently free only by accident of
      the broken kernel; with Term B small the optimum returns to ~7-9 from an apparent 11-13.
- [x] **2.2** **DEPLOYED, 2026-08-20 — the server has now run it, and this is the first time.**
      Chose the staged checkpoint over a `--head` flag, and the engine's own source is the argument:
      on 0731-REAP all 2977 `mtp.*` tensors are EMBEDDED in shards 46-48 and those shards hold
      nothing else, so `src/engine.cu` reads the head out of the main `WeightStore` precisely
      because a second store "would duplicate ~6.5 GiB against ~16 GiB of headroom". A flag ADDS a
      mapping; staging REPLACES one. `scripts/stage_head.sh` builds
      `~/models/ckpt-head-s3` as a symlink farm — 45 shards, the tokenizer and config linked to the
      read-only base **at the same inode**, the three head shards linked to
      `~/model-backups/heads/s3/` — so the resident set is bit-for-bit a production load, the
      checkpoint is never written, no disk is spent, and **the A/B runs an identical binary**.
      `config/live_ckpt` (tracked, one line) is what `serve.sh` reads; every launcher in the repo
      execs `serve.sh`, so this is the single switch. `tools/verify_staged_ckpt.py` gates it on the
      CPU before any load: **45,821/45,821 tensors drop-in identical** in dtype/shape/byte-length,
      45 files the same inode as base, the 3 head shards' sha256 equal to what `head_card.json`
      recorded at promotion, and — the check that matters most — the head shards proven to be
      DIFFERENT bytes from base, because a no-op stage would have measured the shipped head twice
      and called it a deployment.

      **The archived numbers were not admissible as a before-arm** (rev `85dbea6c` / `2632540`,
      before 1.0/1.2/1.3/1.4/1.5), so both heads were re-measured back to back on `93699e6`, same
      binary, frozen protocol (8-prompt suite, block 6, adaptK 1.50, NGEN0 200, clean, cache dropped
      before each), LOSSLESS GATE PASS and first-token GATE PASS on both arms:

      | | base AR | suite mean tau | suite mean tok/s | tok/s / base AR |
      |---|---|---|---|---|
      | shipped | 11.41 | **3.5362** | 22.1425 | 1.9406 |
      | `s3` | 11.33 | **3.8438** | 24.2512 | 2.1404 |
      | delta | **-0.70 %** | **+0.3076 (+8.70 %)** | **+9.52 %** | **+10.30 %** |

      The -0.70 % on base AR is the drift control: base AR never touches the draft head, so it
      measures the between-load spread (§19 puts that at 5.7 %) that per-prompt pairing cannot
      cancel. It came in at 0.7 %, so the two loads were unusually well matched and the tok/s gain
      is not drift. Paired per prompt: **`d tau = +0.3075 +/- 0.4814`, 5/8 legs positive** — and
      that band is NOT measurement error, it is between-PROMPT heterogeneity, because `tau` is exact
      (below). Read it as: decided on this corpus, weakly established on an arbitrary prompt.

      **What improved is the floor, not the ceiling, and that is the finding.** Per-prompt `tau`
      shipped `5.15 4.43 1.84 1.75 5.54 4.46 1.85 3.27` -> s3 `3.64 4.20 2.49 3.06 3.94 6.09 3.85
      3.48`. The three prompts where the shipped head barely speculated at all (`tau` < 2 of a
      possible 5) are the three largest gains; **after s3 no suite prompt is below 2**; s.d. across
      the suite falls **1.570 -> 1.056, -32.8 %** while the mean rises; and prompt 7 goes 14.39 ->
      26.33 tok/s, **+83 %**, which no suite mean will ever show. On a real workload the worst
      prompts dominate wall-clock, so mean `tau` understates this.

      **Proven live, not asserted.** Server started via `scripts/run_server.sh` (detached, ppid
      systemd, tty `?`, `detach_audit.sh` green), log shows `live checkpoint pinned by
      config/live_ckpt -> /home/patrickd/models/ckpt-head-s3`, and a completion on the gate prompt
      returned **`tokens_per_verify = 2.857`** against the shipped head's **3.61** and s3's **2.87**
      for that same prompt offline. That is behavioural proof of which weights are resident, not a
      path in a log line. Evidence: `evidence/decode_loop/2p2_arm{A_shipped,B_s3}.log`,
      `2p2_ab.txt`, `2p2_live_proof.txt`, `2p2_live_probe.json`. The server was then stopped — the
      deployment is the tracked pointer file, not a running process, and leaving 100 GiB resident
      would block the next iteration's GPU work.

      Wiki: [`kernel-optimisations.md` §2.8], [`draft-head-finetuning.md` §8],
      [`measurement-and-traps.md` §22], `HEAD_REGISTRY.md` "What is actually being served",
      README state table.

      **Two things this turned up that are NOT part of 2.2 and are queued as 2.4 and 2.5 below.**
- [ ] **2.3** Use the confidence head at verify time (EVICT-style `argmax E[A(T_k)]/C(k)`). It
      exists and is unused.
- [ ] **2.4** **`promote_head.py`'s selection rule compares the wrong column, across the wrong
      axis.** Its docstring criterion 4 says "suite-mean **tau** must exceed the incumbent's by more
      than the 3.5 % run-to-run spread"; the code compares `suite_tok_s`, and against an incumbent
      row read out of `HEAD_REGISTRY.md` — i.e. a number recorded on whatever engine revision was
      current when THAT head was measured. 2.2 measured the drift on that axis directly: on the same
      frozen protocol, suite `tau` reproduced the archived value to **four decimal places for both
      heads** across 8 days and five decode-kernel rewrites, while suite tok/s moved -2.3 % and
      -5.0 % and base AR moved **-17.4 %**. `s2` was refused by a 2.5 % margin on exactly this
      comparison. Fix: require the incumbent to be **re-measured in the same session** as the
      candidate (which is what 2.2 did, and what makes the drift common-mode), and use `tau` as the
      cross-session anchor that says the re-measurement was faithful. Three of the registry's seven
      rows are `not promoted` verdicts of this shape and should be re-adjudicated, not overturned by
      assertion. **Instrument-shaped, so it needs a named unblock: it decides whether s2 and the
      three ce/tv ablation heads — all measured, all archived, all currently refused — are actually
      worse than s1, and one of them may already beat the head 2.2 just deployed.**
      [`measurement-and-traps.md` §22]
- [ ] **2.5** **Base AR fell 72.5 -> 87.7/88.2 ms/tok on the identical `WARM decode` measurement,
      and nothing explains it.** Measured twice today in two independent loads, same gate prompt,
      same 7-step average, -17.4 %. **Do not treat it as a 17 % regression without checking the
      measurement first**: speculative throughput on the same suite in the same runs is flat
      (22.655 -> 22.1425, inside the 3.5 % spread), and spec runs the same forward — a real 17 %
      forward regression cannot leave spec unchanged. Likeliest reading is that a 7-step warm
      average is too short for a path that 1.0 put a main-KV cache on. Either way **every "N.NNx vs
      base AR" figure in this repo is a ratio whose denominator moved 17 % while its numerator did
      not**, and the shipped head's own ratio inflated 1.6464 -> 1.9406 across the ladder without
      the head changing. Term A is 83 % of the forward, so if it IS the engine this is larger than
      everything the ladder has won put together — which is the reason it gets its own item and not
      a footnote. [`measurement-and-traps.md` §22.2]

      **ANSWERED 2026-08-20 by 3.1, and this entry's own reading was the right one: it is the
      measurement.** `WARM decode` is a 7-step average taken immediately after load — i.e. inside the
      ~2 s governor ramp — so it times the governor, not the engine. Six loads today, same binary,
      same gate prompt, same protocol, differing only in whether the rails were pinned:

      | machine state | base AR (`WARM decode`) |
      |---|---|
      | governed | **88.0 / 88.0 / 88.7 / 88.4 ms/tok** (11.36 / 11.36 / 11.27 / 11.31 tok/s) |
      | pinned | **72.7 / 72.9 ms/tok** (13.75 / 13.72 tok/s) |

      That is **+21.0 to +21.7 %**, it reproduces to a tenth of a millisecond in *both* states, and
      it is the whole of the 72.5 -> 87.7/88.2 "fall". The earlier numbers were taken on a pinned
      box and the later ones were not. HARDWARE.md predicted exactly this at **+20.7 %** on
      2026-08-07 and named the mechanism — "that window is measured immediately after load, before
      the governor has ramped" — and nobody connected the two because **no run recorded its clock
      state**. There is no 17.4 % engine regression. This entry's own reason for doubting one holds
      up perfectly: spec throughput was flat across the same runs because the suite runs *after* the
      ramp, so it never saw it.

      **Left unchecked deliberately.** 3.1 supplies the mechanism and the six-load table; what 2.5
      still owns is the consequence it names — every "N.NNx vs base AR" ratio in this repo has a
      denominator that depends on a clock state nobody recorded, and those need re-stating against
      the pinned ~72.8 ms/tok now that `run_model.sh` pins by default. That is a sweep through the
      archive, not a measurement, and it should be its own iteration.

## Phase 3 — clocks, then hand back

- [x] **3.1 DONE 2026-08-20 — the lever was already applied. The governor holds BOTH rails at
      their ceiling for 97.7 % of the compute window, so `jetson_clocks` buys the ramp and not the
      ceiling: +2.0 to +3.0 % on the suite, below the 3.5 % spread, NOT counted as a win.** Adopted
      anyway, in the launchers, because of what it does to the *measurement*.

      **Both halves of the item text were wrong about the workload, and one sampler settled it.**
      The item read "GPU 1386 -> 1575 MHz, EMC 2750 -> 4266". Sampling both rails every 2 s across
      three governed arms (`clocks_emc_*.samples`):

      | arm | machine state | compute-window samples | EMC at 4266 | gpc at 1386 |
      |---|---|---|---|---|
      | A2 | governed | 43 | **97.7 %** (mean 4231 MHz) | **97.7 %** (mean 1378 MHz) |
      | A2p | governed | 42 | **97.6 %** (mean 4241 MHz) | **97.6 %** (mean 1380 MHz) |
      | B2 | pinned 120W | 89 | 100 % | 100 % |

      The 315 MHz samples everyone took for "the run" are the ~90 s checkpoint load, when the GPU is
      idle and the CPU is reading 100 GiB. EMC's 2750 is its *idle* rate; it ramps to 4266 within
      ~2 s of the GPU going busy, **without any pin**. So the remaining 2.3 % is the ramp, and that
      is the entire size of the lever.

      **1575 MHz is worth nothing, which is what a bandwidth-bound engine should say about a core
      clock.** Arm C ran at a verified 1575 MHz for 18/18 samples (`nvpmodel -m 0`, which is the only
      way to reach it — at nvpmodel 1 `/etc/nvpmodel.conf` caps GPU MAX_FREQ at 1386, so
      `jetson_clocks` alone provably cannot get there) and measured **+1.68 +/- 5.83 % paired**
      against pinned-120W. The deployment therefore pins at the **current** power mode and does not
      touch `nvpmodel`.

      **The first run of this item was VOID and said so itself.** `clocks_ab_run.sh` pre-registered
      "if A' does not come back to A the whole comparison is void", and A' came back **+8.19 %** on
      the suite mean — larger than the +5.70 % pinning was supposed to be worth. Rerun on a quiet
      box with an ABA bracket, the drift control passes (+1.78 %) and the answer shrinks to:

      | comparison | suite mean tok/s | paired d % | `tau` |
      |---|---|---|---|
      | B2 vs A2p (closest in time, conservative) | 25.585 vs 25.0912 | **+1.99 +/- 0.15 %**, 8/8 legs | 3.8438 both |
      | B2 vs a time-interpolated governed baseline | 25.585 vs 24.8484 | **+2.96 %** | 3.8438 both |
      | A2p vs A2 (drift control) | 25.0912 vs 24.6525 | +2.03 +/- 0.99 % | 3.8438 both |

      `tau` is **3.8438 in all six arms across both runs, to four decimal places** — the negative
      control this item needed, since clocks cannot change which token wins an argmax.

      **Adopted for the measurement, not the 2 %.** `scripts/pin_clocks.sh` (pin | restore | show |
      pin-maxn), called by `run_model.sh` and `run_server.sh` unless `DSV4_PIN_CLOCKS=0`, which the
      clock arms set so they can still measure a governed baseline. Both launchers now also write a
      `<log>.clocks` sidecar — HARDWARE.md has demanded since 2026-08-07 that "every future
      measurement must state whether clocks were pinned", and nothing was recording it, which is why
      this ladder ran every one of its A/Bs against a ramping governor without knowing.

      **It also resolves 2.5 outright — see the note there.** Evidence:
      `evidence/decode_loop/clocks_3p1_ab.txt` (the void run, kept), `clocks_emc_report.txt`,
      `clocks_emc_{A2,B2,A2p}.{log,samples}`. Wiki: [`negative-results.md` §4e],
      [`measurement-and-traps.md` §23, §24], [`hardware-sm110a.md` §5], README state table.
- [ ] **3.2** Final `PERF.md` re-run and both coefficients recorded. **Then continue into the P2
      ladder below — do NOT stop.** (Amended 2026-08-20 on the operator's explicit instruction: the
      kernel phase closes here, the programme does not. Everything after this point is `tau` and
      serving, not `ms/forward`.)

## P2 — the acceptance programme

**Authorised 2026-08-20.** The kernel path is at the bandwidth floor and `tau` is the only
multiplier left. Full reasoning, arithmetic and citations: [`PHASE2_PLAN.md`](PHASE2_PLAN.md).
These items are written to the same rules as the ladder above — pre-registered ceiling, named
instrument, a gate that can fail, evidence path — with two additions the kernel items did not need:

- **Every long stage is DETACHED** (`CLAUDE.md`). Captures, regenerations and fine-tunes are
  launched with `setsid` and *polled* by a later iteration. An item whose work does not fit the 4 h
  `ITER_TIMEOUT` must launch, record the pid and log path in its ladder entry, and return; the next
  iteration picks it up. Add every new stage to `detach_audit.sh`'s `PATTERNS` in the same commit —
  a stage the audit does not know about reports as detached by never being looked at.
- **Nothing is ever deleted from `~/model-backups/heads/`.** P2.0 makes that checkable.

The one number to hold onto: today is **24.7 tok/s at suite `tau` 3.84**; block-6 perfect acceptance
is **40.6**; the defensible target is **30–33** and it is reached by lifting the *constructive*
categories (`explanation` 1.75, `code_gen` 1.84, `reasoning` 1.85) without losing the
*reconstructive* ones (`long_context` 5.54, `agentic_format` 5.15) — which is exactly what every
training session so far has failed to do (F117).

### P2.0 — preserve what exists, before anything trains

- [ ] **P2.0** **Make the head archive self-proving, and keep it that way.** `tools/verify_head_archive.py`
      exists and passes: 10 directories, 46 files, 0 problems, and the registry and the directory
      tree agree in both directions. Two things it does not yet do.
      (a) **The `--full` sha256 pass has never been run** — the quick pass proves presence and size,
      which catches deletion and truncation but not bit-rot. Run it **only when the loop is idle**
      (it reads ~30 GB and will contend with any benchmark), record the result under
      `evidence/head_archive_full.json`, and treat that file as the baseline every later run is
      compared against.
      (b) **Nothing runs it automatically.** Wire it into `s5_session.sh` as a preflight (refuse to
      start a session if the archive is already damaged — training on top of a broken archive is how
      you lose the only copy) and into the P2.9 release step as a postflight.
      **Gate: `--full` exits 0, and the baseline JSON is committed.** No head trains until it does.

### P2.1 — the instrument, before the corpus

- [ ] **P2.1** **The pattern labeller.** For every generated token, `recon(i) = 1` iff the 8-gram
      ending at `t_i` appears anywhere in `prompt + committed output before i`; label each 64-token
      span by its mean into `R-high` (>= 0.75), `R-mid` (0.35–0.75), `C-low` (< 0.35).
      **Extend `DSV4_SUFFIXPROBE`, do not write a second matcher** — it already maintains the match
      machinery, is read-only, and does not perturb GATE or LOSSLESS.
      **Why this is item one:** it is the same function offline and online. Offline it labels the
      corpus; online it drives block width (P2.6). If they are two implementations they will
      disagree, and the disagreement will be discovered in a trained head.
      **Gate: labelling the frozen 8-category suite reproduces the pattern ordering in
      `PHASE2_PLAN.md` §1** — the reconstructive categories must land `R-high`/`R-mid` and the
      constructive ones `C-low`. If it does not, the labeller is wrong, not the table.

- [ ] **P2.2** **Falsify H1: `tau` depends on the generation pattern, not the harness.** Capture the
      same 40 tasks through **two structurally different harnesses** — a plan-then-edit loop and a
      single-shot whole-file rewriter — and regress measured `tau` on pattern mix.
      **Pre-registered: H1 predicts no harness term survives, and the residual is inside the 3.5 %
      run-to-run spread.** If a harness term does survive, H1 is false, the corpus must be
      harness-stratified, and the whole programme costs more — which is why this is measured on 40
      tasks now rather than discovered in a finished head.
      **This item also prices S6 for free.** Run it with `DSV4_SUFFIXPROBE=1`. F80 retired
      prompt-lookup at an oracle ceiling of **+0.0 %**, but scoped its own refutation: it was
      measured on a period-8 degenerate decode, and reopening *"needs a long-repeated-context prompt
      on which `mlen` routinely reaches the block size — that is the agentic regime SuffixDecoding
      actually reports."* This capture is that prompt. Zero marginal cost, and acceptance is a
      counted integer so it is immune to the 1.5 % timing floor (trap 25).
      Detached. Evidence: `evidence/p2/h1_{harnessA,harnessB}.jsonl`, `h1_regression.txt`,
      `s6_suffixprobe.txt`.

### P2.3 — the corpus

- [ ] **P2.3** **Harvest `(h_40/41/42, p_target)` from live verify forwards.** The verifier has
      already computed both, so marginal compute is **zero** and the data is on-policy by
      construction. This replaces teacher-forced prefill capture, which cost 29.6 h per 5 K
      sequences and is bounded by a prefill rate P2.7 has not yet fixed.
      **The constraint inverts and the plan must respect it.** The eval battery alone produced
      **3,463,648 completion tokens**; at the measured 33 KB/token that is 114 GB against 279 GB
      free. The problem is now **selection and storage, not collection** — so this item ships a
      *sampling policy* keyed on the P2.1 buckets, not a capture schedule. fp8 hidden states halve
      the disk column and cost one numerics check against bf16 on a small sample.
      **Guard the feedback loop:** version every shard with the sha of the head that produced it,
      and hold the frozen 8-category suite **completely out** of the harvest. Harvesting from a
      server running the head you are about to retrain is self-training.
      **Gate: shard `tau` by bucket reproduces `PHASE2_PLAN.md` §1 within spread** — the proof that
      the labeller and the engine agree about what the engine just did.
      Detached. Evidence: `evidence/p2/harvest_manifest.json`, `harvest_tau_by_bucket.txt`.

### P2.4 — training

- [ ] **P2.4** **Fix `promote_head.py` before any P2 head is measured.** This is ladder **2.4** and
      it is a prerequisite, not a parallel task: its docstring says suite `tau` must beat the
      incumbent, its code compares `suite_tok_s`, and it reads the incumbent out of
      `HEAD_REGISTRY.md` — a number recorded on whatever engine revision was current when *that*
      head was measured. 2.2 measured the drift directly: across 8 days and five decode-kernel
      rewrites suite `tau` reproduced to **four decimal places for both heads** while suite tok/s
      moved −2.3 % and −5.0 % and base AR moved **−17.4 %**. `s2` and three ablations were refused
      on exactly that comparison, one at **25.10 tok/s against an incumbent 24.52**.
      Fix: require the incumbent to be **re-measured in the same session** as the candidate, and use
      `tau` as the cross-session anchor proving the re-measurement was faithful. Then re-adjudicate
      the four refused rows — do not overturn them by assertion.
      **Gate: re-adjudication reproduces `s3` as incumbent, or names a better head with a
      same-session re-measurement.** Grading P2 with the current rule grades it with a broken ruler.

- [ ] **P2.5** **The `beta` sweep — the one recipe change, aimed at F117.** Every session so far has
      traded the strong categories for the weak ones: `s1` (reasoning-only) lost `long_context`
      **−1.01**, `s2` (8-way balanced) lost it **−1.19**, and balancing did nothing for
      `agentic_format` either way. F119 then falsified the CE/TV explanation. The current loss has
      **no term that penalises drift on inputs the head already handles**. Add one:
      `L = SUM_s w(s)*L_dspark(s) + beta*KL(q_new || q_frozen)` over `R-high` spans, with `q_frozen`
      the incumbent head cached at capture time (no extra train-time forward), and
      `w(s) = (1 - tau_bucket/6) / mean(1 - tau/6)` — measured weights `long_context` **0.19**
      through `explanation` **1.72**, a 9x range, which is the *opposite* of what `make_corpus.py`
      does today.
      **Sweep `beta` in {0, 0.1, 0.5}, 1 epoch, everything else per `S5_RECIPE.md` §2.**
      `beta = 0` reproduces the current recipe exactly, so the sweep contains its own control.
      **Gate: a `beta > 0` arm holds `long_context` within −0.2 of run-0 WHILE lifting the
      constructive mean.** If none does, F117 is deeper than the loss, say so, and Track 1 caps at
      `tau` ~4.0. Detached, one arm at a time. **Archive every arm whether or not it is promoted** —
      a rejected head is still a measured point on the acceptance-vs-corpus curve.

- [ ] **P2.6** **HASS on draft steps 2–3, and pattern-gated block width.** Two changes, sequenced
      after P2.5 because both act on the same positions and would confound the `beta` sweep.
      (a) **HASS** trains later draft steps on imperfect *draft* features rather than clean target
      features (+8–20 % over EAGLE-2); our chain is depth 3, so steps 2 and 3 already run on their
      own predecessor's output. Training-loop change, no extra capture.
      (b) **Block width gated on the P2.1 signal.** The static sweep is done and pooled: F93/F94
      measured `tau` rising 5 -> 6 on all four realistic prompts and **falling 6 -> 8 on three of
      four**. But that pools a saturating category against a starving one — `long_context` at 5.54/6
      is pressing the ceiling while `explanation` at 1.75/6 wastes four verify slots per forward.
      Re-run the sweep **per bucket**: wide (8–10) on `R-high`, narrow (4–5) on `C-low`.
      **Caveat, pre-registered (F129):** the suite is verify-dominated — the same NVFP4 change moved
      M=1 by −1.3 % and M>=2 by **−14 %**. Widening moves more time into M>=2, and the retired
      `m16` B-repack needs **5.05 rows/expert** against block 6's **1.67** but pays around
      **block 40**. If width moves materially, **re-price `m16` — do not assume it stayed dead.**
      **Gate: pattern-gated width beats static block 6 on the suite, outside the 3.5 % spread.**

### P2.7 — the agentic daily driver

- [ ] **P2.7** **Prefill and TTFT — the item that decides whether this is usable at all.** Measured
      (B9): **52.6 / 50.3 / 47.7 / 43.8 tok/s at PS = 255 / 511 / 1023 / 2047** — it *declines* with
      prompt size — and today's ladder sweeps reproduce it: **12,282 prompt tokens in 227.6 s**,
      i.e. 54 tok/s and **3.8 minutes to first token**. A coding agent carrying 12 k of repo context
      cannot use that, no matter what decode does.
      **The anomaly is quantified and unexplained.** A batched forward amortising 12.26 GB of weights
      over 255 positions should be compute-bound and is running at only **3.4x the M=1 decode rate**.
      Every kernel in the path was tuned for M=1: `RB` is fitted to a 1.71-rows-per-expert histogram
      prefill does not have, and the grouped GEMV is chosen over the mma GEMM on a decision that
      inverts at prefill row counts — the retired `m16` repack is alive for prefill at **+16.7 %**
      (F85).
      **First measurement, and nobody has ever taken it: `DSV4_DPROF` on a PS=1023 prefill.** One
      checkpoint load. **Gate: it names the sub-op holding prefill at 3.4x decode. If the cost is
      diffuse across many sub-ops, say so and stop** — that is a real answer and it retires the
      lever instead of funding a rewrite on hope.
      **Pre-registered target, so the item can fail:** an agentic daily driver needs **TTFT under
      30 s at 12 k uncached context**, i.e. prefill >= 410 tok/s, a **7.6x** improvement. Anything
      short of that keeps the harness dependent on cache hits.

- [ ] **P2.8** **Prefix caching under an agentic turn structure, and what happens when context is
      compressed.** The battery measured per-task prefix-cache hit rates, but an agentic loop is the
      one access pattern it never exercised: a *growing* prefix re-sent every turn, with tool output
      appended in the middle, and — once long-horizon memory and context compression are in play —
      a prefix that is periodically **rewritten**, which invalidates every block after the edit
      point. That is the difference between a 3.8 min cold prefill amortised over a session and one
      paid repeatedly.
      **Three measurements, in one harness run:** (a) hit rate and TTFT across a 20-turn agentic
      session with a growing prefix; (b) the same with a mid-context compression event at turn 10;
      (c) the KV-eviction behaviour when a harness abandons a generation mid-stream — an unfreed
      cache degrades the next turn.
      **Gate: state the measured TTFT distribution across a real 20-turn session, p50 and p90.**
      Pass is p90 under 30 s; anything else is reported as the number it is, not as a plan.
      **Also decide `1b.2`'s default here, not on the ladder's terms.** It is banked and OFF:
      2048 -> 720 B/row, **2.844x**, bit-exact, +3.23 ms flat against −0.50 ms/1000 ctx, break-even
      **ctx 6,471**. A decode ladder optimising the flat term was right to leave it off; a
      long-context agentic profile lives above the break-even and gets 3x the context in the same
      pool. **Re-decide against the production context distribution and record which distribution
      decided it.**

### P2.9 — the deliverable

- [ ] **P2.9** **Name and package the best head, and make "best" mean something specific.** The
      programme's output is one uploadable artifact, and today nothing in the repo defines which head
      that is once `tau` stops being a single number.
      **The promotion rule, written before the candidates exist (rule 2):** the released head is the
      one with the highest **suite mean `tau`** that ALSO satisfies both floors —
      **(i)** no *reconstructive* category (`long_context`, `agentic_format`, `code_edit`) below its
      run-0 value minus 0.2, and **(ii)** no *constructive* category (`explanation`, `code_gen`,
      `reasoning`) below the incumbent's. A head that wins on the mean by hollowing out the agentic
      categories is **not** the best head for this engine's purpose, and the mean alone cannot say so.
      Both arms measured **in the same session** as the incumbent (P2.4).
      **Packaging** follows the convention already in `~/model-backups/releases/`: HF model-card
      frontmatter in `README.md`, `SHA256SUMS`, `provenance.json`, `head_card.json`,
      `mtp_trained.safetensors`, `train_metrics.json`, `eval.log`. Directory name states the claim:
      `dspark-mtp-draft-head-v2.0-<name>`, and the bundle README must carry **the per-category table,
      not just the mean** — it is the number a downstream user needs to predict their own workload.
      **Gate: `tools/verify_head_archive.py --full` exits 0 over the new bundle, and
      `tools/verify_staged_ckpt.py` proves the farm is the base checkpoint with exactly one head
      swapped.** Then, and only then, is it uploadable.

### P2.10 — close

- [ ] **P2.10** **Final `PERF.md`, `HEAD_REGISTRY.md` and `PHASE2_PLAN.md` reconciliation.** Re-run
      the battery on the released head, record both fit coefficients and both `tau` columns **in one
      session**, and grade `PHASE2_PLAN.md` §10's pre-registered bands against what actually
      happened — including the ones that were wrong. Then stop and hand back.
