// dscratch.h — decode scratch arena. At M=1 the per-call cudaMalloc/cudaFree/cudaStreamSynchronize in every
// sub-function dominate. When the arena is ON (decode), dmalloc bumps from a pre-allocated slab (reset per
// layer), dfree/dsync are no-ops, and the whole token runs as one stream with a single final sync. When OFF
// (gates / prefill forward), it falls back to real cudaMalloc/cudaFree/cudaStreamSynchronize — zero behavior
// change for existing callers. Single-stream, single-threaded decode only.
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
extern bool   g_arena_on;
extern char*  g_arena;
extern size_t g_arena_off, g_arena_cap, g_arena_hwm;

static inline void* dmalloc(size_t n){
    if(g_arena_on){ n=(n+255)&~((size_t)255); void* p=g_arena+g_arena_off; g_arena_off+=n;
        if(g_arena_off>g_arena_hwm) g_arena_hwm=g_arena_off;
        if(g_arena_off>g_arena_cap){ fprintf(stderr,"[dscratch] arena overflow %zu>%zu\n",g_arena_off,g_arena_cap); abort(); }
        return p; }
    void* p; cudaMalloc(&p,n); return p;
}
static inline void dfree(void* p){ if(!g_arena_on && p) cudaFree(p); }

// LOOP_LOG Finding 53. A kernel LAUNCH failure — gridDim 0, too much dynamic shared memory, a bad
// block shape — is reported only through the thread's last-error slot. `cudaStreamSynchronize` does
// NOT return it (measured on this box: a gridDim-0 launch leaves cudaErrorInvalidValue pending and
// the following cudaDeviceSynchronize returns cudaSuccess), and this engine never called
// cudaGetLastError anywhere. So a kernel that never ran was indistinguishable from one that did, and
// the stale code sat in the slot until some unrelated CU() picked it up — which is how a fault gets
// attributed to a line that only happens to hold the next sync.
//
// dsync() is the one call every sub-function already ends with, so it is the right drain point. The
// SYNC stays a no-op under the arena (that is the whole point of the arena); only the last-error
// read is added, which is a TLS read, not a device round-trip. It REPORTS by default and aborts only
// under DSV4_STRICT_LAUNCH=1 — a stale code must never be able to kill a 15-minute model run on its
// own, but it must never be silent either.
void dsync_at(cudaStream_t s, const char* file, int line);
#define dsync(s) dsync_at((s), __FILE__, __LINE__)

// FAULT LOCALISATION (DSV4_SYNCPROBE=1). Off by default and compiled to one predicted branch.
//
// Why this exists. Three cycles chased "an illegal memory access at kernels/indexer.cu:91" and then
// ":96" as if the line named a kernel. It does not: both are `CUI(cudaStreamSynchronize(stream))` at
// the end of indexer_forward, and that is the FIRST REAL SYNC in the whole layer's attention path —
// every launch from compressed_attn.cu:36 onward (the q chain, the kv chain, compressor_forward, the
// indexer's own GEMMs) is still in flight when it runs, because the sub-functions' own dsync() calls
// are no-ops while the arena is on. An asynchronous fault from ANY of ~20 launches surfaces there.
// So the line number was never evidence about the location, and every hypothesis built on "the fault
// is in the top-k" or "the fault is in the indexer" was reading a sync as a stack frame.
//
// dprobe() is a real, checked sync placed after individual launches. Under DSV4_SYNCPROBE it turns
// the prefill path into a serialised, fully-attributed walk and stops at the FIRST fault with the
// exact file:line of the launch that caused it. It destroys prefill throughput, which is fine —
// prefill timing is not a measured quantity here, and decode is untouched because nothing on the
// decode path calls it.
void dprobe_at(cudaStream_t s, const char* file, int line);
#define dprobe(s) dprobe_at((s), __FILE__, __LINE__)

// N1 LOCALISATION (Finding 60). Zeroing the prefill's raw cudaMalloc scratch moved 8/8 distinct
// first-verify margin vectors to 5/8, so SOMETHING in that path reads its output before writing it —
// but zeroing masks the bug instead of naming the kernel. These let a unit gate POISON scratch with
// two different bit patterns and diff the outputs: if a forward is sensitive to the poison, it reads
// uninitialised memory, and poisoning ONE allocation at a time says which.
//   g_scratch_poison_idx: -2 = normal (zero, the adopted default), -1 = poison every allocation,
//                         >=0 = poison only the allocation with that sequence number, zero the rest
//   g_scratch_alloc_seq : incremented by every scratch allocation; reset by the gate before a forward
extern int      g_scratch_poison_idx;
extern unsigned char g_scratch_poison_val;
extern int      g_scratch_alloc_seq;
extern unsigned long long g_scratch_first_addr;  // address of allocation #0 since the last reset

void arena_init(size_t cap);   // allocate the slab once, set g_arena_on
void arena_reset();            // g_arena_off = 0 (call at the top of each layer's work)

// INTRA-LAYER CONCURRENCY (LOOP_LOG Finding 55). A decode-sized kernel cannot saturate this memory
// system on its own — tools/overlap_probe.cu measures wkv [512,4096] at M=5 running 47.8 GB/s alone
// and 150.4 GB/s when four independent copies run on four streams, and wq_a at 86.4 -> 194.2. The
// engine issues one kernel at a time down a serialised layer chain, so every small kernel leaves
// 1.4-3.2x of the memory system idle. Where a layer contains two genuinely independent chains, they
// belong on two streams.
//
// One secondary stream, created ONCE inside arena_init so it can never be created during a graph
// capture (that is illegal and would kill the base-AR graph). Callers fork with g_side_fork and
// join with g_side_join, which is the capturable fork/join pattern. `g_side` is null when
// arena_init was never called — every gate that links these kernels without an arena keeps working
// on the single-stream path, which is also the NO_MOESPLIT=1 fallback.
// TWO independent side streams, because the forks NEST. `compressed_verify_step_indexer` forks the
// compressor emits onto g_side and then calls build_qKV *inside* that region, so a second fork in
// build_qKV must not reuse g_side's event pair — recording g_side_fork again while the first pair is
// still outstanding silently rewires the dependency graph and still produces plausible numbers.
// Each fork site owns exactly one pair. Add a third pair here, do not share one.
extern cudaStream_t g_side, g_side2;
extern cudaEvent_t  g_side_fork,  g_side_join;
extern cudaEvent_t  g_side2_fork, g_side2_join;
