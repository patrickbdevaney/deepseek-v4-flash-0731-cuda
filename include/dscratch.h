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
extern size_t g_arena_off, g_arena_cap;

static inline void* dmalloc(size_t n){
    if(g_arena_on){ n=(n+255)&~((size_t)255); void* p=g_arena+g_arena_off; g_arena_off+=n;
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
extern cudaStream_t g_side;
extern cudaEvent_t  g_side_fork, g_side_join;
