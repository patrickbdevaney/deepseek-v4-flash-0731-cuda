// dprof.h — named-phase GPU timing, for attributing a composite step to its sub-operations.
//
// Built for LOOP_LOG Finding 15: ~70 ms of the M>=2 verify step is unattributed after three
// refuted hypotheses (MoE expert-union, DSA, ogroup/HC). The layer-flavour split narrowed it to
// "something every layer does"; this narrows it to WHICH sub-op, by timing all 8 phases of
// block_verify_step across all 43 layers and comparing K=1 against K=5.
//
// Design notes:
//  - Off by default and compiled to nothing but a branch; enabled with DSV4_DPROF=1.
//  - Events are recorded into a preallocated pool and NOT synchronised until dprof_report(), so
//    the instrumented path stays asynchronous (a sync per phase would itself create the stalls we
//    are hunting, which is exactly the trap dspark_forward_head fell into).
//  - Pool is fixed-size; overflow stops recording rather than reallocating mid-measurement.
#pragma once
#include <cuda_runtime.h>

enum DProfId {
    DP_HC_PRE_ATTN = 0, DP_RMSNORM_ATTN, DP_ATTN, DP_HC_POST_ATTN,
    DP_HC_PRE_FFN, DP_RMSNORM_FFN, DP_MOE, DP_HC_POST_FFN, DP_KV_XIN,
    // second level: inside the attention phase (Finding 15 narrowed the M>=2 step to ATTENTION)
    DP_A_QPROJ, DP_A_KV, DP_A_SPARSE, DP_A_OGROUP, DP_A_MISC,
    DP_N
};

extern bool g_dprof_on;

void dprof_init(int max_marks = 8192);
void dprof_begin(int id, cudaStream_t s = 0);
void dprof_end(int id, cudaStream_t s = 0);
void dprof_reset();
// Sync, sum elapsed per id, print, and reset. `tag` labels the row block (e.g. "K=1").
void dprof_report(const char* tag);
