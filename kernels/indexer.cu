// indexer.cu — DSA Indexer primitives, correctness-first (Gate K: ref/gen_units gen_hadamard/gen_index_score).
#include "indexer.h"
#include "topk_radix.h"
#include "dscratch.h"

// Hadamard: y[r,j] = D^-0.5 * Σ_i x[r,i] * (-1)^popcount(i&j). One thread per (row, j).
__global__ void hadamard_kernel(float* __restrict__ y, const float* __restrict__ x, int rows, int D, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= rows * D) return;
    int r = idx / D, j = idx % D;
    const float* xr = x + (size_t)r * D;
    float acc = 0.f;
    for (int i = 0; i < D; ++i) acc += (__popc(i & j) & 1) ? -xr[i] : xr[i];
    y[idx] = acc * scale;
}
void hadamard(float* y, const float* x, int rows, int D, cudaStream_t stream) {
    float scale = rsqrtf((float)D);
    hadamard_kernel<<<(rows * D + 255) / 256, 256, 0, stream>>>(y, x, rows, D, scale);
}

// index_score[s,t] = Σ_h relu(Σ_d q[s,h,d]*kv[t,d]) * weights[s,h]. One thread per (s,t).
__global__ void index_score_kernel(float* __restrict__ score, const float* __restrict__ q,
                                   const float* __restrict__ kv, const float* __restrict__ weights,
                                   int S, int T, int H, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= S * T) return;
    int s = idx / T, t = idx % T;
    const float* kvt = kv + (size_t)t * d;
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
        const float* qh = q + (((size_t)s * H + h) * d);
        float dot = 0.f; for (int e = 0; e < d; ++e) dot += qh[e] * kvt[e];
        acc += fmaxf(dot, 0.f) * weights[(size_t)s * H + h];      // relu * head weight
    }
    score[(size_t)s * T + t] = acc;
}
// WARP PER OUTPUT (LOOP_LOG Finding 71). The kernel above puts one THREAD on each (query, row) pair,
// and at decode there are only S*T ~ 95 of them — a single block, three warps, one SM, each thread
// walking H*d = 1024 MACs serially with a stride-1 read that no other lane shares. Measured in situ
// it is 6.05 ms of the indexer's 9.14, i.e. 4.2% of the whole K=5 verify, for 97k MACs.
//
// One warp per pair instead: 32x the threads, the d-loop is lane-strided so consecutive lanes read
// consecutive floats of both operands, and the per-head dot finishes in a shuffle tree.
//
// NOT bit-exact: the dot over `d` changes from serial to tree order. That is why it ships behind the
// LOSSLESS gate (Finding 68) rather than a cosine — a tolerance that is fine for one dot product says
// nothing about what a different top-k selection does 43 layers later. NO_IXWARP=1 restores the
// scalar kernel for A/B.
__global__ void index_score_warp_kernel(float* __restrict__ score, const float* __restrict__ q,
                                        const float* __restrict__ kv, const float* __restrict__ weights,
                                        int S, int T, int H, int d) {
    const int gid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (gid >= S * T) return;
    const int s = gid / T, t = gid % T, lane = threadIdx.x & 31;
    const float* kvt = kv + (size_t)t * d;
    float acc = 0.f;
    for (int h = 0; h < H; ++h) {
        const float* qh = q + (((size_t)s * H + h) * d);
        float dot = 0.f;
        for (int e = lane; e < d; e += 32) dot += qh[e] * kvt[e];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_down_sync(0xffffffff, dot, o);
        dot = __shfl_sync(0xffffffff, dot, 0);
        acc += fmaxf(dot, 0.f) * weights[(size_t)s * H + h];   // same relu, same head order
    }
    if (lane == 0) score[(size_t)s * T + t] = acc;
}
void index_score(float* score, const float* q, const float* kv, const float* weights,
                 int S, int T, int H, int d, cudaStream_t stream) {
    static const bool warpk = getenv("NO_IXWARP") == nullptr;
    if (warpk && (d % 32) == 0) {
        const int threads = 256, wpb = threads >> 5;
        index_score_warp_kernel<<<((size_t)S * T + wpb - 1) / wpb, threads, 0, stream>>>(score, q, kv, weights, S, T, H, d);
        return;
    }
    index_score_kernel<<<(S * T + 255) / 256, 256, 0, stream>>>(score, q, kv, weights, S, T, H, d);
}

// ================= DSA Indexer forward =================
#include "fp8_block_gemm.h"
#include "mla_attn.h"      // act_quant_fp8, rope_interleaved, act_quant_fp4sim
#include "compressor.h"    // gemm_fp32, compressor_forward
#include <cstdio>
#define CUI(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

// N1, ADOPTED (default ON; NO_ZERO_SCRATCH=1 disables). Finding 60: the engine is nondeterministic on identical input
// and DSV4_ARENA_ZERO exonerated the arena. This is the analogue the arena test could not reach —
// the PREFILL path allocates its scratch with raw cudaMalloc, once per layer per call, and never
// zeroes it, so every buffer starts life holding whatever the allocator last had at that address.
// Any kernel here that accumulates into its output rather than assigning it, or that writes fewer
// rows than it later reads, would inherit that and produce different numbers run to run.
//
// CLAIM RETRACTED (Finding 61). This was adopted on "zeroing moves distinct first-verify margin
// vectors from 8/8 to 5/8", read as evidence of an uninitialised read. tests/gate_scratch_init then
// tested the thing directly — same weights, same input, scratch filled with 0x00 vs 0xFF vs 0x3C,
// arena included — and compressed_attn_forward is bitwise IDENTICAL at every length 1..29. The
// prefill chain does not read uninitialised scratch. The 8/8 -> 5/8 was a 5-cycle in the engine
// (Finding 61) sampled 8 times, not a change caused by zeroing: with the sharper instrument the
// engine produces the SAME hash sequence zeroed and unzeroed.
//
// Kept ON anyway, and only because it is free: decode runs entirely out of the arena and never
// touches these allocations, so this costs the measured number nothing and removes a whole class of
// future doubt. It is NOT a fix for anything, and nothing should be attributed to it.
static inline cudaError_t zalloc(void** p, size_t n){
    static const bool z = getenv("NO_ZERO_SCRATCH") == nullptr;   // DEFAULT ON, see below
    cudaError_t e = cudaMalloc(p, n);
    if(e != cudaSuccess || !n) return e;
    const int idx = g_scratch_alloc_seq++;
    if(idx == 0) g_scratch_first_addr = (unsigned long long)*p;
    if(g_scratch_poison_idx == -1 || g_scratch_poison_idx == idx) cudaMemset(*p, g_scratch_poison_val, n);
    else if(z)                                                    cudaMemset(*p, 0, n);
    return e;
}

__global__ void k_scale(float* y, float sc, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]*=sc; }

// causal mask: score[si,t] = -inf where t >= (si+1)/ratio.
__global__ void k_causal_mask(float* score, int s, int T, int ratio){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=s*T) return; int si=i/T, t=i%T;
    if (t >= (si+1)/ratio) score[i] = -1e30f;
}
// per query: descending top-k of score[si,:T]; then idx = (t >= (si+1)/ratio) ? -1 : t+offset.
__global__ void k_topk_offset(int* out, const float* score, int s, int T, int topk, int ratio, int offset){
    int si=blockIdx.x; if(si>=s) return;
    extern __shared__ float sh[]; const int L=threadIdx.x;
    for(int t=L;t<T;t+=32) sh[t]=score[(size_t)si*T+t];
    __syncwarp();
    int thr=(si+1)/ratio;
    for(int k=0;k<topk;++k){
        float best=-1e30f; int bi=T;
        // NOTE the asymmetry, preserved verbatim: the scan covers the FULL row, and out-of-range
        // picks are rejected only at the OUTPUT. They still consume a slot. Bounding the scan by
        // `thr` here would change which rows land in later slots.
        for(int t=L;t<T;t+=32) if(sh[t]>best){best=sh[t];bi=t;}
        warp_argmax(best,bi);
        if(L==0){
            if(bi<T) sh[bi]=-1e30f;
            out[(size_t)si*topk+k] = (bi>=T || bi>=thr) ? -1 : bi+offset;
        }
        __syncwarp();
    }
}

// Radix-select twin (item 1.2). The asymmetry above is preserved: the SELECTION covers the full row
// and out-of-range picks are rejected only at the OUTPUT, still consuming a slot.
__global__ void k_topk_offset_rx(int* out, const float* score, int s, int T, int topk, int ratio, int offset){
    int si=blockIdx.x; if(si>=s) return;
    __shared__ TopkRadixSmem S;
    int* o = out + (size_t)si*topk;
    topk_radix_select<TOPK_RADIX_NT>(o, score+(size_t)si*T, T, topk, -1e30f, S);
    const int thr=(si+1)/ratio;
    for(int k=threadIdx.x;k<topk;k+=TOPK_RADIX_NT){ int b=o[k]; o[k] = (b<0 || b>=thr)? -1 : b+offset; }
}

void indexer_forward(float* index_score_out, int* topk_idxs, const float* x, const float* qr,
                     const unsigned char* wq_b, const float* wq_b_s, const float* weights_proj,
                     const float* c_wkv, const float* c_wgate, const float* c_ape, const float* c_norm,
                     const float* q_cos, const float* q_sin, const float* c_cos, const float* c_sin,
                     int s, int dim, int q_lora, int n_heads, int idx_hd, int rd, int ratio,
                     int index_topk, int offset, float eps, cudaStream_t stream) {
    int T = s / ratio, QD = n_heads * idx_hd;
    // Same defect as compressor_forward's (LOOP_LOG Finding 53), one layer up: with T = s/ratio == 0
    // there is no compressed row to score, both outputs are empty ([s,0] and [s,min(topk,0)]), and
    // index_score / k_causal_mask were being launched with gridDim (s*0+255)/256 = 0 — a launch that
    // fails and leaves cudaErrorInvalidValue behind. Reached at prompt lengths <= ratio.
    if (T <= 0) return;
    float softmax_scale = rsqrtf((float)idx_hd), wscale = softmax_scale * rsqrtf((float)n_heads);
    unsigned char* qrq; float *qrs, *q, *qtmp, *ckv, *weights;
    CUI(zalloc((void**)&qrq,(size_t)s*q_lora)); CUI(zalloc((void**)&qrs,(size_t)s*(q_lora/128)*4));
    CUI(zalloc((void**)&q,(size_t)s*QD*4)); CUI(zalloc((void**)&qtmp,(size_t)s*QD*4));
    CUI(zalloc((void**)&ckv,(size_t)T*idx_hd*4)); CUI(zalloc((void**)&weights,(size_t)s*n_heads*4));

    act_quant_fp8(qrq, qrs, qr, s, q_lora, 128, stream); dprobe(stream);
    fp8_block_gemm(q, qrq, qrs, wq_b, wq_b_s, s, QD, q_lora, stream); dprobe(stream);              // [s, n_heads*idx_hd]
    rope_interleaved(q + (idx_hd - rd), q_cos, q_sin, s*n_heads, rd, false, idx_hd, n_heads, stream); dprobe(stream);
    hadamard(qtmp, q, s*n_heads, idx_hd, stream); dprobe(stream);                                 // out!=in
    act_quant_fp4sim(qtmp, s*n_heads, idx_hd, 32, idx_hd, stream); dprobe(stream);                // fp4-sim
    compressor_forward(ckv, x, c_wkv, c_wgate, c_ape, c_norm, c_cos, c_sin, s, dim, idx_hd, ratio, true, rd, eps, true, stream); dprobe(stream);
    gemm_fp32(weights, x, weights_proj, s, n_heads, dim, stream); dprobe(stream);
    k_scale<<<(s*n_heads+255)/256,256,0,stream>>>(weights, wscale, s*n_heads); dprobe(stream);
    index_score(index_score_out, qtmp, ckv, weights, s, T, n_heads, idx_hd, stream); dprobe(stream);
    k_causal_mask<<<(s*T+255)/256,256,0,stream>>>(index_score_out, s, T, ratio); dprobe(stream);
    int topk = index_topk < T ? index_topk : T;
    if(topk_radix_on() && topk<=TOPK_RADIX_CAP) k_topk_offset_rx<<<s, TOPK_RADIX_NT, 0, stream>>>(topk_idxs, index_score_out, s, T, topk, ratio, offset);
    else                                        k_topk_offset<<<s, 32, topk_scan_smem(T), stream>>>(topk_idxs, index_score_out, s, T, topk, ratio, offset);
    dprobe(stream);
    CUI(cudaStreamSynchronize(stream));
    cudaFree(qrq);cudaFree(qrs);cudaFree(q);cudaFree(qtmp);cudaFree(ckv);cudaFree(weights);
}
