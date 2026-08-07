// compressor.cu — KV Compressor gated-pooling core, correctness-first (Gate K: ref/gen_units gen_compressor).
#include "compressor.h"
// When true, `wkv`/`wgate` pointers are BF16 storage, not f32. Set once at engine init.
bool g_compressor_bf16 = false;
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdlib.h>
#include "dscratch.h"

// C[M,N] = A[M,K] @ B[N,K]^T. One warp per (m,n).
//
// LOOP_LOG Finding 34 (Opt #10). The original launched <<<dim3(N,M), 32>>> with a scalar ILP=1
// K-loop, and so lost twice over:
//   * one warp per block caps occupancy at 50% — the identical defect Finding 21 fixed in the MoE
//     and Finding 26 fixed in `gemm_bf16w`; this call site was simply never revisited.
//   * `acc += a[k]*b[k]` is a single dependent load->fma chain, i.e. one outstanding request per
//     lane. Little's Law says achieved bandwidth = MSHRs x bytes / latency, so ILP=1 leaves most
//     of the memory level parallelism on the floor. Opt #7 measured +3.4% end-to-end from exactly
//     this fix on the dense FP8 GEMV.
// float4 gives 4 independent accumulator chains and turns each lane's 4 B loads into 1. This is
// the "fix the inner loop BEFORE narrowing the format" step that Finding 32 concluded with: the
// BF16-native compressor lost because it added conversion ALU to a kernel that was ALU/issue
// bound, not bandwidth bound. Make it bandwidth bound first, then the byte saving pays.
//
// float4 needs 16-byte alignment. B is a mapped safetensors tensor, so it is checked at runtime
// and the kernel falls back to the scalar loop — Finding 25 is why that is not optional.
__global__ void gemm_fp32_kernel(float* __restrict__ C, const float* __restrict__ A,
                                 const float* __restrict__ B, int M, int N, int K, int vec4) {
    const int n = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    const int m = blockIdx.y;
    if (m >= M || n >= N) return;
    const int lane = threadIdx.x & 31;
    const float* a = A + (size_t)m * K; const float* b = B + (size_t)n * K;
    float acc = 0.f;
    if (vec4) {
        const float4* a4 = (const float4*)a; const float4* b4 = (const float4*)b;
        const int n4 = K >> 2;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int k = lane; k < n4; k += 32) {
            const float4 av = a4[k], bv = b4[k];
            a0 = fmaf(av.x, bv.x, a0); a1 = fmaf(av.y, bv.y, a1);
            a2 = fmaf(av.z, bv.z, a2); a3 = fmaf(av.w, bv.w, a3);
        }
        acc = (a0 + a1) + (a2 + a3);
    } else {
        for (int k = lane; k < K; k += 32) acc += a[k] * b[k];
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}
// 16-byte alignment of a *row* needs the base pointer aligned AND the row stride K*4 a multiple
// of 16, i.e. K a multiple of 4 — which the K%4==0 test already implies.
static inline int gemm_vec4_ok(const void* A, const void* B, int K) {
    return ((K & 3) == 0) && ((((uintptr_t)A) & 15) == 0) && ((((uintptr_t)B) & 15) == 0);
}
void gemm_fp32(float* C, const float* A, const float* B, int M, int N, int K, cudaStream_t stream) {
    const int wpb = 4, threads = 32*wpb;
    dim3 grid((N + wpb - 1)/wpb, M);
    gemm_fp32_kernel<<<grid, threads, 0, stream>>>(C, A, B, M, N, K, gemm_vec4_ok(A, B, K));
}

// ---------------------------------------------------------------------------------------------
// BF16-WEIGHT GEMV: C[M,N] = A[M,K] (f32) @ B[N,K]^T with B read NATIVELY as bf16.
//
// LOOP_LOG Finding 26. `lm_head` [129280, 4096] and the DSpark markov heads [129280, 256] ship as
// BF16 but were being materialised to f32 by `Loader::bf16` and then fed to gemm_fp32. That
// doubled both the resident footprint and, more importantly, the bytes read every single step:
// lm_head alone went 1059 -> 2118 MB, ~19 ms of a 108 ms decode. The markov head is worse in the
// draft, where it is re-read once per block position (5x).
//
// Two defects fixed together, since they are one code path:
//   1. read bf16 directly (halves the bytes, frees 2.1 GiB of headroom)
//   2. several warps per block — gemm_fp32 launched <<<dim3(N,M), 32>>>, i.e. 646,400 one-warp
//      blocks for lm_head, capping occupancy at 50% exactly as Finding 21 did for the MoE.
//
// bf16x2 loads need 4-byte alignment; B is a mapped safetensors tensor (>=8-byte offsets) and the
// row stride n*K*2 is a multiple of 4 for any even K, but it is checked at runtime anyway —
// Finding 25 is the reason that is not optional here.
__global__ void gemm_bf16w_kernel(float* __restrict__ C, const float* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ B, int M, int N, int K, int vec2) {
    const int n = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    const int m = blockIdx.y;
    if (m >= M || n >= N) return;
    const int lane = threadIdx.x & 31;
    const float* a = A + (size_t)m * K;
    const __nv_bfloat16* b = B + (size_t)n * K;
    float acc = 0.f;
    if (vec2 == 2) {
        // Finding 34: 8 bf16 per B load (one float4) + 4 independent accumulators. The vec2 path
        // below was still ILP=1 on a dependent load->fma chain; this is what makes the kernel
        // bandwidth bound rather than issue bound, which is the precondition for the BF16 byte
        // saving to pay at all (Finding 32).
        const float4* b4 = (const float4*)b; const float4* a4 = (const float4*)a;
        const int n8 = K >> 3;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int k = lane; k < n8; k += 32) {
            const float4 bv = b4[k];
            const float4 av0 = a4[2*k], av1 = a4[2*k+1];
            const __nv_bfloat162* bp = (const __nv_bfloat162*)&bv;
            const float2 p0 = __bfloat1622float2(bp[0]), p1 = __bfloat1622float2(bp[1]);
            const float2 p2 = __bfloat1622float2(bp[2]), p3 = __bfloat1622float2(bp[3]);
            a0 = fmaf(av0.x, p0.x, a0); a1 = fmaf(av0.y, p0.y, a1);
            a2 = fmaf(av0.z, p1.x, a2); a3 = fmaf(av0.w, p1.y, a3);
            a0 = fmaf(av1.x, p2.x, a0); a1 = fmaf(av1.y, p2.y, a1);
            a2 = fmaf(av1.z, p3.x, a2); a3 = fmaf(av1.w, p3.y, a3);
        }
        acc = (a0 + a1) + (a2 + a3);
    } else if (vec2) {
        const __nv_bfloat162* b2 = (const __nv_bfloat162*)b;
        const int n2 = K >> 1;
        for (int k = lane; k < n2; k += 32) {
            const float2 bv = __bfloat1622float2(b2[k]);
            acc += a[2*k] * bv.x + a[2*k+1] * bv.y;
        }
    } else {
        for (int k = lane; k < K; k += 32) acc += a[k] * __bfloat162float(b[k]);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) C[(size_t)m * N + n] = acc;
}
// M=K variant (LOOP_LOG Finding 42). The kernel above is one warp per OUTPUT ELEMENT with
// grid(N/wpb, M), so at M>1 it reads the whole B matrix M times. For every other caller B is a few
// MB and nobody noticed; for the lm_head B is [129280, 4096] bf16 = 1.06 GB, and the spec-decode
// verify calls it at M=BLK. Five passes over 1.06 GB is 5.3 GB of DRAM per verify — more traffic
// than any single MoE phase — to compute 5 rows.
//
// Same shape of fix as ogroup_gemv_mk (Finding 40): one warp per n, load the B row ONCE, dot it
// against all MM activation rows. B traffic becomes 1x. A is re-read per warp, but A is MM*K*4 =
// 80 KB at the verify shape — an L2 resident, not DRAM — so it trades 4.2 GB of DRAM for L2 hits.
// V8=true reads 8 bf16 per B load (float4, needs a 16-byte-aligned base); V8=false reads 2 (one
// unsigned, needs 4). Both variants exist because B here is a MAPPED tensor whose alignment nothing
// guarantees — the same fact that crashed the fp8 staging kernel (Finding 41). Falling back to the
// M=1 kernel when the base is only 4-byte aligned would silently give the M-times-B-traffic
// behaviour this whole change exists to remove, which is the worst of the three outcomes.
template<int MM, bool V8>
__global__ void gemm_bf16w_mk_kernel(float* __restrict__ C, const float* __restrict__ A,
                                     const __nv_bfloat16* __restrict__ B, int N, int K){
    const int n = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    if (n >= N) return;
    const int lane = threadIdx.x & 31;
    const __nv_bfloat16* b = B + (size_t)n*K;
    float acc[MM];
    #pragma unroll
    for (int m=0;m<MM;++m) acc[m]=0.f;
    if (V8) {
        const float4* b4 = (const float4*)b;
        const int n8 = K >> 3;
        for (int k = lane; k < n8; k += 32) {
            const float4 bv = b4[k];
            const __nv_bfloat162* bp = (const __nv_bfloat162*)&bv;
            const float2 p0=__bfloat1622float2(bp[0]), p1=__bfloat1622float2(bp[1]);
            const float2 p2=__bfloat1622float2(bp[2]), p3=__bfloat1622float2(bp[3]);
            #pragma unroll
            for (int m=0;m<MM;++m){
                const float4* a4 = (const float4*)(A + (size_t)m*K);
                const float4 av0 = a4[2*k], av1 = a4[2*k+1];
                // Same accumulation order as the M=1 vec2==2 path, so a single row is bit-identical.
                float s0=fmaf(av0.x,p0.x,0.f), s1=fmaf(av0.y,p0.y,0.f), s2=fmaf(av0.z,p1.x,0.f), s3=fmaf(av0.w,p1.y,0.f);
                s0=fmaf(av1.x,p2.x,s0); s1=fmaf(av1.y,p2.y,s1); s2=fmaf(av1.z,p3.x,s2); s3=fmaf(av1.w,p3.y,s3);
                acc[m] += (s0+s1)+(s2+s3);
            }
        }
    } else {
        const __nv_bfloat162* b2 = (const __nv_bfloat162*)b;
        const int n2 = K >> 1;
        for (int k = lane; k < n2; k += 32) {
            const float2 bv = __bfloat1622float2(b2[k]);
            #pragma unroll
            for (int m=0;m<MM;++m){
                const float* a = A + (size_t)m*K;
                acc[m] += a[2*k]*bv.x + a[2*k+1]*bv.y;
            }
        }
    }
    #pragma unroll
    for (int m=0;m<MM;++m){ float a=acc[m];
        #pragma unroll
        for (int o=16;o>0;o>>=1) a += __shfl_down_sync(0xffffffff,a,o);
        if (lane==0) C[(size_t)m*N + n] = a; }
}
void gemm_bf16w(float* C, const float* A, const void* Bbf16, int M, int N, int K, cudaStream_t stream) {
    const int wpb = 4, threads = 32*wpb;
    // 2 = float4 (8 bf16) path, 1 = bf16x2 path, 0 = scalar. B rows are 16-byte aligned iff the
    // base is and K*2 is a multiple of 16, i.e. K%8==0; A rows likewise need K*4 %16 == 0.
    const int vec2 = (((K & 7) == 0) && ((((uintptr_t)Bbf16) & 15) == 0) && ((((uintptr_t)A) & 15) == 0)) ? 2
                   : ((((K & 1) == 0) && ((((uintptr_t)Bbf16) & 3) == 0)) ? 1 : 0);
    // M=K path: only when the vectorised alignment holds (it is the only variant implemented) and
    // only up to M=8, past which MM accumulators plus 2 float4 A loads per m stop paying.
    if (vec2 >= 1 && M >= 2 && M <= 8 && getenv("NO_BF16MK")==nullptr) {
        const int blocks = (N + wpb - 1)/wpb; const bool v8 = (vec2 == 2);
        #define BFMK(MM) case MM: if(v8) gemm_bf16w_mk_kernel<MM,true ><<<blocks,threads,0,stream>>>(C,A,(const __nv_bfloat16*)Bbf16,N,K); \
                                  else   gemm_bf16w_mk_kernel<MM,false><<<blocks,threads,0,stream>>>(C,A,(const __nv_bfloat16*)Bbf16,N,K); return;
        switch(M){ BFMK(2) BFMK(3) BFMK(4) BFMK(5) BFMK(6) BFMK(7) BFMK(8) default: break; }
        #undef BFMK
    }
    dim3 grid((N + wpb - 1)/wpb, M);
    gemm_bf16w_kernel<<<grid, threads, 0, stream>>>(C, A, (const __nv_bfloat16*)Bbf16, M, N, K, vec2);
}
// device-conditional gemm (CUDA-graph emit): launch stays static but the block early-exits when this step is
// NOT a group-completion (so the expensive K-loop only runs on commit steps ~ every `ratio` tokens).
// grid-STRIDE (small fixed grid) so non-commit steps schedule few blocks (all early-return), not M*N of them.
__global__ void gemm_fp32_cond_kernel(float* __restrict__ C, const float* __restrict__ A, const float* __restrict__ B,
                                      int M, int N, int K, const int* __restrict__ d_pos, int ratio, int vec4){
    if(((*d_pos)+1)%ratio != 0) return;
    int lane=threadIdx.x&31, wpb=blockDim.x>>5, warp=blockIdx.x*wpb + (threadIdx.x>>5), nw=gridDim.x*wpb, tot=M*N;
    for(int idx=warp; idx<tot; idx+=nw){ int m=idx/N, n=idx%N;
        const float* a=A+(size_t)m*K; const float* b=B+(size_t)n*K; float acc=0.f;
        if(vec4){                                                  // Finding 34: same ILP fix as gemm_fp32
            const float4* a4=(const float4*)a; const float4* b4=(const float4*)b; const int n4=K>>2;
            float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
            for(int k=lane;k<n4;k+=32){ const float4 av=a4[k], bv=b4[k];
                a0=fmaf(av.x,bv.x,a0); a1=fmaf(av.y,bv.y,a1); a2=fmaf(av.z,bv.z,a2); a3=fmaf(av.w,bv.w,a3); }
            acc=(a0+a1)+(a2+a3);
        } else for(int k=lane;k<K;k+=32) acc+=a[k]*b[k];
        #pragma unroll
        for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o);
        if(lane==0) C[idx]=acc; }
}
void gemm_fp32_cond(float* C, const float* A, const float* B, int M, int N, int K, const int* d_pos, int ratio, cudaStream_t stream){
    gemm_fp32_cond_kernel<<<256,256,0,stream>>>(C,A,B,M,N,K,d_pos,ratio,gemm_vec4_ok(A,B,K));   // 256*8=2048 warps grid-stride M*N
}

// pooled[g,e] = Σ_p softmax_p(score[g*ratio+p,e]+ape[p,e]) * kv[g*ratio+p,e]. One thread per (g,e).
__global__ void compressor_pool_kernel(float* __restrict__ pooled, const float* __restrict__ kv,
                                       const float* __restrict__ score, const float* __restrict__ ape,
                                       int groups, int ratio, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= groups * d) return;
    int g = i / d, e = i % d;
    float mx = -1e30f;
    for (int p = 0; p < ratio; ++p) { float s = score[((size_t)(g * ratio + p)) * d + e] + ape[(size_t)p * d + e]; mx = fmaxf(mx, s); }
    float sum = 0.f, acc = 0.f;
    for (int p = 0; p < ratio; ++p) {
        float s = score[((size_t)(g * ratio + p)) * d + e] + ape[(size_t)p * d + e];
        float w = expf(s - mx); sum += w;
        acc += w * kv[((size_t)(g * ratio + p)) * d + e];
    }
    pooled[i] = acc / sum;
}
void compressor_pool(float* pooled, const float* kv, const float* score, const float* ape,
                     int groups, int ratio, int d, cudaStream_t stream) {
    compressor_pool_kernel<<<(groups * d + 255) / 256, 256, 0, stream>>>(pooled, kv, score, ape, groups, ratio, d);
}

// ---------------- overlap pooling (ratio==4) ----------------
// kv,score:[s,2d] (s=groups*ratio); ape:[ratio,2d]. Slot q in [0,2*ratio): q>=ratio -> current group token
// (q-ratio), dims [d:2d]; q<ratio -> previous group (g-1) token q, dims [0:d] (masked for g=0).
__global__ void compressor_pool_overlap_kernel(float* __restrict__ pooled, const float* __restrict__ kv,
                                               const float* __restrict__ score, const float* __restrict__ ape,
                                               int groups, int ratio, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= groups * d) return;
    int g = i / d, e = i % d; int twod = 2 * d, nslot = 2 * ratio;
    float mx = -1e30f;
    for (int q = 0; q < nslot; ++q) {
        float sc;
        if (q >= ratio) { int tok = g * ratio + (q - ratio); sc = score[(size_t)tok * twod + d + e] + ape[(size_t)(q - ratio) * twod + d + e]; }
        else if (g >= 1) { int tok = (g - 1) * ratio + q; sc = score[(size_t)tok * twod + e] + ape[(size_t)q * twod + e]; }
        else sc = -1e30f;
        mx = fmaxf(mx, sc);
    }
    float sum = 0.f, acc = 0.f;
    for (int q = 0; q < nslot; ++q) {
        float sc, kvv;
        if (q >= ratio) { int tok = g * ratio + (q - ratio); sc = score[(size_t)tok * twod + d + e] + ape[(size_t)(q - ratio) * twod + d + e]; kvv = kv[(size_t)tok * twod + d + e]; }
        else if (g >= 1) { int tok = (g - 1) * ratio + q; sc = score[(size_t)tok * twod + e] + ape[(size_t)q * twod + e]; kvv = kv[(size_t)tok * twod + e]; }
        else continue;
        float w = expf(sc - mx); sum += w; acc += w * kvv;
    }
    pooled[i] = acc / sum;
}
void compressor_pool_overlap(float* pooled, const float* kv, const float* score, const float* ape,
                             int groups, int ratio, int d, cudaStream_t stream) {
    compressor_pool_overlap_kernel<<<(groups * d + 255) / 256, 256, 0, stream>>>(pooled, kv, score, ape, groups, ratio, d);
}

// ---------------- full Compressor forward ----------------
#include "mla_attn.h"     // rmsnorm, rope_interleaved, act_quant_fp8sim/fp4sim
#include "indexer.h"      // hadamard
#include <cstdio>
#define CU2(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
void compressor_forward(float* out, const float* x, const float* wkv, const float* wgate,
                        const float* ape, const float* norm_w, const float* cosT, const float* sinT,
                        int s, int dim, int d, int ratio, bool overlap, int rope_dim, float eps,
                        bool rotate, cudaStream_t stream) {
    int coff = overlap ? 2 : 1, groups = s / ratio, od = coff * d;
    float *kv, *score;
    kv=(decltype(kv))dmalloc( (size_t)s * od * 4); score=(decltype(score))dmalloc( (size_t)s * od * 4);
    // BF16-NATIVE (LOOP_LOG Finding 32): wkv/wgate ship as BF16 and were being expanded to f32 by
    // Loader::bf16, doubling 526 MB/step to 1052. gemm_bf16w reads them natively. Same defect class
    // as the lm_head fix (Opt #4); same proven template.
    if (g_compressor_bf16) { gemm_bf16w(kv, x, (const void*)wkv, s, od, dim, stream);
                             gemm_bf16w(score, x, (const void*)wgate, s, od, dim, stream); }
    else                   { gemm_fp32 (kv, x, wkv, s, od, dim, stream);
                             gemm_fp32 (score, x, wgate, s, od, dim, stream); }
    if (overlap) compressor_pool_overlap(out, kv, score, ape, groups, ratio, d, stream);
    else         compressor_pool(out, kv, score, ape, groups, ratio, d, stream);
    rmsnorm(out, out, norm_w, groups, d, eps, true, stream);
    rope_interleaved(out + (d - rope_dim), cosT, sinT, groups, rope_dim, false, d, 1, stream);
    if (rotate) { hadamard(out, out, groups, d, stream); act_quant_fp4sim(out, groups, d, 32, d, stream); }  // indexer compressor
    else        { act_quant_fp8sim(out, groups, d - rope_dim, 64, d, stream); }                             // main compressor NoPE
    dsync(stream);
    dfree(kv); dfree(score);
}

// ---- incremental single-group emit (STRUCTURAL_PLAN Step 4 decode) ----
// Emit ONE compressed row = compressor_forward's out[g], from just this group's tokens (append-only KV: a
// compressed row finalizes once its `ratio` tokens exist and never changes). Non-overlap (ratio!=4): pools
// x[g*ratio .. g*ratio+ratio-1]. Overlap (ratio==4): pools the 2 local groups [(g-1)*ratio .. g*ratio+ratio-1]
// (prev half masked for g==0) and takes the current one. Bit-exact vs compressor_forward (same per-group math).
void compressor_emit_group(float* out_row, const float* x, int g, int ratio, const float* wkv,
                           const float* wgate, const float* ape, const float* norm_w,
                           const float* cc_cos, const float* cc_sin, int dim, int d, bool overlap,
                           int rope_dim, float eps, bool rotate, cudaStream_t stream){
    int coff = overlap ? 2 : 1, od = coff * d;
    int ntok, tok0, localg;
    if(overlap){ tok0 = (g>=1) ? (g-1)*ratio : 0; ntok = (g>=1) ? 2*ratio : ratio; localg = (g>=1) ? 1 : 0; }
    else       { tok0 = g*ratio; ntok = ratio; localg = 0; }
    const float* xg = x + (size_t)tok0*dim;
    float *kv,*score,*pooled;
    kv=(decltype(kv))dmalloc((size_t)ntok*od*4); score=(decltype(score))dmalloc((size_t)ntok*od*4);
    pooled=(decltype(pooled))dmalloc((size_t)(localg+1)*d*4);
    if (g_compressor_bf16) { gemm_bf16w(kv, xg, (const void*)wkv, ntok, od, dim, stream);
                             gemm_bf16w(score, xg, (const void*)wgate, ntok, od, dim, stream); }
    else                   { gemm_fp32 (kv, xg, wkv, ntok, od, dim, stream);
                             gemm_fp32 (score, xg, wgate, ntok, od, dim, stream); }
    if(overlap) compressor_pool_overlap(pooled, kv, score, ape, localg+1, ratio, d, stream);
    else        compressor_pool(pooled, kv, score, ape, 1, ratio, d, stream);
    float* prow = pooled + (size_t)localg*d;                 // the target group's pooled row
    rmsnorm(out_row, prow, norm_w, 1, d, eps, true, stream);
    rope_interleaved(out_row + (d - rope_dim), cc_cos + (size_t)g*(rope_dim/2), cc_sin + (size_t)g*(rope_dim/2),
                     1, rope_dim, false, d, 1, stream);
    if(rotate){ hadamard(out_row, out_row, 1, d, stream); act_quant_fp4sim(out_row, 1, d, 32, d, stream); }
    else      { act_quant_fp8sim(out_row, 1, d - rope_dim, 64, d, stream); }
    dsync(stream);
    dfree(kv); dfree(score); dfree(pooled);
}
