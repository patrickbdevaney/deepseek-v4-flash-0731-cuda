// mla_attn.cu — MLA attention primitives, correctness-first (Gate K oracle: ref/gen_units.py).
// Optimization (mma, smem KV staging, bf16) comes AFTER these pass their gate (CONSTITUTION Art. I).
#include <cuda_fp16.h>
#include "dscratch.h"
#include "mla_attn.h"

// ---------------- sparse_attn ----------------
// One warp per (b, m, head): stream over the top-k gathered KV with online (running max/sum) softmax,
// then fold in the learnable sink (denominator-only). MLA => single latent KV shared across heads.
// Lanes hold d/32 elements of the accumulator and of q; scores are warp-reduced dot products.
__global__ void sparse_attn_kernel(float* __restrict__ o, const float* __restrict__ q,
                                   const float* __restrict__ kv, const float* __restrict__ attn_sink,
                                   const int* __restrict__ topk_idxs,
                                   int b, int m, int h, int d, int n, int topk, float scale) {
    int gid = blockIdx.x;                       // (b*m*h) index
    int head = gid % h; int bm = gid / h; int mi = bm % m; int bi = bm / m;
    int lane = threadIdx.x & 31;
    int per = (d + 31) / 32;                    // elems per lane along d
    const float* qp = q + (((size_t)(bi * m + mi) * h + head) * d);
    const int*   ip = topk_idxs + ((size_t)(bi * m + mi) * topk);

    float qreg[32];                             // per<=32 for d<=1024
    #pragma unroll
    for (int r = 0; r < 32; ++r) { int j = lane + r * 32; qreg[r] = (r < per && j < d) ? qp[j] : 0.f; }

    float acc[32]; for (int r = 0; r < 32; ++r) acc[r] = 0.f;
    float run_max = -1e30f, run_sum = 0.f;

    for (int t = 0; t < topk; ++t) {
        int idx = ip[t];
        if (idx < 0) continue;                  // masked slot
        const float* kp = kv + (((size_t)bi * n + idx) * d);
        // score = scale * dot(q, kv[idx])
        float part = 0.f;
        #pragma unroll
        for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) part += qreg[r] * kp[j]; }
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) part += __shfl_down_sync(0xffffffff, part, o2);
        float score = __shfl_sync(0xffffffff, part, 0) * scale;   // broadcast to all lanes
        // online softmax update
        float new_max = fmaxf(run_max, score);
        float corr = expf(run_max - new_max);
        float p = expf(score - new_max);
        run_sum = run_sum * corr + p;
        #pragma unroll
        for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) acc[r] = acc[r] * corr + p * kp[j]; }
        run_max = new_max;
    }
    // sink: contributes exp(sink-max) to denominator only
    run_sum += expf(attn_sink[head] - run_max);
    float inv = (run_sum > 0.f) ? 1.f / run_sum : 0.f;
    float* op = o + (((size_t)(bi * m + mi) * h + head) * d);
    #pragma unroll
    for (int r = 0; r < 32; ++r) { int j = lane + r * 32; if (r < per && j < d) op[j] = acc[r] * inv; }
}

void sparse_attn(float* o, const float* q, const float* kv, const float* attn_sink,
                 const int* topk_idxs, int b, int m, int h, int d, int n, int topk,
                 float scale, cudaStream_t stream) {
    int blocks = b * m * h;
    sparse_attn_kernel<<<blocks, 32, 0, stream>>>(o, q, kv, attn_sink, topk_idxs, b, m, h, d, n, topk, scale);
}

// ---------------- rope_interleaved ----------------
// x[rows, rope_dim]; pairs (2j,2j+1) rotated by (cos_j, sin_j). inverse => sin -> -sin.
__global__ void rope_kernel(float* __restrict__ x, const float* __restrict__ cosT,
                            const float* __restrict__ sinT, int rows, int rope_dim, int inv,
                            int row_stride, int cos_stride_rows) {
    int row = blockIdx.x; if (row >= rows) return;
    int half = rope_dim / 2;
    float* xr = x + (size_t)row * row_stride;
    int crow = row / cos_stride_rows;
    const float* c = cosT + (size_t)crow * half;
    const float* s = sinT + (size_t)crow * half;
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        float a = xr[2 * j], bb = xr[2 * j + 1];
        float sj = inv ? -s[j] : s[j], cj = c[j];
        xr[2 * j]     = a * cj - bb * sj;
        xr[2 * j + 1] = a * sj + bb * cj;
    }
}

void rope_interleaved(float* x, const float* cosT, const float* sinT,
                      int rows, int rope_dim, bool inverse, int row_stride, int cos_stride_rows,
                      cudaStream_t stream) {
    if (row_stride < 0) row_stride = rope_dim;
    rope_kernel<<<rows, 64, 0, stream>>>(x, cosT, sinT, rows, rope_dim, inverse ? 1 : 0,
                                         row_stride, cos_stride_rows);
}
// DEVICE-POS rope (for CUDA-graph capture): cos row = (*d_pos) + row/cos_stride_rows, read from the FULL table.
// M=1 decode: q rows=N_HEADS,stride=N_HEADS -> crow=pos; kv rows=1,stride=1 -> crow=pos. No pos baked into args.
__global__ void rope_kernel_dp(float* __restrict__ x, const float* __restrict__ cosT, const float* __restrict__ sinT,
                               int rows, int rope_dim, int inv, int row_stride, int cos_stride_rows, const int* __restrict__ d_pos){
    int row=blockIdx.x; if(row>=rows) return; int half=rope_dim/2;
    float* xr = x + (size_t)row*row_stride; int crow=(*d_pos) + row/cos_stride_rows;
    const float* c=cosT + (size_t)crow*half; const float* s=sinT + (size_t)crow*half;
    for(int j=threadIdx.x;j<half;j+=blockDim.x){ float a=xr[2*j], bb=xr[2*j+1]; float sj=inv?-s[j]:s[j], cj=c[j];
        xr[2*j]=a*cj-bb*sj; xr[2*j+1]=a*sj+bb*cj; }
}
void rope_interleaved_dp(float* x, const float* cosT, const float* sinT, int rows, int rope_dim, bool inverse,
                         int row_stride, int cos_stride_rows, const int* d_pos, cudaStream_t stream){
    if(row_stride<0) row_stride=rope_dim;
    rope_kernel_dp<<<rows,64,0,stream>>>(x,cosT,sinT,rows,rope_dim,inverse?1:0,row_stride,cos_stride_rows,d_pos);
}

// ---------------- rmsnorm ----------------
__global__ void rmsnorm_kernel(float* __restrict__ y, const float* __restrict__ x,
                               const float* __restrict__ w, int rows, int dim, float eps, int has_w) {
    int row = blockIdx.x; if (row >= rows) return;
    const float* xr = x + (size_t)row * dim; float* yr = y + (size_t)row * dim;
    __shared__ float red[256];
    float ss = 0.f; for (int j = threadIdx.x; j < dim; j += blockDim.x) ss += xr[j] * xr[j];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) yr[j] = xr[j] * inv * (has_w ? w[j] : 1.f);
}

void rmsnorm(float* y, const float* x, const float* weight, int rows, int dim,
             float eps, bool has_weight, cudaStream_t stream) {
    rmsnorm_kernel<<<rows, 256, 0, stream>>>(y, x, weight, rows, dim, eps, has_weight ? 1 : 0);
}

// ---------------- act_quant_fp8sim ----------------
// Per (row, block-of-`block`): amax -> pow2 scale (ue8m0) -> clamp(x/scale) to e4m3 -> dequant*scale.
// One block per (row, group); threads cover the group. Matches kernel.py act_quant(inplace, round_scale).
#include <cuda_fp8.h>
__global__ void act_quant_fp8sim_kernel(float* __restrict__ x, int rows, int active_dim, int block, int row_stride) {
    int ng = active_dim / block; int gid = blockIdx.x; if (gid >= rows * ng) return;
    int row = gid / ng, g = gid % ng;
    float* xr = x + (size_t)row * row_stride + (size_t)g * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 1e-4f);
    float scale = exp2f(ceilf(log2f(amax * (1.f / 448.f))));      // pow2 (ue8m0)
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -448.f), 448.f);
        __nv_fp8_e4m3 e = __nv_fp8_e4m3(q);
        __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)e.__x, __NV_E4M3);
        xr[threadIdx.x] = __half2float(*reinterpret_cast<__half*>(&hr)) * scale;
    }
}
void act_quant_fp8sim(float* x, int rows, int active_dim, int block, int row_stride, cudaStream_t stream) {
    if (row_stride < 0) row_stride = active_dim;
    int threads = block < 32 ? 32 : block;
    act_quant_fp8sim_kernel<<<rows * (active_dim / block), threads, threads * sizeof(float), stream>>>(x, rows, active_dim, block, row_stride);
}

// Real activation quant -> fp8 bytes + f32 pow2 scale (the activation half of an fp8 linear).
__global__ void act_quant_fp8_kernel(uint8_t* __restrict__ a, float* __restrict__ as,
                                     const float* __restrict__ x, int rows, int K, int block) {
    int nb = K / block; int gid = blockIdx.x; if (gid >= rows * nb) return;
    int row = gid / nb, b = gid % nb;
    const float* xr = x + (size_t)row * K + (size_t)b * block;
    uint8_t* ar = a + (size_t)row * K + (size_t)b * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 1e-4f);
    float scale = exp2f(ceilf(log2f(amax * (1.f / 448.f))));
    if (threadIdx.x == 0) as[(size_t)row * nb + b] = scale;
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -448.f), 448.f);
        ar[threadIdx.x] = __nv_fp8_e4m3(q).__x;
    }
}
void act_quant_fp8(uint8_t* a_fp8, float* a_s, const float* x, int rows, int K, int block, cudaStream_t stream) {
    int threads = block < 32 ? 32 : block;
    act_quant_fp8_kernel<<<rows * (K / block), threads, threads * sizeof(float), stream>>>(a_fp8, a_s, x, rows, K, block);
}

// ---------------- ogroup_gemm ----------------
// out[bs,G,R] = sum_d o[bs,G,d] * wo_a[G,R,d]. One warp per (bs,G,R) reducing over d.
__global__ void ogroup_gemm_kernel(float* __restrict__ out, const float* __restrict__ o,
                                   const float* __restrict__ wo_a, int bs, int G, int R, int Kd) {
    int gid = blockIdx.x; int r = gid % R; int bg = gid / R; int gg = bg % G; int bb = bg / G;
    int lane = threadIdx.x & 31;
    const float* op = o + (((size_t)bb * G + gg) * Kd);
    const float* wp = wo_a + (((size_t)gg * R + r) * Kd);
    float acc = 0.f;
    for (int d = lane; d < Kd; d += 32) acc += op[d] * wp[d];
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffff, acc, s);
    if (lane == 0) out[((size_t)bb * G + gg) * R + r] = acc;
}
// --- TC version: fp16 mma per group (matches the bf16-einsum reference; ~13% warp-per-output -> tensor core) ---
// o,wo_a fp32 -> fp16 (transient, memory-neutral), m16n8k16 fp16 GEMM per group. Dispatched by g_tc_ogroup.
__device__ __forceinline__ void ogm_mma(float* c, const unsigned* a, const unsigned* b){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
      :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1])); }
__global__ void k_f2h(__half* o, const float* in, size_t n){ size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) o[i]=__float2half(in[i]); }
__global__ void tc_ogroup_kernel(float* out, const __half* o16, const __half* wo16, int bs, int G, int R, int Kd){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int gg=blockIdx.y, n0=blockIdx.x*8; if(n0>=R) return;
    const __half* xg0 = o16 + ((size_t)gid*G+gg)*Kd;          // A row bb=gid (stride G*Kd), group gg
    const __half* xg8 = o16 + ((size_t)(gid+8)*G+gg)*Kd;
    const __half* Bg  = wo16 + (size_t)gg*R*Kd;               // B = wo16[gg] [R,Kd]
    bool m0=gid<bs, m8=(gid+8)<bs; float c[4]={0,0,0,0};
    for(int k0=0;k0<Kd;k0+=16){
        unsigned a[4],b[2];
        a[0]=m0?*(const unsigned*)(xg0+k0+2*t4):0u; a[1]=m8?*(const unsigned*)(xg8+k0+2*t4):0u;
        a[2]=m0?*(const unsigned*)(xg0+k0+2*t4+8):0u; a[3]=m8?*(const unsigned*)(xg8+k0+2*t4+8):0u;
        const __half* wr=Bg+(size_t)(n0+gid)*Kd;
        b[0]=(n0+gid<R)?*(const unsigned*)(wr+k0+2*t4):0u; b[1]=(n0+gid<R)?*(const unsigned*)(wr+k0+2*t4+8):0u;
        ogm_mma(c,a,b);
    }
    int cn=2*t4;
    if(gid<bs   && n0+cn  <R) out[((size_t)gid*G+gg)*R + n0+cn  ]=c[0];
    if(gid<bs   && n0+cn+1<R) out[((size_t)gid*G+gg)*R + n0+cn+1]=c[1];
    if(gid+8<bs && n0+cn  <R) out[((size_t)(gid+8)*G+gg)*R + n0+cn ]=c[2];
    if(gid+8<bs && n0+cn+1<R) out[((size_t)(gid+8)*G+gg)*R + n0+cn+1]=c[3];
}
#include <cuda_fp8.h>
__device__ __forceinline__ float ogm_e4m3(uint8_t b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
// FUSED fp8 wo_a TC ogroup: decode fp8 wo_a * e8m0 block-scale -> f16 IN the mma inner loop (no wo16 buffer,
// no per-token full-tensor conversion). Reads fp8 (half the bytes of f16). Bit-identical to convert-then-mma.
__global__ void tc_ogroup_fp8_kernel(float* out, const __half* o16, const uint8_t* wo, const uint8_t* wsc,
                                     int bs, int G, int R, int Kd){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int gg=blockIdx.y, n0=blockIdx.x*8; if(n0>=R) return;
    const __half* xg0 = o16 + ((size_t)gid*G+gg)*Kd;
    const __half* xg8 = o16 + ((size_t)(gid+8)*G+gg)*Kd;
    const uint8_t* Bg = wo + (size_t)gg*R*Kd; int scw=Kd/128;
    bool m0=gid<bs, m8=(gid+8)<bs; float c[4]={0,0,0,0};
    int n=n0+gid; const uint8_t* wr = Bg + (size_t)n*Kd; size_t grow=(size_t)gg*R+n;
    for(int k0=0;k0<Kd;k0+=16){
        unsigned a[4],b[2];
        a[0]=m0?*(const unsigned*)(xg0+k0+2*t4):0u; a[1]=m8?*(const unsigned*)(xg8+k0+2*t4):0u;
        a[2]=m0?*(const unsigned*)(xg0+k0+2*t4+8):0u; a[3]=m8?*(const unsigned*)(xg8+k0+2*t4+8):0u;
        if(n<R){ int kk=k0+2*t4;
            float s0=exp2f((float)wsc[(grow/128)*scw + kk/128]-127.f);
            float s1=exp2f((float)wsc[(grow/128)*scw + (kk+8)/128]-127.f);
            __half2 p0=__halves2half2(__float2half(ogm_e4m3(wr[kk])*s0),   __float2half(ogm_e4m3(wr[kk+1])*s0));
            __half2 p1=__halves2half2(__float2half(ogm_e4m3(wr[kk+8])*s1), __float2half(ogm_e4m3(wr[kk+9])*s1));
            b[0]=*(unsigned*)&p0; b[1]=*(unsigned*)&p1;
        } else { b[0]=0u; b[1]=0u; }
        ogm_mma(c,a,b);
    }
    int cn=2*t4;
    if(gid<bs   && n0+cn  <R) out[((size_t)gid*G+gg)*R + n0+cn  ]=c[0];
    if(gid<bs   && n0+cn+1<R) out[((size_t)gid*G+gg)*R + n0+cn+1]=c[1];
    if(gid+8<bs && n0+cn  <R) out[((size_t)(gid+8)*G+gg)*R + n0+cn ]=c[2];
    if(gid+8<bs && n0+cn+1<R) out[((size_t)(gid+8)*G+gg)*R + n0+cn+1]=c[3];
}
bool g_tc_ogroup = false;   // forward.cu sets true; gates use the warp-per-output oracle
// M=1 ogroup GEMV: one warp per output (g,r), out[g*R+r] = sum_d o[g][d]*fp8(wo[gr][d])*e8m0scale. Coalesced
// strided reads (lanes read consecutive bytes), NO mma waste (bs=1). Kills the tc_ogroup's scalar-byte m16 mma
// (~32 ms/tok -> bandwidth). Gated cosine vs ogroup_gemm oracle (tests/gate_ogroup_gemv.cu).
//
// LOOP_LOG Finding 35 (Opt #11). The inner loop above read ONE BYTE per lane:
//     for(dd=kb*128+lane; dd<(kb+1)*128; dd+=32) acc += og[dd]*ogm_e4m3(wr[dd])*ws;
// 32 lanes x 1 byte = a 32-byte request, so streaming `wo_a` costs 4x the requests a 128-byte
// access would. This is the SAME defect Finding 15 traced under the M>=2 verify penalty — there
// the m16 tile issued 8x32B where the GEMV issued 1x128B — and it was sitting in the M=1 path all
// along. `wo_a` is [G*R, Kd] fp8 = ~33 MB/layer, ~1.44 GB/token: by the dprof sub-marks this
// kernel is ~22 ms of a ~98 ms step, i.e. ~65 GB/s against 240 achievable.
//
// One uint32_t per lane makes the warp cover 128 contiguous bytes in a single request, and gives
// 4 independent accumulator chains for free (Little's Law: ILP=1 leaves MLP on the floor).
// `ws` is a power of two, so folding it into the weight instead of the product is exact.
__global__ void ogroup_gemv_fp8_kernel(float* __restrict__ out, const float* __restrict__ o,
                                       const uint8_t* __restrict__ wo, const uint8_t* __restrict__ wsc, int G, int R, int Kd, int vec4){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int total=G*R; if(warp>=total) return;
    int gr=warp, g=gr/R; int lane=threadIdx.x&31, scw=Kd/128;
    const uint8_t* wr=wo+(size_t)gr*Kd; const uint8_t* sr=wsc+(size_t)(gr/128)*scw; const float* og=o+(size_t)g*Kd;
    float acc=0.f;
    if(vec4){
        float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
        for(int kb=0; kb<Kd/128; ++kb){
            const float ws=exp2f((float)sr[kb]-127.f);
            const int base=kb*128+lane*4;
            const uint32_t w4=*(const uint32_t*)(wr+base);
            const float4  o4=*(const float4*)(og+base);
            a0=fmaf(o4.x, ogm_e4m3((uint8_t)(w4    ))*ws, a0);
            a1=fmaf(o4.y, ogm_e4m3((uint8_t)(w4>> 8))*ws, a1);
            a2=fmaf(o4.z, ogm_e4m3((uint8_t)(w4>>16))*ws, a2);
            a3=fmaf(o4.w, ogm_e4m3((uint8_t)(w4>>24))*ws, a3);
        }
        acc=(a0+a1)+(a2+a3);
    } else {
        for(int kb=0; kb<Kd/128; ++kb){ float ws=exp2f((float)sr[kb]-127.f);
            for(int dd=kb*128+lane; dd<(kb+1)*128; dd+=32) acc += og[dd]*ogm_e4m3(wr[dd])*ws; }
    }
    #pragma unroll
    for(int o2=16;o2>0;o2>>=1) acc+=__shfl_down_sync(0xffffffff,acc,o2);
    if(lane==0) out[gr]=acc;
}
// M=K ogroup GEMV (LOOP_LOG Finding 40). At the M=5 spec-decode verify, `cattn:ogroup` measured
// 64.34 ms against 16.63 at M=1 — a 3.87x scaling on weights that are IDENTICAL for all M tokens
// and should therefore cost ~1.0x. The m16 mma tile re-reads `wo_a` per 16-row tile and issues
// 8x32B requests per fragment (Finding 15's mechanism), so at M=5 it both wastes 11 of 16 rows and
// quadruples the request count. Reading each weight row ONCE and dotting it against all M
// activation rows removes both at a stroke — the same shape as `fp8_gemv_mkT_kernel` (Finding 28),
// templated on M so `acc[M]` is a real register array rather than 16 spilled ones.
// MULTI-ROW (Finding 43). Finding 40's kernel reads the weight once per M rows, which fixed the
// weight traffic — and left `cattn:ogroup` at 1.73x from K=1 to K=5 on weights that do not change.
// The residual is the OTHER operand. Per warp per 128-byte K-block this reads 32x4 = 128 weight
// bytes and M x 32 x 16 = M*512 bytes of `o`, so the f32 activation traffic is 4M times the fp8
// weight traffic — 20x at M=5. `o` is 655 KB, i.e. L2-resident, so none of it is DRAM; it is L2
// bandwidth, and at 27 GB per verify across 41 layers that is the binding constraint.
//
// `o` depends only on the group g, and gr = g*R + r, so NR consecutive output rows share it
// exactly. Give each warp NR rows: the `o` float4 is loaded once and used against NR weights, and
// the ratio falls from 4M to 4M/NR. Accumulators are NR*M registers, which is why NR is templated
// alongside M rather than fixed — 4x5 = 20 is comfortable, 8x16 would spill.
template<int M, int NR>
__global__ void ogroup_gemv_mk_kernel(float* __restrict__ out, const float* __restrict__ o,
                                      const uint8_t* __restrict__ wo, const uint8_t* __restrict__ wsc,
                                      int G, int R, int Kd){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)>>5; int total=G*R;
    int gr0=warp*NR; if(gr0>=total) return;
    int g=gr0/R; int lane=threadIdx.x&31, scw=Kd/128;
    // NR rows share a group only if the run does not straddle a g boundary; R%NR==0 guarantees it.
    const float* og=o+(size_t)g*Kd;
    float acc[NR][M];
    #pragma unroll
    for(int r=0;r<NR;++r){
        #pragma unroll
        for(int m=0;m<M;++m) acc[r][m]=0.f; }
    for(int kb=0; kb<Kd/128; ++kb){
        const int base=kb*128+lane*4;
        float d[NR][4];
        #pragma unroll
        for(int r=0;r<NR;++r){
            const int rr = (gr0+r<total) ? (gr0+r) : gr0;      // tail warps: harmless duplicate, store is masked
            const uint8_t* wr=wo+(size_t)rr*Kd;
            const uint8_t* sr=wsc+(size_t)(rr/128)*scw;
            const float ws=exp2f((float)sr[kb]-127.f);      // power of two: folding it in is exact
            const uint32_t w4=*(const uint32_t*)(wr+base);
            d[r][0]=ogm_e4m3((uint8_t)(w4    ))*ws; d[r][1]=ogm_e4m3((uint8_t)(w4>> 8))*ws;
            d[r][2]=ogm_e4m3((uint8_t)(w4>>16))*ws; d[r][3]=ogm_e4m3((uint8_t)(w4>>24))*ws;
        }
        #pragma unroll
        for(int m=0;m<M;++m){
            const float4 o4=*(const float4*)(og+(size_t)m*G*Kd+base);   // loaded ONCE for all NR
            #pragma unroll
            for(int r=0;r<NR;++r){
                acc[r][m]=fmaf(o4.x,d[r][0],acc[r][m]); acc[r][m]=fmaf(o4.y,d[r][1],acc[r][m]);
                acc[r][m]=fmaf(o4.z,d[r][2],acc[r][m]); acc[r][m]=fmaf(o4.w,d[r][3],acc[r][m]);
            }
        }
    }
    #pragma unroll
    for(int r=0;r<NR;++r){
        #pragma unroll
        for(int m=0;m<M;++m){ float a=acc[r][m];
            #pragma unroll
            for(int o2=16;o2>0;o2>>=1) a+=__shfl_down_sync(0xffffffff,a,o2);
            if(lane==0 && gr0+r<total) out[(size_t)m*total+gr0+r]=a; }
    }
}
// NR falls as M rises to keep NR*M accumulators in registers. NO_OGNR=1 pins NR=1, which is
// exactly Finding 40's kernel, for A/B.
#define OG_MK_CASE(M) case M: \
    if(ognr==4)      ogroup_gemv_mk_kernel<M,4><<<(nb+3)/4,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd); \
    else if(ognr==2) ogroup_gemv_mk_kernel<M,2><<<(nb+1)/2,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd); \
    else             ogroup_gemv_mk_kernel<M,1><<<nb,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd); break;
// fp8-native TC ogroup: o(f32)->f16 + fused fp8 wo_a decode in the mma. No per-token wo16 conversion.
void ogroup_gemm_fp8(float* out, const float* o, const uint8_t* wo_fp8, const uint8_t* wo_sc,
                     int bs, int G, int R, int Kd, cudaStream_t stream){
    if(bs==1 && getenv("NO_OGGEMV")==nullptr){          // M=1 decode: bandwidth-bound GEMV (no mma waste). bs>1
        // uint32 weight / float4 activation loads need Kd%128==0 (already required by the e8m0
        // block-scale stride) plus real pointer alignment — `wo_fp8` is a mapped safetensors
        // tensor, and Finding 25 is why that is checked rather than assumed.
        const int vec4 = ((Kd % 128)==0) && ((((uintptr_t)wo_fp8)&3)==0) && ((((uintptr_t)o)&15)==0)
                      && (getenv("NO_OGVEC4")==nullptr);
        int threads=256; ogroup_gemv_fp8_kernel<<<((size_t)G*R*32+threads-1)/threads,threads,0,stream>>>(out,o,wo_fp8,wo_sc,G,R,Kd,vec4);
        dsync(stream); return; }
    // M=K GEMV for the spec-decode verify (Finding 40). The old note here said "M=K GEMV A/B'd
    // SLOWER: acc[bs] array kills occupancy" — that was the UNTEMPLATED version, whose acc[] was
    // sized to the maximum M regardless of the real M; Finding 28 established the same fix for the
    // dense fp8 GEMV. Templated on M, acc[] is M registers, not 16.
    // The M=K kernel is vec4-only (uint32 weight + float4 activation). Both alignments are
    // structural here — wo_a rows are Kd-strided with Kd%128==0, and `o` is arena-allocated — but
    // check anyway: a mapped weight that is merely 1-byte aligned would fault, and Finding 41 is
    // the reminder that "structurally aligned" is a claim about code that has since been rewritten.
    // Unaligned falls through to the f16 tensor-core path below, not to a wrong answer.
    const int ogvec4 = ((Kd % 128)==0) && ((((uintptr_t)wo_fp8)&3)==0) && ((((uintptr_t)o)&15)==0)
                    && (getenv("NO_OGVEC4")==nullptr);
    if(bs>1 && bs<=16 && ogvec4 && getenv("NO_OGMK")==nullptr){
        // NR consecutive output rows share `o` only if the run stays inside one group: R%NR==0.
        int ognr = (getenv("NO_OGNR")!=nullptr) ? 1 : (bs<=8 ? 4 : 2);
        if((R % ognr)!=0) ognr = ((R%2)==0)?2:1;
        const int threads=256; const size_t nb=((size_t)G*R*32+threads-1)/threads;
        switch(bs){ OG_MK_CASE(2)  OG_MK_CASE(3)  OG_MK_CASE(4)  OG_MK_CASE(5)
                    OG_MK_CASE(6)  OG_MK_CASE(7)  OG_MK_CASE(8)  OG_MK_CASE(9)
                    OG_MK_CASE(10) OG_MK_CASE(11) OG_MK_CASE(12) OG_MK_CASE(13)
                    OG_MK_CASE(14) OG_MK_CASE(15) OG_MK_CASE(16) default: break; }
        dsync(stream); return;
    }
    __half* o16; o16=(__half*)dmalloc((size_t)bs*G*Kd*2);
    k_f2h<<<((size_t)bs*G*Kd+255)/256,256,0,stream>>>(o16,o,(size_t)bs*G*Kd);
    dim3 grid((R+7)/8, G); tc_ogroup_fp8_kernel<<<grid,32,0,stream>>>(out,o16,wo_fp8,wo_sc,bs,G,R,Kd);
    dsync(stream); dfree(o16);
}
void ogroup_gemm(float* out, const float* o, const float* wo_a,
                 int bs, int G, int R, int Kd, cudaStream_t stream) {
    if (g_tc_ogroup && Kd%16==0) {
        __half *o16,*wo16; o16=(__half*)dmalloc((size_t)bs*G*Kd*2); wo16=(__half*)dmalloc((size_t)G*R*Kd*2);
        k_f2h<<<((size_t)bs*G*Kd+255)/256,256,0,stream>>>(o16,o,(size_t)bs*G*Kd);
        k_f2h<<<((size_t)G*R*Kd+255)/256,256,0,stream>>>(wo16,wo_a,(size_t)G*R*Kd);
        dim3 grid((R+7)/8, G); tc_ogroup_kernel<<<grid,32,0,stream>>>(out,o16,wo16,bs,G,R,Kd);
        dsync(stream); dfree(o16); dfree(wo16); return;
    }
    ogroup_gemm_kernel<<<bs * G * R, 32, 0, stream>>>(out, o, wo_a, bs, G, R, Kd);
}

// ---------------- act_quant_fp4sim ----------------
// FP4 e2m1 QAT-sim (quant->dequant), pow2 scale, fp4_max=6, block=32. Matches kernel.py fp4_act_quant inplace.
__device__ __forceinline__ float round_e2m1(float v) {   // nearest signed E2M1 grid value {0,.5,1,1.5,2,3,4,6}
    float a = fabsf(v), m;
    if (a < 0.25f) m = 0.f; else if (a < 0.75f) m = 0.5f; else if (a < 1.25f) m = 1.f;
    else if (a < 1.75f) m = 1.5f; else if (a < 2.5f) m = 2.f; else if (a < 3.5f) m = 3.f;
    else if (a < 5.f) m = 4.f; else m = 6.f;
    return (v < 0.f) ? -m : m;
}
__global__ void act_quant_fp4sim_kernel(float* __restrict__ x, int rows, int active_dim, int block, int row_stride) {
    int ng = active_dim / block; int gid = blockIdx.x; if (gid >= rows * ng) return;
    int row = gid / ng, g = gid % ng;
    float* xr = x + (size_t)row * row_stride + (size_t)g * block;
    extern __shared__ float red[];
    float v = (threadIdx.x < block) ? fabsf(xr[threadIdx.x]) : 0.f;
    red[threadIdx.x] = v; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = fmaxf(red[0], 6.f * 7.5231631e-37f);              // 6*2^-126
    float scale = exp2f(ceilf(log2f(amax * (1.f / 6.f))));
    if (threadIdx.x < block) {
        float q = fminf(fmaxf(xr[threadIdx.x] / scale, -6.f), 6.f);
        xr[threadIdx.x] = round_e2m1(q) * scale;
    }
}
void act_quant_fp4sim(float* x, int rows, int active_dim, int block, int row_stride, cudaStream_t stream) {
    if (row_stride < 0) row_stride = active_dim;
    int threads = block < 32 ? 32 : block;
    act_quant_fp4sim_kernel<<<rows * (active_dim / block), threads, threads * sizeof(float), stream>>>(x, rows, active_dim, block, row_stride);
}
