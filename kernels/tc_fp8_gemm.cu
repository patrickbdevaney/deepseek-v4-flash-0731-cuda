// tc_fp8_gemm.cu — native FP8 tensor-core GEMM (W8A8) via mma.sync.m16n8k32.e4m3 (2x fp16 on Thor sm_110a).
// C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T, per-128 act scale a_s[M,K/128], per-128x128 wt scale b_s[N/128,K/128].
// Drop-in for fp8_block_gemm; fp8 acts/weights feed the tensor core directly (no fp16 upconvert).
// *** UNGATED until it passes cosine vs fp8_block_gemm (tests: gate_units [fp8_block_gemm+TC]). ***
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// mma D[16,8] += A[16,32] @ B[8,32]^T ; A/B = e4m3 (4 fp8 per reg), C/D = f32 (4 per lane).
__device__ __forceinline__ void mma_m16n8k32_e4m3(float* c, const unsigned* a, const unsigned* b){
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
}

// grid = (ceil(N/8), ceil(M/16)); one warp -> one [16-row, 8-col] tile. Accumulate raw per 128-K-block, then
// scale (a_s per-row, b_s per-128-N-block) and add — matches fp8_block_gemm's per-block scale application.
__global__ void tc_fp8_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                              const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    int lane=threadIdx.x&31, gid=lane>>2, tid4=lane&3;
    int nb=blockIdx.x, mt=blockIdx.y; int n0=nb*8, m0=mt*16; if(n0>=N) return;
    int r0=m0+gid, r1=m0+gid+8, nn=n0+gid; int KB=K/128;
    float acc[4]={0.f,0.f,0.f,0.f};
    for(int kblk=0; kblk<KB; ++kblk){
        float cb[4]={0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int kt=0; kt<4; ++kt){ int k0=kblk*128+kt*32;
            unsigned a[4], b[2];
            a[0]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4)      : 0u;
            a[1]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4)      : 0u;
            a[2]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4+16)   : 0u;
            a[3]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4+16)   : 0u;
            b[0]=(nn<N)? *(const unsigned*)(B+(size_t)nn*K+k0+tid4*4)      : 0u;
            b[1]=(nn<N)? *(const unsigned*)(B+(size_t)nn*K+k0+tid4*4+16)   : 0u;
            mma_m16n8k32_e4m3(cb, a, b);
        }
        float bsc = bs[(size_t)(n0/128)*KB + kblk];
        float as0 = (r0<M)? as[(size_t)r0*KB + kblk] : 0.f;
        float as1 = (r1<M)? as[(size_t)r1*KB + kblk] : 0.f;
        acc[0]+=cb[0]*as0*bsc; acc[1]+=cb[1]*as0*bsc; acc[2]+=cb[2]*as1*bsc; acc[3]+=cb[3]*as1*bsc;
    }
    int cn=tid4*2;
    if(r0<M && n0+cn  <N) C[(size_t)r0*N + n0+cn  ]=acc[0];
    if(r0<M && n0+cn+1<N) C[(size_t)r0*N + n0+cn+1]=acc[1];
    if(r1<M && n0+cn  <N) C[(size_t)r1*N + n0+cn  ]=acc[2];
    if(r1<M && n0+cn+1<N) C[(size_t)r1*N + n0+cn+1]=acc[3];
}

// ---------------------------------------------------------------------------------------------
// SMEM-STAGED B (LOOP_LOG Finding 41). Measured on wq_b [32768,1024], 12 rotating copies so every
// call reads memory it has not just touched:
//
//            M=1     M=2     M=3     M=5     M=8
//   HOT     .145    .248    .353    .153    .203     <- m16 at M=5/8; compute is NOT the problem
//   COLD    .157    .499    .505    .488    .503     <- same kernel, 3.2x slower off DRAM
//
// A kernel that is 3.2x slower cold than hot is not compute-bound and not bandwidth-bound; it is
// fetching bytes it does not use. `tc_fp8_kernel` issues its B operand as `*(const unsigned*)(B +
// nn*K + k0 + tid4*4)` — four lanes covering 16 CONTIGUOUS bytes of one row, eight rows per warp,
// strided by K. Sixteen bytes is half a 32-byte sector, and the other half arrives on a separate
// instruction, so cold every sector is fetched for half its payload.
//
// The mma tiling is right; only the path from DRAM is wrong. Stage a [64 x 128] B tile through shared
// memory with eight consecutive lanes covering 128 contiguous bytes of one row — one full 128-byte
// line per eight lanes, nothing over-fetched — then read mma fragments back out of smem. The tile
// is not reused (each warp reads only its own 8 rows), so this buys exactly one thing: the global
// access pattern. That is the whole 3.2x.
//
// Row stride is padded 128 -> 144 bytes (36 words). The fragment read has lane group g = lane>>2
// touching row g at word (kt*8 + (lane&3)); at stride 36 words row g starts at bank (36g)%32 = 4g,
// so the eight groups land on banks 4g..4g+3 — a permutation of all 32, conflict-free. At stride 32
// every row would start at bank 0 and all eight groups would collide 8 ways.
// Tile height is adaptive. With W warps a block stages 8W B-rows, so the grid is ceil(N/8W) blocks
// — and at W=8 the small-N shapes do not fill the device: wkv [512,4096] gives 8 blocks for 20 SMs
// and measured 38 GB/s where wo_b [4096,4096] measured 199. Total warp count is N/8 either way, so
// shrinking W costs nothing and buys blocks. W is chosen so the grid is >= 64 blocks where N allows.
// Staging is 32 bytes/thread at every W (8W rows x 128 B over 32W threads), so the mapping below is
// independent of W and eight consecutive lanes always walk one row contiguously.
//
// ALIGNMENT (LOOP_LOG Finding 25, re-learned the hard way here). B is a mapped safetensors tensor.
// The old kernel loaded it with `*(const unsigned*)`, so 4-byte alignment was all it ever needed and
// nothing in the engine guaranteed more. A uint4 stage crashed prefill on the first real weight
// (`misaligned address`, mla_forward.cu:81 -> wo_b) and the unit gate did NOT catch it, because a
// gate that allocates B with cudaMalloc always gets a 256-byte-aligned pointer. So the staging loop
// is templated on alignment: 1 x uint4 when the base is 16-byte aligned, else 4 x unsigned covering
// the same 16 bytes. The four-instruction form touches the identical cache lines back to back, so
// the coalescing — the entire point of the staging — survives; only the instruction count grows.
#define TCB_LD  144                                  // padded row stride in bytes
template<int WARPS, bool AL16>
__global__ void tc_fp8_smemB_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                    const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    __shared__ uint8_t sB[WARPS*8*TCB_LD];
    const int t=threadIdx.x, warp=t>>5, lane=t&31, gid=lane>>2, tid4=lane&3;
    const int n0blk=blockIdx.x*(WARPS*8), m0=blockIdx.y*16;
    const int n0=n0blk+warp*8;
    const int r0=m0+gid, r1=m0+gid+8, KB=K/128;
    float acc[4]={0.f,0.f,0.f,0.f};
    for(int kblk=0; kblk<KB; ++kblk){
        // stage B[n0blk .. n0blk+8W)[kblk*128 .. +128) — 2 x uint4 per thread.
        // chunk c: row=c>>3, byte offset=(c&7)*16 -> lanes 8g..8g+7 walk one row contiguously.
        __syncthreads();
        #pragma unroll
        for(int h=0; h<2; ++h){
            const int c=t+h*(WARPS*32), row=c>>3, off=(c&7)*16, gn=n0blk+row;
            uint4 v=make_uint4(0,0,0,0);
            if(gn<N){
                const uint8_t* src=B+(size_t)gn*K+kblk*128+off;
                if(AL16) v=*(const uint4*)src;
                else { v.x=*(const unsigned*)(src);    v.y=*(const unsigned*)(src+4);
                       v.z=*(const unsigned*)(src+8);  v.w=*(const unsigned*)(src+12); }
            }
            *(uint4*)&sB[row*TCB_LD+off]=v;
        }
        __syncthreads();
        float cb[4]={0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int kt=0; kt<4; ++kt){ const int k0=kblk*128+kt*32;
            unsigned a[4], b[2];
            a[0]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4)    : 0u;
            a[1]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4)    : 0u;
            a[2]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4+16) : 0u;
            a[3]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4+16) : 0u;
            const int so=(warp*8+gid)*TCB_LD + kt*32 + tid4*4;
            b[0]=*(const unsigned*)&sB[so];
            b[1]=*(const unsigned*)&sB[so+16];
            mma_m16n8k32_e4m3(cb, a, b);
        }
        const float bsc = bs[(size_t)(n0/128)*KB + kblk];
        const float as0 = (r0<M)? as[(size_t)r0*KB + kblk] : 0.f;
        const float as1 = (r1<M)? as[(size_t)r1*KB + kblk] : 0.f;
        acc[0]+=cb[0]*as0*bsc; acc[1]+=cb[1]*as0*bsc; acc[2]+=cb[2]*as1*bsc; acc[3]+=cb[3]*as1*bsc;
    }
    const int cn=tid4*2;
    if(r0<M && n0+cn  <N) C[(size_t)r0*N + n0+cn  ]=acc[0];
    if(r0<M && n0+cn+1<N) C[(size_t)r0*N + n0+cn+1]=acc[1];
    if(r1<M && n0+cn  <N) C[(size_t)r1*N + n0+cn  ]=acc[2];
    if(r1<M && n0+cn+1<N) C[(size_t)r1*N + n0+cn+1]=acc[3];
}

// -1 = read NO_TCSMEM from the environment on first use; the setter exists so gates and benches can
// A/B both paths in ONE process (a cached getenv cannot be re-read, which is how the stale
// "COLD+GEMV_MK" row in gemm_bench came to be measuring the default path for months).
int g_tc_smem = -1;
void tc_fp8_set_smem(int on){ g_tc_smem = on; }
void tc_fp8_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp8, const float* b_s,
                 int M, int N, int K, cudaStream_t s){
    if(g_tc_smem<0) g_tc_smem = getenv("NO_TCSMEM")==nullptr;
    if(g_tc_smem && (K%128)==0){
        int W=8; while(W>1 && (N+8*W-1)/(8*W) < 64) W>>=1;      // >= 64 blocks where N allows
        const int mt=(M+15)/16;
        // Row stride K is a multiple of 128 and the staging offset a multiple of 16, so the base
        // pointer alone decides whether every staged address is 16-byte aligned.
        const bool al16 = (((uintptr_t)B_fp8) & 15) == 0;
        #define TCB_LAUNCH(WW) { dim3 g((N+8*(WW)-1)/(8*(WW)), mt); \
            if(al16) tc_fp8_smemB_kernel<WW,true ><<<g, 32*(WW), 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K); \
            else     tc_fp8_smemB_kernel<WW,false><<<g, 32*(WW), 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K); }
        switch(W){ case 8: TCB_LAUNCH(8) break; case 4: TCB_LAUNCH(4) break;
                   case 2: TCB_LAUNCH(2) break; default: TCB_LAUNCH(1) break; }
        #undef TCB_LAUNCH
        return;
    }
    dim3 grid((N+7)/8, (M+15)/16);
    tc_fp8_kernel<<<grid, 32, 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K);
}
