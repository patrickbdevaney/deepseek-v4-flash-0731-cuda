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
// K-CHUNKED STAGING (LOOP_LOG Finding 74). The staging loop above used to run once per 128-K-block:
// two uint4 per thread, then __syncthreads, then the mma, then __syncthreads again. That is ONE
// round trip to DRAM per barrier pair with nothing else in flight, and the total warp count of this
// kernel is structurally N/8 — so the small-N shapes have neither enough warps NOR enough bytes per
// warp to cover the latency. Measured in situ (evidence/moescan.log, same weight bytes at both K):
//
//     mark     bytes/step   K=1 (GEMV path)   K=5 (this tile)
//     q:wq_a     0.172 GB    1.49 ms (115 GB/s)   6.71 ms ( 26 GB/s)   <- N=1024 -> 64 blk x 2 warps
//     q:wq_b     1.376 GB    7.04 ms (195 GB/s)  11.18 ms (123 GB/s)
//     o:wo_b     1.376 GB    7.43 ms (185 GB/s)   9.76 ms (141 GB/s)
//
// Same bytes, 1.3-4.5x the time. The M=1 GEMV proves 185-195 GB/s is reachable on these exact rows,
// so this is not a roofline; it is memory-level parallelism.
//
// Fix: stage KC consecutive K-blocks per barrier pair. Per-thread staging is 32*KC bytes and is
// INDEPENDENT of WARPS (8W rows x KC*128 B over 32W threads), so KC alone sets bytes-in-flight, and
// the barrier count drops by KC. Loads are issued into a register array FIRST and stored to smem
// after — a load->store->load loop would serialise at one outstanding miss no matter how large KC is.
//
// The accumulation is unchanged: `cb` is still built per 128-block from four mma, and `acc` still
// takes them in kblk order 0,1,2,..., so this is BIT-EXACT, not merely close (contrast Finding 68).
//
// KC is picked from the smem budget, which automatically gives the starved shapes the most: the
// staged tile is 8W*(KC*128+16) bytes, so W=8 -> KC=2 and W=2 -> KC=8. The starved shapes also have
// the registers to spare — wq_a runs 128 warps on the whole device, so 8*KC registers of staging
// costs nothing there.
#define TCB_PAD 16                                   // row-stride pad, in bytes
template<int WARPS, int KC, bool AL16>
__global__ void tc_fp8_smemB_kernel(float* __restrict__ C, const uint8_t* __restrict__ A, const float* __restrict__ as,
                                    const uint8_t* __restrict__ B, const float* __restrict__ bs, int M, int N, int K){
    constexpr int LD = KC*128 + TCB_PAD;             // padded row stride in bytes
    constexpr int CPR = KC*8;                        // 16-byte chunks per staged row
    constexpr int NH  = 2*KC;                        // uint4 staged per thread per round
    __shared__ uint8_t sB[WARPS*8*LD];
    const int t=threadIdx.x, warp=t>>5, lane=t&31, gid=lane>>2, tid4=lane&3;
    const int n0blk=blockIdx.x*(WARPS*8), m0=blockIdx.y*16;
    const int n0=n0blk+warp*8;
    const int r0=m0+gid, r1=m0+gid+8, KB=K/128;
    float acc[4]={0.f,0.f,0.f,0.f};
    for(int kb0=0; kb0<KB; kb0+=KC){
        // stage B[n0blk .. n0blk+8W)[kb0*128 .. +KC*128) — NH x uint4 per thread.
        // chunk c: row=c/CPR, byte offset=(c%CPR)*16 -> lanes 0..CPR-1 walk one row contiguously.
        // At LD=(32*KC+4) words, row g starts at bank 4g, so the eight fragment groups are a
        // permutation of all 32 banks for every KC — the same conflict-free property as LD=144.
        __syncthreads();
        uint4 v[NH];
        #pragma unroll
        for(int h=0; h<NH; ++h){
            const int c=t+h*(WARPS*32), row=c/CPR, off=(c%CPR)*16, gn=n0blk+row;
            v[h]=make_uint4(0,0,0,0);
            if(gn<N){
                const uint8_t* src=B+(size_t)gn*K+(size_t)kb0*128+off;
                if(AL16) v[h]=*(const uint4*)src;
                else { v[h].x=*(const unsigned*)(src);    v[h].y=*(const unsigned*)(src+4);
                       v[h].z=*(const unsigned*)(src+8);  v[h].w=*(const unsigned*)(src+12); }
            }
        }
        #pragma unroll
        for(int h=0; h<NH; ++h){
            const int c=t+h*(WARPS*32), row=c/CPR, off=(c%CPR)*16;
            *(uint4*)&sB[row*LD+off]=v[h];
        }
        __syncthreads();
        #pragma unroll 1
        for(int kc=0; kc<KC; ++kc){
            const int kblk=kb0+kc;
            float cb[4]={0.f,0.f,0.f,0.f};
            #pragma unroll
            for(int kt=0; kt<4; ++kt){ const int k0=kblk*128+kt*32;
                unsigned a[4], b[2];
                a[0]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4)    : 0u;
                a[1]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4)    : 0u;
                a[2]=(r0<M)? *(const unsigned*)(A+(size_t)r0*K+k0+tid4*4+16) : 0u;
                a[3]=(r1<M)? *(const unsigned*)(A+(size_t)r1*K+k0+tid4*4+16) : 0u;
                const int so=(warp*8+gid)*LD + kc*128 + kt*32 + tid4*4;
                b[0]=*(const unsigned*)&sB[so];
                b[1]=*(const unsigned*)&sB[so+16];
                mma_m16n8k32_e4m3(cb, a, b);
            }
            const float bsc = bs[(size_t)(n0/128)*KB + kblk];
            const float as0 = (r0<M)? as[(size_t)r0*KB + kblk] : 0.f;
            const float as1 = (r1<M)? as[(size_t)r1*KB + kblk] : 0.f;
            acc[0]+=cb[0]*as0*bsc; acc[1]+=cb[1]*as0*bsc; acc[2]+=cb[2]*as1*bsc; acc[3]+=cb[3]*as1*bsc;
        }
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
// K-chunk depth for the staged tile. -1 = auto from the smem budget (see the kernel comment);
// TCB_KC=<n> or tc_fp8_set_kc(<n>) pins it, and KC=1 is exactly the pre-Finding-74 kernel, so the
// A/B is one env var. 0 also means auto.
static int g_tc_kc = -1;
void tc_fp8_set_kc(int kc){ g_tc_kc = kc; }
static int tcb_kc_env(){
    static int v = -2;
    if(v==-2){ const char* e=getenv("TCB_KC"); v = e ? atoi(e) : -1; }
    return v;
}
// KC=2 everywhere except W=1, where 8 wins. Measured COLD on the real verify shapes at M=5, the
// `m16+smem B+4` (4-byte-aligned weights = what the engine actually has) column, ms(GB/s)
// (evidence/kchunk_bench.log, clocks pinned):
//
//   shape                 KC=1        KC=2        KC=4        KC=8      W
//   wq_a  [1024,4096]  .0501( 84)  .0402(104)  .0521( 81)  .0438( 96)  2
//   wq_b  [4096,1024]  .0339(124)  .0309(136)  .0332(126)  .0328(128)  8
//   wkv   [ 512,4096]  .0434( 48)  .0329( 64)  .0361( 58)  .0266( 79)  1   <- the W=1 exception
//   wo_b  [4096,4096]  .1131(148)  .1054(159)  .1072(156)  .1074(156)  8
//   sw1/3 [2048,4096]  .0676(124)  .0588(143)  .0671(125)  .0606(139)  8
//   sw2   [4096,2048]  .0616(136)  .0571(147)  .0584(144)  .0597(140)  8
//
// KC=4 is consistently WORSE than both 2 and 8 — non-monotone, so this is a lookup, not a rule, and
// it is fitted to one bench (LEVERS.md trap 3: gemm_bench ranks, it does not predict end-to-end).
// TCB_KC=<n> overrides; TCB_KC=1 is exactly the pre-Finding-74 kernel.
//
// HARD cap at KC<=4 for W=8: 64 rows x 8 K-blocks would be a 66 KB static __shared__, which does not
// compile, so that instantiation must not exist even on a forced path.
static int tcb_kc_max(int W){ return W>=8 ? 4 : 8; }
static int tcb_pick_kc(int W, int KB){
    const int hi = tcb_kc_max(W);
    const int forced = (g_tc_kc>0) ? g_tc_kc : tcb_kc_env();
    const int budget = forced>0 ? (40<<10) : 20480;      // forced may exceed the occupancy budget
    int want = forced>0 ? forced : (W==1 ? 8 : 2);
    if(want>hi) want=hi;
    int KC=1;
    while(KC<want && (KB%(KC*2))==0 && (size_t)W*8*((size_t)(KC*2)*128+16) <= (size_t)budget) KC*=2;
    return KC;
}
void tc_fp8_gemm(float* C, const uint8_t* A_fp8, const float* a_s, const uint8_t* B_fp8, const float* b_s,
                 int M, int N, int K, cudaStream_t s){
    if(g_tc_smem<0) g_tc_smem = getenv("NO_TCSMEM")==nullptr;
    if(g_tc_smem && (K%128)==0){
        int W=8; while(W>1 && (N+8*W-1)/(8*W) < 64) W>>=1;      // >= 64 blocks where N allows
        const int mt=(M+15)/16;
        const int KC=tcb_pick_kc(W, K/128);
        // Row stride K is a multiple of 128 and the staging offset a multiple of 16, so the base
        // pointer alone decides whether every staged address is 16-byte aligned.
        const bool al16 = (((uintptr_t)B_fp8) & 15) == 0;
        #define TCB_KCASE(WW,KK) case KK: { dim3 g((N+8*(WW)-1)/(8*(WW)), mt); \
            if(al16) tc_fp8_smemB_kernel<WW,KK,true ><<<g, 32*(WW), 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K); \
            else     tc_fp8_smemB_kernel<WW,KK,false><<<g, 32*(WW), 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K); } break;
        #define TCB_LAUNCH(WW) switch(KC){ TCB_KCASE(WW,2) TCB_KCASE(WW,4) TCB_KCASE(WW,8) \
                                           default: TCB_KCASE(WW,1) }
        #define TCB_LAUNCH8     switch(KC){ TCB_KCASE(8,2)  TCB_KCASE(8,4) \
                                           default: TCB_KCASE(8,1) }
        switch(W){ case 8: TCB_LAUNCH8 break; case 4: TCB_LAUNCH(4) break;
                   case 2: TCB_LAUNCH(2) break; default: TCB_LAUNCH(1) break; }
        #undef TCB_LAUNCH8
        #undef TCB_LAUNCH
        #undef TCB_KCASE
        return;
    }
    dim3 grid((N+7)/8, (M+15)/16);
    tc_fp8_kernel<<<grid, 32, 0, s>>>(C, A_fp8, a_s, B_fp8, b_s, M, N, K);
}
