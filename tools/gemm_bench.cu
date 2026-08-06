// gemm_bench.cu — standalone micro-benchmark for the decode-critical GEMMs, on SYNTHETIC weights.
//
// Why this exists: every measurement so far cost a ~10-minute cold load of a 100 GiB checkpoint.
// That is fatal to an optimization loop that wants one change per measurement. These kernels do not
// care what the weight bytes contain, only their shapes — so allocate the representative shapes,
// run the real kernels, and report achieved bandwidth against the measured 240 GB/s roofline.
//
// Shapes are the ACTUAL per-layer shapes read from docs/hdrs (see MODEL_INVENTORY.md), not invented.
//
//   build: nvcc -O3 -std=c++17 -arch=sm_110a -I include tools/gemm_bench.cu \
//            kernels/fp8_block_gemm.cu kernels/tc_fp8_gemm.cu kernels/mla_attn.cu kernels/moe.cu \
//            kernels/tc_moe_gemm.cu kernels/dscratch.cu kernels/hc.cu kernels/hc_sinkhorn.cu \
//            kernels/compressor.cu kernels/indexer.cu -o build/gemm_bench
//   run:   ./build/gemm_bench [reps]
#include "deepseek_v4.h"
#include "fp8_block_gemm.h"
#include "dscratch.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

using namespace dsv4;
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

extern bool g_tc_fp8;

static void* dalloc(size_t n){ void* p; CU(cudaMalloc(&p,n)); CU(cudaMemset(p,0x11,n)); return p; }

// Achieved bandwidth counts the WEIGHT bytes (they dominate; activations are negligible at small M).
struct Row { std::string name; int M,N,K; double ms; double wbytes; };

static void report(std::vector<Row>& rows, double achievable_gbs) {
    printf("\n%-26s %4s %7s %6s %10s %10s %9s %8s\n",
           "kernel / shape", "M", "N", "K", "w_MB", "ms", "GB/s", "%%roof");
    printf("--------------------------------------------------------------------------------------------\n");
    for (auto& r : rows) {
        double gbs = r.wbytes / (r.ms*1e-3) / 1e9;
        printf("%-26s %4d %7d %6d %10.2f %10.4f %9.1f %7.1f%%\n",
               r.name.c_str(), r.M, r.N, r.K, r.wbytes/1e6, r.ms, gbs, 100.0*gbs/achievable_gbs);
    }
}

template <class F>
static double timeit(F&& f, int reps) {
    cudaEvent_t a,b; CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    f(); CU(cudaDeviceSynchronize());                      // warm
    CU(cudaEventRecord(a));
    for (int i=0;i<reps;++i) f();
    CU(cudaEventRecord(b)); CU(cudaEventSynchronize(b));
    float ms; CU(cudaEventElapsedTime(&ms,a,b));
    CU(cudaEventDestroy(a)); CU(cudaEventDestroy(b));
    return ms/reps;
}

int main(int argc, char** argv) {
    const int reps = argc>1 ? atoi(argv[1]) : 50;
    const double ROOF = 240.0;                              // measured achievable, HARDWARE.md §2
    g_tc_fp8 = true;
    arena_init((size_t)512<<20);

    printf("gemm_bench — decode-critical GEMMs on synthetic weights of the REAL shapes\n");
    printf("roofline: %.0f GB/s achievable (tools/bw_probe.cu)   reps=%d\n", ROOF, reps);

    // ---- the five MLA per-layer dense GEMMs (41.1%% of B_tok, ROOFLINE.md §2) ----
    struct Shape { const char* n; int N, K; };
    const Shape mla[] = {
        {"MLA wq_a",   Q_LORA,            DIM},                 // [1024, 4096]
        {"MLA wq_b",   N_HEADS*HEAD_DIM,  Q_LORA},              // [32768, 1024]
        {"MLA wkv",    HEAD_DIM,          DIM},                 // [512, 4096]
        {"MLA wo_b",   DIM,               O_GROUPS*O_LORA},     // [4096, 8192]
    };

    std::vector<Row> rows;
    for (int mi = 0; mi < 4; ++mi) {
        const auto& sh = mla[mi];
        const size_t wbytes = (size_t)sh.N * sh.K;              // fp8 = 1 byte/elt
        uint8_t* B  = (uint8_t*)dalloc(wbytes);
        float*   bs = (float*)  dalloc((size_t)(sh.N/128 + 1)*(sh.K/128 + 1)*4);
        for (int M : {1, 2, 3, 5, 8, 16}) {
            uint8_t* A  = (uint8_t*)dalloc((size_t)M*sh.K);
            float*   as = (float*)  dalloc((size_t)M*(sh.K/128)*4);
            float*   C  = (float*)  dalloc((size_t)M*sh.N*4);
            double ms = timeit([&]{ fp8_block_gemm(C,A,as,B,bs,M,sh.N,sh.K,0); }, reps);
            rows.push_back({std::string(sh.n)+(M==1?" (GEMV)":" (mma)"), M, sh.N, sh.K, ms, (double)wbytes});
            CU(cudaFree(A)); CU(cudaFree(as)); CU(cudaFree(C));
        }
        CU(cudaFree(B)); CU(cudaFree(bs));
    }
    report(rows, ROOF);

    // ---- the M=1 -> M=2 cliff, isolated (LOOP_LOG Finding 15) ----
    printf("\nM=1 -> M>=2 step (Finding 15): per-token cost of the SAME weight read\n");
    printf("%-14s %10s %10s %10s %10s %10s %10s\n","shape","M=1","M=2","M=3","M=5","M=8","M=16");
    for (int mi = 0; mi < 4; ++mi) {
        printf("%-14s", mla[mi].n);
        for (int M : {1,2,3,5,8,16}) {
            double ms = 0;
            for (auto& r : rows) if (r.N==mla[mi].N && r.K==mla[mi].K && r.M==M) ms = r.ms;
            printf(" %9.4f", ms);
        }
        printf("   ms/call\n");
    }
    printf("\nIf the M=1 GEMV is bandwidth-optimal and the m16 mma is latency-bound, M=2 costs\n"
           "MUCH more than M=1 for the same weight bytes, then stays ~flat to M=16.\n");

    // ---- MXFP4 grouped MoE: the other 30.8%% of B_tok ----
    // Emulates the decode-time dispatch: M tokens x top-6, gathered and counting-sorted by expert.
    // At M=1 that is 6 experts x 1 row; at M=K it is |union| experts with 1-2 rows each. We feed the
    // WORST case (all rows on distinct experts) so the weight traffic matches |union| = min(6M, nr).
    {
        const int nr = N_ROUTED, inter = MOE_INTER, dim = DIM;
        const size_t w13n = (size_t)inter*(dim/2), w13s = (size_t)inter*(dim/32);
        std::vector<uint8_t*> hw(nr); std::vector<uint8_t*> hs(nr);
        for (int e = 0; e < nr; ++e) { hw[e] = (uint8_t*)dalloc(w13n); hs[e] = (uint8_t*)dalloc(w13s); }
        const uint8_t** wptr; const uint8_t** sptr;
        CU(cudaMalloc(&wptr, nr*sizeof(void*))); CU(cudaMalloc(&sptr, nr*sizeof(void*)));
        CU(cudaMemcpy(wptr, hw.data(), nr*sizeof(void*), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(sptr, hs.data(), nr*sizeof(void*), cudaMemcpyHostToDevice));

        printf("\nMXFP4 grouped MoE (w1 shape [%d,%d], %d experts, top-%d)\n", inter, dim, nr, N_ACT);
        printf("%-22s %5s %8s %10s %10s %9s %8s\n","path","M","|union|","w_MB","ms","GB/s","%%roof");
        printf("--------------------------------------------------------------------------------\n");
        for (int M : {1,2,3,5,8}) {
            const int U = (6*M < nr) ? 6*M : nr;              // distinct-expert worst case
            std::vector<int> hoff(nr+1,0);
            for (int e = 0; e <= nr; ++e) hoff[e] = (e <= U) ? e : U;   // 1 row on each of the first U
            int *off_d,*tile_e,*tile_row0,*ntiles_d;
            CU(cudaMalloc(&off_d,(nr+1)*4)); CU(cudaMemcpy(off_d,hoff.data(),(nr+1)*4,cudaMemcpyHostToDevice));
            CU(cudaMalloc(&tile_e,(U+8)*4)); CU(cudaMalloc(&tile_row0,(U+8)*4)); CU(cudaMalloc(&ntiles_d,4));
            extern void tc_build_tiles(int*,int*,int*,const int*,int,cudaStream_t);
            tc_build_tiles(tile_e,tile_row0,ntiles_d,off_d,nr,0); CU(cudaDeviceSynchronize());

            __half* x16 = (__half*)dalloc((size_t)U*dim*2);
            uint8_t* xq = (uint8_t*)dalloc((size_t)U*dim);
            float*  xs  = (float*) dalloc((size_t)U*(dim/128)*4);
            float*  out = (float*) dalloc((size_t)U*inter*4);
            const double wb = (double)U * w13n;               // one read per activated expert

            extern void tc_fp4_grouped_gemm_e8m0(float*,const __half*,const uint8_t* const*,const uint8_t* const*,
                                                 const int*,const int*,const int*,const int*,int,int,int,cudaStream_t);
            extern void tc_fp4_grouped_gemv_e8m0(float*,const uint8_t*,const float*,const uint8_t* const*,const uint8_t* const*,
                                                 const int*,const int*,const int*,const int*,int,int,int,cudaStream_t);
            double ms_mma = timeit([&]{ tc_fp4_grouped_gemm_e8m0(out,x16,wptr,sptr,off_d,tile_e,tile_row0,ntiles_d,U,inter,dim,0); }, reps);
            double ms_gv  = timeit([&]{ tc_fp4_grouped_gemv_e8m0(out,xq,xs,wptr,sptr,off_d,tile_e,tile_row0,ntiles_d,U,inter,dim,0); }, reps);
            printf("%-22s %5d %8d %10.2f %10.4f %9.1f %7.1f%%\n","grouped mma (default)",M,U,wb/1e6,ms_mma,wb/(ms_mma*1e-3)/1e9,100.0*(wb/(ms_mma*1e-3)/1e9)/ROOF);
            printf("%-22s %5d %8d %10.2f %10.4f %9.1f %7.1f%%\n","grouped GEMV (MOE_GEMV)",M,U,wb/1e6,ms_gv, wb/(ms_gv *1e-3)/1e9,100.0*(wb/(ms_gv *1e-3)/1e9)/ROOF);
            CU(cudaFree(x16)); CU(cudaFree(xq)); CU(cudaFree(xs)); CU(cudaFree(out));
            CU(cudaFree(off_d)); CU(cudaFree(tile_e)); CU(cudaFree(tile_row0)); CU(cudaFree(ntiles_d));
        }
        printf("\nMoE is 30.8%% of B_tok. If this sits far below the MLA GEMMs' efficiency, the\n"
               "ROOFLINE.md §6 priority order (MLA #1, MoE #2) is inverted by measurement.\n");
    }
    return 0;
}
