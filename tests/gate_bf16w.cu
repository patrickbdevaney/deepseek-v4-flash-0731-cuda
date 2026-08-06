// gate_bf16w.cu — gemm_bf16w (native BF16 weights) vs gemm_fp32 on the SAME weights dequantised to
// f32, which is exactly what the engine did before Finding 26. BF16->F32 is lossless, so the only
// admissible difference is fp32 accumulation reassociation (the bf16x2 path pairs elements).
//   build: nvcc -O3 -std=c++17 -arch=sm_110a -I include tests/gate_bf16w.cu kernels/compressor.cu \
//            kernels/dscratch.cu -o build/gate_bf16w
#include "compressor.h"
#include "dscratch.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#define CU(x) do{cudaError_t e=(x); if(e){printf("cuda %s\n",cudaGetErrorString(e));return 2;} }while(0)

__global__ void k_bf2f(float* o, const __nv_bfloat16* i, size_t n){
    size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(k<n) o[k]=__bfloat162float(i[k]); }

int main(){
    arena_init((size_t)64<<20);
    int fail=0;
    struct Case{ const char* n; int M,N,K; };
    // the shapes that actually matter: lm_head at decode (M=1) and verify (M=5), and markov.
    const Case cs[] = {{"lm_head M=1",1,129280,4096},{"lm_head M=5",5,129280,4096},
                       {"markov  M=1",1,129280,256},{"markov  M=5",5,129280,256}};
    for(const auto& c : cs){
        std::vector<__nv_bfloat16> hb((size_t)c.N*c.K);
        std::vector<float> ha((size_t)c.M*c.K);
        for(size_t i=0;i<hb.size();++i) hb[i]=__float2bfloat16(((float)rand()/RAND_MAX-0.5f)*0.1f);
        for(size_t i=0;i<ha.size();++i) ha[i]=((float)rand()/RAND_MAX-0.5f);
        __nv_bfloat16* B; float *Bf,*A,*C1,*C2;
        CU(cudaMalloc(&B,hb.size()*2)); CU(cudaMalloc(&Bf,hb.size()*4));
        CU(cudaMalloc(&A,ha.size()*4)); CU(cudaMalloc(&C1,(size_t)c.M*c.N*4)); CU(cudaMalloc(&C2,(size_t)c.M*c.N*4));
        CU(cudaMemcpy(B,hb.data(),hb.size()*2,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(A,ha.data(),ha.size()*4,cudaMemcpyHostToDevice));
        k_bf2f<<<(hb.size()+255)/256,256>>>(Bf,B,hb.size());          // what Loader::bf16 used to do
        CU(cudaDeviceSynchronize());
        gemm_fp32 (C1,A,Bf,c.M,c.N,c.K,0);                            // old path
        gemm_bf16w(C2,A,(const void*)B,c.M,c.N,c.K,0);                // new path
        CU(cudaDeviceSynchronize());
        std::vector<float> r1((size_t)c.M*c.N), r2((size_t)c.M*c.N);
        CU(cudaMemcpy(r1.data(),C1,r1.size()*4,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(r2.data(),C2,r2.size()*4,cudaMemcpyDeviceToHost));
        double dot=0,n1=0,n2=0,mx=0,amx=0; int argmax1=0,argmax2=0;
        for(size_t i=0;i<r1.size();++i){
            dot+=(double)r1[i]*r2[i]; n1+=(double)r1[i]*r1[i]; n2+=(double)r2[i]*r2[i];
            mx=fmax(mx,fabs(r1[i]-r2[i])); amx=fmax(amx,fabs((double)r1[i]));
            if(r1[i]>r1[argmax1]) argmax1=(int)i;
            if(r2[i]>r2[argmax2]) argmax2=(int)i;
        }
        const double cos=dot/sqrt(n1*n2), rel=mx/amx;
        const bool ok = cos>0.9999999 && rel<1e-5 && argmax1==argmax2;
        printf("[%-12s] cosine=%.9f max_abs=%.3e rel=%.2e argmax %s -> %s\n",
               c.n, cos, mx, rel, argmax1==argmax2?"MATCH":"DIFFER", ok?"PASS":"FAIL");
        if(!ok) ++fail;
        cudaFree(B);cudaFree(Bf);cudaFree(A);cudaFree(C1);cudaFree(C2);
    }
    printf("\nGate BF16W: %s\n", fail?"FAIL":"PASS");
    return fail?1:0;
}
