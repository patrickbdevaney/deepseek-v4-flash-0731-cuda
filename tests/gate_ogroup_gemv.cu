// gate_ogroup_gemv.cu — the M=1 fp8 ogroup GEMV, which had NO unit coverage at all.
//
// `tests/gate_ogroup.cu` exercises `ogroup_gemm` (f32 weights) at bs=8, i.e. the tensor-core tile
// path. The kernel that actually runs at decode is `ogroup_gemv_fp8_kernel`, reached only through
// `ogroup_gemm_fp8` with bs==1 — a different kernel, a different weight format, a different M.
// mla_attn.cu claimed "Gated cosine vs ogroup_gemm oracle (tests/gate_ogroup_gemv.cu)"; that file
// did not exist. By the dprof sub-marks this kernel is ~22% of the decode step, so it was both the
// least-tested and among the most expensive things in the engine.
//
// Two comparisons, because they fail in different ways:
//   A. vec4 (uint32 weight / float4 activation loads, Finding 35) vs the scalar byte loop on the
//      SAME fp8 bytes — isolates the vectorisation, which should differ only by summation order.
//   B. both vs an f32 oracle over the dequantised weights — catches a wrong dequant or scale index.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#define CU(x) do{auto e=(x); if(e){printf("ERR %s:%d %d\n",__FILE__,__LINE__,(int)e);return 1;}}while(0)

void ogroup_gemm(float*, const float*, const float*, int,int,int,int, cudaStream_t);
void ogroup_gemm_fp8(float* out, const float* o, const uint8_t* wo_fp8, const uint8_t* wo_sc,
                     int bs, int G, int R, int Kd, cudaStream_t stream);
extern bool g_tc_ogroup;

// Dequantise exactly as ogroup_gemv_fp8_kernel does: e4m3 byte * 2^(scale_byte-127), scales laid
// out [G*R/128, Kd/128] and indexed (gr/128, kb).
__global__ void k_deq(float* out, const uint8_t* w, const uint8_t* sc, int rows, int Kd){
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; if(i >= (size_t)rows*Kd) return;
    int gr = i/Kd, d = i%Kd, scw = Kd/128;
    __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)w[i], __NV_E4M3);
    out[i] = __half2float(*reinterpret_cast<__half*>(&r)) * exp2f((float)sc[(size_t)(gr/128)*scw + d/128] - 127.f);
}

struct Cmp { double cos, rms_rel, rel_max; };
static Cmp compare(const std::vector<float>& a, const std::vector<float>& b){
    double d=0,na=0,nb=0,se=0,md=0,amax=0;
    for(size_t i=0;i<a.size();++i){ d+=(double)a[i]*b[i]; na+=(double)a[i]*a[i]; nb+=(double)b[i]*b[i];
        double e=(double)a[i]-b[i]; se+=e*e; md=fmax(md,fabs(e)); amax=fmax(amax,fabs((double)a[i])); }
    Cmp c; c.cos=d/(sqrt(na)*sqrt(nb)+1e-30);
    c.rms_rel=sqrt(se/a.size())/(sqrt(na/a.size())+1e-30); c.rel_max=md/(amax+1e-30);
    return c;
}
// The deep-composition metric this project standardised on (see tests/gate_fp4_gemv.cu): elementwise
// absolute tolerance is the wrong instrument for a reduction over thousands of terms.
static bool ok(Cmp c){ return c.cos>0.9999 && c.rms_rel<1e-2 && c.rel_max<5e-3; }

int main(){
    const int G=8, R=1024, Kd=4096, rows=G*R;          // the real decode shape
    std::vector<uint8_t> w((size_t)rows*Kd), sc((size_t)(rows/128)*(Kd/128));
    srand(11);
    // 0x7f/0xff are e4m3 NaN; a NaN weight would poison every metric and tell us nothing.
    for(auto&b:w){ uint8_t v; do { v=(uint8_t)(rand()&0xff); } while(v==0x7f||v==0xff); b=v; }
    for(auto&b:sc) b=(uint8_t)(120 + rand()%15);       // 2^-7 .. 2^7, no overflow of the f32 oracle
    std::vector<float> o((size_t)G*Kd);
    for(auto&x:o) x=(rand()%2000-1000)/1000.f;

    uint8_t *dw,*dsc; float *doo,*dwf,*Cref,*Cvec,*Cscl;
    CU(cudaMalloc(&dw,w.size())); CU(cudaMalloc(&dsc,sc.size()));
    CU(cudaMalloc(&doo,o.size()*4)); CU(cudaMalloc(&dwf,(size_t)rows*Kd*4));
    CU(cudaMalloc(&Cref,(size_t)rows*4)); CU(cudaMalloc(&Cvec,(size_t)rows*4)); CU(cudaMalloc(&Cscl,(size_t)rows*4));
    CU(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dsc,sc.data(),sc.size(),cudaMemcpyHostToDevice));
    CU(cudaMemcpy(doo,o.data(),o.size()*4,cudaMemcpyHostToDevice));
    k_deq<<<(((size_t)rows*Kd)+255)/256,256>>>(dwf,dw,dsc,rows,Kd); CU(cudaDeviceSynchronize());

    g_tc_ogroup=false;
    ogroup_gemm(Cref,doo,dwf,1,G,R,Kd,0); CU(cudaDeviceSynchronize());          // f32 oracle, warp-per-output
    unsetenv("NO_OGVEC4"); ogroup_gemm_fp8(Cvec,doo,dw,dsc,1,G,R,Kd,0); CU(cudaDeviceSynchronize());
    setenv("NO_OGVEC4","1",1); ogroup_gemm_fp8(Cscl,doo,dw,dsc,1,G,R,Kd,0); CU(cudaDeviceSynchronize());
    unsetenv("NO_OGVEC4");

    std::vector<float> cr(rows), cv(rows), cs(rows);
    CU(cudaMemcpy(cr.data(),Cref,rows*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(cv.data(),Cvec,rows*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(cs.data(),Cscl,rows*4,cudaMemcpyDeviceToHost));

    Cmp a=compare(cs,cv), b=compare(cr,cv), c=compare(cr,cs);
    printf("[oggemv vec4 vs scalar] cos=%.9f rms_rel=%.2e rel_max=%.2e -> %s\n", a.cos,a.rms_rel,a.rel_max, ok(a)?"PASS":"FAIL");
    printf("[oggemv vec4 vs f32   ] cos=%.9f rms_rel=%.2e rel_max=%.2e -> %s\n", b.cos,b.rms_rel,b.rel_max, ok(b)?"PASS":"FAIL");
    printf("[oggemv scal vs f32   ] cos=%.9f rms_rel=%.2e rel_max=%.2e -> %s\n", c.cos,c.rms_rel,c.rel_max, ok(c)?"PASS":"FAIL");
    bool pass = ok(a) && ok(b) && ok(c);
    printf("\nGate OGGEMV: %s\n", pass?"PASS":"FAIL");
    return pass?0:1;
}
