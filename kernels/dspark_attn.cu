// dspark_attn.cu — DSparkAttention. See dspark_attn.h / DSPARK_HEAD_BUILD.md piece 4.
#include "dspark_attn.h"
#include "fp8_block_gemm.h"
#include "mla_attn.h"
#include "deepseek_v4.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include "dscratch.h"
#define CU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)
using namespace dsv4;

// main-KV from main_x: wkv -> kv_norm -> rope(per-position) -> act_quant fp8sim(nope). [s,dim]->[s,HEAD_DIM].
void dspark_main_kv(float* main_kv, const float* main_x, const MLAWeights& w, int s, float eps, cudaStream_t stream){
    uint8_t* xq; float* xs;
    CU(dkmalloc(&xq,(size_t)s*DIM)); CU(dkmalloc(&xs,(size_t)s*(DIM/128)*4));
    act_quant_fp8(xq, xs, main_x, s, DIM, 128, stream);
    fp8_block_gemm(main_kv, xq, xs, w.wkv, w.wkv_s, s, HEAD_DIM, DIM, stream);
    rmsnorm(main_kv, main_kv, w.kv_norm, s, HEAD_DIM, eps, true, stream);
    rope_interleaved(main_kv + NOPE_DIM, w.cosT, w.sinT, s, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(main_kv, s, NOPE_DIM, 64, HEAD_DIM, stream);
    dksync(stream); dkfree(xq); dkfree(xs);
}

void dspark_attn_forward(float* out, const float* xin, const float* main_kv, int t,
                         const MLAWeights& w, const float* cosB, const float* sinB,
                         int block, int win, float eps, cudaStream_t stream){
    const int Kd = N_HEADS*HEAD_DIM, GKd = Kd/O_GROUPS, OB = O_GROUPS*O_LORA;
    const float scale = 1.f/sqrtf((float)HEAD_DIM);
    int nwin = (t+1 < win) ? t+1 : win; int wstart = t+1-nwin; int n = nwin + block;

    uint8_t *xq,*qrq,*ogq; float *xs,*qrs,*ogs,*qr,*q,*bkv,*kv_all,*o,*og;
    CU(dkmalloc(&xq,(size_t)block*DIM)); CU(dkmalloc(&xs,(size_t)block*(DIM/128)*4));
    CU(dkmalloc(&qr,(size_t)block*Q_LORA*4)); CU(dkmalloc(&qrq,(size_t)block*Q_LORA)); CU(dkmalloc(&qrs,(size_t)block*(Q_LORA/128)*4));
    CU(dkmalloc(&q,(size_t)block*Kd*4)); CU(dkmalloc(&bkv,(size_t)block*HEAD_DIM*4));
    CU(dkmalloc(&kv_all,(size_t)n*HEAD_DIM*4)); CU(dkmalloc(&o,(size_t)block*Kd*4)); CU(dkmalloc(&og,(size_t)block*OB*4));
    CU(dkmalloc(&ogq,(size_t)block*OB)); CU(dkmalloc(&ogs,(size_t)block*(OB/128)*4));

    // q
    act_quant_fp8(xq, xs, xin, block, DIM, 128, stream);
    fp8_block_gemm(qr, xq, xs, w.wq_a, w.wq_a_s, block, Q_LORA, DIM, stream);
    rmsnorm(qr, qr, w.q_norm, block, Q_LORA, eps, true, stream);
    act_quant_fp8(qrq, qrs, qr, block, Q_LORA, 128, stream);
    fp8_block_gemm(q, qrq, qrs, w.wq_b, w.wq_b_s, block, Kd, Q_LORA, stream);
    rmsnorm(q, q, nullptr, block*N_HEADS, HEAD_DIM, eps, false, stream);
    rope_interleaved(q + NOPE_DIM, cosB, sinB, block*N_HEADS, ROPE_DIM, false, HEAD_DIM, N_HEADS, stream);
    // block kv
    fp8_block_gemm(bkv, xq, xs, w.wkv, w.wkv_s, block, HEAD_DIM, DIM, stream);
    rmsnorm(bkv, bkv, w.kv_norm, block, HEAD_DIM, eps, true, stream);
    rope_interleaved(bkv + NOPE_DIM, cosB, sinB, block, ROPE_DIM, false, HEAD_DIM, 1, stream);
    act_quant_fp8sim(bkv, block, NOPE_DIM, 64, HEAD_DIM, stream);
    // kv = [main-KV window ⊕ block-KV]
    CU(cudaMemcpyAsync(kv_all, main_kv + (size_t)wstart*HEAD_DIM, (size_t)nwin*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    CU(cudaMemcpyAsync(kv_all + (size_t)nwin*HEAD_DIM, bkv, (size_t)block*HEAD_DIM*4, cudaMemcpyDeviceToDevice, stream));
    // dense idxs [block, n]: every block query attends to all n (window ⊕ block), per get_dspark_topk_idxs
    std::vector<int> hidx((size_t)block*n); for(int m=0;m<block;++m) for(int k=0;k<n;++k) hidx[(size_t)m*n+k]=k;
    int* idx; CU(dkmalloc(&idx,(size_t)block*n*4)); CU(cudaMemcpyAsync(idx,hidx.data(),(size_t)block*n*4,cudaMemcpyHostToDevice,stream));
    sparse_attn(o, q, kv_all, w.attn_sink, idx, 1, block, N_HEADS, HEAD_DIM, n, n, scale, stream);
    rope_interleaved(o + NOPE_DIM, cosB, sinB, block*N_HEADS, ROPE_DIM, true, HEAD_DIM, N_HEADS, stream);
    ogroup_gemm(og, o, w.wo_a, block, O_GROUPS, O_LORA, GKd, stream);
    act_quant_fp8(ogq, ogs, og, block, OB, 128, stream);
    fp8_block_gemm(out, ogq, ogs, w.wo_b, w.wo_b_s, block, DIM, OB, stream);
    dksync(stream);
    dkfree(xq);dkfree(xs);dkfree(qr);dkfree(qrq);dkfree(qrs);dkfree(q);dkfree(bkv);
    dkfree(kv_all);dkfree(o);dkfree(og);dkfree(ogq);dkfree(ogs);dkfree(idx);
}

// ---- DSparkBlock forward (block_forward with dspark_attn) ----
#include "hc.h"
#include "moe.h"
// F103 localised the port's 18557x divergence to dspark_block_forward. These stage dumps take it
// one level deeper: whichever sub-stage first disagrees is the bug. Written only when decode.cu
// points g_dspark_dump at a file, and only for the block it wants -- otherwise a no-op branch.
FILE* g_dspark_dump = nullptr;
static void dsp_dump(const char* tag, const float* dev, size_t n){
    if(!g_dspark_dump) return;
    std::vector<float> hv(n);
    if(cudaMemcpy(hv.data(),dev,n*4,cudaMemcpyDeviceToHost)!=cudaSuccess) return;
    uint32_t L=(uint32_t)strlen(tag), N=(uint32_t)n;
    fwrite(&L,4,1,g_dspark_dump); fwrite(tag,1,L,g_dspark_dump);
    fwrite(&N,4,1,g_dspark_dump); fwrite(hv.data(),4,n,g_dspark_dump);
}
void dspark_block_forward(float* out, const float* x, const int* input_ids, const float* main_kv, int t,
                          const BlockWeights& w, const float* cosB, const float* sinB, int block, int win,
                          int iters, float eps, cudaStream_t stream){
    const int d=w.dim, hc=w.hc;
    float *x1,*post,*comb,*sub,*res2;
    CU(dkmalloc(&x1,(size_t)block*d*4)); CU(dkmalloc(&post,(size_t)block*hc*4)); CU(dkmalloc(&comb,(size_t)block*hc*hc*4));
    CU(dkmalloc(&sub,(size_t)block*d*4)); CU(dkmalloc(&res2,(size_t)block*hc*d*4));
    hc_pre(x1, post, comb, x, w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, block, hc, d, iters, eps, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream);
        dsp_dump("b_hcpre_attn_x1", x1, (size_t)block*d);
        dsp_dump("b_hcpre_attn_post", post, (size_t)block*hc);
        dsp_dump("b_hcpre_attn_comb", comb, (size_t)block*hc*hc); }
    rmsnorm(x1, x1, w.attn_norm, block, d, eps, true, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_rmsnorm_attn", x1, (size_t)block*d); }
    dspark_attn_forward(sub, x1, main_kv, t, w.attn, cosB, sinB, block, win, eps, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_attn_out", sub, (size_t)block*d); }
    hc_post(res2, sub, x, post, comb, block, hc, d, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_hcpost_attn", res2, (size_t)block*hc*d); }
    hc_pre(x1, post, comb, res2, w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, block, hc, d, iters, eps, stream);
    rmsnorm(x1, x1, w.ffn_norm, block, d, eps, true, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_rmsnorm_ffn", x1, (size_t)block*d); }
    moe_forward(sub, x1, input_ids, w.ffn, block, stream);
    if(g_dspark_dump){ cudaStreamSynchronize(stream); dsp_dump("b_moe_out", sub, (size_t)block*d); }
    hc_post(out, sub, res2, post, comb, block, hc, d, stream);
    dksync(stream);
    dkfree(x1);dkfree(post);dkfree(comb);dkfree(sub);dkfree(res2);
}
