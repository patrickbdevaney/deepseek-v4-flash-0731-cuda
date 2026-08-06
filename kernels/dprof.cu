// dprof.cu — see include/dprof.h.
#include "dprof.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

bool g_dprof_on = false;

namespace {
struct Mark { int id; cudaEvent_t a, b; };
std::vector<Mark> g_pool;
int  g_used = 0;
int  g_open[DP_N];              // slot currently open per id (-1 = none)
bool g_inited = false;

const char* kName[DP_N] = {
    "hc_pre  (attn)", "rmsnorm (attn)", "ATTENTION", "hc_post (attn)",
    "hc_pre  (ffn)",  "rmsnorm (ffn)",  "MoE",       "hc_post (ffn)", "kv xin copy",
    "  attn:q_proj",  "  attn:kv_write", "  attn:sparse", "  attn:ogroup", "  attn:misc",
};
} // namespace

void dprof_init(int max_marks){
    if (g_inited) return;
    g_dprof_on = getenv("DSV4_DPROF") != nullptr;
    if (!g_dprof_on) { g_inited = true; return; }
    g_pool.resize(max_marks);
    for (auto& m : g_pool) { cudaEventCreate(&m.a); cudaEventCreate(&m.b); m.id = -1; }
    for (int i = 0; i < DP_N; ++i) g_open[i] = -1;
    g_used = 0; g_inited = true;
    printf("[dprof] enabled, %d marks\n", max_marks);
}

void dprof_begin(int id, cudaStream_t s){
    if (!g_dprof_on || g_used >= (int)g_pool.size()) return;
    Mark& m = g_pool[g_used];
    m.id = id; cudaEventRecord(m.a, s);
    g_open[id] = g_used; ++g_used;
}

void dprof_end(int id, cudaStream_t s){
    if (!g_dprof_on) return;
    const int slot = g_open[id];
    if (slot < 0) return;
    cudaEventRecord(g_pool[slot].b, s);
    g_open[id] = -1;
}

void dprof_reset(){ g_used = 0; for (int i = 0; i < DP_N; ++i) g_open[i] = -1; }

void dprof_report(const char* tag){
    if (!g_dprof_on || !g_used) return;
    cudaDeviceSynchronize();
    double sum[DP_N] = {0}; int cnt[DP_N] = {0};
    for (int i = 0; i < g_used; ++i){
        float ms = 0.f;
        if (cudaEventElapsedTime(&ms, g_pool[i].a, g_pool[i].b) != cudaSuccess) continue;
        sum[g_pool[i].id] += ms; ++cnt[g_pool[i].id];
    }
    double tot = 0; for (int i = 0; i < DP_N; ++i) tot += sum[i];
    printf("\n[dprof] %s — verify step by sub-op, summed over all layers (%d marks)\n", tag, g_used);
    printf("[dprof] %-16s %10s %8s %10s\n", "phase", "ms", "%", "calls");
    for (int i = 0; i < DP_N; ++i)
        if (cnt[i]) printf("[dprof] %-16s %10.2f %7.1f%% %10d\n", kName[i], sum[i], 100.0*sum[i]/tot, cnt[i]);
    printf("[dprof] %-16s %10.2f\n", "TOTAL", tot);
    dprof_reset();
}
