// dscratch.cu — decode scratch arena globals. See dscratch.h.
#include "dscratch.h"
bool   g_arena_on  = false;
char*  g_arena     = nullptr;
size_t g_arena_off = 0;
size_t g_arena_cap = 0;
void arena_init(size_t cap){
    if(g_arena){ cudaFree(g_arena); }
    cudaMalloc((void**)&g_arena, cap); g_arena_cap = cap; g_arena_off = 0; g_arena_on = true;
}
void arena_reset(){ g_arena_off = 0; }

// See the comment on dsync() in dscratch.h. Drains the thread's last-error slot so a launch failure
// is named where it happened instead of surfacing at the next unrelated CU(). Reports every time,
// because a rate limit is how the first one gets lost; aborts only under DSV4_STRICT_LAUNCH=1.
void dsync_at(cudaStream_t s, const char* file, int line){
    if(!g_arena_on) cudaStreamSynchronize(s);
    cudaError_t e = cudaGetLastError();
    if(e == cudaSuccess) return;
    static const bool strict = getenv("DSV4_STRICT_LAUNCH") != nullptr;
    fprintf(stderr, "[launch] %s:%d pending CUDA error: %s (%d)\n", file, line, cudaGetErrorString(e), (int)e);
    if(strict) abort();
}
