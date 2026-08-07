// dscratch.cu — decode scratch arena globals. See dscratch.h.
#include "dscratch.h"
bool   g_arena_on  = false;
char*  g_arena     = nullptr;
size_t g_arena_off = 0;
size_t g_arena_cap = 0;
cudaStream_t g_side       = nullptr;
cudaEvent_t  g_side_fork  = nullptr;
cudaEvent_t  g_side_join  = nullptr;
cudaStream_t g_side2      = nullptr;
cudaEvent_t  g_side2_fork = nullptr;
cudaEvent_t  g_side2_join = nullptr;
void arena_init(size_t cap){
    if(g_arena){ cudaFree(g_arena); }
    cudaMalloc((void**)&g_arena, cap); g_arena_cap = cap; g_arena_off = 0; g_arena_on = true;
    // Created here, once, because arena_init runs long before any cudaStreamBeginCapture and
    // creating a stream or an event during capture is illegal. cudaEventDisableTiming is required:
    // a timing-enabled event used purely for ordering forces a host-visible timestamp and cannot be
    // captured into a graph.
    if(!g_side){
        if(cudaStreamCreateWithFlags(&g_side, cudaStreamNonBlocking) != cudaSuccess) g_side = nullptr;
        if(cudaEventCreateWithFlags(&g_side_fork, cudaEventDisableTiming) != cudaSuccess ||
           cudaEventCreateWithFlags(&g_side_join, cudaEventDisableTiming) != cudaSuccess){
            g_side = nullptr;                       // no events -> no safe fork/join; stay serial
        }
    }
    if(!g_side2){
        if(cudaStreamCreateWithFlags(&g_side2, cudaStreamNonBlocking) != cudaSuccess) g_side2 = nullptr;
        if(cudaEventCreateWithFlags(&g_side2_fork, cudaEventDisableTiming) != cudaSuccess ||
           cudaEventCreateWithFlags(&g_side2_join, cudaEventDisableTiming) != cudaSuccess){
            g_side2 = nullptr;
        }
    }
}
void arena_reset(){ g_arena_off = 0; }

// See the comment on dsync() in dscratch.h. Drains the thread's last-error slot so a launch failure
// is named where it happened instead of surfacing at the next unrelated CU(). Reports every time,
// because a rate limit is how the first one gets lost; aborts only under DSV4_STRICT_LAUNCH=1.
// See the comment on dprobe() in dscratch.h. Reports the LAUNCH SITE, which is the thing three
// cycles of chasing indexer.cu:91/:96 never had.
void dprobe_at(cudaStream_t s, const char* file, int line){
    static const bool probe = getenv("DSV4_SYNCPROBE") != nullptr;
    if(!probe) return;
    cudaError_t e = cudaStreamSynchronize(s);
    cudaError_t l = cudaGetLastError();
    if(e == cudaSuccess) e = l;
    if(e == cudaSuccess) return;
    fprintf(stderr, "\n[syncprobe] FAULT ATTRIBUTED TO %s:%d -> %s (%d)\n", file, line, cudaGetErrorString(e), (int)e);
    fprintf(stderr, "[syncprobe] this is the launch that faulted, not the next sync after it.\n");
    fflush(stderr);
    exit(7);
}

void dsync_at(cudaStream_t s, const char* file, int line){
    if(!g_arena_on) cudaStreamSynchronize(s);
    cudaError_t e = cudaGetLastError();
    if(e == cudaSuccess) return;
    static const bool strict = getenv("DSV4_STRICT_LAUNCH") != nullptr;
    fprintf(stderr, "[launch] %s:%d pending CUDA error: %s (%d)\n", file, line, cudaGetErrorString(e), (int)e);
    if(strict) abort();
}
