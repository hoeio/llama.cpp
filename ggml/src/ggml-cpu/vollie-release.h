// Vollie (GGML_REPACK_RELEASE=1): once a tensor has been repacked into its own buffer, the
// pages of the memory-mapped source are never read again, yet they stay in the page cache
// until memory pressure evicts them: on an 8 GB phone that is ~0.4 GB the MoE experts could
// use. Ask the kernel to reclaim the whole pages inside the source range (MADV_PAGEOUT,
// Linux 5.4+; MADV_COLD or MADV_DONTNEED as fallbacks). Data and results are unchanged.
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#ifndef MADV_COLD
#define MADV_COLD 20
#endif
#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif
static inline void ggml_cpu_release_source(const void * data, size_t size) {
    static size_t announced = 0; // bytes released so far, printed once per 256 MB
    const char * s = getenv("GGML_REPACK_RELEASE"); // read per tensor: an app toggles it between loads
    if (!s || atoi(s) != 1) {
        return;
    }
    const size_t before = announced;
    announced += size;
    if (before / (256u << 20) != announced / (256u << 20)) {
        fprintf(stderr, "repack-release: %zu MB released so far\n", announced >> 20);
    }
    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t b = ((uintptr_t) data + page - 1) & ~(page - 1);
    const uintptr_t e = ((uintptr_t) data + size) & ~(page - 1);
    if (e <= b) {
        return;
    }
#if defined(__linux__)
    if (madvise((void *) b, e - b, MADV_PAGEOUT) != 0 && madvise((void *) b, e - b, MADV_COLD) != 0) {
        madvise((void *) b, e - b, MADV_DONTNEED);
    }
#else
    madvise((void *) b, e - b, MADV_DONTNEED);
#endif
}
#else
static inline void ggml_cpu_release_source(const void * data, size_t size) { (void) data; (void) size; }
#endif
