#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"
#include "unary-ops.h"
#include "binary-ops.h"
#include "vec.h"
#include "ops.h"
#include "ggml.h"
#include "common.h"
#include "tiled/tiled.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#    include "spacemit/ime.h"
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

// precomputed f32 table for ue4m3 (1 KB) (simd-mappings.h)
float ggml_table_f32_ue4m3[1 << 8];

// Signed FP8 needs its own table: UE4M3 ignores the sign, halves values for NVFP4, and maps NaN to zero.
float ggml_table_f32_f8_e4m3[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/resource.h> // Vollie: getrusage, the profile's majflt / minflt
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_0] = {
        .from_float               = quantize_row_q2_0,
        .vec_dot                  = ggml_vec_dot_q2_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_F8_E4M3] = {
        .from_float               = quantize_row_f8_e4m3,
        .vec_dot                  = ggml_vec_dot_f8_e4m3_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64) && !defined(_M_ARM64EC))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

// Vollie: env GGML_HEAD2=<M> (0 / unset = off, stock path; NOT bit-exact, R28, reports/memos/head2stage_fix.md):
// two-stage lm_head. On the MUL_MAT whose src0 is the model's output head (output.weight, or token_embd.weight when
// the head is tied) and src1 is one token (ne11 == 1; a prompt's head is one token too, llama keeps only the last
// row), a low-bit copy of src0 scores every row into dst, the M best rows are recomputed in place with the Q6_K
// vec_dot on the same Q8_K activations, the rest keep the screen logit. Every other MUL_MAT, and the head at
// ne11 > 1, takes the stock path untouched. The exact rows equal stock's to the bit when the head is not repacked
// (useExtraBuffers off); a repacked head (CPU_REPACK q6_K_8x4 / 8x8 gemv) sums the same products in another order,
// so there they match it to float rounding, and are read from the tied token_embd rows in the file mmap.
// GGML_HEAD2_Q = q2_k (default) | q3_k | q4_0: the copy's type. It is made once, at the first such op (and again if
// the head or the type changes), into anonymous memory; the served file is untouched. Cost for Granite-4.0-H-Tiny's
// 100,352 x 1,536 Q6_K head (126.4 MB): Q2_K 50.6 MB, Q3_K 66.2 MB, Q4_0 86.7 MB resident; the load time is printed
// once under GGML_CPU_PROF ("cpu-prof head2 requant ...").
// Vollie r8: q4_0 also repacks the copy into the 4-row interleave the stock Q4_0 weights get with extra buffers (A78:
// 4x4 dotprod gemv, M4: 4x8 i8mm; repack.cpp ggml_cpu_head2_q4_0_*); same size, layout only. q4_0_plain (test only)
// keeps r7's row-major Q4_0 copy with the vec_dot screen, for the repacked-vs-plain logits check.
// GGML_HEAD2_BATCH=1 (test only, for llama-perplexity): the same per column at ne11 > 1, so the full-set gate can be
// run at -b 512; each column gets exactly what a one-token graph would compute from that hidden state.
static bool    ggml_cpu_prof_on; // Vollie: defined with its knob below
static int64_t ggml_cpu_head2_m = 0;
static enum ggml_type ggml_cpu_head2_type = GGML_TYPE_Q2_K;
static bool    ggml_cpu_head2_batch = false;
static bool    ggml_cpu_head2_prep  = false; // GGML_CPU_AB=knob:HEAD2:..: make the copy on the prompt graph, use it per side
static const struct ggml_tensor * ggml_cpu_head2_plain = NULL; // this graph's token_embd (GET_ROWS src0), see below
static const void * ggml_cpu_head2_key = NULL; // the data the copy was made from
static enum ggml_type ggml_cpu_head2_qtype = GGML_TYPE_COUNT;
static bool    ggml_cpu_head2_rp = false; // GGML_HEAD2_Q=q4_0: repack the copy (when this CPU has the 4-row gemv)
static int     ggml_cpu_head2_il = 0;     // the copy's interleave: 0 = plain rows, 4 / 8 = block_q4_0x4 groups
int  ggml_cpu_head2_q4_0_il(void);                                                              // repack.cpp
void ggml_cpu_head2_q4_0_pack4(void * dst, const void * src, int64_t nb, int il);                 // repack.cpp
void ggml_cpu_head2_q4_0_gemv(int il, int n, float * s, const void * vx, const void * vy, int nc); // repack.cpp
static char *  ggml_cpu_head2_q   = NULL; // the copy, rows of ggml_row_size(type, ne00)
static char *  ggml_cpu_head2_act = NULL; // per column: Q8 activations for src0's vec_dot, then the screen's if different
static size_t  ggml_cpu_head2_act_cap = 0;
static int32_t * ggml_cpu_head2_idx = NULL; // per column: the M row ids recomputed
static size_t  ggml_cpu_head2_idx_cap = 0;
static int32_t * ggml_cpu_head2_hist = NULL; // the top-M select: per thread 4096 bins; candidates (value, id); picks
static float   * ggml_cpu_head2_cv   = NULL;
static int32_t * ggml_cpu_head2_cid  = NULL, * ggml_cpu_head2_tmp = NULL;
static int64_t ggml_cpu_head2_sel_rows = 0;
static int     ggml_cpu_head2_sel_nth  = 0;
// Vollie: env GGML_HEAD2_LOWRANK=<r> (C2, round14_plan.md Part 2; off unless set, "" = unset; a DECLARED NUMERICS EXPERIMENT,
// needs GGML_HEAD2_Q=q4_0 on a CPU with the 4-row gemv): head2's screen becomes rank r. The tied head E ~= (U_r S_r) W_r^T:
// per column z = W_r^T x (F16 rows, ggml_vec_dot_f16 on the F16-cast hidden state), quantised to Q8_0, then the repacked
// Q4_0 gemv over the 100,352 x r matrix U_r S_r replaces the 100,352 x 1,536 Q4_0 screen. Top-M selection, the exact
// recompute of the M rows, and the screen logit for the rest are unchanged. The factors come from GGML_HEAD2_LOWRANK_FILE
// (lab/head2_lowrank.py's format; its key = the sum of the head's uint64 words, checked here); a missing or mismatched
// file prints one line and leaves the plain Q4_0 screen. ponytail: the runtime reads a host-made factorisation; the
// phone-side SVD (the owner's rule: computed by the runtime from the served file) is not written.
// GGML_HEAD2_LOWRANK_CHECK=1 (host test): after each one-column head, thread 0 computes all rows exactly and prints
// "cpu-prof head2 check in_set=<exact argmax among the M picked> overlap=<|exact top-M  picked| / M>" (any screen).
static int     ggml_cpu_head2_lr = 0;       // GGML_HEAD2_LOWRANK this graph
static int     ggml_cpu_head2_lr_made = 0;  // the r the copy was made with (0 = the full-width screen)
static int     ggml_cpu_head2_lr_tried = 0; // the r last tried for this copy (a failed file is not re-read every token)
static ggml_fp16_t * ggml_cpu_head2_vt = NULL; // W_r^T, r x ne00
static bool    ggml_cpu_head2_check = false;

// the low-rank screen from the file: U_r S_r into the repacked Q4_0 copy (4 rows at a time, the plain screen's path), W_r^T
// into F16. Thread 0 alone, between two barriers of ggml_cpu_head2. Returns false (nothing changed) on any mismatch.
static bool ggml_cpu_head2_lr_load(const struct ggml_tensor * src0, int r, int il) {
    const char * path = getenv("GGML_HEAD2_LOWRANK_FILE");
    FILE * fp = path && *path ? fopen(path, "rb") : NULL;
    char magic[4] = {0};
    int32_t fr = 0; int64_t rows = 0, cols = 0; uint64_t key = 0, nbytes = 0; float ms = 0;
    bool ok = fp && fread(magic, 4, 1, fp) == 1 && memcmp(magic, "VLR1", 4) == 0 && fread(&fr, 4, 1, fp) == 1 && fread(&rows, 8, 1, fp) == 1 &&
              fread(&cols, 8, 1, fp) == 1 && fread(&key, 8, 1, fp) == 1 && fread(&nbytes, 8, 1, fp) == 1 && fread(&ms, 4, 1, fp) == 1 &&
              fr == r && rows == src0->ne[1] && cols == src0->ne[0] && nbytes == ggml_nbytes(src0) && r % 32 == 0 && rows % 4 == 0;
    if (ok) {
        uint64_t k = 0;
        const uint64_t * w = (const uint64_t *) src0->data;
        for (size_t i = 0; i < nbytes/8; i++) {
            k += w[i];
        }
        ok = k == key;
    }
    const size_t qrow = ggml_row_size(GGML_TYPE_Q4_0, r);
    float * f = ok ? malloc(4*r*sizeof(float)) : NULL;
    char * t4 = ok ? malloc(4*qrow) : NULL;
    char * q = ok ? malloc(qrow*rows) : NULL;
    ggml_fp16_t * vt = ok ? malloc((size_t) r*cols*sizeof(ggml_fp16_t)) : NULL;
    ok = ok && f && t4 && q && vt;
    for (int64_t g = 0; ok && g < rows/4; g++) {
        ok = fread(f, sizeof(float), 4*r, fp) == (size_t) (4*r);
        for (int i = 0; ok && i < 4; i++) {
            ggml_quantize_chunk(GGML_TYPE_Q4_0, f + i*r, t4 + i*qrow, 0, 1, r, NULL);
        }
        if (ok) {
            ggml_cpu_head2_q4_0_pack4(q + 4*g*qrow, t4, r/QK4_0, il);
        }
    }
    for (int64_t j = 0; ok && j < r; j++) {
        float row[4096];
        ok = cols <= 4096 && fread(row, sizeof(float), cols, fp) == (size_t) cols;
        if (ok) {
            ggml_cpu_fp32_to_fp16(row, vt + j*cols, cols);
        }
    }
    if (fp) {
        fclose(fp);
    }
    free(f); free(t4);
    if (!ok) {
        free(q); free(vt);
        fprintf(stderr, "cpu-prof head2 lowrank r=%d file=%s missing or not this head's: plain q4_0 screen\n", r, path ? path : "(GGML_HEAD2_LOWRANK_FILE unset)");
        return false;
    }
    free(ggml_cpu_head2_q); free(ggml_cpu_head2_vt);
    ggml_cpu_head2_q  = q;
    ggml_cpu_head2_vt = vt;
    fprintf(stderr, "cpu-prof head2 lowrank r=%d screen_mb=%.1f (u %.1f + vt %.1f) svd_ms=%.0f cached=1\n", r,
            (qrow*rows + (size_t) r*cols*2)/1e6, qrow*rows/1e6, r*cols*2/1e6, (double) ms);
    return true;
}

// the top 12 bits of a float's order-preserving key (larger float, larger bin)
static inline int ggml_cpu_head2_bin(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return (int) (((u & 0x80000000u) ? ~u : u | 0x80000000u) >> 20);
}

void ggml_cpu_head2_topm(const float * x, int64_t n, int64_t m, int32_t * out); // ops.cpp

static bool ggml_cpu_is_head(const struct ggml_tensor * node) {
    const char * n = node->src[0]->name;
    return node->op == GGML_OP_MUL_MAT && (strcmp(n, "output.weight") == 0 || strcmp(n, "token_embd.weight") == 0);
}

// returns false (nothing done, every thread alike) when the op is not the head or the knob is off
static bool ggml_cpu_head2(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src1 = dst->src[1];
    if ((ggml_cpu_head2_m <= 0 && !ggml_cpu_head2_prep) || !ggml_cpu_is_head(dst)) {
        return false;
    }
    // the rows in their file layout: src0 itself, or, when src0 is a repacked copy (CPU_REPACK: interleaved rows,
    // no per-row vec_dot), the tied token_embd the graph's embedding lookup reads (same name, type and shape: the
    // same file bytes). ponytail: an untied head that is repacked has no plain rows here, so it keeps the stock path.
    const struct ggml_tensor * src0 = dst->src[0];
    if (src0->extra != NULL) {
        const struct ggml_tensor * p = ggml_cpu_head2_plain;
        src0 = p && p->extra == NULL && p->type == src0->type && ggml_are_same_shape(p, src0) && strcmp(p->name, src0->name) == 0 ? p : NULL;
        if (!src0) {
            return false;
        }
    }
    const enum ggml_type st = ggml_cpu_head2_type, et = src0->type;
    const int64_t M = ggml_cpu_head2_m, ne00 = src0->ne[0], nr = src0->ne[1], nc = src1->ne[1];
    if (params->use_ref || src1->type != GGML_TYPE_F32 ||
        et == st || (nc > 1 && !ggml_cpu_head2_batch) || src1->ne[2]*src1->ne[3] != 1 || src0->ne[2]*src0->ne[3] != 1 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || M >= nr || ne00 % ggml_blck_size(st) != 0 ||
        !ggml_get_type_traits(et)->to_float || !type_traits_cpu[type_traits_cpu[et].vec_dot_type].from_float) {
        return false;
    }
    const int ith = params->ith, nth = params->nth;
    const size_t qrow = ggml_row_size(st, ne00);
    const int il = st == GGML_TYPE_Q4_0 && ggml_cpu_head2_rp && nr % 4 == 0 && ne00 % 8 == 0 ? ggml_cpu_head2_q4_0_il() : 0;
    const int lr = il && ggml_cpu_head2_lr > 0 ? ggml_cpu_head2_lr : 0; // Vollie: GGML_HEAD2_LOWRANK (repacked q4_0 only)
    if (lr && ggml_cpu_head2_key == src0->data && ggml_cpu_head2_qtype == st && ggml_cpu_head2_il == il && ggml_cpu_head2_lr_tried != lr) {
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            ggml_cpu_head2_lr_tried = lr;
            if (ggml_cpu_head2_lr_load(src0, lr, il)) {
                ggml_cpu_head2_lr_made = lr;
            }
        }
        ggml_barrier(params->threadpool);
    }
    if (ggml_cpu_head2_key != src0->data || ggml_cpu_head2_qtype != st || ggml_cpu_head2_il != il ||
        (ggml_cpu_head2_lr_made && ggml_cpu_head2_lr_made != lr)) {
        // the copy: rows split across the threads, dequantised (to_float) and requantised without an imatrix
        const int64_t t0 = ggml_time_us();
        if (ith == 0) {
            free(ggml_cpu_head2_q);
            ggml_cpu_head2_q = malloc(qrow*nr);
            GGML_ASSERT(ggml_cpu_head2_q && "GGML_HEAD2: no memory for the head copy");
        }
        ggml_barrier(params->threadpool);
        float * f = malloc(ne00*sizeof(float));
        GGML_ASSERT(f);
        const ggml_to_float_t to_float = ggml_get_type_traits(et)->to_float;
        if (il) { // 4 rows at a time: quantise into a 4-row scratch, interleave into the copy (whole groups per thread)
            char * t4 = malloc(4*qrow);
            GGML_ASSERT(t4);
            for (int64_t g = ith*(nr/4)/nth; g < (ith + 1)*(nr/4)/nth; g++) {
                for (int i = 0; i < 4; i++) {
                    to_float((const char *) src0->data + (4*g + i)*src0->nb[1], f, ne00);
                    ggml_quantize_chunk(st, f, t4 + i*qrow, 0, 1, ne00, NULL);
                }
                ggml_cpu_head2_q4_0_pack4(ggml_cpu_head2_q + 4*g*qrow, t4, ne00/ggml_blck_size(st), il);
            }
            free(t4);
        } else {
            for (int64_t r = ith*nr/nth; r < (ith + 1)*nr/nth; r++) {
                to_float((const char *) src0->data + r*src0->nb[1], f, ne00);
                ggml_quantize_chunk(st, f, ggml_cpu_head2_q + r*qrow, 0, 1, ne00, NULL);
            }
        }
        free(f);
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            ggml_cpu_head2_key   = src0->data;
            ggml_cpu_head2_qtype = st;
            ggml_cpu_head2_il    = il;
            ggml_cpu_head2_lr_made = 0;
            ggml_cpu_head2_lr_tried = lr;
            if (lr && ggml_cpu_head2_lr_load(src0, lr, il)) {
                ggml_cpu_head2_lr_made = lr;
            }
            if (ggml_cpu_prof_on) {
                fprintf(stderr, "cpu-prof head2 requant %s -> %s%s rows=%lld threads=%d copy_mb=%.1f ms=%.1f\n", ggml_type_name(et),
                        ggml_type_name(st), il == 8 ? "x4x8" : il == 4 ? "x4x4" : "", (long long) nr, nth, qrow*nr/1e6, (ggml_time_us() - t0)/1e3);
            }
        }
    }
    if (lr) {
        ggml_barrier(params->threadpool); // Vollie: GGML_HEAD2_LOWRANK, thread 0 may have just swapped the screen in
    }
    if (M <= 0) {
        return false; // GGML_CPU_AB prep only: this side runs the stock head
    }
    const enum ggml_type evt = type_traits_cpu[et].vec_dot_type, svt = type_traits_cpu[st].vec_dot_type;
    const int lrm = ggml_cpu_head2_lr_made; // Vollie: the screen's width is r (low-rank) or ne00
    const int64_t sk = lrm ? lrm : ne00;
    const size_t erow = ggml_row_size(evt, ne00), srow = svt == evt && !lrm ? 0 : ggml_row_size(svt, sk);
    if (ith == 0) {
        if (ggml_cpu_head2_act_cap < nc*(erow + srow)) {
            free(ggml_cpu_head2_act);
            ggml_cpu_head2_act_cap = nc*(erow + srow);
            ggml_cpu_head2_act = malloc(ggml_cpu_head2_act_cap);
        }
        if (ggml_cpu_head2_idx_cap < (size_t) (nc*M)) {
            free(ggml_cpu_head2_idx);
            ggml_cpu_head2_idx_cap = nc*M;
            ggml_cpu_head2_idx = malloc(ggml_cpu_head2_idx_cap*sizeof(int32_t));
        }
        if (ggml_cpu_head2_sel_rows < nr || ggml_cpu_head2_sel_nth < nth) { // the select's scratch, sized by the head
            free(ggml_cpu_head2_hist); free(ggml_cpu_head2_cv); free(ggml_cpu_head2_cid); free(ggml_cpu_head2_tmp);
            ggml_cpu_head2_sel_rows = nr;
            ggml_cpu_head2_sel_nth  = nth;
            ggml_cpu_head2_hist = malloc(nth*4096*sizeof(int32_t));
            ggml_cpu_head2_cv   = malloc(nr*sizeof(float));
            ggml_cpu_head2_cid  = malloc(nr*sizeof(int32_t));
            ggml_cpu_head2_tmp  = malloc(nr*sizeof(int32_t));
        }
        GGML_ASSERT(ggml_cpu_head2_act && ggml_cpu_head2_idx && ggml_cpu_head2_hist && ggml_cpu_head2_cv && ggml_cpu_head2_cid && ggml_cpu_head2_tmp);
    }
    ggml_barrier(params->threadpool);
    // activations, whole rows (Q8_K / Q8_0 blocks are independent: the same bytes as the stock split by blocks)
    for (int64_t c = ith; c < nc; c += nth) {
        const float * x = (const float *) ((const char *) src1->data + c*src1->nb[1]);
        char * a = ggml_cpu_head2_act + c*(erow + srow);
        type_traits_cpu[evt].from_float(x, a, ne00);
        if (lrm) { // Vollie: GGML_HEAD2_LOWRANK, z = W_r^T x in F16 rows, then the screen's Q8_0
            ggml_fp16_t xh[4096];
            float z[4096];
            GGML_ASSERT(ne00 <= 4096 && lrm <= 4096);
            ggml_cpu_fp32_to_fp16(x, xh, ne00);
            for (int j = 0; j < lrm; j++) {
                ggml_vec_dot_f16((int) ne00, z + j, 0, ggml_cpu_head2_vt + (size_t) j*ne00, 0, xh, 0, 1);
            }
            type_traits_cpu[svt].from_float(z, a + erow, lrm);
        } else if (srow) {
            type_traits_cpu[svt].from_float(x, a + erow, ne00);
        }
    }
    ggml_barrier(params->threadpool);
    // stage 1: the screen over every row, rows split across the threads, blocks of 64 rows x every column
    const ggml_vec_dot_t sdot = type_traits_cpu[st].vec_dot, edot = type_traits_cpu[et].vec_dot;
    // (repacked: whole 4-row groups per thread, one gemv per 64 rows)
    const int64_t sr0 = il ? 4*(ith*(nr/4)/nth) : ith*nr/nth, sr1 = il ? 4*((ith + 1)*(nr/4)/nth) : (ith + 1)*nr/nth;
    for (int64_t r0 = sr0; r0 < sr1; r0 += 64) {
        for (int64_t c = 0; c < nc; c++) {
            float * d = (float *) ((char *) dst->data + c*dst->nb[1]);
            const char * a = ggml_cpu_head2_act + c*(erow + srow) + (srow ? erow : 0);
            if (il) {
                ggml_cpu_head2_q4_0_gemv(il, (int) sk, d + r0, ggml_cpu_head2_q + r0*ggml_row_size(st, sk), a, MIN(r0 + 64, sr1) - r0);
                continue;
            }
            for (int64_t r = r0; r < MIN(r0 + 64, sr1); r++) {
                sdot(ne00, d + r, 0, ggml_cpu_head2_q + r*qrow, 0, a, 0, 1);
            }
        }
    }
    ggml_barrier(params->threadpool);
    // the top M of every column (by value, ties to the lower id: the set a full nth_element gives), on all threads:
    // a 4096-bin histogram of the order-preserving float bits over each thread's own rows, the bin b where the count
    // from the top reaches M (every thread sums the histograms itself, no extra barrier), rows above b go straight
    // to idx, rows in b are candidates, and thread 0 picks the missing ones among those. One nth_element over all
    // 100,352 floats on one thread was 0.55 ms on the M4 host (~1.2-1.6 ms on an A78, estimated).
    int32_t * hist = ggml_cpu_head2_hist;
    const int64_t r0 = ith*nr/nth, r1 = (ith + 1)*nr/nth;
    for (int64_t c = 0; c < nc; c++) {
        const float * d = (const float *) ((const char *) dst->data + c*dst->nb[1]);
        int32_t * h = hist + ith*4096;
        memset(h, 0, 4096*sizeof(int32_t));
        for (int64_t r = r0; r < r1; r++) {
            h[ggml_cpu_head2_bin(d[r])]++;
        }
        ggml_barrier(params->threadpool);
        int64_t above = 0, b = 4095;
        for (;; b--) {
            int64_t n = 0;
            for (int t = 0; t < nth; t++) {
                n += hist[t*4096 + b];
            }
            if (above + n >= M || b == 0) {
                break;
            }
            above += n;
        }
        int64_t ka = 0, kb = 0; // this thread's offsets: rows above b, rows in b, of the threads before it
        for (int t = 0; t < ith; t++) {
            for (int64_t i = b + 1; i < 4096; i++) {
                ka += hist[t*4096 + i];
            }
            kb += hist[t*4096 + b];
        }
        int32_t * idx = ggml_cpu_head2_idx + c*M;
        for (int64_t r = r0; r < r1; r++) {
            const int bin = ggml_cpu_head2_bin(d[r]);
            if (bin > b) {
                idx[ka++] = (int32_t) r;
            } else if (bin == b) {
                ggml_cpu_head2_cv[kb] = d[r];
                ggml_cpu_head2_cid[kb++] = (int32_t) r;
            }
        }
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            int64_t nb = 0;
            for (int t = 0; t < nth; t++) {
                nb += hist[t*4096 + b];
            }
            const int64_t need = M - above;
            if (need < nb) {
                ggml_cpu_head2_topm(ggml_cpu_head2_cv, nb, need, ggml_cpu_head2_tmp); // ties: lower position = lower id
                for (int64_t i = 0; i < need; i++) {
                    idx[above + i] = ggml_cpu_head2_cid[ggml_cpu_head2_tmp[i]];
                }
            } else {
                memcpy(idx + above, ggml_cpu_head2_cid, need*sizeof(int32_t));
            }
        }
        ggml_barrier(params->threadpool);
    }
    // stage 2: the M rows of every column with src0's own kernel, split evenly across the threads
    for (int64_t k = ith*nc*M/nth; k < (ith + 1)*nc*M/nth; k++) {
        const int64_t c = k / M, r = ggml_cpu_head2_idx[k];
        edot(ne00, (float *) ((char *) dst->data + c*dst->nb[1]) + r, 0, (const char *) src0->data + r*src0->nb[1], 0,
             ggml_cpu_head2_act + c*(erow + srow), 0, 1);
    }
    if (ggml_cpu_head2_check && nc == 1) { // Vollie: GGML_HEAD2_LOWRANK_CHECK, host test
        ggml_barrier(params->threadpool);
        if (ith == 0) {
            float * ex = malloc(nr*sizeof(float));
            int32_t * top = malloc(M*sizeof(int32_t));
            GGML_ASSERT(ex && top);
            int64_t am = 0;
            for (int64_t r = 0; r < nr; r++) {
                edot(ne00, ex + r, 0, (const char *) src0->data + r*src0->nb[1], 0, ggml_cpu_head2_act, 0, 1);
                am = ex[r] > ex[am] ? r : am;
            }
            ggml_cpu_head2_topm(ex, nr, M, top);
            char * picked = calloc(nr, 1);
            GGML_ASSERT(picked);
            for (int64_t i = 0; i < M; i++) {
                picked[ggml_cpu_head2_idx[i]] = 1;
            }
            int64_t ov = 0;
            for (int64_t i = 0; i < M; i++) {
                ov += picked[top[i]];
            }
            fprintf(stderr, "cpu-prof head2 check lowrank=%d in_set=%d overlap=%.4f\n", lrm, picked[am], (double) ov/M);
            free(ex); free(top); free(picked);
        }
        ggml_barrier(params->threadpool);
    }
    return true;
}

// Vollie: env GGML_MM_CHUNK=<rows> (default 64 = stock; "" = unset; bit-identical, 05_compute_less.md §7.11 (i)): rows per
// chunk of a one-column MUL_MAT (nr1 == 1, decode) and of the one-token MUL_MAT_ID paths (stock: 64; GGML_MMID bit 1: 32,
// so there the rows are MIN(32, n)). A dst row is one full-K vec_dot on one thread whatever the chunk, so only the work
// split changes: finer chunks shorten the tail the other threads wait for at the barrier. Rounded down to a multiple
// of 4 (>= 4), which GGML_EXPERT_Q4_LAYERS' 4-row gemv needs.
static int ggml_cpu_mm_chunk = 64;
static int ggml_cpu_chunk_rows(int v) {
    return v >= 4 ? v & ~3 : 64;
}

// Vollie: env GGML_ROUTER_CAST=f16|q8_0 (C1, round14_plan.md Part 2; off unless set, "" = unset; a DECLARED NUMERICS
// EXPERIMENT, NOT bit-exact when on): every MoE router (F32 "*ffn_gate_inp*" src0 of a MUL_MAT) gets a converted copy in
// anonymous memory, made in ggml_graph_compute before the threads start (ggml_cpu_router_prep), and its MUL_MAT runs the
// stock F16 / Q8_0 path on the copy. The served file is untouched; unset = the stock F32 path.
#define GGML_CPU_ROUTER_MAX 128
static struct ggml_cpu_router_ent { const void * key; enum ggml_type type; void * q; } ggml_cpu_router[GGML_CPU_ROUTER_MAX];
static int ggml_cpu_router_n = 0;
static enum ggml_type ggml_cpu_router_type = GGML_TYPE_COUNT; // this graph's cast (COUNT = off)

static enum ggml_type ggml_cpu_router_env(void) { // read where needed: ggml_graph_plan runs before ggml_graph_compute
    const char * e = getenv("GGML_ROUTER_CAST");
    return !e || !*e ? GGML_TYPE_COUNT : strcmp(e, "f16") == 0 ? GGML_TYPE_F16 : strcmp(e, "q8_0") == 0 ? GGML_TYPE_Q8_0 : GGML_TYPE_COUNT;
}

// Vollie r15 (R46): GGML_CPU_AB=knob:ROUTER_CAST:0:1 (1 = the F16 copy, 0 = the stock F32 path; a DECLARED NUMERICS EXPERIMENT on
// side 1): every graph makes the F16 copies (the prompt graph first) and the plan sizes the router's work buffer for F16 on both
// sides (unused bytes on the F32 side), so a decode graph can take either path. ponytail: f16 only (q8_0 failed R43)
static bool ggml_cpu_router_ab(void) {
    const char * ab = getenv("GGML_CPU_AB");
    return ab && strncmp(ab, "knob:", 5) == 0 && strstr(ab, "ROUTER_CAST:") != NULL;
}

static enum ggml_type ggml_cpu_router_plan_type(void) {
    return ggml_cpu_router_ab() ? GGML_TYPE_F16 : ggml_cpu_router_env();
}

static bool ggml_cpu_is_router(const struct ggml_tensor * node) {
    const struct ggml_tensor * w = node->src[0];
    return node->op == GGML_OP_MUL_MAT && w->type == GGML_TYPE_F32 && w->extra == NULL && w->data && ggml_is_contiguous(w) &&
           strstr(w->name, "ffn_gate_inp") != NULL;
}

static void * ggml_cpu_router_find(const struct ggml_tensor * w, enum ggml_type t) {
    for (int i = 0; i < ggml_cpu_router_n; i++) {
        if (ggml_cpu_router[i].key == w->data && ggml_cpu_router[i].type == t) {
            return ggml_cpu_router[i].q;
        }
    }
    return NULL;
}

// Calling thread, before the graph's threads start: copies for this graph's routers that have none yet.
static void ggml_cpu_router_prep(const struct ggml_cgraph * cgraph, enum ggml_type t) {
    if (t == GGML_TYPE_COUNT) {
        return;
    }
    int made = 0;
    size_t b0 = 0, b1 = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * n = cgraph->nodes[i];
        if (!ggml_cpu_is_router(n) || ggml_cpu_router_find(n->src[0], t) || ggml_cpu_router_n == GGML_CPU_ROUTER_MAX ||
            n->src[0]->ne[0] % ggml_blck_size(t) != 0) {
            continue;
        }
        const struct ggml_tensor * w = n->src[0];
        const int64_t nrows = ggml_nrows(w);
        void * q = malloc(ggml_row_size(t, w->ne[0])*nrows);
        GGML_ASSERT(q && "GGML_ROUTER_CAST: no memory for the router copy");
        ggml_quantize_chunk(t, (const float *) w->data, q, 0, nrows, w->ne[0], NULL); // F16: a plain fp32 -> fp16 cast
        ggml_cpu_router[ggml_cpu_router_n++] = (struct ggml_cpu_router_ent) { w->data, t, q };
        made++;
        b0 += ggml_nbytes(w);
        b1 += ggml_row_size(t, w->ne[0])*nrows;
    }
    if (made) {
        fprintf(stderr, "cpu-prof router cast=%s tensors=%d mb=%.2f->%.2f\n", ggml_type_name(t), made, b0/1e6, b1/1e6);
        fflush(stderr);
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    if (ggml_cpu_router_type != GGML_TYPE_COUNT && ggml_cpu_is_router(dst)) { // Vollie: GGML_ROUTER_CAST
        void * q = ggml_cpu_router_find(dst->src[0], ggml_cpu_router_type);
        if (q) { // the same op on a view of the copy: stack copies, every thread its own
            struct ggml_tensor w = *dst->src[0], d = *dst;
            w.type  = ggml_cpu_router_type;
            w.data  = q;
            w.nb[0] = ggml_type_size(w.type);
            w.nb[1] = ggml_row_size(w.type, w.ne[0]);
            w.nb[2] = w.nb[1]*w.ne[1];
            w.nb[3] = w.nb[2]*w.ne[2];
            d.src[0] = &w;
            ggml_compute_forward_mul_mat(params, &d);
            return;
        }
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

    // If tiled is supported, it will execute the full op here and we return
    if (ggml_compute_forward_mul_mat_tiled(params, dst)) {
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type    const vec_dot_type     = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t const from_float       = type_traits_cpu[vec_dot_type].from_float;
    int64_t           const vec_dot_num_rows = type_traits_cpu[src0->type].nrows;

    // TODO: a more efficient F16 x F16 path can be used here if ggml_prec_get_src(dst, 1) allows converting src1 to F16
    //       the code needs to be refactored to enable this (src1 is currently converted based on vec_dot_type alone)

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (!params->use_ref && src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        // src1 is either packed from F32 into vec_dot_type, or widened from F16 or BF16 into the F32 work buffer
        const bool widen = src1->type != GGML_TYPE_F32;

        GGML_ASSERT(!widen || vec_dot_type == GGML_TYPE_F32);
        GGML_ASSERT(!widen || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        const int64_t bs = ggml_blck_size(vec_dot_type);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    const void * src1_row  = (const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10;
                    void       * wdata_row =                    wdata  + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0;

                    const int64_t ne10_block_size = (ne10_block_end - ne10_block_start) * bs;

                    if (src1->type == GGML_TYPE_F16) {
                        ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) src1_row, (float *) wdata_row, ne10_block_size);
                    } else if (src1->type == GGML_TYPE_BF16) {
                        ggml_cpu_bf16_to_fp32((const ggml_bf16_t *) src1_row, (float *) wdata_row, ne10_block_size);
                    } else {
                        from_float((const float *) src1_row, wdata_row, ne10_block_size);
                    }
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (!params->use_ref && src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = nr1 == 1 ? ggml_cpu_mm_chunk : 64; // Vollie: GGML_MM_CHUNK
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static int ggml_cpu_q3k_x4; // Vollie: defined with its knob below

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                int64_t ir0 = iir0;
#if defined(__aarch64__) && defined(__ARM_NEON)
                if (type == GGML_TYPE_Q3_K && ggml_cpu_q3k_x4) {
                    for (; ir0 + 4 <= iir0 + blck_0 && ir0 + 4 <= ir0_end; ir0 += 4) {
                        ggml_vec_dot_q3_K_q8_K_x4(ne00, &tmp[ir0 - iir0], src0_cur + ir0*nb01, nb01, src1_col);
                    }
                }
#endif
                for (; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

// ---- Vollie: MUL_MAT_ID additions (knobs documented where the variables are defined below) ----
static int ggml_cpu_mmid, ggml_cpu_mmid_touch, ggml_cpu_mmid_fuse, ggml_cpu_mmid_chunk2;
static bool ggml_cpu_mmid_test1;
static bool ggml_cpu_disable_fusion = false;  // initialized once in ggml_cpu_init(), read-only afterwards
static int ggml_cpu_q3k_x4;
static int ggml_cpu_expert_skip, ggml_cpu_expert_skip_renorm;
static int ggml_cpu_expert_topp, ggml_cpu_expert_gap, ggml_cpu_expert_mink;
static int64_t ggml_cpu_skip_cnt[2], ggml_cpu_skip_total[2];
#define GGML_CPU_SKIP_MAX 4096
#define GGML_CPU_MMID_MAX_AS 512

// Thread-local view of the expert-skip decision for the MUL_MAT_ID being computed: every
// thread derives it from the same (read-only) weights, so no thread writes shared state.
static _Thread_local uint8_t ggml_cpu_skip_mask_tl[GGML_CPU_SKIP_MAX]; // [token*n_ids + slot]
static _Thread_local float   ggml_cpu_skip_scale_tl[GGML_CPU_SKIP_MAX / 2]; // per token, renorm factor
static _Thread_local bool    ggml_cpu_skip_on_tl = false, ggml_cpu_skip_down_tl = false;
static _Thread_local const struct ggml_tensor * ggml_cpu_skip_weights_tl = NULL; // set by the compute loop
static _Thread_local int     ggml_cpu_mmid_seq_tl = 0; // MUL_MAT_ID ops run so far in this graph

// Dynamic chunk counters for the no-barrier path, one cache line per expert, two banks so
// thread 0 can reset the next op's bank before the barrier while slow threads finish the
// current op (a thread is never more than one MUL_MAT_ID behind thread 0).
static struct { atomic_int v; char pad[60]; } ggml_cpu_mmid_ctr[2][GGML_CPU_MMID_MAX_AS];

// Vollie: env GGML_EXPERT_ANON=<N> (0 / unset = off, nothing allocated; N <= 64): the N most used
// experts of every expert tensor (gate, up, down of each layer) are copied into anonymous memory,
// same bytes and layout, and MUL_MAT_ID reads them there instead of through the file mmap (no page
// faults, not evicted with the page cache); results are identical. Window-LFU per tensor: for each
// expert, how many of the last 64 tokens routed to it; after each MUL_MAT_ID (thread 0, behind the
// barrier that ends it) a missing expert used more often than the coldest cached one replaces it,
// at most 4 per decode token, up to N per prompt batch (so the prompt fills it). The cache lives for
// the process, keyed by tensor name (a model reload refills it); a different N frees it first.
// MADV_HUGEPAGE where the kernel has it. cpu-prof shows "anon=N/hit%/copies" (routed slots of
// decode tokens found cached, skipped slots included; tensor copies per decode token) and anon_mb.
#define GGML_CPU_ANON_MAXT 256
#define GGML_CPU_ANON_MAXE 64
#define GGML_CPU_ANON_WIN  64
struct ggml_cpu_anon_ent {
    char         name[GGML_MAX_NAME];
    const char * data;                          // the tensor's data the slots were copied from
    size_t       nb2;
    char       * buf;                           // N slots of nb2 bytes (NULL: mmap failed)
    int8_t       slot_of[GGML_CPU_ANON_MAXE];   // expert -> slot, -1 = not cached
    int8_t       slot_e[GGML_CPU_ANON_MAXE];    // slot -> expert, -1 = empty
    uint64_t     ring[GGML_CPU_ANON_WIN];       // experts each of the last 64 tokens used
    int          ring_pos;
    uint8_t      cnt[GGML_CPU_ANON_MAXE];       // tokens in the window that used the expert
};
static struct ggml_cpu_anon_ent ggml_cpu_anon[GGML_CPU_ANON_MAXT];
static int     ggml_cpu_anon_n = 0;          // entries in use (published with release order)
static int     ggml_cpu_expert_anon = 0;     // N asked for this graph
static int     ggml_cpu_anon_alloc_n = 0;    // N the buffers were made with
static bool    ggml_cpu_anon_use = false;    // this graph reads the copies (GGML_CPU_AB: per side)
static int64_t ggml_cpu_anon_hit = 0, ggml_cpu_anon_routed = 0, ggml_cpu_anon_copies = 0, ggml_cpu_anon_dtok = 0;
static size_t  ggml_cpu_anon_bytes = 0;

// Graph start (no worker running): a new N frees the old cache.
static void ggml_cpu_anon_config(int n) {
    n = MAX(0, MIN(n, GGML_CPU_ANON_MAXE));
    if (n == ggml_cpu_anon_alloc_n) {
        return;
    }
#if !defined(_WIN32)
    for (int i = 0; i < ggml_cpu_anon_n; i++) {
        if (ggml_cpu_anon[i].buf) {
            munmap(ggml_cpu_anon[i].buf, (size_t) ggml_cpu_anon_alloc_n*ggml_cpu_anon[i].nb2);
        }
    }
#endif
    ggml_cpu_anon_n = 0;
    ggml_cpu_anon_bytes = 0;
    ggml_cpu_anon_alloc_n = n;
}

static const struct ggml_cpu_anon_ent * ggml_cpu_anon_find(const struct ggml_tensor * w) {
    if (!ggml_cpu_anon_use) {
        return NULL;
    }
    const int n = __atomic_load_n(&ggml_cpu_anon_n, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++) {
        // ponytail: plain reads of data/slot_of; thread 0 only rewrites an entry behind the barrier
        // that ends its tensor's op, and that tensor is next read in a later graph
        if (ggml_cpu_anon[i].data == w->data && ggml_cpu_anon[i].buf) {
            return &ggml_cpu_anon[i];
        }
    }
    return NULL;
}

static inline const char * ggml_cpu_anon_src(const struct ggml_cpu_anon_ent * a, const struct ggml_tensor * w, int64_t e) {
    const int s = a ? a->slot_of[e] : -1;
    return s >= 0 ? a->buf + (size_t) s*a->nb2 : (const char *) w->data + e*w->nb[2];
}

// Thread 0, after the barrier that ends MUL_MAT_ID `node`: window counts, hit stats, promotions.
static void ggml_cpu_anon_update(const struct ggml_tensor * node) {
#if !defined(_WIN32)
    const struct ggml_tensor * w = node->src[0], * ids = node->src[2];
    const int N = MIN(ggml_cpu_anon_alloc_n, (int) w->ne[2]);
    if (N <= 0 || w->ne[2] > GGML_CPU_ANON_MAXE || w->data == NULL || ids->type != GGML_TYPE_I32) {
        return;
    }
    struct ggml_cpu_anon_ent * a = NULL;
    for (int i = 0; i < ggml_cpu_anon_n && !a; i++) {
        if (ggml_cpu_anon[i].nb2 == w->nb[2] && strcmp(ggml_cpu_anon[i].name, w->name) == 0) {
            a = &ggml_cpu_anon[i];
        }
    }
    if (!a) {
        if (ggml_cpu_anon_n == GGML_CPU_ANON_MAXT) {
            return;
        }
        a = &ggml_cpu_anon[ggml_cpu_anon_n];
        memset(a, 0, sizeof *a);
        snprintf(a->name, sizeof a->name, "%s", w->name);
        a->nb2 = w->nb[2];
        const size_t size = (size_t) ggml_cpu_anon_alloc_n*a->nb2;
        void * p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
#ifdef MADV_HUGEPAGE
            madvise(p, size, MADV_HUGEPAGE);
#endif
            a->buf = p;
            ggml_cpu_anon_bytes += size;
        }
        memset(a->slot_of, -1, sizeof a->slot_of);
        memset(a->slot_e,  -1, sizeof a->slot_e);
        __atomic_store_n(&ggml_cpu_anon_n, ggml_cpu_anon_n + 1, __ATOMIC_RELEASE);
    }
    if (!a->buf) {
        return;
    }
    if (a->data != w->data) { // first use, or the model was reloaded: start empty
        memset(a->slot_of, -1, sizeof a->slot_of);
        memset(a->slot_e,  -1, sizeof a->slot_e);
        memset(a->ring, 0, sizeof a->ring);
        memset(a->cnt,  0, sizeof a->cnt);
        a->ring_pos = 0;
        a->data = w->data;
    }
    const int64_t n_as = w->ne[2], n_tok = ids->ne[1];
    const bool dec = n_tok == 1;
    for (int64_t t = 0; t < n_tok; t++) {
        uint64_t mask = 0;
        for (int64_t k = 0; k < ids->ne[0]; k++) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + k*ids->nb[0]);
            if (e >= 0 && e < n_as) {
                mask |= 1ull << e;
                if (dec) {
                    ggml_cpu_anon_hit    += a->slot_of[e] >= 0;
                    ggml_cpu_anon_routed += 1;
                }
            }
        }
        const uint64_t old = a->ring[a->ring_pos];
        for (int e = 0; e < n_as; e++) {
            a->cnt[e] += (int) (mask >> e & 1) - (int) (old >> e & 1);
        }
        a->ring[a->ring_pos] = mask;
        a->ring_pos = (a->ring_pos + 1) % GGML_CPU_ANON_WIN;
    }
    for (int64_t budget = dec ? 4 : MIN(N, 4*n_tok); budget > 0; budget--) {
        int best = -1, vs = -1;
        for (int e = 0; e < n_as; e++) {
            if (a->slot_of[e] < 0 && a->cnt[e] && (best < 0 || a->cnt[e] > a->cnt[best])) {
                best = e;
            }
        }
        for (int s = 0; s < N && best >= 0; s++) { // an empty slot, else the coldest
            if (a->slot_e[s] < 0) {
                vs = s;
                break;
            }
            if (vs < 0 || a->cnt[a->slot_e[s]] < a->cnt[a->slot_e[vs]]) {
                vs = s;
            }
        }
        if (best < 0 || (a->slot_e[vs] >= 0 && a->cnt[best] <= a->cnt[a->slot_e[vs]])) {
            break;
        }
        if (a->slot_e[vs] >= 0) {
            a->slot_of[a->slot_e[vs]] = -1;
        }
        memcpy(a->buf + (size_t) vs*a->nb2, a->data + (size_t) best*a->nb2, a->nb2);
        a->slot_e[vs]   = (int8_t) best;
        a->slot_of[best] = (int8_t) vs;
        ggml_cpu_anon_copies += dec;
    }
#else
    GGML_UNUSED(node);
#endif
}

// Vollie: env GGML_EXPERT_Q4_LAYERS=<n> (0 / unset = off, nothing allocated; NOT bit-exact, budgeted: 05_compute_less.md
// §7.11 (ii)): the routed experts (gate / up / down *_exps; Q3_K in the served file) of the n first layers of
// ggml_cpu_expq4_rank get an anonymous copy, dequantised and requantised to Q4_0 without an imatrix (quantize_row_q4_0_ref,
// as lab/tools/exps_requant.py) and repacked 4 rows at a time into block_q4_0x4 (GGML_HEAD2_Q=q4_0's layout and helpers,
// repack.cpp). MUL_MAT_ID of those tensors then quantises its activations to Q8_0 and runs the 4-row sdot gemv (R31: the
// byte-rate kernel class on the A78) instead of the Q3_K vec_dot, at one token and in prompt batches alike, so a -b 512
// perplexity run measures the decode numerics. Each tensor's copy is made at its first MUL_MAT_ID by all threads (the app
// runs a warm-up prompt at load), freed when n goes to 0 or changes; "cpu-prof expq4 requant ..." once per graph that made
// any (GGML_CPU_PROF). Every other layer, and every layer at n = 0, is the stock path to the bit. Granite-4.0-H-Tiny: 85 MB
// per layer. GGML_CPU_AB=knob:EXPERT_Q4_LAYERS:A:B makes max(A, B) layers' copies (on the prompt graph), a side reads n.
// Ranked by kept expert slots per token at GGML_EXPERT_SKIP=117 (the "expert-budget layers" exit line; host, wikitext-2
// 40 chunks, lab/logs/r32/rank). GGML_EXPERT_Q4_LIST=<l,l,...> ("" = unset) replaces that order (another model, or
// another choice of layers); n still takes its first n. ponytail: the compiled list is one model's; a routing-profile pass
// at load would make the default general.
static int ggml_cpu_expq4_list[128], ggml_cpu_expq4_list_n = 0; // GGML_EXPERT_Q4_LIST, parsed at graph start
static const int ggml_cpu_expq4_rank[] = { 39, 38, 9, 31, 33, 32, 15, 28, 29, 21, 20, 37, 23, 17, 24, 35, 19, 22, 10, 27, 8, 11, 16, 26, 18, 34, 30, 25, 36, 12, 7, 14, 2, 6, 1, 13, 0, 4, 3, 5 };
#define GGML_CPU_EXPQ4_MAX 192
struct ggml_cpu_expq4_ent {
    const void * data; // the tensor's data the copy was made from
    char       * q;    // n_expert x ne01 rows of Q4_0, 4-row groups interleaved
    size_t       esz;  // bytes per expert
    int          il;   // 4 or 8 (block_q4_0x4 interleave)
};
static struct ggml_cpu_expq4_ent ggml_cpu_expq4[GGML_CPU_EXPQ4_MAX];
static int     ggml_cpu_expq4_n = 0;      // copies made (entries)
static int     ggml_cpu_expq4_make = 0;   // layers that get a copy this graph (knob, or max(A, B))
static int     ggml_cpu_expq4_use = 0;    // layers that read it (this graph's side)
static int     ggml_cpu_expq4_made = 0;   // the make value the entries belong to
static char *  ggml_cpu_expq4_pending = NULL;
static size_t  ggml_cpu_expq4_bytes = 0;
static int64_t ggml_cpu_expq4_us = 0;     // requant + repack time of this graph's new copies (thread 0)
static int     ggml_cpu_expq4_new = 0;    // copies made in this graph

static int ggml_cpu_expq4_nrank(void) {
    return ggml_cpu_expq4_list_n ? ggml_cpu_expq4_list_n : (int) (sizeof ggml_cpu_expq4_rank / sizeof ggml_cpu_expq4_rank[0]);
}
static int ggml_cpu_expq4_layer(int r) {
    return ggml_cpu_expq4_list_n ? ggml_cpu_expq4_list[r] : ggml_cpu_expq4_rank[r];
}

// rank of w's layer in the list if w is a routed-expert tensor, else -1
static int ggml_cpu_expq4_rank_of(const struct ggml_tensor * w) {
    int layer = -1;
    if (!strstr(w->name, "_exps") || sscanf(w->name, "blk.%d.", &layer) != 1) {
        return -1;
    }
    for (int r = 0; r < ggml_cpu_expq4_nrank(); r++) {
        if (ggml_cpu_expq4_layer(r) == layer) {
            return r;
        }
    }
    return -1;
}

// thread 0 (expert prefetch): w's copy if this side reads one
static const struct ggml_cpu_expq4_ent * ggml_cpu_expq4_find(const struct ggml_tensor * w) {
    const int r = ggml_cpu_expq4_use > 0 ? ggml_cpu_expq4_rank_of(w) : -1;
    for (int i = 0; r >= 0 && r < ggml_cpu_expq4_use && i < ggml_cpu_expq4_n; i++) {
        if (ggml_cpu_expq4[i].data == w->data) {
            return &ggml_cpu_expq4[i];
        }
    }
    return NULL;
}

// Graph start (no worker running): a new layer count or list frees the copies.
static void ggml_cpu_expq4_config(const char * list) {
    static char last[512] = "";
    bool changed = false;
    if (strcmp(list ? list : "", last) != 0) {
        snprintf(last, sizeof last, "%s", list ? list : "");
        ggml_cpu_expq4_list_n = 0;
        for (const char * p = last; *p && ggml_cpu_expq4_list_n < 128; ) {
            char * e;
            const long l = strtol(p, &e, 10);
            if (e == p) {
                break;
            }
            ggml_cpu_expq4_list[ggml_cpu_expq4_list_n++] = (int) l;
            p = *e == ',' ? e + 1 : e;
        }
        changed = true;
    }
    if (ggml_cpu_expq4_make != ggml_cpu_expq4_made || changed) {
        for (int i = 0; i < ggml_cpu_expq4_n; i++) {
            free(ggml_cpu_expq4[i].q);
        }
        ggml_cpu_expq4_n = 0;
        ggml_cpu_expq4_bytes = 0;
        ggml_cpu_expq4_made = ggml_cpu_expq4_make;
    }
    ggml_cpu_expq4_us = 0;
    ggml_cpu_expq4_new = 0;
}

// Every thread, at the top of MUL_MAT_ID (same decision on all of them): w's copy, made here if missing; NULL = the stock
// kernels. The table only grows inside this op, behind its first barrier, and is read again only after the op's last.
static const struct ggml_cpu_expq4_ent * ggml_cpu_expq4_get(const struct ggml_compute_params * params, const struct ggml_tensor * dst) {
    const struct ggml_tensor * w = dst->src[0], * src1 = dst->src[1];
    const int r = ggml_cpu_expq4_make > 0 ? ggml_cpu_expq4_rank_of(w) : -1;
    if (r < 0 || r >= ggml_cpu_expq4_make || params->use_ref || ggml_cpu_expert_skip_renorm) {
        return NULL;
    }
    const int n = ggml_cpu_expq4_n;
    for (int i = 0; i < n; i++) {
        if (ggml_cpu_expq4[i].data == w->data) {
            return r < ggml_cpu_expq4_use ? &ggml_cpu_expq4[i] : NULL;
        }
    }
    const int il = ggml_cpu_head2_q4_0_il();
    const int64_t ne00 = w->ne[0], ne01 = w->ne[1], ne02 = w->ne[2];
    const ggml_to_float_t to_float = ggml_get_type_traits(w->type)->to_float;
    if (!il || n == GGML_CPU_EXPQ4_MAX || w->type == GGML_TYPE_Q4_0 || !to_float || !ggml_is_contiguous(w) || w->ne[3] != 1 ||
        ne00 % QK4_0 != 0 || ne01 % 4 != 0 || src1->type != GGML_TYPE_F32 ||
        ggml_row_size(GGML_TYPE_Q8_0, ne00) > ggml_row_size(type_traits_cpu[w->type].vec_dot_type, ne00)) {
        return NULL; // the activations are requantised into the stock path's own work buffer
    }
    const int ith = params->ith, nth = params->nth;
    const size_t qrow = ggml_row_size(GGML_TYPE_Q4_0, ne00), esz = qrow*ne01;
    const int64_t t0 = ggml_time_us();
    if (ith == 0) {
        ggml_cpu_expq4_pending = malloc(esz*ne02);
        GGML_ASSERT(ggml_cpu_expq4_pending && "GGML_EXPERT_Q4_LAYERS: no memory for the expert copy");
    }
    ggml_barrier(params->threadpool);
    char * q = ggml_cpu_expq4_pending;
    float * f = malloc(ne00*sizeof(float));
    char * t4 = malloc(4*qrow);
    GGML_ASSERT(f && t4);
    const int64_t ng = ne02*(ne01/4); // 4-row groups, whole groups per thread
    for (int64_t g = ith*ng/nth; g < (ith + 1)*ng/nth; g++) {
        const int64_t e = g/(ne01/4), r0 = 4*(g % (ne01/4));
        for (int i = 0; i < 4; i++) {
            to_float((const char *) w->data + e*w->nb[2] + (r0 + i)*w->nb[1], f, ne00);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, f, t4 + i*qrow, 0, 1, ne00, NULL); // = quantize_row_q4_0_ref
        }
        ggml_cpu_head2_q4_0_pack4(q + e*esz + r0*qrow, t4, ne00/QK4_0, il);
    }
    free(f);
    free(t4);
    if (ith == 0) {
        ggml_cpu_expq4[n] = (struct ggml_cpu_expq4_ent) { w->data, q, esz, il };
    }
    ggml_barrier(params->threadpool);
    if (ith == 0) {
        ggml_cpu_expq4_n = n + 1;
        ggml_cpu_expq4_bytes += esz*ne02;
        ggml_cpu_expq4_us += ggml_time_us() - t0;
        ggml_cpu_expq4_new++;
    }
    return r < ggml_cpu_expq4_use ? &ggml_cpu_expq4[n] : NULL;
}

static bool ggml_cpu_skip_prepare(const struct ggml_tensor * dst) {
    const struct ggml_tensor * ids = dst->src[2];
    const struct ggml_tensor * w   = ggml_cpu_skip_weights_tl;
    const int     n_ids = ids->ne[0];
    const int64_t n_tok = ids->ne[1];
    ggml_cpu_skip_on_tl = false;
    if (!(ggml_cpu_expert_skip || ggml_cpu_expert_topp || ggml_cpu_expert_gap) || !w || n_ids > 64 || n_ids*n_tok > GGML_CPU_SKIP_MAX ||
        w->type != GGML_TYPE_F32 || w->ne[0] != 1 || w->ne[1] != n_ids || w->ne[2] != n_tok) {
        return false;
    }
    const float t = ggml_cpu_expert_skip / 1000.0f;
    const float p = ggml_cpu_expert_topp ? ggml_cpu_expert_topp / 1000.0f : 2.0f; // 2: never reached
    const float g = ggml_cpu_expert_gap / 1000.0f;
    const bool  ranked = ggml_cpu_expert_topp || ggml_cpu_expert_gap;
    for (int64_t tok = 0; tok < n_tok; tok++) {
        float v[64];
        int   ord[64]; // slots by weight, descending (insertion sort: stable, same on every thread)
        for (int s = 0; s < n_ids; s++) {
            v[s] = *(const float *) ((const char *) w->data + tok*w->nb[2] + s*w->nb[1]);
            int j = s;
            for (; j > 0 && v[ord[j - 1]] < v[s]; j--) {
                ord[j] = ord[j - 1];
            }
            ord[j] = s;
        }
        float dropped = 0.0f, cum = 0.0f; // cum: weight of the higher-ranked slots
        for (int r = 0; r < n_ids; r++) {
            const int s = ord[r];
            // SKIP: plain threshold (unchanged); TOPP: past the shortest prefix reaching mass p;
            // GAP: under g x the top weight. The two ranked rules keep at least MINK slots.
            const bool skip = v[s] < t || (ranked && r >= ggml_cpu_expert_mink && (cum >= p || v[s] < g*v[ord[0]]));
            cum += v[s];
            ggml_cpu_skip_mask_tl[tok*n_ids + s] = skip;
        }
        for (int s = 0; s < n_ids; s++) { // slot order, as before (renorm stays bit-identical)
            if (ggml_cpu_skip_mask_tl[tok*n_ids + s]) {
                dropped += v[s];
            }
        }
        ggml_cpu_skip_scale_tl[tok] = (ggml_cpu_expert_skip_renorm && dropped > 0.0f && dropped < 1.0f) ? 1.0f/(1.0f - dropped) : 1.0f;
    }
    ggml_cpu_skip_on_tl   = true;
    ggml_cpu_skip_down_tl = dst->src[1]->ne[1] == n_ids; // the down projection: its input has one row per slot
    return true;
}

// Builds counts[n_as] and the MMID_MATRIX_ROW map from ids, leaving out the (token, slot)
// pairs the skip mask marks; if zero_skipped, their dst rows are set to zero (what the
// mixture multiplies by the routing weight). Same layout and order as the stock loop.
static void ggml_cpu_mmid_build_map(const struct ggml_tensor * dst, const struct ggml_tensor * ids, int64_t n_as,
                                    int64_t * counts, struct mmid_row_mapping * matrix_rows, bool zero_skipped) {
    const int n_ids = ids->ne[0];
    memset(counts, 0, n_as*sizeof(int64_t));
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
            assert(i02 == -1 || (i02 >= 0 && i02 < n_as));
            // an id of -1 (upstream #26631) is handled like a skipped slot: no compute, zero row
            if (i02 == -1 || (ggml_cpu_skip_on_tl && ggml_cpu_skip_mask_tl[iid1*n_ids + id])) {
                if (zero_skipped) {
                    memset((char *) dst->data + id*dst->nb[1] + iid1*dst->nb[2], 0, dst->ne[0]*sizeof(float));
                }
                continue;
            }
            MMID_MATRIX_ROW(i02, counts[i02]) = (struct mmid_row_mapping) {id, iid1};
            counts[i02] += 1;
        }
    }
}

// One chunk, then the renormalisation of the down projection's rows when experts were skipped.
static void ggml_cpu_mmid_chunk(
    struct ggml_tensor * dst, const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * ids,
    int64_t cur_a, int64_t ir0_start, int64_t ir0_end, int64_t ir1_start, int64_t ir1_end,
    const char * src0_cur, const struct mmid_row_mapping * matrix_rows, size_t row_size, bool src1_cont, const void * wdata) {
    ggml_compute_forward_mul_mat_id_one_chunk(dst, src0, src1, ids, cur_a, ir0_start, ir0_end, ir1_start, ir1_end,
                                              src0_cur, matrix_rows, row_size, src1_cont, wdata);
    if (ggml_cpu_skip_on_tl && ggml_cpu_skip_down_tl && ggml_cpu_expert_skip_renorm) {
        for (int64_t ir1 = ir1_start; ir1 < ir1_end; ++ir1) {
            const struct mmid_row_mapping m = MMID_MATRIX_ROW(cur_a, ir1);
            const float scale = ggml_cpu_skip_scale_tl[m.i2];
            if (scale != 1.0f) {
                float * row = (float *) ((char *) dst->data + m.i1*dst->nb[1] + m.i2*dst->nb[2]);
                for (int64_t i = ir0_start; i < ir0_end; i++) {
                    row[i] *= scale;
                }
            }
        }
    }
}

// Vollie: GGML_EXPERT_Q4_LAYERS, one chunk (rows ir0_start..ir0_end, a multiple of 4 from a multiple of 4) of expert
// cur_a for its routed rows ir1: the 4-row Q4_0 gemv on the Q8_0 activation row (wdata: rows of row_size, as the stock path)
static void ggml_cpu_expq4_chunk(const struct ggml_cpu_expq4_ent * q4, struct ggml_tensor * dst, const struct ggml_tensor * ids,
                                 int64_t cur_a, int64_t ir0_start, int64_t ir0_end, int64_t ir1_start, int64_t ir1_end,
                                 const struct mmid_row_mapping * matrix_rows, size_t row_size, const char * wdata) {
    const int64_t ne00 = dst->src[0]->ne[0], ne11 = dst->src[1]->ne[1];
    const size_t  qrow = ggml_row_size(GGML_TYPE_Q4_0, ne00);
    for (int64_t ir1 = ir1_start; ir1 < ir1_end; ++ir1) {
        const struct mmid_row_mapping m = MMID_MATRIX_ROW(cur_a, ir1);
        float * dst_col = (float *) ((char *) dst->data + m.i1*dst->nb[1] + m.i2*dst->nb[2]);
        ggml_cpu_head2_q4_0_gemv(q4->il, (int) ne00, dst_col + ir0_start, q4->q + cur_a*q4->esz + ir0_start*qrow,
                                 wdata + (m.i1 % ne11 + m.i2*ne11)*row_size, (int) (ir0_end - ir0_start));
    }
}

// Vollie: the no-barrier path's next chunk range [*c0, *c1) of one expert for this thread; false when none is left. *pos starts
// at 0 (dynamic: the counter is known to be at least *pos) or ith (static round-robin). GGML_MMID_CHUNK2: two chunks per
// fetch_add while >= 2*nth remain, one at the tail. Which thread runs a chunk does not change any result.
// ponytail: "remain" is judged from this thread's own last claim (no extra load of the contended line), so it can over-count by
// the other threads' claims since then and take one pair inside the tail; a fresh relaxed load first would cost a second line transfer
static inline bool ggml_cpu_mmid_claim(bool dyn, atomic_int * ctr, int64_t nchunk, int nth, int64_t * pos, int64_t * c0, int64_t * c1) {
    if (dyn) {
        const int k = ggml_cpu_mmid_chunk2 && nchunk - *pos >= 2*nth ? 2 : 1;
        *c0  = atomic_fetch_add_explicit(ctr, k, memory_order_relaxed);
        *pos = *c0 + k;
        *c1  = MIN(*pos, nchunk);
    } else {
        *c0  = *pos;
        *c1  = *c0 + 1;
        *pos += nth;
    }
    return *c0 < nchunk;
}

// Where the no-barrier path keeps thread ith's quantized activation (GGML_MMID bit 1).
static size_t ggml_cpu_mmid_row_pad(const struct ggml_tensor * dst) {
    const enum ggml_type vec_dot_type = type_traits_cpu[dst->src[0]->type].vec_dot_type;
    return GGML_PAD(ggml_row_size(vec_dot_type, dst->src[1]->ne[0]), 64);
}

static bool ggml_cpu_mmid_fast_ok(const struct ggml_tensor * dst, int nth, size_t wsize) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];
    const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
    return (ggml_cpu_mmid & 1) && (nth > 1 || ggml_cpu_mmid_test1) &&
           ids->ne[1] == 1 && src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           src0->ne[2] <= GGML_CPU_MMID_MAX_AS && src0->ne[2]*ids->ne[0] <= GGML_CPU_SKIP_MAX &&
           src1->type == GGML_TYPE_F32 && src1->type != vec_dot_type && ggml_is_contiguous(src1) &&
           wsize >= (size_t) nth*ggml_cpu_mmid_row_pad(dst);
}

// One-token MUL_MAT_ID without the internal barrier. Results are identical to the stock path:
// the same from_float on the same row, the same vec_dot per output row.
static void ggml_cpu_mmid_decode(const struct ggml_compute_params * params, struct ggml_tensor * dst,
                                 const struct ggml_cpu_expq4_ent * q4) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type    type         = src0->type;
    const enum ggml_type    vec_dot_type = q4 ? GGML_TYPE_Q8_0 : type_traits_cpu[type].vec_dot_type; // Vollie: EXPERT_Q4_LAYERS
    ggml_from_float_t const from_float   = type_traits_cpu[vec_dot_type].from_float;

    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], n_as = src0->ne[2];
    const size_t  row_size = ggml_row_size(vec_dot_type, ne00);

    char * my_q = (char *) params->wdata + ith*ggml_cpu_mmid_row_pad(dst);
    from_float((const float *) src1->data, my_q, ne00);

    int64_t counts[GGML_CPU_MMID_MAX_AS];
    struct mmid_row_mapping rows_buf[GGML_CPU_SKIP_MAX];
    struct mmid_row_mapping * matrix_rows = rows_buf;
    ggml_cpu_mmid_build_map(dst, ids, n_as, counts, matrix_rows, ith == 0);

    const bool    dyn    = (ggml_cpu_mmid & 2) != 0;
    const int     bank   = ggml_cpu_mmid_seq_tl & 1;
    const struct ggml_cpu_anon_ent * anon = ggml_cpu_anon_find(src0); // Vollie: GGML_EXPERT_ANON
    const int64_t chunk  = MIN(32, ggml_cpu_mm_chunk); // Vollie: GGML_MM_CHUNK (stock 64 -> this path's 32)
    const int64_t nchunk = (ne01 + chunk - 1)/chunk;

    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        if (counts[cur_a] == 0) {
            continue;
        }
        const char * src0_cur = ggml_cpu_anon_src(anon, src0, cur_a);
        atomic_int * ctr = &ggml_cpu_mmid_ctr[bank][cur_a].v;
        for (int64_t pos = dyn ? 0 : ith, c0, c1; ggml_cpu_mmid_claim(dyn, ctr, nchunk, nth, &pos, &c0, &c1);)
        for (int64_t c = c0; c < c1; c++) {
            if (q4) {
                ggml_cpu_expq4_chunk(q4, dst, ids, cur_a, c*chunk, MIN((c + 1)*chunk, ne01), 0, counts[cur_a], matrix_rows, row_size, my_q);
            } else {
                if (ggml_cpu_mmid_touch) { // Vollie: GGML_MMID_TOUCH
                    ggml_cpu_page_touch(src0_cur + c*chunk*src0->nb[1], (size_t) (MIN((c + 1)*chunk, ne01) - c*chunk)*src0->nb[1]);
                }
                ggml_cpu_mmid_chunk(dst, src0, src1, ids, cur_a, c*chunk, MIN((c + 1)*chunk, ne01), 0, 1,
                                    src0_cur, matrix_rows, row_size, true, my_q);
            }
        }
    }
}

// Vollie: GGML_MMID_FUSE, decided by every thread alike from the graph and the knobs: nodes j, j+1 are the one-token gate and
// up MUL_MAT_IDs of one expert layer (either order; same activation, same ids, same weight type and shape, both on the
// no-barrier path) and j+2 is the SWIGLU of the two, which alone reads them. Not with the EXPERT_Q4_LAYERS or EXPERT_ANON
// copies (their per-tensor bookkeeping runs per op), and only when every chunk boundary is a multiple of 16 floats, so
// ggml_vec_swiglu_f32 on a chunk takes the same SIMD/scalar path per element as on the whole row (bit-identical).
static bool ggml_cpu_mmid_fuse_ok(const struct ggml_cgraph * cgraph, int j, const struct ggml_cplan * cplan, int nth) {
    if (!ggml_cpu_mmid_fuse || ggml_cpu_disable_fusion || cplan->use_ref || ggml_cpu_expq4_make > 0 || ggml_cpu_expert_anon > 0 ||
        ggml_cpu_anon_alloc_n > 0 || j + 2 >= cgraph->n_nodes) {
        return false;
    }
    const struct ggml_tensor * a = cgraph->nodes[j], * b = cgraph->nodes[j + 1], * g = cgraph->nodes[j + 2];
    const enum ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    const int outs[] = { j + 2 };
    if (a->op != GGML_OP_MUL_MAT_ID || b->op != GGML_OP_MUL_MAT_ID || g->op != GGML_OP_GLU ||
        ggml_get_glu_op(g) != GGML_GLU_OP_SWIGLU || !((g->src[0] == a && g->src[1] == b) || (g->src[0] == b && g->src[1] == a)) ||
        !ggml_can_fuse_subgraph(cgraph, j, 3, ops, outs, 1)) {
        return false;
    }
    const int64_t chunk = MIN(32, ggml_cpu_mm_chunk), ne01 = a->src[0]->ne[1];
    return a->src[1] == b->src[1] && a->src[2] == b->src[2] && a->src[0]->type == b->src[0]->type &&
           ggml_are_same_shape(a->src[0], b->src[0]) && ggml_are_same_stride(a->src[0], b->src[0]) &&
           ggml_are_same_shape(a, b) && ggml_are_same_shape(a, g) && ggml_are_same_stride(a, b) && ggml_are_same_stride(a, g) &&
           a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && g->type == GGML_TYPE_F32 &&
           a->nb[0] == sizeof(float) && b->nb[0] == sizeof(float) && g->nb[0] == sizeof(float) &&
           a->ne[0] == ne01 && chunk % 16 == 0 && ne01 % 16 == 0 &&
           ggml_cpu_mmid_fast_ok(a, nth, cplan->work_size) && ggml_cpu_mmid_fast_ok(b, nth, cplan->work_size);
}

// Vollie: GGML_MMID_FUSE, the fused gate + up + SWIGLU (glu = the GLU node; its src[0] is the gate, src[1] the up). Per
// claimed chunk of one expert: the gate rows and the up rows into their own tensors exactly as the two ops write them (same
// from_float once, same vec_dot per row), then glu = ggml_vec_swiglu_f32 over that chunk's elements of each routed slot.
// Skipped slots (expert skip): thread 0 zeroes their gate and up rows as each op would and runs the same swiglu on the row.
// One chunk counter (the first op's bank) covers both weights; the caller counts the group as one MUL_MAT_ID.
static void ggml_cpu_mmid_decode_fused(const struct ggml_compute_params * params, struct ggml_tensor * glu) {
    struct ggml_tensor * gate = glu->src[0], * up = glu->src[1];
    const struct ggml_tensor * src1 = gate->src[1], * ids = gate->src[2];
    const struct ggml_tensor * w_g = gate->src[0], * w_u = up->src[0];
    const int ith = params->ith, nth = params->nth;

    ggml_cpu_skip_prepare(gate); // same ids and weights as up's
    const enum ggml_type vec_dot_type = type_traits_cpu[w_g->type].vec_dot_type;
    const int64_t ne00 = w_g->ne[0], ne01 = w_g->ne[1], n_as = w_g->ne[2];
    const size_t  row_size = ggml_row_size(vec_dot_type, ne00);

    char * my_q = (char *) params->wdata + ith*ggml_cpu_mmid_row_pad(gate);
    type_traits_cpu[vec_dot_type].from_float((const float *) src1->data, my_q, ne00);

    int64_t counts[GGML_CPU_MMID_MAX_AS];
    struct mmid_row_mapping rows_buf[GGML_CPU_SKIP_MAX];
    struct mmid_row_mapping * matrix_rows = rows_buf;
    ggml_cpu_mmid_build_map(gate, ids, n_as, counts, matrix_rows, ith == 0);
    if (ith == 0 && ggml_cpu_skip_on_tl) {
        ggml_cpu_mmid_build_map(up, ids, n_as, counts, matrix_rows, true); // the same map again, zeroing up's skipped rows
        for (int64_t t = 0; t < ids->ne[1]; t++) {
            for (int64_t s = 0; s < ids->ne[0]; s++) {
                if (ggml_cpu_skip_mask_tl[t*ids->ne[0] + s]) {
                    const size_t o = s*glu->nb[1] + t*glu->nb[2];
                    ggml_vec_swiglu_f32((int) ne01, (float *) ((char *) glu->data + o), (const float *) ((const char *) gate->data + o),
                                        (const float *) ((const char *) up->data + o));
                }
            }
        }
    }

    const bool    dyn    = (ggml_cpu_mmid & 2) != 0;
    const int     bank   = ggml_cpu_mmid_seq_tl & 1;
    const int64_t chunk  = MIN(32, ggml_cpu_mm_chunk);
    const int64_t nchunk = (ne01 + chunk - 1)/chunk;

    for (int64_t cur_a = 0; cur_a < n_as; ++cur_a) {
        if (counts[cur_a] == 0) {
            continue;
        }
        const char * g_cur = ggml_cpu_anon_src(NULL, w_g, cur_a), * u_cur = ggml_cpu_anon_src(NULL, w_u, cur_a);
        atomic_int * ctr = &ggml_cpu_mmid_ctr[bank][cur_a].v;
        for (int64_t pos = dyn ? 0 : ith, c0, c1; ggml_cpu_mmid_claim(dyn, ctr, nchunk, nth, &pos, &c0, &c1);)
        for (int64_t c = c0; c < c1; c++) {
            const int64_t r0 = c*chunk, r1 = MIN((c + 1)*chunk, ne01);
            if (ggml_cpu_mmid_touch) { // Vollie: GGML_MMID_TOUCH
                ggml_cpu_page_touch(g_cur + r0*w_g->nb[1], (size_t) (r1 - r0)*w_g->nb[1]);
                ggml_cpu_page_touch(u_cur + r0*w_u->nb[1], (size_t) (r1 - r0)*w_u->nb[1]);
            }
            ggml_cpu_mmid_chunk(gate, w_g, src1, ids, cur_a, r0, r1, 0, counts[cur_a], g_cur, matrix_rows, row_size, true, my_q);
            ggml_cpu_mmid_chunk(up,   w_u, src1, ids, cur_a, r0, r1, 0, counts[cur_a], u_cur, matrix_rows, row_size, true, my_q);
            for (int64_t ir1 = 0; ir1 < counts[cur_a]; ir1++) {
                const struct mmid_row_mapping m = MMID_MATRIX_ROW(cur_a, ir1);
                const size_t o = m.i1*glu->nb[1] + m.i2*glu->nb[2] + r0*sizeof(float);
                ggml_vec_swiglu_f32((int) (r1 - r0), (float *) ((char *) glu->data + o), (const float *) ((const char *) gate->data + o),
                                    (const float *) ((const char *) up->data + o));
            }
        }
    }
}

// Thread 0, before the barrier that precedes MUL_MAT_ID node `next`: reset the chunk counters
// that op will use (bank of the op index), so no in-op sync is needed.
static void ggml_cpu_mmid_prepare_next(const struct ggml_tensor * next, int nth, size_t wsize) {
    if ((ggml_cpu_mmid & 2) && ggml_cpu_mmid_fast_ok(next, nth, wsize)) {
        const int bank = ggml_cpu_mmid_seq_tl & 1; // thread 0 has run seq ops; the next is op seq
        for (int64_t a = 0; a < next->src[0]->ne[2]; a++) {
            atomic_store_explicit(&ggml_cpu_mmid_ctr[bank][a].v, 0, memory_order_relaxed);
        }
    }
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    // Vollie: GGML_EXPERT_Q4_LAYERS (the copy is made here, by every thread, at the tensor's first op): Q8_0 activations
    const struct ggml_cpu_expq4_ent * q4 = ggml_cpu_expq4_get(params, dst);
    enum ggml_type    const vec_dot_type    = q4 ? GGML_TYPE_Q8_0 : type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // TODO: a more efficient F16 x F16 path can be used here if ggml_prec_get_src(dst, 1) allows converting src1 to F16
    //       the code needs to be refactored to enable this (src1 is currently converted based on vec_dot_type alone)

    // Vollie: expert skip (same decision on every thread) and the no-barrier one-token path
    ggml_cpu_skip_prepare(dst);
    if (ggml_cpu_mmid_fast_ok(dst, nth, params->wsize)) {
        ggml_cpu_mmid_decode(params, dst, q4);
        return;
    }

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    // Tiled matmul (see tiled.h); per-thread work buffers, 0 bytes when disabled. The
    // reservation is unconditional, the per expert eligibility is decided at dispatch time
    char * tiled_scratch = incr_ptr_aligned(&wdata_cur, ggml_tiled_wdata_size(nth, dst), 64);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    if (ith == 0) {
        // Vollie: the stock grouping loop, through the shared builder (applies the expert-skip mask)
        ggml_cpu_mmid_build_map(dst, ids, n_as, matrix_row_counts, matrix_rows, true);
    }
    GGML_UNUSED(n_ids);

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    const struct ggml_cpu_anon_ent * anon = ggml_cpu_anon_find(src0); // Vollie: GGML_EXPERT_ANON
    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        if (q4) { // Vollie: GGML_EXPERT_Q4_LAYERS, every routed row of the expert per chunk of src0 rows
            const int64_t ch = cne1 == 1 ? ggml_cpu_mm_chunk : 16;
            const int64_t nch = (ne01 + ch - 1)/ch;
            atomic_int * ctr = (atomic_int *)(atomic_current_chunk + cur_a);
            for (int64_t c = ith; c < nch; c = atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed)) {
                ggml_cpu_expq4_chunk(q4, dst, ids, cur_a, c*ch, MIN((c + 1)*ch, ne01), 0, cne1, matrix_rows,
                                     ggml_row_size(vec_dot_type, ne10), params->wdata);
            }
            continue;
        }

        // tiled takes over if profitable for this expert (see tiled.h)
        if (ggml_compute_forward_mul_mat_id_tiled(params, dst, cur_a, cne1, (const int32_t *) &MMID_MATRIX_ROW(cur_a, 0), tiled_scratch)) {
            continue;
        }

        const char * src0_cur = ggml_cpu_anon_src(anon, src0, cur_a);
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = nr1 == 1 ? ggml_cpu_mm_chunk : 64; // Vollie: GGML_MM_CHUNK
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_cpu_mmid_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

// Vollie: env GGML_SHEXP_PRUNE=<file> (C4, round14_plan.md §F (3); off unless set, "" = unset; a DECLARED NUMERICS EXPERIMENT,
// NOT bit-exact when on): static shared-expert neuron pruning from lab/shexp_prune.py's artefact (kept neuron ids per layer,
// the kept gate/up Q4_0 rows, the gathered down columns in F32; key = the routers' bytes). Loaded once by the calling thread
// before a graph's threads start (ggml_cpu_shexp_prep): down is re-quantised to its served type with ggml_quantize_chunk.
// The three shared-expert MUL_MATs of a pruned layer run on anon copies, before the extra-buffer dispatch (the served
// weights may be repacked): gate/up = the stock mul_mat over the k kept rows into a scratch, scattered into the kept
// channels of the op's dst (pruned channels set to 0, so the SWIGLU between is untouched); down = the kept channels of its
// input gathered into the scratch, then the stock mul_mat with the k-column copy. ponytail: the copies are plain rows, so
// on a CPU that repacks Q4_0 these three ops take the plain vec_dot path (other float order, slower; host only); the
// artefact is host-made (the runtime does not derive the importance itself).
#define GGML_CPU_SHEXP_MAXL 128
static struct ggml_cpu_shexp_ent { int k; enum ggml_type dt; int32_t * kept; char * g, * u, * d; } ggml_cpu_shexp[GGML_CPU_SHEXP_MAXL];
static char    ggml_cpu_shexp_path[1024] = ""; // the file loaded ("" = none)
static bool    ggml_cpu_shexp_on = false;      // this graph uses the copies
static float * ggml_cpu_shexp_scr = NULL;      // k x n_tokens floats
static size_t  ggml_cpu_shexp_scr_n = 0;

static int ggml_cpu_shexp_layer(const struct ggml_tensor * node, int * which) { // layer of a pruned shexp MUL_MAT, else -1
    // no sscanf "%[": every thread runs this, and Apple's scanf char class was seen to disagree between threads (C4)
    const char * name = node->op == GGML_OP_MUL_MAT ? node->src[0]->name : "";
    char * end;
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    const long l = strtol(name + 4, &end, 10);
    if (end == name + 4 || l < 0 || l >= GGML_CPU_SHEXP_MAXL || !ggml_cpu_shexp[l].k) {
        return -1;
    }
    *which = strcmp(end, ".ffn_gate_shexp.weight") == 0 ? 0 : strcmp(end, ".ffn_up_shexp.weight") == 0 ? 1 :
             strcmp(end, ".ffn_down_shexp.weight") == 0 ? 2 : -1;
    return *which < 0 ? -1 : (int) l;
}

static void ggml_cpu_shexp_prep(const struct ggml_cgraph * cgraph) {
    const char * path = getenv("GGML_SHEXP_PRUNE");
    ggml_cpu_shexp_on = false;
    if (!path || !*path) {
        return;
    }
    if (strcmp(path, ggml_cpu_shexp_path) != 0) {
        uint64_t key = 0; // the routers' bytes, as the artefact's key
        int nr = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const struct ggml_tensor * w = cgraph->nodes[i]->src[0];
            int l;
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT && w->type == GGML_TYPE_F32 && strstr(w->name, ".ffn_gate_inp.weight") && sscanf(w->name, "blk.%d.", &l) == 1) {
                for (size_t j = 0; j < ggml_nbytes(w)/8; j++) {
                    key += ((const uint64_t *) w->data)[j];
                }
                nr++;
            }
        }
        FILE * fp = fopen(path, "rb");
        char magic[4] = {0};
        float frac = 0; int32_t nl = 0; uint64_t fkey = 0;
        bool ok = nr > 0 && fp && fread(magic, 4, 1, fp) == 1 && memcmp(magic, "VSP1", 4) == 0 && fread(&frac, 4, 1, fp) == 1 &&
                  fread(&nl, 4, 1, fp) == 1 && fread(&fkey, 8, 1, fp) == 1 && fkey == key;
        size_t b0 = 0, b1 = 0;
        for (int i = 0; ok && i < nl; i++) {
            int32_t h[3];
            ok = fread(h, 4, 3, fp) == 3 && h[0] >= 0 && h[0] < GGML_CPU_SHEXP_MAXL && h[1] > 0 && h[1] % 32 == 0 && h[1] <= 4096;
            if (!ok) break;
            const int l = h[0], k = h[1];
            const enum ggml_type dt = (enum ggml_type) h[2];
            const size_t grow = ggml_row_size(GGML_TYPE_Q4_0, 1536), drow = ggml_row_size(dt, k);
            struct ggml_cpu_shexp_ent * e = &ggml_cpu_shexp[l];
            e->kept = malloc(k*sizeof(int32_t)); e->g = malloc(k*grow); e->u = malloc(k*grow); e->d = malloc(1536*drow);
            float * f = malloc((size_t) 1536*k*sizeof(float));
            ok = e->kept && e->g && e->u && e->d && f && fread(e->kept, 4, k, fp) == (size_t) k && fread(e->g, grow, k, fp) == (size_t) k &&
                 fread(e->u, grow, k, fp) == (size_t) k && fread(f, sizeof(float)*k, 1536, fp) == 1536;
            if (ok) {
                ggml_quantize_chunk(dt, f, e->d, 0, 1536, k, NULL);
                e->k = k; e->dt = dt;
                b0 += 2*1024*grow + 1536*ggml_row_size(dt, 1024);
                b1 += 2*k*grow + 1536*drow;
            }
            free(f);
        }
        if (fp) fclose(fp);
        snprintf(ggml_cpu_shexp_path, sizeof ggml_cpu_shexp_path, "%s", path);
        if (!ok) { // ponytail: a bad file leaves the layers read so far half-set; the knob is host-only
            for (int l = 0; l < GGML_CPU_SHEXP_MAXL; l++) ggml_cpu_shexp[l].k = 0;
            fprintf(stderr, "cpu-prof shexp prune file=%s missing or not this model's: stock shared experts\n", path);
        } else {
            fprintf(stderr, "cpu-prof shexp prune=%.3f layers=%d mb=%.1f->%.1f\n", (double) frac, nl, b0/1e6, b1/1e6);
        }
        fflush(stderr);
    }
    // scratch: the largest k x n_tokens of this graph's pruned ops
    size_t need = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        int w;
        const int l = ggml_cpu_shexp_layer(cgraph->nodes[i], &w);
        if (l >= 0) {
            need = MAX(need, (size_t) ggml_cpu_shexp[l].k*cgraph->nodes[i]->src[1]->ne[1]);
            ggml_cpu_shexp_on = true;
        }
    }
    if (need > ggml_cpu_shexp_scr_n) {
        free(ggml_cpu_shexp_scr);
        ggml_cpu_shexp_scr = malloc(need*sizeof(float));
        GGML_ASSERT(ggml_cpu_shexp_scr);
        ggml_cpu_shexp_scr_n = need;
    }
}

void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

static bool ggml_cpu_shexp_forward(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    int w;
    const int l = ggml_cpu_shexp_on ? ggml_cpu_shexp_layer(dst, &w) : -1;
    const struct ggml_tensor * src1 = dst->src[1];
    if (l < 0 || src1->type != GGML_TYPE_F32 || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst) || src1->ne[2]*src1->ne[3] != 1) {
        return false;
    }
    const struct ggml_cpu_shexp_ent * e = &ggml_cpu_shexp[l];
    const int k = e->k, ith = params->ith, nth = params->nth;
    const int64_t T = src1->ne[1];
    struct ggml_tensor wt = *dst->src[0], d = *dst, x;
    wt.extra = NULL;
    if (w < 2) { // gate / up: the kept rows, outputs to the scratch, then scattered into dst's kept channels
        wt.ne[1] = k; wt.data = w == 0 ? e->g : e->u;
        wt.nb[1] = ggml_row_size(wt.type, wt.ne[0]); wt.nb[2] = wt.nb[3] = wt.nb[1]*k;
        d.ne[0] = k; d.data = ggml_cpu_shexp_scr;
        d.nb[1] = k*sizeof(float); d.nb[2] = d.nb[3] = d.nb[1]*T;
        d.src[0] = &wt;
        ggml_compute_forward_mul_mat(params, &d);
        ggml_barrier(params->threadpool);
        for (int64_t t = ith; t < T; t += nth) {
            float * o = (float *) ((char *) dst->data + t*dst->nb[1]);
            memset(o, 0, dst->ne[0]*sizeof(float));
            for (int i = 0; i < k; i++) {
                o[e->kept[i]] = ggml_cpu_shexp_scr[t*k + i];
            }
        }
        return true;
    }
    // down: the kept input channels gathered into the scratch, then the k-column copy
    for (int64_t t = ith; t < T; t += nth) {
        const float * in = (const float *) ((const char *) src1->data + t*src1->nb[1]);
        for (int i = 0; i < k; i++) {
            ggml_cpu_shexp_scr[t*k + i] = in[e->kept[i]];
        }
    }
    ggml_barrier(params->threadpool);
    x = *src1;
    x.ne[0] = k; x.data = ggml_cpu_shexp_scr; x.nb[1] = k*sizeof(float); x.nb[2] = x.nb[3] = x.nb[1]*T;
    wt.type = e->dt; wt.ne[0] = k; wt.data = e->d;
    wt.nb[0] = ggml_type_size(e->dt); wt.nb[1] = ggml_row_size(e->dt, k); wt.nb[2] = wt.nb[3] = wt.nb[1]*wt.ne[1];
    d.src[0] = &wt; d.src[1] = &x;
    ggml_compute_forward_mul_mat(params, &d);
    return true;
}

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // Vollie: GGML_HEAD2, the output head only; before the extra buffers, since the head is usually repacked
    if (tensor->op == GGML_OP_MUL_MAT && (ggml_cpu_head2(params, tensor) || ggml_cpu_shexp_forward(params, tensor))) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_compute_forward_col2im_1d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                ggml_compute_forward_lightning_indexer(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_COMB:
            {
                ggml_compute_forward_dsv4_hc_comb(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                ggml_compute_forward_dsv4_hc_pre(params, tensor);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                ggml_compute_forward_dsv4_hc_post(params, tensor);
            } break;
        case GGML_OP_TOP_K_MASK:
            {
                ggml_compute_forward_top_k_mask(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_TOP_K_MASK:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
        case GGML_OP_SET_ROWS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_COL2IM_1D:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

#if defined(__wasi__)
    // WASI doesn't support parallelism yet
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type) || node->src[0]->type == GGML_TYPE_F8_E4M3) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type rc = ggml_cpu_is_router(node) ? ggml_cpu_router_plan_type() : GGML_TYPE_COUNT; // Vollie: GGML_ROUTER_CAST
                        const enum ggml_type vec_dot_type = type_traits_cpu[rc != GGML_TYPE_COUNT ? rc : node->src[0]->type].vec_dot_type;
                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // Vollie: one quantized activation row per thread for the no-barrier path (GGML_MMID)
                        if (ids->ne[1] == 1 && ggml_nelements(src1) == src1->ne[0]) {
                            cur += n_tasks*GGML_PAD(ggml_row_size(vec_dot_type, src1->ne[0]), 64) + 64;
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                        // Workspace for tiled (see tiled.h)
                        cur = GGML_PAD(cur, 64);
                        cur += ggml_tiled_wdata_size(n_tasks, node);
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type) || node->src[0]->type == GGML_TYPE_F8_E4M3 ||
                            node->src[0]->type == GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SET_ROWS:
                    {
                        if (node->src[0]->type == GGML_TYPE_F16 && node->type != GGML_TYPE_F16) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne10*ne11;
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In
                        const int64_t ne13 = node->src[1]->ne[3]; // Batch

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += sizeof(float) * ne10 * ne11 * ne12 * ne13;
                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = ggml_get_op_params_i32(node, 0);
                        const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                case GGML_OP_LIGHTNING_INDEXER:
                    {
                        // temp buffer for dequantizing lightning indexer keys
                        const int64_t ne10 = node->src[1]->ne[0];
                        cur += sizeof(float)*ne10*n_tasks;
                    } break;
                default:
                    break;
            }
        }

        if (node->op == GGML_OP_MUL_MAT && strstr(node->src[0]->name, "_shexp.") && getenv("GGML_SHEXP_PRUNE") && *getenv("GGML_SHEXP_PRUNE")) {
            // Vollie: GGML_SHEXP_PRUNE runs these on plain copies: at least the stock (non-repacked) activation buffer
            const enum ggml_type vdt = type_traits_cpu[node->src[0]->type].vec_dot_type;
            cur = MAX(cur, GGML_PAD(ggml_row_size(vdt, ggml_nelements(node->src[1])), 64) + 64);
        }
        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}


// Try to fuse the current node with subsequent nodes for better performance.
// Returns the number of nodes skipped by fusion (>=1), or 0 if no fusion was applied.
// ggml_cpu_disable_fusion: declared with the MUL_MAT_ID additions (Vollie: GGML_MMID_FUSE reads it too)

// Vollie: RMS_NORM + MUL fusion check (was inline in ggml_cpu_try_fuse_ops), shared
// by the compute loop and the tiny-op lookahead so both can tell
// that a node will run fused (multi-threaded) and keep the barrier before it.
static bool ggml_cpu_can_fuse_rms_norm_mul(
        const struct ggml_cgraph * cgraph,
        const int node_n,
        const struct ggml_cplan * cplan) {

    if (ggml_cpu_disable_fusion || cplan->use_ref) {
        return false;
    }

    const struct ggml_tensor * node = cgraph->nodes[node_n];
    const enum ggml_op fuse_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
    if (node->op != GGML_OP_RMS_NORM || !ggml_can_fuse(cgraph, node_n, fuse_ops, 2)) {
        return false;
    }
    const struct ggml_tensor * mul_node = cgraph->nodes[node_n + 1];
    const struct ggml_tensor * mul_w = (mul_node->src[0] == node)
        ? mul_node->src[1] : mul_node->src[0];
    return node->src[0]->type  == GGML_TYPE_F32 &&
           mul_node->type      == GGML_TYPE_F32 &&
           mul_w->type         == GGML_TYPE_F32 &&
           mul_w->ne[0]        == node->ne[0]   &&
           mul_w->nb[0]        == sizeof(float);
}


// Vollie: an op on a few thousand elements takes less time than the barrier
// that follows it, and a decode graph is mostly such ops (Granite 4.0 H Tiny:
// ~3300 nodes per token). Tiny ops run on thread 0 alone, and the barrier
// between two consecutive tiny ops is skipped: the next multi-threaded op is
// still preceded by a barrier, so every thread sees thread 0's results first.
// Threshold in elements, from env GGML_TINY_OP (unset or 0 = off, stock behaviour).
// Re-read every graph so an app can sweep it per device (barrier ~1 us on M4,
// unknown on SD888) without reloading the model.
static int64_t ggml_cpu_tiny_thresh = 0;

// Vollie: env GGML_CPU_PROF=1 times one-token (decode) graphs on thread 0, per op
// type: its compute and its wait at the barrier after the op (the other threads
// finishing, plus the sync itself). Every 32 graphs one line goes to stderr
// ("cpu-prof ..."), µs per graph. Thread 0 is the caller of ggml_graph_compute,
// so the accumulators below need no locking.
static bool    ggml_cpu_prof_on   = false;
static bool    ggml_cpu_prof_this = false; // this graph is a decode graph
// [variant]: env GGML_CPU_AB alternates a compute setting between decode graphs (q3k, tiny: every graph;
// knob: ABBA blocks, see ggml_cpu_ab_warm), so heat, clocks and background load fall on both sides alike;
// B's stats go to [1].
//   GGML_CPU_AB=q3k       A = vectorized Q3_K dot (ggml_cpu_q3k_vec = 1), B = upstream loop
//   GGML_CPU_AB=tiny:<n>  A = GGML_TINY_OP, B = <n>
// Not for GGML_EXPERT_IO: one token's prefetch warms the cache for the next.
extern int ggml_cpu_q3k_vec;
extern int ggml_cpu_q3k_ilp, ggml_cpu_q3k_pf, ggml_cpu_q3k_kern; // Vollie: arch/arm/quants.c
static int     ggml_cpu_ab_mode = 0, ggml_cpu_ab_var = 0, ggml_cpu_ab_count = 0;
// Vollie: knob mode runs ABBA blocks (r5, 05_compute_less.md §7.5): the first GGML_CPU_AB_WARM decode graphs of a
// run (default 16) are side A, then blocks of GGML_CPU_AB_BLOCK graphs (default 8) go A B B A A B B A ..., which
// cancels a linear drift (page-cache fill, clocks) instead of handing it to one side. The profile window closes
// at every block; its line carries "ab=A|B blk=<n>" (blk=0: the warm-up). The count restarts at each prompt.
static int     ggml_cpu_ab_warm = 16, ggml_cpu_ab_block = 8, ggml_cpu_ab_blk = 0;
// Vollie: env GGML_CPU_AB_BAAB=1 (r13, R39): the post-warm-up blocks go B A A B B A A B ... instead (warm-up still A; ab= keeps
// naming the side, so A is always the first value), so position 1 of each group of four is B's.
static bool    ggml_cpu_ab_baab = false;
// Vollie: env GGML_CPU_AB_TOKEN=1 (r15, R46; off unless set): the knob flips per decode graph (= per token), A B B A A B B A ...
// (block 1, warm-up 0 unless GGML_CPU_AB_WARM is set), so pair k = tokens (2k-1, 2k) is one A and one B; the window closes every
// token with one short line "cpu-prof ab=A|B tok=<n> ..." (ggml_cpu_prof_dump_one: side, faults, cpu4/cpu7 clock, counts, the
// scored ops only: MUL_MAT_ID / MUL_MAT / GLU / SSM_SCAN, by cost). Knob values are re-read at op dispatch, so per graph is exact.
// knob:<NAME>:0:0 is the A/A null: same code path, both labels side A's value.
static bool    ggml_cpu_ab_token = false;
static int64_t ggml_cpu_ab_tiny = 0;
static int64_t ggml_cpu_prof_comp[2][GGML_OP_COUNT], ggml_cpu_prof_wait[2][GGML_OP_COUNT];
static int     ggml_cpu_prof_cnt[2][GGML_OP_COUNT], ggml_cpu_prof_graphs[2], ggml_cpu_prof_tokens[2], ggml_cpu_prof_nodes[2], ggml_cpu_prof_barriers[2];
static int64_t ggml_cpu_prof_total[2];
static int64_t ggml_cpu_prof_kept[2], ggml_cpu_prof_ktok[2]; // kept expert slots / tokens, down projections
static int64_t ggml_cpu_prof_adv[2]; // Vollie: expert slices given MADV_WILLNEED (EXPERT_IO bit 1), profiled graphs, per A/B side
static bool    ggml_cpu_prof_anon[2];                        // the side read the GGML_EXPERT_ANON copies
static int     ggml_cpu_prof_ilp[2], ggml_cpu_prof_pf[2];    // GGML_Q3K_ILP / _PF on that side
static int     ggml_cpu_prof_kern[2];                        // Vollie: GGML_Q3K_KERN on that side
static int     ggml_cpu_prof_touch[2];                       // Vollie: GGML_MMID_TOUCH on that side
static int     ggml_cpu_prof_fuse[2], ggml_cpu_prof_chunk2[2]; // Vollie: GGML_MMID_FUSE / GGML_MMID_CHUNK2 on that side
static int     ggml_cpu_prof_skip[2];                        // GGML_EXPERT_SKIP on that side
static int64_t ggml_cpu_prof_flt[2][2]; // Vollie: page faults (major, minor) of the process during profiled graphs, per A/B side
static int64_t ggml_cpu_prof_h2[2];   // GGML_HEAD2 on that side
static int64_t ggml_cpu_prof_head[2]; // Vollie: µs of the output head's MUL_MAT (compute + wait), profiled graphs, per A/B side
static int64_t ggml_cpu_prof_router[2];  // Vollie r15: µs of the MoE routers' MUL_MATs (compute + wait), as head
static int     ggml_cpu_prof_rcast[2];   // Vollie r15: GGML_ROUTER_CAST type on that side (GGML_TYPE_COUNT = stock F32)
static int     ggml_cpu_prof_chunk[2], ggml_cpu_prof_expq4[2]; // Vollie: GGML_MM_CHUNK / GGML_EXPERT_Q4_LAYERS on that side

// Vollie: env GGML_EXPERT_IO, bit mask (0 = off, stock). MoE expert weights are the
// only large tensors still read through the mmap, and on a phone whose page cache is
// smaller than them every token faults some back in from flash, one page at a time on
// whichever compute thread touches it (SD888, Granite: ~600 major faults/token).
//   1: when a layer's routing is known (its first MUL_MAT_ID), MADV_WILLNEED exactly the
//      chosen experts' slices of every MUL_MAT_ID that shares those ids (gate, up, down),
//      so the reads are in flight together instead of serialized by faults;
//   2: MADV_RANDOM on expert tensors, so a fault doesn't read around into experts that
//      weren't chosen (read-around is doubled by llama.cpp's FADV_SEQUENTIAL).
// Advice only: results are unchanged.
static int ggml_cpu_expert_io = 0;
// Lab only: env GGML_EXPERT_LOG=1 prints "eid <tensor> <ids...>" per routed MUL_MAT_ID of
// one-token graphs (needs GGML_EXPERT_IO != 0), to measure expert overlap across tokens.
static bool ggml_cpu_expert_log = false;

// Vollie: env GGML_MMID, bit mask for one-token MUL_MAT_ID (0 = stock path; default 3 since
// R18: -7.5 ms/token on the SD888, bit-identical; "" counts as unset):
//   1: no barrier inside the op. Stock quantizes the activation across threads and has
//      thread 0 build the expert -> row map, then syncs (120 extra barriers per token on
//      Granite). Here every thread quantizes the one activation row itself (~1.7 KB) and
//      builds the map itself; identical results.
//   2: 32-row chunks claimed dynamically (needs 1) instead of one static chunk per thread,
//      so a throttled or preempted core does not stall the others. The per-expert claim
//      counters are reset by thread 0 before the barrier that precedes the op.
//   Without 2, chunks are assigned round-robin (thread i takes chunks i, i+nth, ...).
static int ggml_cpu_mmid = 0;

// Vollie: env GGML_MMID_TOUCH=1 (r13, R39; off unless set, "" = unset): in the no-barrier path, after a chunk is claimed and
// before its kernel runs, one demand load of the first byte of each 4-KB page the chunk's weight rows span (ggml_cpu_page_touch),
// so its 2-6 page walks overlap instead of serialising inside the kernel. Bit-identical (reads only; the touch takes the
// minor fault the kernel would have taken). Not on the GGML_EXPERT_Q4_LAYERS copies.
static int ggml_cpu_mmid_touch = 0;
// Vollie: env GGML_MMID_FUSE=1 (r14, R42; default since R47 (device pass, r15), =0 off, "" = unset = on): a one-token expert layer's gate and up MUL_MAT_IDs and
// their SWIGLU run as one op (ggml_cpu_mmid_fuse_ok / ggml_cpu_mmid_decode_fused): one activation quantize, one chunk loop
// per expert over both weights with the swiglu applied per chunk, no barriers between the three. Bit-identical (same vec_dot
// per row, same ggml_vec_swiglu_f32 per element). The GLU line of the profile loses the fused ops, MUL_MAT_ID absorbs them.
// Vollie: env GGML_MMID_CHUNK2=1 (r14, R42; default since R47, =0 off): the no-barrier path claims two chunks per fetch_add while
// >= 2*nth remain (ggml_cpu_mmid_claim). Bit-identical. GGML_CPU_AB=knob:MMID_FUSE+CHUNK2:0:1 flips both.
// Vollie: env GGML_MMID_TEST1=1 (r13, host test only; off unless set, the kit never sets it): the no-barrier path also at
// 1 thread, so the MMID_TOUCH class-K fault test can run without concurrent faults. Same results (same from_float, same vec_dot).
static bool ggml_cpu_mmid_test1 = false;

// Vollie: env GGML_Q3K_X4=1 (default since R18): Q3_K expert rows four at a time, sharing the
// activation loads (bit-identical: each row's integer sum and float accumulation order are unchanged).
static int ggml_cpu_q3k_x4 = 0;

// Vollie: env GGML_EXPERT_SKIP=<permille> (0 = off; NOT bit-exact, see 04_campaign.md E1):
// an expert whose normalized routing weight is under permille/1000 is not computed; its
// output rows are zero. GGML_EXPERT_SKIP_RENORM=1 also rescales the remaining weights by
// 1/(1 - dropped weight) so the mixture keeps its mass. Thread 0 decides before the barrier
// that precedes the layer's first expert op, so every thread and every sibling op (gate,
// up, down) sees the same mask.
static int ggml_cpu_expert_skip = 0, ggml_cpu_expert_skip_renorm = 0;
static int64_t ggml_cpu_skip_cnt[2], ggml_cpu_skip_total[2]; // (token, slot) pairs skipped / routed per A/B side, profiled graphs, thread 0

// Vollie: per-token expert budget from the router's own weights (NOT bit-exact; 05_compute_less.md S1),
// decided in the same place as GGML_EXPERT_SKIP and combined with it (a slot is skipped if any rule says so):
// GGML_EXPERT_TOPP=<permille>: keep the fewest top-weighted slots whose cumulative weight reaches p/1000;
// GGML_EXPERT_GAP=<permille>: skip a slot under g/1000 of the top slot's weight;
// GGML_EXPERT_MINK=<n> (default 1): TOPP and GAP never keep fewer than n slots. No renormalization (R17).
// When any skip knob is on, the run's totals (down projections: one per layer and token) print at exit:
// "expert-budget routed=<slots> skipped=<slots> kavg=<kept slots per layer and token>".
static int ggml_cpu_expert_topp = 0, ggml_cpu_expert_gap = 0, ggml_cpu_expert_mink = 1;
static int64_t ggml_cpu_budget_routed = 0, ggml_cpu_budget_skipped = 0, ggml_cpu_budget_tok = 0;
static bool ggml_cpu_budget_atexit = false;
static int64_t ggml_cpu_budget_lkept[128], ggml_cpu_budget_ltok[128]; // Vollie: the same per layer (GGML_EXPERT_Q4_LAYERS' ranking)
static void ggml_cpu_budget_print(void) {
    fprintf(stderr, "expert-budget routed=%lld skipped=%lld kavg=%.2f\n", (long long) ggml_cpu_budget_routed,
            (long long) ggml_cpu_budget_skipped,
            ggml_cpu_budget_tok ? (double) (ggml_cpu_budget_routed - ggml_cpu_budget_skipped) / ggml_cpu_budget_tok : 0.0);
    fprintf(stderr, "expert-budget layers kept/token:");
    for (int l = 0; l < 128; l++) {
        if (ggml_cpu_budget_ltok[l]) {
            fprintf(stderr, " %d:%.3f", l, (double) ggml_cpu_budget_lkept[l] / ggml_cpu_budget_ltok[l]);
        }
    }
    fprintf(stderr, "\n");
}

// Vollie: env GGML_SSM_INPLACE (default on since r5, "" = unset = on, 0 = off; bit-identical: the same floats land in the same cache
// bytes, only the 1.5 MB/layer bounce through the scan output and back is gone). For a one-token,
// one-sequence, no-snapshot SSM_SCAN whose final state is CPY'd straight into the recurrent cache
// (llm_build_mamba_base::build_mamba2_layer), the scan writes its state into the CPY's destination
// (pointer in the scan's op_params[2..3], read in ops.cpp) and the CPY is not run. Applied per graph
// and undone after it, so a reused graph is unchanged. Refused if a node between the scan and the CPY
// touches that cache tensor, or anything other than the CPY reads the scan's state part.
// ponytail: at most 64 layers per graph; more layers fall back to the stock CPY
static int ggml_cpu_ssm_inplace = 0;
static struct ggml_tensor * ggml_cpu_ssm_scan[64], * ggml_cpu_ssm_cpy[64];
static int ggml_cpu_ssm_n = 0;

static inline const struct ggml_tensor * ggml_cpu_base(const struct ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

static void ggml_cpu_ssm_apply(struct ggml_cgraph * g) {
    ggml_cpu_ssm_n = 0;
    for (int j = 0; j < g->n_nodes && ggml_cpu_ssm_n < 64; j++) {
        struct ggml_tensor * c = g->nodes[j];
        if (c->op != GGML_OP_CPY || !c->src[0]->view_src || c->src[0]->view_src->op != GGML_OP_SSM_SCAN) {
            continue;
        }
        struct ggml_tensor * scan = c->src[0]->view_src;
        const struct ggml_tensor * s = scan->src[0], * x = scan->src[1];
        const size_t s_off = ggml_nelements(x) * ggml_element_size(x);
        const size_t s_bytes = (size_t) s->ne[0] * s->ne[1] * s->ne[2] * sizeof(float);
        if (ggml_get_op_params_i32(scan, 0) != 1 || x->ne[2] != 1 || x->ne[3] != 1 || scan->src[6]->ne[0] != 1 ||
            s->type != GGML_TYPE_F32 || c->src[0]->view_offs != s_off || ggml_nbytes(c->src[0]) != s_bytes ||
            !(c->flags & GGML_TENSOR_FLAG_COMPUTE) ||
            !ggml_is_contiguous(c->src[0]) || c->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(c->src[1]) ||
            ggml_nbytes(c->src[1]) != s_bytes) {
            continue;
        }
        const struct ggml_tensor * cache = ggml_cpu_base(c->src[1]);
        int i = j - 1;
        while (i >= 0 && g->nodes[i] != scan) i--;
        bool ok = i >= 0;
        for (int k = i + 1; ok && k < j; k++) {
            const struct ggml_tensor * n = g->nodes[k];
            if (ggml_op_is_empty(n->op) || ggml_is_empty(n)) continue;
            ok = ggml_cpu_base(n) != cache;
            for (int q = 0; ok && q < GGML_MAX_SRC && n->src[q]; q++) {
                ok = ggml_cpu_base(n->src[q]) != cache;
            }
        }
        if (ok) {
            ggml_cpu_ssm_scan[ggml_cpu_ssm_n] = scan;
            ggml_cpu_ssm_cpy[ggml_cpu_ssm_n++] = c;
        }
    }
    // drop a pair if any compute node other than its CPY reads the scan's state part
    for (int k = 0; ggml_cpu_ssm_n && k < g->n_nodes; k++) {
        const struct ggml_tensor * n = g->nodes[k];
        if (ggml_op_is_empty(n->op)) continue;
        for (int q = 0; q < GGML_MAX_SRC && n->src[q]; q++) {
            const struct ggml_tensor * t = n->src[q], * b = ggml_cpu_base(t);
            if (b->op != GGML_OP_SSM_SCAN) continue;
            for (int p = 0; p < ggml_cpu_ssm_n; p++) {
                if (ggml_cpu_ssm_scan[p] != b || n == ggml_cpu_ssm_cpy[p]) continue;
                const size_t s_off = ggml_nelements(b->src[1]) * ggml_element_size(b->src[1]);
                if (t == b || t->view_offs + ggml_nbytes(t) > s_off) {
                    ggml_cpu_ssm_scan[p] = ggml_cpu_ssm_scan[--ggml_cpu_ssm_n];
                    ggml_cpu_ssm_cpy[p]  = ggml_cpu_ssm_cpy[ggml_cpu_ssm_n];
                }
                break;
            }
        }
    }
    static int logged = 0;
    if (!logged++) {
        fprintf(stderr, "ssm-inplace layers=%d\n", ggml_cpu_ssm_n); // once per process
    }
    for (int p = 0; p < ggml_cpu_ssm_n; p++) {
        void * dst = ggml_cpu_ssm_cpy[p]->src[1]->data;
        memcpy((char *) ggml_cpu_ssm_scan[p]->op_params + 8, &dst, sizeof(dst));
        ggml_cpu_ssm_cpy[p]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    }
}

static void ggml_cpu_ssm_undo(void) {
    for (int p = 0; p < ggml_cpu_ssm_n; p++) {
        memset((char *) ggml_cpu_ssm_scan[p]->op_params + 8, 0, sizeof(void *));
        ggml_cpu_ssm_cpy[p]->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_cpu_ssm_n = 0;
}

// Vollie: env GGML_CONV_RING (default on since r6, "" = unset = on, 0 = off; bit-identical, R25). One-token Mamba2 conv: stock builds
// conv_x = CONCAT(old conv state, x^T), CPYs conv_x's last d_conv-1 columns into the conv cache, and
// SSM_CONV dots conv_x with the weights. Here SSM_CONV reads the old state and x where they already
// are (same products summed in the same order), writes the shifted state into the cache slot itself
// (pointer in its op_params[0..1], read in ops.cpp), and the CONCAT and the CPY are not run. Not a
// real ring buffer: the cache keeps its layout, so prompt graphs and every other reader are unchanged.
// Refused unless the CONCAT feeds only that SSM_CONV and one CPY, no node between the CONCAT and the
// SSM_CONV writes over the old state or x, and none between the CPY and the SSM_CONV touches the cache.
// The fused SSM_CONV runs on thread 0 alone (3.3k rows x 4 on Granite: tiny-op runs it so anyway).
// Applied per graph and undone after it, like GGML_SSM_INPLACE.
// ponytail: at most 64 layers per graph; more layers fall back to the stock ops
static int ggml_cpu_conv_ring = 0;
static struct ggml_tensor * ggml_cpu_conv_op[64], * ggml_cpu_conv_cat[64], * ggml_cpu_conv_cpy[64];
static int ggml_cpu_conv_n = 0;

static bool ggml_cpu_overlaps(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    const char * pa = a->data, * pb = b->data;
    return pa && pb && pa < pb + ggml_nbytes(b) && pb < pa + ggml_nbytes(a);
}

static void ggml_cpu_conv_apply(struct ggml_cgraph * g) {
    ggml_cpu_conv_n = 0;
    for (int j = 0; j < g->n_nodes && ggml_cpu_conv_n < 64; j++) {
        struct ggml_tensor * sc = g->nodes[j], * cc = sc->src[0];
        if (sc->op != GGML_OP_SSM_CONV || cc->op != GGML_OP_CONCAT || !(cc->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            continue;
        }
        const struct ggml_tensor * st = cc->src[0], * xt = cc->src[1], * w = sc->src[1];
        if (ggml_get_op_params_i32(cc, 0) != 0 || cc->type != GGML_TYPE_F32 || st->type != GGML_TYPE_F32 ||
            xt->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(st) ||
            cc->ne[0] != st->ne[0] + 1 || st->ne[0] > 8 || cc->ne[2] != 1 || xt->ne[0] != 1 || xt->ne[1] != cc->ne[1] || xt->ne[2] != 1 ||
            w->ne[0] != cc->ne[0] || w->nb[0] != sizeof(float) || sc->ne[1] != 1) {
            continue;
        }
        int ci = j - 1, cpi = -1;
        while (ci >= 0 && g->nodes[ci] != cc) ci--;
        bool ok = ci >= 0;
        struct ggml_tensor * cp = NULL;
        for (int k = 0; ok && k < g->n_nodes; k++) { // who reads conv_x: only sc and one matching CPY
            struct ggml_tensor * n = g->nodes[k];
            if (n == sc || ggml_op_is_empty(n->op) || ggml_is_empty(n)) continue;
            for (int q = 0; ok && q < GGML_MAX_SRC && n->src[q]; q++) {
                if (ggml_cpu_base(n->src[q]) != cc) continue;
                const struct ggml_tensor * v = n->src[0];
                ok = !cp && n->op == GGML_OP_CPY && q == 0 && v->view_src == cc && v->view_offs == sizeof(float) &&
                     (n->flags & GGML_TENSOR_FLAG_COMPUTE) && k > ci && k < j &&
                     v->ne[0] == st->ne[0] && v->ne[1] == st->ne[1] && v->ne[2] == 1 &&
                     n->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(n->src[1]) &&
                     ggml_nelements(n->src[1]) == ggml_nelements(st) && !ggml_cpu_overlaps(n->src[1], st) &&
                     !ggml_cpu_overlaps(n->src[1], xt);
                if (ok) { cp = n; cpi = k; }
            }
        }
        ok = ok && cp;
        for (int k = ci + 1; ok && k <= j; k++) { // the old state and x survive until sc, the cache is not read early
            const struct ggml_tensor * n = g->nodes[k];
            if (n == cp || ggml_op_is_empty(n->op) || ggml_is_empty(n) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE)) continue;
            // sc's own output may start at or before the old state (the allocator reuses the CONCAT's freed
            // input): ops.cpp runs it on one thread, rows ascending, each row's inputs read before its output
            // is written, so it only overwrites state rows already done
            ok = (!ggml_cpu_overlaps(n, st) || (n == sc && (const char *) n->data <= (const char *) st->data)) &&
                 !ggml_cpu_overlaps(n, xt);
                for (int q = 0; ok && k > cpi && k < j && q < GGML_MAX_SRC && n->src[q]; q++) {
                ok = ggml_cpu_base(n->src[q]) != ggml_cpu_base(cp->src[1]);
            }
            ok = ok && (k < cpi || k == j || ggml_cpu_base(n) != ggml_cpu_base(cp->src[1]));
        }
        if (ok) {
            ggml_cpu_conv_op[ggml_cpu_conv_n]  = sc;
            ggml_cpu_conv_cat[ggml_cpu_conv_n] = cc;
            ggml_cpu_conv_cpy[ggml_cpu_conv_n++] = cp;
        }
    }
    static int logged = 0;
    if (!logged++) {
        fprintf(stderr, "conv-ring layers=%d\n", ggml_cpu_conv_n); // once per process
    }
    for (int p = 0; p < ggml_cpu_conv_n; p++) {
        void * dst = ggml_cpu_conv_cpy[p]->src[1]->data;
        memcpy(ggml_cpu_conv_op[p]->op_params, &dst, sizeof(dst));
        ggml_cpu_conv_cat[p]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        ggml_cpu_conv_cpy[p]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
    }
}

static void ggml_cpu_conv_undo(void) {
    for (int p = 0; p < ggml_cpu_conv_n; p++) {
        memset(ggml_cpu_conv_op[p]->op_params, 0, sizeof(void *));
        ggml_cpu_conv_cat[p]->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_cpu_conv_cpy[p]->flags |= GGML_TENSOR_FLAG_COMPUTE;
    }
    ggml_cpu_conv_n = 0;
}

// Finds the mixture weights a layer's MUL_MAT_ID ops are combined with: src[1] of the MUL
// named ffn_moe_weighted that follows (llama.cpp's build_moe_ffn), [1, n_ids, n_tokens] F32.
static const struct ggml_tensor * ggml_cpu_moe_weights(const struct ggml_cgraph * cgraph, int node_n) {
    for (int j = node_n + 1; j < cgraph->n_nodes && j < node_n + 64; j++) {
        const struct ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_MUL && strncmp(n->name, "ffn_moe_weighted", 16) == 0) {
            return n->src[1];
        }
    }
    return NULL;
}

// Vollie: env GGML_BARRIER_BENCH=<n>: once per process, before the first graph's first op,
// all threads run n empty barriers and thread 0 prints the cost per barrier.
static int  ggml_cpu_bbench_n = 0; // last value run

// Vollie: env GGML_CPU_BENCH=<n> (r10): once per distinct n > 0, on the thread calling the first graph after it is set
// (before the graph runs), a 1-thread kernel timing of the Q3_K expert dot: two "cpu-bench q3k ..." lines (arch/arm/quants.c).
static int  ggml_cpu_kbench_n = 0; // last value run
static bool ggml_cpu_bbench_this = false;

// Vollie: env GGML_HELPER_CPUS="0-3" or "0,1,2,3": cores for the expert-guess helper thread
// (it otherwise inherits thread 0's big-core affinity and competes with the compute threads).
static bool ggml_cpu_helper_mask[GGML_MAX_N_THREADS];
static bool ggml_cpu_helper_mask_set = false;

// Per-thread compute time inside MUL_MAT_ID (GGML_CPU_PROF): shows imbalance between cores.
static struct { int64_t us; char pad[56]; } ggml_cpu_prof_mmid_thr[GGML_MAX_N_THREADS];

#define GGML_CPU_RANDOM_MAX 1024
static struct { const void * data; size_t size; } ggml_cpu_random_done[GGML_CPU_RANDOM_MAX];
static int ggml_cpu_random_n = 0;

static void ggml_cpu_madvise(const void * p, size_t n, int advice) {
#if !defined(_WIN32)
    static uintptr_t page = 0;
    if (!page) {
        page = (uintptr_t) sysconf(_SC_PAGESIZE);
    }
    const uintptr_t b = (uintptr_t) p & ~(page - 1);
    const uintptr_t e = ((uintptr_t) p + n + page - 1) & ~(page - 1);
    madvise((void *) b, e - b, advice);
#else
    GGML_UNUSED(p); GGML_UNUSED(n); GGML_UNUSED(advice);
#endif
}

// Vollie: env GGML_DENSE_WILLNEED=1 (0 / unset = off; advice only, results unchanged; 05_compute_less.md §7.11 (iii)): at
// every prompt graph (n_tokens > 1; once per reply), MADV_WILLNEED over every model weight the graph reads except the
// routed experts (*_exps), so the dense set (Granite: ≈ 562 MB, the tied head included: GGML_HEAD2 still recomputes its
// top rows from it) is read back from flash in large requests while the prompt runs, instead of a page per fault in the
// first decode tokens. "cpu-prof willneed tensors=<n> mb=<MB> us=<µs>" under GGML_CPU_PROF. Calling thread, graph start.
static bool ggml_cpu_dense_willneed = false;
static void ggml_cpu_dense_advise(const struct ggml_cgraph * g) {
    const int64_t t0 = ggml_time_us();
    const void * seen[2048];
    int n = 0;
    size_t bytes = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        for (int j = 0; j < GGML_MAX_SRC && g->nodes[i]->src[j]; j++) {
            const struct ggml_tensor * t = g->nodes[i]->src[j];
            if (!t->buffer || !t->data || ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                strstr(t->name, "_exps")) {
                continue;
            }
            int k = 0;
            while (k < n && seen[k] != t->data) {
                k++;
            }
            if (k < n || n == (int) (sizeof seen / sizeof seen[0])) { // ponytail: past 2,048 weights, the rest are not advised
                continue;
            }
            seen[n++] = t->data;
            bytes += ggml_nbytes(t);
#ifdef MADV_WILLNEED
            ggml_cpu_madvise(t->data, ggml_nbytes(t), MADV_WILLNEED);
#endif
        }
    }
    if (ggml_cpu_prof_on) {
        fprintf(stderr, "cpu-prof willneed tensors=%d mb=%.1f us=%lld\n", n, bytes/1e6, (long long) (ggml_time_us() - t0));
    }
}

#if !defined(_WIN32)
static void ggml_cpu_expert_random(const struct ggml_tensor * w) {
    for (int i = 0; i < ggml_cpu_random_n; i++) {
        if (ggml_cpu_random_done[i].data == w->data) {
            return;
        }
    }
    if (ggml_cpu_random_n == GGML_CPU_RANDOM_MAX) {
        return;
    }
    ggml_cpu_random_done[ggml_cpu_random_n].data = w->data;
    ggml_cpu_random_done[ggml_cpu_random_n].size = ggml_nbytes(w);
    ggml_cpu_random_n++;
    ggml_cpu_madvise(w->data, ggml_nbytes(w), MADV_RANDOM);
}

// Bit 4: also guess the NEXT layer's experts and prefetch them a layer early, so their
// reads overlap this layer's MoE and the next layer's Mamba block instead of stalling
// the expert mat-muls. The guess applies layer L+1's router (and its RMS-norm weight) to
// layer L's FFN input; a helper thread does the mat-vec and the madvise calls, off the
// compute threads (Vollie runs 3 threads on the SD888's 4 big cores). Wrong guesses cost
// only a wasted read: the real routing still prefetches as above. The hit rate shows in
// the GGML_CPU_PROF line (pred=: share of the real experts that were guessed).
#define GGML_CPU_PRED_N 8192
#define GGML_CPU_PRED_Q 4
typedef struct {
    float           x[GGML_CPU_PRED_N];
    int             n, n_exp, k, nw;
    float           eps;
    const float   * router; // [n, n_exp], F32
    const float   * gamma;  // [n], F32
    const char    * wdata[3];
    size_t          wnb2[3];
    const void    * key;    // the next layer's first expert tensor
} ggml_cpu_pred_job;

static ggml_cpu_pred_job ggml_cpu_pred_q[GGML_CPU_PRED_Q];
static atomic_int        ggml_cpu_pred_head = 0, ggml_cpu_pred_tail = 0; // one producer (thread 0), one consumer
static pthread_mutex_t   ggml_cpu_pred_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t    ggml_cpu_pred_cv = PTHREAD_COND_INITIALIZER;
static bool              ggml_cpu_pred_started = false;

#define GGML_CPU_PRED_KEYS 128
static struct { const void * key; _Atomic uint64_t mask; } ggml_cpu_pred_masks[GGML_CPU_PRED_KEYS];
static int64_t ggml_cpu_pred_hit = 0, ggml_cpu_pred_total = 0;

static void ggml_cpu_pred_store(const void * key, uint64_t mask) {
    for (int i = 0; i < GGML_CPU_PRED_KEYS; i++) {
        if (ggml_cpu_pred_masks[i].key == key || ggml_cpu_pred_masks[i].key == NULL) {
            ggml_cpu_pred_masks[i].key = key;
            atomic_store(&ggml_cpu_pred_masks[i].mask, mask);
            return;
        }
    }
}

static uint64_t ggml_cpu_pred_take(const void * key) {
    for (int i = 0; i < GGML_CPU_PRED_KEYS && ggml_cpu_pred_masks[i].key; i++) {
        if (ggml_cpu_pred_masks[i].key == key) {
            return atomic_exchange(&ggml_cpu_pred_masks[i].mask, 0);
        }
    }
    return 0;
}

static void ggml_cpu_pred_run(const ggml_cpu_pred_job * j) {
    float ss = 0.0f;
    for (int i = 0; i < j->n; i++) {
        ss += j->x[i]*j->x[i];
    }
    const float scale = 1.0f/sqrtf(ss/j->n + j->eps);
    float v[GGML_CPU_PRED_N];
    for (int i = 0; i < j->n; i++) {
        v[i] = j->x[i]*scale*j->gamma[i];
    }
    float logit[64];
    for (int e = 0; e < j->n_exp; e++) {
        const float * w = j->router + (size_t) e*j->n;
        float d = 0.0f;
        for (int i = 0; i < j->n; i++) {
            d += w[i]*v[i];
        }
        logit[e] = d;
    }
    uint64_t mask = 0;
    for (int t = 0; t < j->k; t++) { // top-k (softmax keeps the order)
        int best = -1;
        for (int e = 0; e < j->n_exp; e++) {
            if (!(mask >> e & 1) && (best < 0 || logit[e] > logit[best])) {
                best = e;
            }
        }
        mask |= 1ull << best;
        for (int w = 0; w < j->nw; w++) {
            ggml_cpu_madvise(j->wdata[w] + (size_t) best*j->wnb2[w], j->wnb2[w], MADV_WILLNEED);
        }
    }
    ggml_cpu_pred_store(j->key, mask);
}

// GGML_HELPER_CPUS: "0-3", "0,1,2,3" or a mix. Re-read per job so an app can change it on a
// running helper; the affinity call happens only when the value changes.
static void ggml_cpu_helper_parse(void) {
    static char last[64] = {0};
    const char * s = getenv("GGML_HELPER_CPUS");
    if (!s) {
        s = "";
    }
    if (strncmp(s, last, sizeof last - 1) == 0) {
        ggml_cpu_helper_mask_set = false; // unchanged: nothing to apply
        return;
    }
    snprintf(last, sizeof last, "%s", s);
    memset(ggml_cpu_helper_mask, 0, sizeof ggml_cpu_helper_mask);
    if (!*s) {
        // back to every core (the compute threads' pinning no longer applies to the helper)
        for (int c = 0; c < GGML_MAX_N_THREADS; c++) {
            ggml_cpu_helper_mask[c] = true;
        }
        ggml_cpu_helper_mask_set = true;
        return;
    }
    int n = 0;
    while (*s) {
        char * end;
        long a = strtol(s, &end, 10), b = a;
        if (end == s) {
            break;
        }
        s = end;
        if (*s == '-') {
            b = strtol(s + 1, &end, 10);
            s = end;
        }
        for (long c = a; c <= b && c < GGML_MAX_N_THREADS; c++) {
            if (c >= 0) {
                ggml_cpu_helper_mask[c] = true;
                n++;
            }
        }
        if (*s == ',') {
            s++;
        }
    }
    ggml_cpu_helper_mask_set = n > 0;
}

static void * ggml_cpu_pred_main(void * arg) {
    GGML_UNUSED(arg);
    for (;;) {
        pthread_mutex_lock(&ggml_cpu_pred_mu);
        while (atomic_load(&ggml_cpu_pred_tail) == atomic_load(&ggml_cpu_pred_head)) {
            pthread_cond_wait(&ggml_cpu_pred_cv, &ggml_cpu_pred_mu);
        }
        pthread_mutex_unlock(&ggml_cpu_pred_mu);
        ggml_cpu_helper_parse();
        if (ggml_cpu_helper_mask_set) {
            ggml_thread_apply_affinity(ggml_cpu_helper_mask);
        }
        const int t = atomic_load(&ggml_cpu_pred_tail);
        ggml_cpu_pred_run(&ggml_cpu_pred_q[t % GGML_CPU_PRED_Q]);
        atomic_store(&ggml_cpu_pred_tail, t + 1); // after the run: the graph waits for this
    }
    return NULL;
}

// Called by thread 0 at the end of each graph: no job may outlive the weights it reads.
static void ggml_cpu_pred_drain(void) {
    while (atomic_load(&ggml_cpu_pred_tail) != atomic_load(&ggml_cpu_pred_head)) {
        sched_yield();
    }
}

static const struct ggml_tensor * ggml_cpu_unview(const struct ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

// At layer L's first MUL_MAT_ID (node_n): queue the guess for layer L+1.
static void ggml_cpu_pred_submit(const struct ggml_cgraph * cgraph, int node_n) {
    // layer L's FFN input: MUL_MAT_ID src1 = view of MUL(RMS_NORM(ffn_inp), gamma)
    const struct ggml_tensor * mul = ggml_cpu_unview(cgraph->nodes[node_n]->src[1]);
    if (!mul || mul->op != GGML_OP_MUL || !mul->src[0] || mul->src[0]->op != GGML_OP_RMS_NORM) {
        return;
    }
    const struct ggml_tensor * x = mul->src[0]->src[0];
    if (x->type != GGML_TYPE_F32 || x->ne[1] != 1 || x->ne[0] > GGML_CPU_PRED_N || !ggml_is_contiguous(x)) {
        return; // one token at a time only
    }
    // layer L+1: its router mat-mul, then its expert mat-muls
    int r = -1;
    for (int j = node_n + 1; j < cgraph->n_nodes && j < node_n + 200; j++) {
        const struct ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_MUL_MAT && strstr(n->src[0]->name, "ffn_gate_inp")) {
            r = j;
            break;
        }
    }
    if (r < 0) {
        return;
    }
    const struct ggml_tensor * router = cgraph->nodes[r]->src[0];
    const struct ggml_tensor * nmul = ggml_cpu_unview(cgraph->nodes[r]->src[1]);
    if (router->type != GGML_TYPE_F32 || router->ne[0] != x->ne[0] || router->ne[1] > 64 ||
        !nmul || nmul->op != GGML_OP_MUL || !nmul->src[0] || nmul->src[0]->op != GGML_OP_RMS_NORM) {
        return;
    }
    const struct ggml_tensor * gamma = nmul->src[1];
    if (gamma->type != GGML_TYPE_F32 || gamma->ne[0] != x->ne[0]) {
        return;
    }
    const int h = atomic_load(&ggml_cpu_pred_head);
    if (h - atomic_load(&ggml_cpu_pred_tail) >= GGML_CPU_PRED_Q) {
        return; // helper behind: skip this guess
    }
    ggml_cpu_pred_job * job = &ggml_cpu_pred_q[h % GGML_CPU_PRED_Q];
    job->nw = 0;
    const struct ggml_tensor * ids = NULL;
    for (int j = r + 1; j < cgraph->n_nodes && j < r + 64 && job->nw < 3; j++) {
        const struct ggml_tensor * n = cgraph->nodes[j];
        if (n->op != GGML_OP_MUL_MAT_ID || n->src[0]->data == NULL || (ids && n->src[2] != ids)) {
            continue;
        }
        ids = n->src[2];
        job->k = (int) ids->ne[0];
        job->wdata[job->nw] = (const char *) n->src[0]->data;
        job->wnb2[job->nw]  = n->src[0]->nb[2];
        if (job->nw == 0) {
            job->key = n->src[0]->data;
        }
        job->nw++;
    }
    // bit 8: guess 2 more than the model routes to (host, en.txt: top-6 covers 83.6% of
    // the real experts, top-8 91.6%, top-10 94.7%), at the cost of reading unused ones
    if (ggml_cpu_expert_io & 8) {
        job->k += 2;
    }
    if (job->nw == 0 || job->k > (int) router->ne[1]) {
        return;
    }
    job->n      = (int) x->ne[0];
    job->n_exp  = (int) router->ne[1];
    job->router = (const float *) router->data;
    job->gamma  = (const float *) gamma->data;
    memcpy(&job->eps, nmul->src[0]->op_params, sizeof(float));
    memcpy(job->x, x->data, x->ne[0]*sizeof(float));
    if (!ggml_cpu_pred_started) {
        pthread_t t;
        pthread_create(&t, NULL, ggml_cpu_pred_main, NULL); // inherits this thread's affinity
        pthread_detach(t);
        ggml_cpu_pred_started = true;
    }
    pthread_mutex_lock(&ggml_cpu_pred_mu);
    atomic_store(&ggml_cpu_pred_head, h + 1);
    pthread_cond_signal(&ggml_cpu_pred_cv);
    pthread_mutex_unlock(&ggml_cpu_pred_mu);
}

// Thread 0, before node_n (a MUL_MAT_ID) runs; its ids are complete (barrier before it).
static void ggml_cpu_expert_prefetch(const struct ggml_cgraph * cgraph, int node_n) {
    const struct ggml_tensor * ids = cgraph->nodes[node_n]->src[2];
    for (int j = node_n - 1; j >= 0 && j >= node_n - 32; j--) {
        if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j]->src[2] == ids) {
            return; // an earlier sibling already covered this layer
        }
    }
    const int64_t n_as = cgraph->nodes[node_n]->src[0]->ne[2];
    if (n_as > 512) {
        return;
    }
    // Vollie (r6): the skip mask is built first (same decision the op makes next), so experts whose
    // every slot is skipped are not read ahead: used[] = routed (bit 4's hit count), need[] = advised
    const bool masked = ggml_cpu_skip_prepare(cgraph->nodes[node_n]);
    uint8_t used[512] = {0}, need[512] = {0};
    for (int64_t t = 0; t < ids->ne[1]; t++) {
        for (int64_t k = 0; k < ids->ne[0]; k++) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + k*ids->nb[0]);
            if (e >= 0 && e < n_as) {
                used[e] = 1;
                need[e] |= !(masked && ggml_cpu_skip_mask_tl[t*ids->ne[0] + k]);
            }
        }
    }
    if (ggml_cpu_expert_log && ids->ne[1] == 1) {
        fprintf(stderr, "eid %s", cgraph->nodes[node_n]->src[0]->name);
        for (int64_t k = 0; k < ids->ne[0]; k++) {
            fprintf(stderr, " %d", *(const int32_t *) ((const char *) ids->data + k*ids->nb[0]));
        }
        fprintf(stderr, "\n");
    }
    if ((ggml_cpu_expert_io & 4) && ids->ne[1] == 1 && n_as <= 64) {
        const uint64_t guess = ggml_cpu_pred_take(cgraph->nodes[node_n]->src[0]->data);
        if (guess) {
            for (int64_t e = 0; e < n_as; e++) {
                ggml_cpu_pred_hit += used[e] && (guess >> e & 1);
            }
            ggml_cpu_pred_total += ids->ne[0];
        }
        ggml_cpu_pred_submit(cgraph, node_n);
    }
    for (int j = node_n; j < cgraph->n_nodes && j < node_n + 32; j++) {
        const struct ggml_tensor * n = cgraph->nodes[j];
        if (n->op != GGML_OP_MUL_MAT_ID || n->src[2] != ids || n->src[0]->data == NULL) {
            continue;
        }
        const struct ggml_tensor * w = n->src[0];
        if (ggml_cpu_expert_io & 2) {
            ggml_cpu_expert_random(w);
        }
        if ((ggml_cpu_expert_io & 1) && !ggml_cpu_expq4_find(w)) { // Vollie: an EXPERT_Q4_LAYERS copy needs no read
            const struct ggml_cpu_anon_ent * anon = ggml_cpu_anon_find(w); // cached experts need no read
            for (int64_t e = 0; e < n_as && e < w->ne[2]; e++) {
                if (need[e] && !(anon && anon->slot_of[e] >= 0)) {
                    ggml_cpu_madvise((const char *) w->data + e*w->nb[2], w->nb[2], MADV_WILLNEED);
                    ggml_cpu_prof_adv[ggml_cpu_ab_var] += ggml_cpu_prof_this;
                }
            }
        }
    }
}
#endif

// Vollie: the process's (major, minor) page faults so far (getrusage RUSAGE_SELF: every thread); 0 on Windows
static void ggml_cpu_faults(int64_t f[2]) {
#if !defined(_WIN32)
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        f[0] = ru.ru_majflt;
        f[1] = ru.ru_minflt;
        return;
    }
#endif
    f[0] = f[1] = 0;
}

// Vollie r15: cpu<n>'s current clock in MHz from sysfs (the kit's mhz_blocks source); -1 where there is none (host)
static int ggml_cpu_mhz(int cpu) {
    char path[64];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE * f = fopen(path, "r");
    long khz = -1000;
    if (f) {
        if (fscanf(f, "%ld", &khz) != 1) khz = -1000;
        fclose(f);
    }
    return (int) (khz / 1000);
}

static void ggml_cpu_prof_dump_one(int v, int n_threads) {
    int idx[GGML_OP_COUNT];
    int n = 0;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        if (ggml_cpu_prof_cnt[v][i]) idx[n++] = i;
    }
    for (int i = 1; i < n; i++) { // by compute + wait, descending
        for (int j = i; j > 0 && ggml_cpu_prof_comp[v][idx[j]] + ggml_cpu_prof_wait[v][idx[j]] >
                                 ggml_cpu_prof_comp[v][idx[j-1]] + ggml_cpu_prof_wait[v][idx[j-1]]; j--) {
            int t = idx[j]; idx[j] = idx[j-1]; idx[j-1] = t;
        }
    }
    const int g = ggml_cpu_prof_graphs[v];
    if (g == 0) {
        return;
    }
    int64_t comp = 0, wait = 0;
    for (int i = 0; i < n; i++) { comp += ggml_cpu_prof_comp[v][idx[i]]; wait += ggml_cpu_prof_wait[v][idx[i]]; }
    char line[2048];
    char ab[32] = "";
    const char * rcast = ggml_cpu_prof_rcast[v] == GGML_TYPE_COUNT ? "f32" : ggml_type_name((enum ggml_type) ggml_cpu_prof_rcast[v]);
    if (ggml_cpu_ab_mode == 3 && ggml_cpu_ab_token) { // Vollie r15: the per-token line, short (200 per story run go to the clipboard)
        // R46 am. 1: no compute= / wait= (the op table has them) and no rcast= (constant per run, in the run's env)
        int len = snprintf(line, sizeof line, "cpu-prof ab=%c tok=%d us/token=%lld head=%lld router=%lld "
            "majflt=%lld minflt=%lld mhz=%d/%d nodes=%d barriers=%d kern=%d fuse=%d chunk2=%d |", v ? 'B' : 'A', ggml_cpu_ab_blk,
            (long long) (ggml_cpu_prof_total[v] / g), (long long) (ggml_cpu_prof_head[v] / g),
            (long long) (ggml_cpu_prof_router[v] / g), (long long) ggml_cpu_prof_flt[v][0], (long long) ggml_cpu_prof_flt[v][1],
            ggml_cpu_mhz(4), ggml_cpu_mhz(7), ggml_cpu_prof_nodes[v] / g, ggml_cpu_prof_barriers[v] / g, ggml_cpu_prof_kern[v],
            ggml_cpu_prof_fuse[v], ggml_cpu_prof_chunk2[v]);
        for (int i = 0; i < n && len < (int) sizeof line - 64; i++) {
            const int o = idx[i];
            if (o == GGML_OP_MUL_MAT_ID || o == GGML_OP_MUL_MAT || o == GGML_OP_GLU || o == GGML_OP_SSM_SCAN) { // R46 am. 1: scored ops only
                len += snprintf(line + len, sizeof line - len, " %s %d:%lld+%lld", ggml_op_name((enum ggml_op) o),
                    ggml_cpu_prof_cnt[v][o] / g, (long long) (ggml_cpu_prof_comp[v][o] / g), (long long) (ggml_cpu_prof_wait[v][o] / g));
            }
        }
        fprintf(stderr, "%s\n", line);
        fflush(stderr);
        return;
    }
    if (ggml_cpu_ab_mode == 3) {
        snprintf(ab, sizeof ab, " ab=%c blk=%d", v ? 'B' : 'A', ggml_cpu_ab_blk);
    } else if (ggml_cpu_ab_mode || v) {
        snprintf(ab, sizeof ab, "%s", v ? " ab=B" : " ab=A"); // Vollie: side B is always labelled
    }
    // Vollie: kavg = kept expert slots per layer and token on this side; anon=N/hit%/tensor copies per
    // decode token (both sides together: one cache); ilp, pf = GGML_Q3K_ILP / GGML_Q3K_PF on this side
    int len = snprintf(line, sizeof line, "cpu-prof%s graphs=%d tokens=%d threads=%d tiny=%lld eio=%d q3k=%d x4=%d ilp=%d pf=%d mmid=%d skip=%d/%lld%% kavg=%.2f anon=%d/%lld%%/%.1f anon_mb=%lld pred=%lld%% adv=%lld us/token=%lld compute=%lld wait=%lld head2=%lld head=%lld majflt=%.1f minflt=%.1f nodes=%d barriers=%d chunk=%d expq4=%d kern=%d touch=%d fuse=%d chunk2=%d router=%lld rcast=%s mmid_thr=",
        ab, g, ggml_cpu_prof_tokens[v], n_threads,
        (long long) (ggml_cpu_ab_mode == 2 && v ? ggml_cpu_ab_tiny : ggml_cpu_tiny_thresh), ggml_cpu_expert_io,
        ggml_cpu_ab_mode == 1 ? (v ? 0 : 1) : ggml_cpu_q3k_vec, ggml_cpu_q3k_x4, ggml_cpu_prof_ilp[v], ggml_cpu_prof_pf[v], ggml_cpu_mmid,
        ggml_cpu_prof_skip[v], (long long) (ggml_cpu_skip_total[v] ? 100*ggml_cpu_skip_cnt[v]/ggml_cpu_skip_total[v] : 0),
        ggml_cpu_prof_ktok[v] ? (double) ggml_cpu_prof_kept[v] / ggml_cpu_prof_ktok[v] : -1.0,
        ggml_cpu_prof_anon[v] ? ggml_cpu_anon_alloc_n : 0,
        (long long) (ggml_cpu_anon_routed ? 100*ggml_cpu_anon_hit/ggml_cpu_anon_routed : -1),
        ggml_cpu_anon_dtok ? (double) ggml_cpu_anon_copies / ggml_cpu_anon_dtok : 0.0, (long long) (ggml_cpu_anon_bytes >> 20),
        (long long) (ggml_cpu_pred_total ? 100*ggml_cpu_pred_hit/ggml_cpu_pred_total : -1),
        (long long) (ggml_cpu_prof_tokens[v] ? ggml_cpu_prof_adv[v] / ggml_cpu_prof_tokens[v] : 0),
        (long long) (ggml_cpu_prof_total[v] / g),
        (long long) (comp / g), (long long) (wait / g), (long long) ggml_cpu_prof_h2[v], (long long) (ggml_cpu_prof_head[v] / g),
        ggml_cpu_prof_tokens[v] ? (double) ggml_cpu_prof_flt[v][0] / ggml_cpu_prof_tokens[v] : 0.0,
        ggml_cpu_prof_tokens[v] ? (double) ggml_cpu_prof_flt[v][1] / ggml_cpu_prof_tokens[v] : 0.0, ggml_cpu_prof_nodes[v] / g, ggml_cpu_prof_barriers[v] / g,
        ggml_cpu_prof_chunk[v], ggml_cpu_prof_expq4[v], ggml_cpu_prof_kern[v], ggml_cpu_prof_touch[v],
        ggml_cpu_prof_fuse[v], ggml_cpu_prof_chunk2[v], (long long) (ggml_cpu_prof_router[v] / g), rcast);
    // per-thread µs computing inside MUL_MAT_ID (both A/B variants together): the spread is the imbalance
    for (int t = 0; t < n_threads && t < GGML_MAX_N_THREADS && len < (int) sizeof line - 64; t++) {
        len += snprintf(line + len, sizeof line - len, "%s%lld", t ? "/" : "",
                        (long long) (ggml_cpu_prof_mmid_thr[t].us / (ggml_cpu_prof_graphs[0] + ggml_cpu_prof_graphs[1])));
    }
    len += snprintf(line + len, sizeof line - len, " |");
    for (int i = 0; i < n && i < 14 && len < (int) sizeof line - 64; i++) {
        const int o = idx[i];
        len += snprintf(line + len, sizeof line - len, " %s %d:%lld+%lld", ggml_op_name((enum ggml_op) o),
            ggml_cpu_prof_cnt[v][o] / g, (long long) (ggml_cpu_prof_comp[v][o] / g), (long long) (ggml_cpu_prof_wait[v][o] / g));
    }
    fprintf(stderr, "%s\n", line);
    fflush(stderr);
}

// Vollie: empties the profile window (after a dump, and at a prompt graph: see ggml_graph_compute)
static void ggml_cpu_prof_reset(void) {
    ggml_cpu_pred_hit = ggml_cpu_pred_total = 0;
    memset(ggml_cpu_skip_cnt, 0, sizeof ggml_cpu_skip_cnt);
    memset(ggml_cpu_skip_total, 0, sizeof ggml_cpu_skip_total);
    memset(ggml_cpu_prof_adv, 0, sizeof ggml_cpu_prof_adv);
    memset(ggml_cpu_prof_head, 0, sizeof ggml_cpu_prof_head);
    memset(ggml_cpu_prof_router, 0, sizeof ggml_cpu_prof_router);
    memset(ggml_cpu_prof_flt, 0, sizeof ggml_cpu_prof_flt);
    ggml_cpu_anon_hit = ggml_cpu_anon_routed = ggml_cpu_anon_copies = ggml_cpu_anon_dtok = 0;
    memset(ggml_cpu_prof_kept, 0, sizeof ggml_cpu_prof_kept);
    memset(ggml_cpu_prof_ktok, 0, sizeof ggml_cpu_prof_ktok);
    memset(ggml_cpu_prof_anon, 0, sizeof ggml_cpu_prof_anon);
    memset(ggml_cpu_prof_mmid_thr, 0, sizeof ggml_cpu_prof_mmid_thr);
    memset(ggml_cpu_prof_comp, 0, sizeof ggml_cpu_prof_comp);
    memset(ggml_cpu_prof_wait, 0, sizeof ggml_cpu_prof_wait);
    memset(ggml_cpu_prof_cnt, 0, sizeof ggml_cpu_prof_cnt);
    memset(ggml_cpu_prof_graphs, 0, sizeof ggml_cpu_prof_graphs);
    memset(ggml_cpu_prof_tokens, 0, sizeof ggml_cpu_prof_tokens);
    memset(ggml_cpu_prof_nodes, 0, sizeof ggml_cpu_prof_nodes);
    memset(ggml_cpu_prof_barriers, 0, sizeof ggml_cpu_prof_barriers);
    memset(ggml_cpu_prof_total, 0, sizeof ggml_cpu_prof_total);
}

static void ggml_cpu_prof_dump(int n_threads) {
    ggml_cpu_prof_dump_one(0, n_threads);
    ggml_cpu_prof_dump_one(1, n_threads);
    ggml_cpu_prof_reset();
}

static bool ggml_cpu_node_is_tiny(const struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_CUSTOM:
        case GGML_OP_MAP_CUSTOM1:
        case GGML_OP_MAP_CUSTOM2:
        case GGML_OP_MAP_CUSTOM3:
        case GGML_OP_ACC:
        case GGML_OP_SET:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_ADD_REL_POS:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_CROSS_ENTROPY_LOSS:
            return false; // real work, or ops that call ggml_barrier themselves
        default:
            break;
    }
    int64_t n = ggml_nelements(node);
    // GET_ROWS reads only the selected rows of src[0] (a whole embedding or
    // state table): size it by its output and index list.
    for (int i = node->op == GGML_OP_GET_ROWS ? 1 : 0; i < GGML_MAX_SRC && node->src[i]; i++) {
        n = MAX(n, ggml_nelements(node->src[i]));
    }
    return n <= ggml_cpu_tiny_thresh;
}

// The compute loop's decision for node j: whether it runs fused with j + 1
// (RMS_NORM + MUL), and whether it runs tiny (thread 0 alone). A fused pair is
// tiny only if both nodes are; the lookahead uses the same rule, so a barrier is
// skipped only before work that thread 0 does alone.
static bool ggml_cpu_runs_tiny(const struct ggml_cgraph * cgraph, int j, const struct ggml_cplan * cplan, int * n_fused) {
    *n_fused = ggml_cpu_can_fuse_rms_norm_mul(cgraph, j, cplan) ? 1 : 0;
    return ggml_cpu_node_is_tiny(cgraph->nodes[j]) && (*n_fused == 0 || ggml_cpu_node_is_tiny(cgraph->nodes[j + 1]));
}

// Same decision the compute loop makes for the next node: fused nodes run on
// every thread, so they need the barrier even when small.
static bool ggml_cpu_next_is_tiny(const struct ggml_cgraph * cgraph, int node_n, const struct ggml_cplan * cplan) {
    for (int j = node_n + 1; j < cgraph->n_nodes; j++) {
        const struct ggml_tensor * next = cgraph->nodes[j];
        if (ggml_op_is_empty(next->op) || (next->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        int n_fused;
        return ggml_cpu_runs_tiny(cgraph, j, cplan, &n_fused);
    }
    return false;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(state->ith);
#else
    set_numa_thread_affinity(state->ith);
#endif

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    // Vollie: barrier micro-benchmark (GGML_BARRIER_BENCH), decided for all threads before the graph
    if (ggml_cpu_bbench_this && params.nth > 1) {
        ggml_barrier(tp);
        const int64_t b0 = ggml_time_us();
        for (int i = 0; i < ggml_cpu_bbench_n; i++) {
            ggml_barrier(tp);
        }
        const int64_t b1 = ggml_time_us();
        if (state->ith == 0) {
            fprintf(stderr, "cpu-bench barriers=%d threads=%d us/barrier=%.2f\n",
                    ggml_cpu_bbench_n, params.nth, (double) (b1 - b0) / ggml_cpu_bbench_n);
            fflush(stderr);
        }
    }
    ggml_cpu_mmid_seq_tl = 0;

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        int n_fused;
        const bool tiny = ggml_cpu_runs_tiny(cgraph, node_n, cplan, &n_fused) && params.nth > 1;
        const bool prof = ggml_cpu_prof_this && state->ith == 0;
        const bool mmid = node->op == GGML_OP_MUL_MAT_ID;
        const bool mfuse = mmid && ggml_cpu_mmid_fuse_ok(cgraph, node_n, cplan, params.nth); // Vollie: GGML_MMID_FUSE
        const int64_t t0 = (prof || (ggml_cpu_prof_this && mmid)) ? ggml_time_us() : 0;
        if (mmid) {
            ggml_cpu_skip_weights_tl = ggml_cpu_expert_skip || ggml_cpu_expert_topp || ggml_cpu_expert_gap ? ggml_cpu_moe_weights(cgraph, node_n) : NULL;
        }
#if !defined(_WIN32)
        if (ggml_cpu_expert_io && state->ith == 0 && mmid) {
            ggml_cpu_expert_prefetch(cgraph, node_n); // after the weights pointer: it applies the skip mask
        }
#endif
        if (tiny && state->ith != 0) {
            // thread 0 does it alone
        } else {
            struct ggml_compute_params p = params;
            if (tiny) {
                p.nth = 1;
            }
            if (mfuse) {
                ggml_cpu_mmid_decode_fused(&p, cgraph->nodes[node_n + 2]);
            } else if (n_fused > 0) {
                ggml_compute_forward_rms_norm_mul_fused(&p, node, cgraph->nodes[node_n + 1]);
            } else {
                ggml_compute_forward(&p, node);
            }
        }
        node_n += mfuse ? 2 : n_fused; // Vollie: a fused gate + up + SWIGLU counts as one MUL_MAT_ID (one bank, one barrier)
        const int64_t t1 = (prof || (ggml_cpu_prof_this && mmid)) ? ggml_time_us() : 0;
        if (mmid) {
            if (ggml_cpu_prof_this) {
                ggml_cpu_prof_mmid_thr[state->ith].us += t1 - t0;
            }
            if (state->ith == 0 && ggml_cpu_skip_on_tl) {
                const struct ggml_tensor * ids = node->src[2];
                int64_t cnt = 0;
                for (int64_t i = 0; i < ids->ne[0]*ids->ne[1]; i++) {
                    cnt += ggml_cpu_skip_mask_tl[i];
                }
                if (ggml_cpu_prof_this) { // Vollie: skip share per A/B side, profiled graphs only
                    ggml_cpu_skip_cnt[ggml_cpu_ab_var]   += cnt;
                    ggml_cpu_skip_total[ggml_cpu_ab_var] += ids->ne[0]*ids->ne[1];
                }
                if (ggml_cpu_skip_down_tl) {
                    ggml_cpu_budget_skipped += cnt;
                    ggml_cpu_budget_routed  += ids->ne[0]*ids->ne[1];
                    ggml_cpu_budget_tok     += ids->ne[1];
                    int l = -1;
                    if (sscanf(node->src[0]->name, "blk.%d.", &l) == 1 && l >= 0 && l < 128) {
                        ggml_cpu_budget_lkept[l] += ids->ne[0]*ids->ne[1] - cnt;
                        ggml_cpu_budget_ltok[l]  += ids->ne[1];
                    }
                }
            }
            if (state->ith == 0 && ggml_cpu_prof_this && node->src[1]->ne[1] == node->src[2]->ne[0]) {
                // Vollie: kept slots per layer of profiled graphs (down projections: one row per slot)
                const struct ggml_tensor * ids = node->src[2];
                int64_t kept = ids->ne[0]*ids->ne[1];
                for (int64_t i = 0; ggml_cpu_skip_on_tl && i < ids->ne[0]*ids->ne[1]; i++) {
                    kept -= ggml_cpu_skip_mask_tl[i];
                }
                ggml_cpu_prof_kept[ggml_cpu_ab_var] += kept;
                ggml_cpu_prof_ktok[ggml_cpu_ab_var] += ids->ne[1];
            }
            ggml_cpu_mmid_seq_tl++;
        }

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        const bool barrier = node_n + 1 < cgraph->n_nodes && !(tiny && ggml_cpu_next_is_tiny(cgraph, node_n, cplan));
        if (barrier) {
            if (state->ith == 0 && (ggml_cpu_mmid & 2)) {
                // Vollie: the next compute node, if a MUL_MAT_ID, gets its chunk counters reset here
                for (int j = node_n + 1; j < cgraph->n_nodes; j++) {
                    const struct ggml_tensor * next = cgraph->nodes[j];
                    if (ggml_op_is_empty(next->op) || (next->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                        continue;
                    }
                    if (next->op == GGML_OP_MUL_MAT_ID) {
                        ggml_cpu_mmid_prepare_next(next, params.nth, cplan->work_size);
                    }
                    break;
                }
            }
            ggml_barrier(state->threadpool);
            if (mmid && state->ith == 0 && ggml_cpu_anon_alloc_n) {
                ggml_cpu_anon_update(node); // Vollie: GGML_EXPERT_ANON, every thread is past this op
            }
        }
        if (prof) {
            const int v = ggml_cpu_ab_var;
            ggml_cpu_prof_comp[v][node->op] += t1 - t0;
            ggml_cpu_prof_wait[v][node->op] += ggml_time_us() - t1;
            if (node->op == GGML_OP_MUL_MAT && ggml_cpu_is_head(node)) {
                ggml_cpu_prof_head[v] += ggml_time_us() - t0; // Vollie: the head alone (GGML_HEAD2's op)
            } else if (node->op == GGML_OP_MUL_MAT && ggml_cpu_is_router(node)) {
                ggml_cpu_prof_router[v] += ggml_time_us() - t0; // Vollie r15: the routers alone (C1-F16's ops)
            }
            ggml_cpu_prof_cnt[v][node->op]++;
            ggml_cpu_prof_nodes[v]++;
            ggml_cpu_prof_barriers[v] += barrier;
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(state->ith);
#endif

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    // Set before any worker reads it for this graph.
    const char * tiny = getenv("GGML_TINY_OP");
    ggml_cpu_tiny_thresh = tiny ? atoll(tiny) : 0;
    const char * eio = getenv("GGML_EXPERT_IO");
    ggml_cpu_expert_io = eio ? atoi(eio) : 0;
    const char * elog = getenv("GGML_EXPERT_LOG");
    ggml_cpu_expert_log = elog && atoi(elog) == 1;
    if (!(ggml_cpu_expert_io & 2) && ggml_cpu_random_n > 0) {
        // switched off: back to normal read-around
        for (int i = 0; i < ggml_cpu_random_n; i++) {
            ggml_cpu_madvise(ggml_cpu_random_done[i].data, ggml_cpu_random_done[i].size, MADV_NORMAL);
        }
        ggml_cpu_random_n = 0;
    }
    const char * prof = getenv("GGML_CPU_PROF");
    // 1: one-token decode graphs; 2: 2-8 token graphs (speculative verify batches; us/token is then per graph)
    const int prof_mode = prof ? atoi(prof) : 0;
    ggml_cpu_prof_on = prof_mode == 1 || prof_mode == 2;
    // One token: the first node is the embedding lookup, {n_embd, n_tokens}.
    int64_t n_tok = cgraph->n_nodes > 0 ? cgraph->nodes[0]->ne[1] : 0;
    const bool decode = n_tok == 1;
    if (prof_mode == 2) {
        // Vollie: 2 counts only the main model's graphs (largest token embedding seen; a draft's is
        // smaller), with the token count read off its embedding lookup, not off the first node
        static int64_t main_embd = 0;
        n_tok = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const struct ggml_tensor * n = cgraph->nodes[i];
            if (n->op == GGML_OP_GET_ROWS && strncmp(n->src[0]->name, "token_embd", 10) == 0) {
                main_embd = MAX(main_embd, n->src[0]->ne[0]);
                n_tok = n->src[0]->ne[0] == main_embd ? n->src[1]->ne[0] : 0;
                break;
            }
        }
    }
    ggml_cpu_prof_this = prof_mode == 1 ? decode : prof_mode == 2 && n_tok >= 2 && n_tok <= 8;
    if (n_tok > 8 && ggml_cpu_prof_graphs[0] + ggml_cpu_prof_graphs[1] > 0) {
        // Vollie: a prompt starts a new run: drop the last run's partial window, else its tail (other
        // knobs, other A/B side) is averaged into this run's first window (R3: B kavg 5.33, not 4.36).
        // ponytail: a prompt of <= 8 tokens does not reset; the kit's prompts are 17 and 155 tokens
        ggml_cpu_prof_reset();
    }
    if (n_tok > 8) {
        ggml_cpu_ab_count = 0; // Vollie: ABBA blocks count from the run's first decode graph
    }
    const char * q3k = getenv("GGML_Q3K_VEC");
    ggml_cpu_q3k_vec = q3k && *q3k ? atoi(q3k) != 0 : 1; // Vollie: "" = unset = on (r3/r4 kits ran it off: q3k=0)
    const char * q3kx = getenv("GGML_Q3K_X4");
    ggml_cpu_q3k_x4 = q3kx && *q3kx ? atoi(q3kx) != 0 : 1;
    const char * mmid = getenv("GGML_MMID");
    ggml_cpu_mmid = mmid && *mmid ? atoi(mmid) : 3;
    const char * mtouch = getenv("GGML_MMID_TOUCH");
    ggml_cpu_mmid_touch = mtouch && *mtouch ? atoi(mtouch) != 0 : 0;
    ggml_cpu_router_type = ggml_cpu_router_env(); // Vollie: GGML_ROUTER_CAST (C1), off unless set
    ggml_cpu_router_prep(cgraph, ggml_cpu_router_plan_type()); // Vollie r15: ROUTER_CAST A/B makes the copies on side A's graphs too
    ggml_cpu_shexp_prep(cgraph); // Vollie: GGML_SHEXP_PRUNE (C4), off unless set
    const char * mfu = getenv("GGML_MMID_FUSE"), * mc2 = getenv("GGML_MMID_CHUNK2");
    ggml_cpu_mmid_fuse   = mfu && *mfu ? atoi(mfu) != 0 : 1; // Vollie: default since R47 (bit-identical, R42); =0 turns it off
    ggml_cpu_mmid_chunk2 = mc2 && *mc2 ? atoi(mc2) != 0 : 1; // Vollie: default since R47 (bit-identical, R42); =0 turns it off
    const char * mt1 = getenv("GGML_MMID_TEST1");
    ggml_cpu_mmid_test1 = mt1 && *mt1 && atoi(mt1) != 0;
    const char * skip = getenv("GGML_EXPERT_SKIP");
    ggml_cpu_expert_skip = skip ? atoi(skip) : 0;
    const char * renorm = getenv("GGML_EXPERT_SKIP_RENORM");
    ggml_cpu_expert_skip_renorm = renorm ? atoi(renorm) : 0;
    const char * topp = getenv("GGML_EXPERT_TOPP");
    ggml_cpu_expert_topp = topp && *topp ? atoi(topp) : 0;
    const char * gap = getenv("GGML_EXPERT_GAP");
    ggml_cpu_expert_gap = gap && *gap ? atoi(gap) : 0;
    const char * mink = getenv("GGML_EXPERT_MINK");
    ggml_cpu_expert_mink = mink && *mink ? atoi(mink) : 1;
    const char * ilp = getenv("GGML_Q3K_ILP");
    ggml_cpu_q3k_ilp = ilp && *ilp ? atoi(ilp) != 0 : 1; // Vollie: default on since r4 (bit-identical, R23)
    const char * pf = getenv("GGML_Q3K_PF");
    ggml_cpu_q3k_pf = pf && *pf ? atoi(pf) != 0 : 0;
    const char * kern = getenv("GGML_Q3K_KERN");
    ggml_cpu_q3k_kern = kern && *kern ? atoi(kern) != 0 : 1; // Vollie: default on since r11 (bit-identical, R34/R35)
    const char * anon = getenv("GGML_EXPERT_ANON");
    ggml_cpu_expert_anon = anon && *anon ? atoi(anon) : 0;
    ggml_cpu_anon_use = ggml_cpu_expert_anon > 0;
    const char * ssmi = getenv("GGML_SSM_INPLACE");
    ggml_cpu_ssm_inplace = ssmi && *ssmi ? atoi(ssmi) != 0 : 1; // Vollie: default on since r5 (bit-identical, R24)
    const char * cring = getenv("GGML_CONV_RING");
    ggml_cpu_conv_ring = cring && *cring ? atoi(cring) != 0 : 1; // Vollie: default on since r6 (bit-identical, R25)
    const char * h2 = getenv("GGML_HEAD2"), * h2q = getenv("GGML_HEAD2_Q"), * h2b = getenv("GGML_HEAD2_BATCH");
    ggml_cpu_head2_m = h2 && *h2 ? atoll(h2) : 0; // Vollie: off unless set (NOT bit-exact, R28)
    ggml_cpu_head2_type = !h2q || !*h2q || strcmp(h2q, "q2_k") == 0 ? GGML_TYPE_Q2_K : strcmp(h2q, "q3_k") == 0 ? GGML_TYPE_Q3_K :
                          strcmp(h2q, "q4_0") == 0 || strcmp(h2q, "q4_0_plain") == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_COUNT;
    GGML_ASSERT(ggml_cpu_head2_type != GGML_TYPE_COUNT && "GGML_HEAD2_Q: q2_k, q3_k, q4_0 or q4_0_plain");
    ggml_cpu_head2_rp = h2q && strcmp(h2q, "q4_0") == 0; // Vollie r8: q4_0 = repacked screen
    ggml_cpu_head2_batch = h2b && *h2b && atoi(h2b) != 0;
    const char * h2lr = getenv("GGML_HEAD2_LOWRANK"), * h2ck = getenv("GGML_HEAD2_LOWRANK_CHECK");
    ggml_cpu_head2_lr    = h2lr && *h2lr ? MAX(0, atoi(h2lr)) : 0; // Vollie: off unless set (C2, NOT bit-exact)
    ggml_cpu_head2_check = h2ck && *h2ck && atoi(h2ck) != 0;
    const char * mmc = getenv("GGML_MM_CHUNK"), * eq4 = getenv("GGML_EXPERT_Q4_LAYERS"), * dwn = getenv("GGML_DENSE_WILLNEED");
    ggml_cpu_mm_chunk = ggml_cpu_chunk_rows(mmc && *mmc ? atoi(mmc) : 64); // Vollie: 64 = stock (bit-identical at any value)
    ggml_cpu_expq4_make = ggml_cpu_expq4_use = eq4 && *eq4 ? MAX(0, atoi(eq4)) : 0; // Vollie: off unless set (NOT bit-exact)
    ggml_cpu_dense_willneed = dwn && *dwn && atoi(dwn) != 0; // Vollie: off unless set (advice only)
    if ((ggml_cpu_expert_skip || ggml_cpu_expert_topp || ggml_cpu_expert_gap) && !ggml_cpu_budget_atexit) {
        ggml_cpu_budget_atexit = true;
        atexit(ggml_cpu_budget_print);
    }
    const char * kbench = getenv("GGML_CPU_BENCH");
    const int kbench_n = kbench && *kbench ? atoi(kbench) : 0;
    if (kbench_n > 0 && kbench_n != ggml_cpu_kbench_n) {
        ggml_cpu_kbench_n = kbench_n;
        ggml_cpu_q3k_bench(); // Vollie r13b: pins itself to cpu 4-6 per shape and checks the core of every trial
    }
    const char * bbench = getenv("GGML_BARRIER_BENCH");
    const int bbench_n = bbench ? atoi(bbench) : 0;
    // runs once per distinct value, so an app can re-arm it (e.g. 10000, 10001) per thread count
    ggml_cpu_bbench_this = bbench_n > 0 && bbench_n != ggml_cpu_bbench_n && decode;
    if (ggml_cpu_bbench_this) {
        ggml_cpu_bbench_n = bbench_n;
    }
    const char * ab = getenv("GGML_CPU_AB");
    // Modes: "q3k", "tiny:<n>", and the generic "knob:<NAME>:<A>:<B>" (NAME = TINY_OP, MMID,
    // Q3K_VEC, Q3K_X4, Q3K_ILP, Q3K_PF, Q3K_KERN, MMID_TOUCH, MMID_FUSE, MMID_CHUNK2, MMID_FUSE+CHUNK2, EXPERT_SKIP, EXPERT_ANON, SSM_INPLACE, CONV_RING, HEAD2, MM_CHUNK, EXPERT_Q4_LAYERS): ABBA blocks (see
    // ggml_cpu_ab_warm); q3k and tiny: A on even decode graphs, B on odd ones. EXPERT_ANON keeps one cache of max(A, B) slots, filled on both sides; a side reads it iff
    // its value is > 0 (so A/B measures reading the copies, not the fill).
    ggml_cpu_ab_mode = !ab ? 0 : strcmp(ab, "q3k") == 0 ? 1 : strncmp(ab, "tiny:", 5) == 0 ? 2 : strncmp(ab, "knob:", 5) == 0 ? 3 : 0;
    ggml_cpu_ab_var = 0;
    if (ggml_cpu_ab_mode == 3) {
        const char * aw = getenv("GGML_CPU_AB_WARM"), * abk = getenv("GGML_CPU_AB_BLOCK");
        ggml_cpu_ab_warm  = aw && *aw ? MAX(0, atoi(aw)) : 16;
        ggml_cpu_ab_block = abk && *abk ? MAX(1, atoi(abk)) : 8;
        const char * baab = getenv("GGML_CPU_AB_BAAB");
        ggml_cpu_ab_baab = baab && *baab && atoi(baab) != 0;
        const char * abt = getenv("GGML_CPU_AB_TOKEN");
        ggml_cpu_ab_token = abt && *abt && atoi(abt) != 0;
        if (ggml_cpu_ab_token) { // Vollie r15: per-token interleaving
            ggml_cpu_ab_warm  = aw && *aw ? MAX(0, atoi(aw)) : 0;
            ggml_cpu_ab_block = 1;
        }
    }
    if (ggml_cpu_ab_mode && decode) {
        const int ig  = ggml_cpu_ab_count++;
        const int blk = ig < ggml_cpu_ab_warm ? -1 : (ig - ggml_cpu_ab_warm) / ggml_cpu_ab_block;
        ggml_cpu_ab_var = ggml_cpu_ab_mode == 3 ? (blk >= 0 && ggml_cpu_ab_baab ? blk % 4 == 0 || blk % 4 == 3 : blk % 4 == 1 || blk % 4 == 2) : ig & 1;
        ggml_cpu_ab_blk = blk + 1;
        if (ggml_cpu_ab_mode == 1) {
            ggml_cpu_q3k_vec = ggml_cpu_ab_var == 0;
        } else if (ggml_cpu_ab_mode == 2) {
            if (ggml_cpu_ab_var) {
                ggml_cpu_ab_tiny = atoll(ab + 5);
                ggml_cpu_tiny_thresh = ggml_cpu_ab_tiny;
            }
        } else {
            // one or more NAME:A:B triples, comma-separated (e.g. knob:MMID:0:3,Q3K_X4:0:1)
            const char * p = ab + 5;
            while (*p) {
                char name[32] = {0};
                long long a = 0, b = 0;
                int used = 0;
                if (sscanf(p, "%31[^:]:%lld:%lld%n", name, &a, &b, &used) < 3) {
                    break;
                }
                const long long v = ggml_cpu_ab_var ? b : a;
                ggml_cpu_ab_tiny = v; // shown in the profile line for B
                if      (strcmp(name, "TINY_OP")     == 0) ggml_cpu_tiny_thresh  = v;
                else if (strcmp(name, "MMID")        == 0) ggml_cpu_mmid         = (int) v;
                else if (strcmp(name, "Q3K_VEC")     == 0) ggml_cpu_q3k_vec      = v != 0;
                else if (strcmp(name, "Q3K_X4")      == 0) ggml_cpu_q3k_x4       = v != 0;
                else if (strcmp(name, "EXPERT_SKIP") == 0) ggml_cpu_expert_skip  = (int) v;
                else if (strcmp(name, "Q3K_ILP")     == 0) ggml_cpu_q3k_ilp      = v != 0;
                else if (strcmp(name, "Q3K_PF")      == 0) ggml_cpu_q3k_pf       = v != 0;
                else if (strcmp(name, "Q3K_KERN")    == 0) ggml_cpu_q3k_kern     = v != 0;
                else if (strcmp(name, "MMID_TOUCH")  == 0) ggml_cpu_mmid_touch   = v != 0;
                else if (strcmp(name, "MMID_FUSE")   == 0) ggml_cpu_mmid_fuse    = v != 0;
                else if (strcmp(name, "MMID_CHUNK2") == 0) ggml_cpu_mmid_chunk2  = v != 0;
                else if (strcmp(name, "MMID_FUSE+CHUNK2") == 0) ggml_cpu_mmid_fuse = ggml_cpu_mmid_chunk2 = v != 0; // Vollie r14: both at once
                else if (strcmp(name, "ROUTER_CAST") == 0) ggml_cpu_router_type = v ? GGML_TYPE_F16 : GGML_TYPE_COUNT; // Vollie r15 (C1-F16)
                else if (strcmp(name, "SSM_INPLACE") == 0) ggml_cpu_ssm_inplace  = v != 0;
                else if (strcmp(name, "CONV_RING")   == 0) ggml_cpu_conv_ring    = v != 0;
                else if (strcmp(name, "HEAD2")       == 0) ggml_cpu_head2_m      = v;
                else if (strcmp(name, "MM_CHUNK")    == 0) ggml_cpu_mm_chunk     = ggml_cpu_chunk_rows((int) v);
                else if (strcmp(name, "EXPERT_Q4_LAYERS") == 0) {
                    ggml_cpu_expq4_make = (int) MAX(0, MAX(a, b)); // both sides' copies exist from the prompt graph on
                    ggml_cpu_expq4_use  = (int) MAX(0, v);
                }
                else if (strcmp(name, "EXPERT_ANON") == 0) {
                    ggml_cpu_expert_anon = (int) MAX(a, b);
                    ggml_cpu_anon_use    = v > 0;
                }
                p += used;
                if (*p == ',') {
                    p++;
                }
            }
        }
    }
    // Vollie: GGML_EXPERT_ANON size for this graph (A/B: prompt graphs fill the same max(A, B) cache)
    const char * ab_anon = ggml_cpu_ab_mode == 3 && !decode ? strstr(ab, "EXPERT_ANON:") : NULL;
    long long anon_a = 0, anon_b = 0;
    if (ab_anon && sscanf(ab_anon, "EXPERT_ANON:%lld:%lld", &anon_a, &anon_b) == 2) {
        ggml_cpu_expert_anon = (int) MAX(anon_a, anon_b);
    }
    ggml_cpu_anon_config(ggml_cpu_expert_anon);
    const char * ab_q4 = ggml_cpu_ab_mode == 3 && !decode ? strstr(ab, "EXPERT_Q4_LAYERS:") : NULL;
    long long q4_a = 0, q4_b = 0;
    if (ab_q4 && sscanf(ab_q4, "EXPERT_Q4_LAYERS:%lld:%lld", &q4_a, &q4_b) == 2) { // Vollie: copies made before side B needs them
        ggml_cpu_expq4_make = (int) MAX(0, MAX(q4_a, q4_b));
        ggml_cpu_expq4_use  = (int) MAX(0, q4_a);
    }
    ggml_cpu_expq4_config(getenv("GGML_EXPERT_Q4_LIST"));
    if (ggml_cpu_dense_willneed && !decode) {
        ggml_cpu_dense_advise(cgraph);
    }
    ggml_cpu_head2_prep = ggml_cpu_ab_mode == 3 && strstr(ab, "HEAD2:") != NULL; // Vollie: the copy is made before side B needs it
    ggml_cpu_head2_plain = NULL;
    for (int i = 0; (ggml_cpu_head2_m > 0 || ggml_cpu_head2_prep) && i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * n = cgraph->nodes[i]; // Vollie: GGML_HEAD2's plain rows (node 0 in llama's graphs)
        if (n->op == GGML_OP_GET_ROWS && strcmp(n->src[0]->name, "token_embd.weight") == 0) {
            ggml_cpu_head2_plain = n->src[0];
            break;
        }
    }
    if (ggml_cpu_head2_m <= 0 && !ggml_cpu_head2_prep && ggml_cpu_head2_q) {
        // Vollie: GGML_HEAD2 off (and no HEAD2 A/B): drop the head copy so it does not outlive its run; an ABBA run keeps
        // it on side A (prep). ponytail: the ~1.2 MB select scratch stays until exit
        free(ggml_cpu_head2_q);
        ggml_cpu_head2_q = NULL; ggml_cpu_head2_key = NULL; ggml_cpu_head2_qtype = GGML_TYPE_COUNT; ggml_cpu_head2_il = 0;
    }
    const int64_t prof_t0 = ggml_cpu_prof_this ? ggml_time_us() : 0;
    int64_t flt0[2] = {0, 0}; // Vollie: faults during this graph go to its A/B side (blocks never split a graph)
    if (ggml_cpu_prof_this) {
        ggml_cpu_faults(flt0);
    }
    if (ggml_cpu_ssm_inplace && decode) {
        ggml_cpu_ssm_apply(cgraph);
    }
    if (ggml_cpu_conv_ring && decode) {
        ggml_cpu_conv_apply(cgraph);
    }

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();
    ggml_cpu_ssm_undo();
    ggml_cpu_conv_undo();
#if !defined(_WIN32)
    ggml_cpu_pred_drain();
#endif

    if (decode && ggml_cpu_anon_alloc_n) {
        ggml_cpu_anon_dtok++;
    }
    if (ggml_cpu_expq4_new && ggml_cpu_prof_on) { // Vollie: GGML_EXPERT_Q4_LAYERS, the copies this graph made
        char layers[256] = "";
        for (int r = 0, len = 0; r < ggml_cpu_expq4_made && r < ggml_cpu_expq4_nrank() && len < (int) sizeof layers - 8; r++) {
            len += snprintf(layers + len, sizeof layers - len, "%s%d", r ? "," : "", ggml_cpu_expq4_layer(r));
        }
        fprintf(stderr, "cpu-prof expq4 requant -> q4_0x4x%d layers=%s tensors=%d/%d threads=%d copy_mb=%.1f ms=%.1f\n",
                ggml_cpu_expq4_n ? ggml_cpu_expq4[0].il : 0, layers, ggml_cpu_expq4_new, ggml_cpu_expq4_n, n_threads,
                ggml_cpu_expq4_bytes/1e6, ggml_cpu_expq4_us/1e3);
    }
    if (ggml_cpu_prof_this) {
        ggml_cpu_prof_anon[ggml_cpu_ab_var] = ggml_cpu_anon_use;
        ggml_cpu_prof_ilp[ggml_cpu_ab_var]  = ggml_cpu_q3k_ilp;
        ggml_cpu_prof_pf[ggml_cpu_ab_var]   = ggml_cpu_q3k_pf;
        ggml_cpu_prof_kern[ggml_cpu_ab_var] = ggml_cpu_q3k_kern;
        ggml_cpu_prof_touch[ggml_cpu_ab_var] = ggml_cpu_mmid_touch;
        ggml_cpu_prof_fuse[ggml_cpu_ab_var]  = ggml_cpu_mmid_fuse;
        ggml_cpu_prof_chunk2[ggml_cpu_ab_var] = ggml_cpu_mmid_chunk2;
        ggml_cpu_prof_rcast[ggml_cpu_ab_var] = ggml_cpu_router_type;
        ggml_cpu_prof_skip[ggml_cpu_ab_var] = ggml_cpu_expert_skip;
        ggml_cpu_prof_h2[ggml_cpu_ab_var]   = ggml_cpu_head2_m;
        ggml_cpu_prof_chunk[ggml_cpu_ab_var] = ggml_cpu_mm_chunk;
        ggml_cpu_prof_expq4[ggml_cpu_ab_var] = ggml_cpu_expq4_use;
        ggml_cpu_prof_total[ggml_cpu_ab_var] += ggml_time_us() - prof_t0;
        int64_t flt1[2];
        ggml_cpu_faults(flt1);
        ggml_cpu_prof_flt[ggml_cpu_ab_var][0] += flt1[0] - flt0[0];
        ggml_cpu_prof_flt[ggml_cpu_ab_var][1] += flt1[1] - flt0[1];
        ggml_cpu_prof_graphs[ggml_cpu_ab_var]++;
        ggml_cpu_prof_tokens[ggml_cpu_ab_var] += (int) n_tok;
        const int done = ggml_cpu_ab_count; // decode graphs of this run so far, this one included
        if (ggml_cpu_ab_mode == 3 ? done == ggml_cpu_ab_warm || (done > ggml_cpu_ab_warm && (done - ggml_cpu_ab_warm) % ggml_cpu_ab_block == 0)
                                  : ggml_cpu_prof_tokens[0] + ggml_cpu_prof_tokens[1] >= (ggml_cpu_ab_mode ? 64 : 32)) {
            ggml_cpu_prof_dump(n_threads);
        }
    }

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#elif defined(__VXE__) || defined(__VXE2__)
    for (; i + 7 < n; i += 8) {
        const uint32x4_t v_yl = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 0));
        const uint32x4_t v_yh = __lzs_f32cx4_to_f16(vec_xl(0, x + i + 4));
        vec_xst(vec_pack(v_yl, v_yh), 0, (uint16_t *)(y + i));
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme2(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME2)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            // initialize UE4M3 table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_ue4m3[i] = ggml_ue4m3_to_fp32(i);
                ggml_table_f32_f8_e4m3[i] = ggml_f8_e4m3_to_fp32(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        {
            const char * env = getenv("GGML_CPU_DISABLE_FUSION");
            ggml_cpu_disable_fusion = (env != NULL && atoi(env) == 1);
        }

        is_first_call = false;
    }

    ggml_critical_section_end();
}
