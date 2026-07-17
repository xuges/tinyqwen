/* ===========================================================================
 * tinyqwen.c — 零依赖、单文件的 Qwen3 系列（GGUF 量化）CPU 推理引擎
 * ===========================================================================
 *
 * 这是一个教学/实验性质的大语言模型推理器，特点：
 *
 *   1. 零依赖 —— 只使用 C 标准库和操作系统自带的基础设施（POSIX 的
 *      mmap/pthread 或 Windows 原生 API），不需要任何第三方库；
 *   2. 单文件 —— 词表解析、BPE 分词、GGUF 解析、Transformer 前向传播、
 *      采样、聊天交互……全部逻辑都在这一个 .c 文件里；
 *   3. 直接读取 GGUF 模型文件 —— 权重、超参数、分词器词表与 BPE 合并规则
 *      全部内嵌在一个 .gguf 文件中，无需其它任何文件；
 *   4. 支持 Qwen3 全系列稠密模型（0.6B/1.7B/4B/8B/14B/32B）—— 层数、
 *      维度、头数等结构差异全部由 GGUF 元数据驱动，代码不区分规格；
 *      张量类型支持 Q4_0/Q4_1/Q8_0/Q6_K/F16/F32，
 *      推理时“逐块反量化 + 点积”融合计算；
 *   5. 跨平台多线程 —— 自带线程池（pthread / Win32 双实现），-j 指定线程数；
 *   6. 运行时 SIMD 检测 —— 启动时用 CPUID 检测 AVX2+FMA，支持则切换到
 *      向量化点积内核，不支持自动退回标量路径（标量内核同时是教学参考）；
 *   7. 纯 CPU 推理，启动后进入命令行多轮问答（ChatML 对话模板）；
 *   8. 可编译成静态/动态库嵌入宿主程序（公共 API 见 tinyqwen.h，
 *      定义 TINYQWEN_LIB 宏即可去掉命令行部分）。
 *
 * 构建（命令行工具）：
 *     gcc -O2 -std=c99 -o tinyqwen tinyqwen.c -lm -pthread
 * 构建（库, 见 Makefile 的 libs 目标）：
 *     gcc -O2 -fPIC -shared -DTINYQWEN_LIB -o libtinyqwen.so tinyqwen.c -lm -pthread
 *
 * 运行：
 *     ./tinyqwen models/Qwen3-0.6B-Q4_0.gguf
 *
 * 参考: 两个实测规格的结构参数（全部从 GGUF 元数据中读取，这里仅作说明）：
 *     Qwen3-0.6B: 28 层, 隐层 1024, 16Q/8KV 头, FFN 3072, 共享输入/输出嵌入
 *     Qwen3-4B:   36 层, 隐层 2560, 32Q/8KV 头, FFN 9728, 共享输入/输出嵌入
 *     公共点: 每头维度 128（带 Qwen3 特有的 QK-RMSNorm）、SwiGLU 激活、
 *     词表 151936（GPT-2 风格字节级 BPE）、RoPE 基频 1e6；
 *     8B 及以上有独立的输出投影层（output.weight），代码已兼容两种情况。
 *
 * 代码地图（按文件内顺序）：
 *     [0]  通用工具           —— 错误处理、fp16 转换等小工具
 *     [0b] 跨平台线程池       —— 平台原生线程 API 上的 parallel_for
 *     [1]  GGUF 文件解析      —— mmap 打开模型文件，解析元数据与张量目录
 *     [2]  分词器             —— GPT-2 字节级 BPE 编码/解码
 *     [3]  模型加载           —— 读取超参数、按名字定位每层权重
 *     [4]  数学内核           —— 反量化点积（矩阵乘）、RMSNorm、Softmax
 *     [4b] AVX2 SIMD 内核     —— x86: CPUID 运行时检测 + 向量化点积
 *     [4c] NEON SIMD 内核     —— aarch64: HWCAP 运行时检测（实验性）
 *     [5]  前向传播           —— 注意力(RoPE/GQA/QK-Norm) + SwiGLU + KV 缓存
 *     [6]  采样器             —— temperature + top-p 核采样
 *     [7]  聊天/嵌入/重排     —— ChatML 模板、逐 token 生成、池化、打分
 *     [8]  库 API             —— tq_load/tq_chat/tq_embed/tq_rerank(tinyqwen.h)
 *     [9]  命令行工具         —— 参数解析、REPL(-DTINYQWEN_LIB 时不编译)
 * ===========================================================================
 */

/* 启用 POSIX.1-2008 接口(clock_gettime / mmap / pthread 等)。
 * -std=c99 严格模式下 glibc 默认隐藏这些声明, 需显式声明特性宏。
 * Windows 下该宏无意义也无害(平台相关代码见下方 _WIN32 分支)。 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

/* 公共库 API 声明(本文件同时是库的实现, TINYQWEN_BUILD 控制符号导出) */
#define TINYQWEN_BUILD
#include "tinyqwen.h"

/* ---- 平台相关头文件 ----
 * 多线程使用各平台的原生 API(不依赖 OpenMP):
 *   - Windows: Win32 线程 + CRITICAL_SECTION + CONDITION_VARIABLE (Vista+)
 *   - 其它(Linux/macOS/BSD): POSIX pthread
 * 文件访问同理: Windows 下用标准 fread 读入, POSIX 下用 mmap 零拷贝。 */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
/* 旧版 MSVC 的 C 模式(未指定 /std:c11)不认识 inline 关键字 */
#if defined(_MSC_VER) && !defined(__cplusplus) && \
    (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L)
#define inline __inline
#endif
#else
#include <fcntl.h>    /* open       */
#include <unistd.h>   /* close      */
#include <sys/mman.h> /* mmap       */
#include <sys/stat.h> /* fstat      */
#include <pthread.h>
#endif

/* ===========================================================================
 * [0] 通用工具
 * ===========================================================================
 */

/* ---- 错误处理约定 ----
 * 可能失败的内部函数一律先调用 fail() 把错误描述写入 g_err_msg,
 * 再通过返回值报告失败(int 返回 -1 / 指针返回 NULL / 浮点返回负值),
 * 由调用方逐层向上传递:
 *   - 作为库使用时, 错误码经公共 API(tq_load 返回 NULL 等)传给宿主,
 *     描述用 tq_last_error() 读取 —— 库永远不会终止宿主进程;
 *   - 作为命令行工具时, main 等入口检查返回值后调用 die()(定义在
 *     命令行段 [9])打印错误并退出。
 * bug() 只用于"理论不可达"的内部不变量破坏(例如加载期已校验过的
 * 张量类型在运行期不被识别), 语义等同断言失败: 打印后 abort。 */
static char g_err_msg[256] = "尚无错误";

static int fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err_msg, sizeof(g_err_msg), fmt, ap);
    va_end(ap);
    return -1;
}

static void bug(const char *what) {
    fprintf(stderr, "TinyQwen 内部错误(不变量被破坏): %s\n", what);
    abort();
}

/* 半精度浮点(fp16, IEEE 754 binary16)转单精度。
 * GGUF 量化块中的缩放因子都是 fp16 存储的，这是推理热路径上的高频操作，
 * 所以除了这个位运算版本，加载时还会预计算一张 65536 项的查找表(见下)。 */
static float f16_to_f32_compute(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16; /* 符号位左移到 f32 位置 */
    uint32_t exp  = (h >> 10) & 0x1F;              /* 5 位指数 */
    uint32_t mant = h & 0x3FF;                     /* 10 位尾数 */
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                           /* ±0 */
        } else {
            /* 次正规数: 逐位左移尾数直到最高位为 1, 换算成 f32 规格化数 */
            int e = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            bits = sign | ((uint32_t)e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);  /* ±inf / NaN */
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* fp16 -> f32 查找表。fp16 只有 65536 种位模式，用 256KB 内存换一次查表，
 * 比每次做位运算换算更快，也让量化点积内核更简洁。 */
static float g_f16_table[65536];

static void f16_table_init(void) {
    for (int i = 0; i < 65536; i++)
        g_f16_table[i] = f16_to_f32_compute((uint16_t)i);
}

#define F16(h) (g_f16_table[(uint16_t)(h)])

/* ===========================================================================
 * [0b] 跨平台线程池 —— 用平台原生线程 API 实现的 "parallel for"
 * ===========================================================================
 *
 * 推理的热点(矩阵×向量、多头注意力)都是"独立行/头的循环", 天然适合
 * 数据并行。这里实现一个常驻线程池, 提供一个原语:
 *
 *     parallel_for(fn, ctx, n)  —— 把 [0,n) 均匀切给各线程执行 fn
 *
 * 设计要点:
 *   - 线程在启动时创建一次, 之后反复复用(每 token 有数百次并行调用,
 *     反复创建线程的开销不可接受);
 *   - 主线程也参与计算(承担第 0 份), 减少一次唤醒等待;
 *   - 同步用互斥锁 + 条件变量: "generation 计数"发布任务, 工人做完自己
 *     那份后递减 pending 计数, 归零时唤醒主线程;
 *   - Windows 用 CRITICAL_SECTION/CONDITION_VARIABLE, 其余平台用 pthread,
 *     通过下面十几行薄封装统一接口, 逻辑只写一份。
 */

/* ---- 平台薄封装: 线程/互斥锁/条件变量 ---- */
#ifdef _WIN32

typedef HANDLE             thread_t;
typedef CRITICAL_SECTION   mutex_t;
typedef CONDITION_VARIABLE cond_t;

static void mutex_init(mutex_t *m)   { InitializeCriticalSection(m); }
static void mutex_lock(mutex_t *m)   { EnterCriticalSection(m); }
static void mutex_unlock(mutex_t *m) { LeaveCriticalSection(m); }
static void cond_init(cond_t *c)     { InitializeConditionVariable(c); }
static void cond_wait_(cond_t *c, mutex_t *m) { SleepConditionVariableCS(c, m, INFINITE); }
static void cond_broadcast_(cond_t *c) { WakeAllConditionVariable(c); }

static DWORD WINAPI win_thread_adapter(LPVOID p); /* 前向声明, 见下 */
static int thread_create(thread_t *t, void *arg) {
    *t = CreateThread(NULL, 0, win_thread_adapter, arg, 0, NULL);
    return *t ? 0 : fail("创建线程失败");
}
static void thread_join(thread_t t) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }

/* 逻辑核数 */
static int cpu_count(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}

#else /* POSIX */

typedef pthread_t       thread_t;
typedef pthread_mutex_t mutex_t;
typedef pthread_cond_t  cond_t;

static void mutex_init(mutex_t *m)   { pthread_mutex_init(m, NULL); }
static void mutex_lock(mutex_t *m)   { pthread_mutex_lock(m); }
static void mutex_unlock(mutex_t *m) { pthread_mutex_unlock(m); }
static void cond_init(cond_t *c)     { pthread_cond_init(c, NULL); }
static void cond_wait_(cond_t *c, mutex_t *m) { pthread_cond_wait(c, m); }
static void cond_broadcast_(cond_t *c) { pthread_cond_broadcast(c); }

static void *posix_thread_adapter(void *p); /* 前向声明, 见下 */
static int thread_create(thread_t *t, void *arg) {
    return pthread_create(t, NULL, posix_thread_adapter, arg) ? fail("创建线程失败") : 0;
}
static void thread_join(thread_t t) { pthread_join(t, NULL); }

static int cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

#endif

/* ---- 线程池本体 ---- */

/* 任务函数: 处理编号 [start, end) 的元素(行/注意力头等) */
typedef void (*PoolTask)(int start, int end, const void *ctx);

typedef struct {
    int         n_threads;   /* 总并行度(含主线程); 1 = 纯单线程       */
    thread_t   *workers;     /* n_threads - 1 个工人线程               */
    int        *worker_ids;  /* 传给每个工人的编号 1..n_threads-1      */
    mutex_t     mu;
    cond_t      cv_work;     /* 有新任务时唤醒工人                     */
    cond_t      cv_done;     /* 全部做完时唤醒主线程                   */
    /* 当前任务(发布后只读, 由 generation 变化通知) */
    PoolTask    fn;
    const void *ctx;
    int         n_items;
    unsigned    generation;  /* 每发布一个任务 +1                      */
    int         n_pending;   /* 尚未完成本代任务的工人数               */
    int         stop;        /* 通知全部工人退出                       */
} Pool;

static Pool g_pool;

/* 第 i 份(共 t 份)负责的区间: 尽量均匀切分 */
static void pool_slice(int n, int t, int i, int *start, int *end) {
    *start = (int)((int64_t)n * i / t);
    *end   = (int)((int64_t)n * (i + 1) / t);
}

/* 工人线程主体: 等任务 -> 干自己那份 -> 报告完成 -> 继续等 */
static void pool_worker(int id) {
    unsigned seen = 0;
    for (;;) {
        mutex_lock(&g_pool.mu);
        while (!g_pool.stop && g_pool.generation == seen)
            cond_wait_(&g_pool.cv_work, &g_pool.mu);
        if (g_pool.stop) { mutex_unlock(&g_pool.mu); return; }
        seen = g_pool.generation;
        PoolTask    fn  = g_pool.fn;
        const void *ctx = g_pool.ctx;
        int         n   = g_pool.n_items;
        mutex_unlock(&g_pool.mu);

        int s, e;
        pool_slice(n, g_pool.n_threads, id, &s, &e);
        if (s < e) fn(s, e, ctx);

        mutex_lock(&g_pool.mu);
        if (--g_pool.n_pending == 0) cond_broadcast_(&g_pool.cv_done);
        mutex_unlock(&g_pool.mu);
    }
}

/* 平台线程入口适配(签名不同, 都转到 pool_worker) */
#ifdef _WIN32
static DWORD WINAPI win_thread_adapter(LPVOID p) { pool_worker(*(int *)p); return 0; }
#else
static void *posix_thread_adapter(void *p) { pool_worker(*(int *)p); return NULL; }
#endif

static void pool_shutdown(void); /* 前向声明: 部分创建失败时回收 */

/* 创建线程池(n_threads 为总并行度, 传 1 则完全不建线程)。
 * 失败返回 -1; 已创建的部分线程按实际数量记录, 可被 pool_shutdown 回收 */
static int pool_init(int n_threads) {
    g_pool.n_threads = n_threads;
    if (n_threads <= 1) return 0;
    mutex_init(&g_pool.mu);
    cond_init(&g_pool.cv_work);
    cond_init(&g_pool.cv_done);
    g_pool.workers    = malloc(sizeof(thread_t) * (n_threads - 1));
    g_pool.worker_ids = malloc(sizeof(int) * (n_threads - 1));
    if (!g_pool.workers || !g_pool.worker_ids) {
        free(g_pool.workers); free(g_pool.worker_ids);
        g_pool.workers = NULL; g_pool.worker_ids = NULL;
        g_pool.n_threads = 1;
        return fail("内存不足(创建线程池)");
    }
    for (int i = 0; i < n_threads - 1; i++) {
        g_pool.worker_ids[i] = i + 1;              /* 工人编号 1..T-1 */
        if (thread_create(&g_pool.workers[i], &g_pool.worker_ids[i]) < 0) {
            g_pool.n_threads = i + 1;              /* 只保留已建成的线程 */
            pool_shutdown();
            return -1;
        }
    }
    return 0;
}

/* 通知全部工人退出并回收线程(进程结束前调用, 保证干净收尾) */
static void pool_shutdown(void) {
    if (g_pool.n_threads <= 1) return;
    mutex_lock(&g_pool.mu);
    g_pool.stop = 1;
    cond_broadcast_(&g_pool.cv_work);
    mutex_unlock(&g_pool.mu);
    for (int i = 0; i < g_pool.n_threads - 1; i++)
        thread_join(g_pool.workers[i]);
    free(g_pool.workers);
    free(g_pool.worker_ids);
    g_pool.n_threads = 1;
}

/* 把 fn 作用到 [0,n) 上并行执行, 返回时保证全部完成。
 * 主线程承担第 0 份; 单线程或任务太小时直接串行, 省掉同步开销。 */
static void parallel_for(PoolTask fn, const void *ctx, int n) {
    int t = g_pool.n_threads;
    if (t <= 1 || n < 4) { fn(0, n, ctx); return; }

    mutex_lock(&g_pool.mu);
    g_pool.fn        = fn;
    g_pool.ctx       = ctx;
    g_pool.n_items   = n;
    g_pool.n_pending = t - 1;
    g_pool.generation++;
    cond_broadcast_(&g_pool.cv_work);
    mutex_unlock(&g_pool.mu);

    int s, e;
    pool_slice(n, t, 0, &s, &e);
    if (s < e) fn(s, e, ctx);                      /* 主线程的那份 */

    mutex_lock(&g_pool.mu);
    while (g_pool.n_pending > 0)
        cond_wait_(&g_pool.cv_done, &g_pool.mu);
    mutex_unlock(&g_pool.mu);
}

/* ===========================================================================
 * [1] GGUF 文件解析
 * ===========================================================================
 *
 * GGUF 是 ggml/llama.cpp 生态的模型文件格式, 布局(全部小端):
 *
 *     +--------------------------------------------------+
 *     | magic "GGUF" (4B) | version (u32) |              |
 *     | n_tensors (u64)   | n_kv (u64)    |              |
 *     +--------------------------------------------------+
 *     | n_kv 个元数据键值对:                             |
 *     |   key(string) + value_type(u32) + value          |
 *     |   string = len(u64) + utf8字节(不含\0)           |
 *     |   array  = elem_type(u32) + n(u64) + n个元素     |
 *     +--------------------------------------------------+
 *     | n_tensors 个张量描述:                            |
 *     |   name(string) + n_dims(u32) + ne[n_dims](u64)   |
 *     |   + ggml_type(u32) + offset(u64)                 |
 *     +--------------------------------------------------+
 *     | (按 alignment 对齐, 默认 32 字节)                |
 *     | 张量数据区 (offset 相对于本区起点)               |
 *     +--------------------------------------------------+
 *
 * 我们 mmap 整个文件, 元数据解析成一个小的 KV 表, 张量数据不复制、
 * 直接引用 mmap 内存(操作系统按需换页, 加载瞬间完成)。
 */

/* ---- GGUF 元数据值类型编号(格式规范定义) ---- */
enum {
    GGUF_U8 = 0, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64
};

/* ---- ggml 张量数据类型编号(只列出本模型会用到的) ---- */
enum {
    GGML_F32  = 0,   /* 32 位浮点(各层 norm 权重)                          */
    GGML_F16  = 1,   /* 16 位浮点                                          */
    GGML_Q4_0 = 2,   /* 4bit 块量化: 32 个权重共享 1 个 fp16 缩放           */
    GGML_Q4_1 = 3,   /* 4bit 块量化: 缩放 + 最小值(部分 ffn_down 层)        */
    GGML_Q8_0 = 8,   /* 8bit 块量化                                        */
    GGML_Q6_K = 14   /* 6bit K-量化: 256 个权重一个超块(词嵌入层)           */
};

#ifndef TINYQWEN_LIB /* 仅 --inspect/--selftest 显示用 */
static const char *ggml_type_name(uint32_t t) {
    switch (t) {
    case GGML_F32:  return "F32";
    case GGML_F16:  return "F16";
    case GGML_Q4_0: return "Q4_0";
    case GGML_Q4_1: return "Q4_1";
    case GGML_Q8_0: return "Q8_0";
    case GGML_Q6_K: return "Q6_K";
    default:        return "?";
    }
}
#endif

/* 每种量化类型的块大小: 一个块含多少个权重元素、占多少字节。
 * 不认识的类型返回 -1(加载期以此校验, 运行期不会再遇到未知类型) */
static int ggml_block_info(uint32_t t, uint64_t *elems, uint64_t *bytes) {
    switch (t) {
    case GGML_F32:  *elems = 1;   *bytes = 4;   return 0;
    case GGML_F16:  *elems = 1;   *bytes = 2;   return 0;
    case GGML_Q4_0: *elems = 32;  *bytes = 18;  return 0; /* fp16 d + 16B 半字节 */
    case GGML_Q4_1: *elems = 32;  *bytes = 20;  return 0; /* fp16 d,m + 16B      */
    case GGML_Q8_0: *elems = 32;  *bytes = 34;  return 0; /* fp16 d + 32B int8   */
    case GGML_Q6_K: *elems = 256; *bytes = 210; return 0; /* 见 [4] 节反量化注释 */
    default: return fail("不支持的张量量化类型(ggml 类型号 %u)", t);
    }
}

/* 一行(ne[0] 个元素)占多少字节 —— 量化块沿行方向排列, 行首总是块对齐的。
 * 返回 0 表示类型不支持或行长不是块的整数倍(错误详情在 g_err_msg) */
static uint64_t ggml_row_bytes(uint32_t type, uint64_t ne0) {
    uint64_t elems = 0, bytes = 0;
    if (ggml_block_info(type, &elems, &bytes) < 0) return 0;
    if (ne0 % elems) { fail("张量行长不是量化块的整数倍"); return 0; }
    return ne0 / elems * bytes;
}

/* ---- 解析后的元数据键值对 ---- */
typedef struct {
    char     key[128];      /* 键名(拷贝为 C 字符串)                        */
    uint32_t type;          /* GGUF_* 值类型                                */
    union {                 /* 标量值                                       */
        uint64_t u; int64_t i; double f;
    } v;
    const char    *str;      /* GGUF_STRING: 指向 mmap 中的字节(非零结尾!)  */
    uint64_t       str_len;
    uint32_t       arr_type; /* GGUF_ARRAY: 元素类型                        */
    uint64_t       arr_n;    /*             元素个数                        */
    const uint8_t *arr_data; /*             首元素位置(字符串数组需逐个走)  */
} GgufKV;

/* ---- 张量目录项 ---- */
typedef struct {
    char        name[96];
    uint32_t    type;       /* GGML_* 数据类型                              */
    uint32_t    n_dims;
    uint64_t    ne[4];      /* 各维元素数; ne[0] 是变化最快的“行内”维度     */
    const void *data;       /* 指向 mmap 中的张量数据                       */
} GTensor;

/* ---- 解析结果 ---- */
typedef struct {
    const uint8_t *base;     /* mmap 起始地址          */
    size_t         size;     /* 文件大小               */
    uint64_t       n_kv, n_tensors;
    GgufKV        *kv;
    GTensor       *tensors;
} Gguf;

/* 带边界检查的顺序读取游标 —— 防止损坏的文件让我们读越界。
 * 采用"粘性错误"设计(类似 stdio 的 ferror): 越界时置 err 标志并返回
 * 指向零缓冲的安全指针, 后续读取全部得到 0 但不会访问非法内存,
 * 解析结束后统一检查一次 err 即可, 避免每次读取都写错误分支。 */
typedef struct { const uint8_t *p, *end; int err; } Cursor;

static const uint8_t g_cur_zeros[16]; /* 越界读取的替身(标量最长 8 字节) */

static const uint8_t *cur_take(Cursor *c, uint64_t n) {
    if ((uint64_t)(c->end - c->p) < n) {
        c->err = 1;
        c->p = c->end;        /* 快进到末尾, 让解析尽快自然结束 */
        return g_cur_zeros;   /* 只被 memcpy <= 8 字节, 安全     */
    }
    const uint8_t *ret = c->p;
    c->p += n;
    return ret;
}
static uint32_t cur_u32(Cursor *c) { uint32_t v; memcpy(&v, cur_take(c, 4), 4); return v; }
static uint64_t cur_u64(Cursor *c) { uint64_t v; memcpy(&v, cur_take(c, 8), 8); return v; }

/* 读取 GGUF 字符串(长度前缀), 返回指向 mmap 的指针(不含 \0), 长度写入 *len。
 * 长度越界时置错误标志并返回空串, 调用方拿到 len=0 不会越界拷贝。 */
static const char *cur_str(Cursor *c, uint64_t *len) {
    uint64_t l = cur_u64(c);
    if ((uint64_t)(c->end - c->p) < l) {
        c->err = 1;
        c->p = c->end;
        *len = 0;
        return (const char *)g_cur_zeros;
    }
    *len = l;
    const char *ret = (const char *)c->p;
    c->p += l;
    return ret;
}

/* 标量值类型的字节宽度(未知类型返回 0, 由调用方置游标错误标志) */
static uint64_t gguf_scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_U8: case GGUF_I8: case GGUF_BOOL:   return 1;
    case GGUF_U16: case GGUF_I16:                 return 2;
    case GGUF_U32: case GGUF_I32: case GGUF_F32:  return 4;
    case GGUF_U64: case GGUF_I64: case GGUF_F64:  return 8;
    default: return 0;
    }
}

/* 解析一个元数据值(已读完键名和类型号之后调用), 结果填入 kv */
static void gguf_read_value(Cursor *c, uint32_t type, GgufKV *kv) {
    kv->type = type;
    switch (type) {
    case GGUF_STRING:
        kv->str = cur_str(c, &kv->str_len);
        break;
    case GGUF_ARRAY: {
        kv->arr_type = cur_u32(c);
        kv->arr_n    = cur_u64(c);
        kv->arr_data = c->p;
        if (kv->arr_type == GGUF_STRING) {
            /* 字符串数组: 逐个跳过(记录首地址, 用到时再重新遍历)。
             * 出错(文件截断)立即停, 防止 arr_n 是垃圾值导致长循环 */
            for (uint64_t i = 0; i < kv->arr_n && !c->err; i++) {
                uint64_t l;
                cur_str(c, &l);
            }
        } else if (kv->arr_type == GGUF_ARRAY) {
            c->err = 1;   /* 嵌套数组: 规范允许但模型文件不会出现 */
        } else {
            uint64_t esz = gguf_scalar_size(kv->arr_type);
            if (!esz) { c->err = 1; break; }
            if (kv->arr_n > (uint64_t)(c->end - c->p) / esz) { c->err = 1; break; }
            cur_take(c, kv->arr_n * esz);
        }
        break;
    }
    case GGUF_F32: { float f;  memcpy(&f, cur_take(c, 4), 4); kv->v.f = f; break; }
    case GGUF_F64: { double d; memcpy(&d, cur_take(c, 8), 8); kv->v.f = d; break; }
    case GGUF_I8:  { kv->v.i = *(const int8_t  *)cur_take(c, 1); break; }
    case GGUF_I16: { int16_t v; memcpy(&v, cur_take(c, 2), 2); kv->v.i = v; break; }
    case GGUF_I32: { int32_t v; memcpy(&v, cur_take(c, 4), 4); kv->v.i = v; break; }
    case GGUF_I64: { int64_t v; memcpy(&v, cur_take(c, 8), 8); kv->v.i = v; break; }
    default: { /* 无符号整数与 bool(未知值类型视为文件损坏) */
        uint64_t sz = gguf_scalar_size(type);
        if (!sz) { c->err = 1; break; }
        uint64_t v = 0;
        memcpy(&v, cur_take(c, sz), sz);
        kv->v.u = v;
        break;
    }
    }
}

static void gguf_close(Gguf *g); /* 前向声明: 失败路径的清理用 */

/* 打开并完整解析一个 GGUF 文件的元数据和张量目录, 结果填入 *out。
 * POSIX 平台用 mmap 零拷贝映射(内核按需换页, "加载"瞬间完成);
 * Windows 平台退化为一次性 fread 读入堆内存(实现最简, 行为等价)。
 * 返回 0 成功, -1 失败(已释放全部中间资源, 错误详情在 g_err_msg)。 */
static int gguf_open(Gguf *out, const char *path) {
    Gguf g = {0};
    memset(out, 0, sizeof(*out));

#ifdef _WIN32
    FILE *f = fopen(path, "rb");
    if (!f) return fail("无法打开模型文件: %s", path);
    /* 用 64 位定位接口, 普通 fseek 的 long 偏移在 Windows 上是 32 位,
     * 处理不了 >2GB 的模型文件 */
    _fseeki64(f, 0, SEEK_END);
    long long sz = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return fail("无法获取模型文件大小"); }
    g.size = (size_t)sz;
    uint8_t *buf = malloc(g.size);
    if (!buf) { fclose(f); return fail("内存不足(读入模型文件)"); }
    if (fread(buf, 1, g.size, f) != g.size) {
        fclose(f);
        free(buf);
        return fail("读取模型文件失败");
    }
    fclose(f);
    g.base = buf;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail("无法打开模型文件: %s", path);
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return fail("无法获取模型文件大小"); }
    g.size = (size_t)st.st_size;

    /* MAP_PRIVATE 只读映射: 权重永远不复制进堆内存, 由内核按需加载页 */
    g.base = (const uint8_t *)mmap(NULL, g.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); /* mmap 建立(或失败)后文件描述符都不再需要 */
    if (g.base == MAP_FAILED) {
        g.base = NULL;
        return fail("mmap 模型文件失败");
    }
#endif

    Cursor c = { g.base, g.base + g.size, 0 };

    /* ---- 头部 ---- */
    if (memcmp(cur_take(&c, 4), "GGUF", 4)) {
        fail("不是 GGUF 文件(魔数不符)");
        goto bad;
    }
    uint32_t version = cur_u32(&c);
    if (version < 2 || version > 3) {
        fail("只支持 GGUF v2/v3(文件版本 %u)", version);
        goto bad;
    }
    g.n_tensors = cur_u64(&c);
    g.n_kv      = cur_u64(&c);
    if (g.n_tensors > 100000 || g.n_kv > 100000) {
        fail("GGUF 头部数值异常");
        goto bad;
    }

    /* ---- 元数据键值对 ---- */
    g.kv = calloc(g.n_kv ? g.n_kv : 1, sizeof(GgufKV));
    if (!g.kv) { fail("内存不足"); goto bad; }
    uint64_t alignment = 32; /* 数据区对齐, 可被 general.alignment 覆盖 */
    for (uint64_t i = 0; i < g.n_kv && !c.err; i++) {
        uint64_t klen;
        const char *k = cur_str(&c, &klen);
        GgufKV *kv = &g.kv[i];
        if (klen >= sizeof(kv->key)) klen = sizeof(kv->key) - 1;
        memcpy(kv->key, k, klen);
        kv->key[klen] = 0;
        gguf_read_value(&c, cur_u32(&c), kv);
        if (!strcmp(kv->key, "general.alignment") && kv->type == GGUF_U32)
            alignment = kv->v.u;
    }
    if (alignment == 0) alignment = 32;

    /* ---- 张量目录 ---- */
    g.tensors = calloc(g.n_tensors ? g.n_tensors : 1, sizeof(GTensor));
    if (!g.tensors) { fail("内存不足"); goto bad; }
    for (uint64_t i = 0; i < g.n_tensors && !c.err; i++) {
        GTensor *t = &g.tensors[i];
        uint64_t nlen;
        const char *n = cur_str(&c, &nlen);
        if (nlen >= sizeof(t->name)) nlen = sizeof(t->name) - 1;
        memcpy(t->name, n, nlen);
        t->name[nlen] = 0;
        t->n_dims = cur_u32(&c);
        if (t->n_dims > 4) { fail("张量维度数超过 4"); goto bad; }
        t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = cur_u64(&c);
        t->type = cur_u32(&c);
        uint64_t off = cur_u64(&c);
        t->data = (const void *)(uintptr_t)off; /* 暂存偏移, 下面再换算成指针 */
    }

    /* 粘性错误统一检查: 任何一步读越界都会走到这里 */
    if (c.err) {
        fail("GGUF 文件被截断或已损坏");
        goto bad;
    }

    /* ---- 张量数据区: 从对齐后的位置开始 ---- */
    uint64_t header_end = (uint64_t)(c.p - g.base);
    uint64_t data_start = (header_end + alignment - 1) / alignment * alignment;

    for (uint64_t i = 0; i < g.n_tensors; i++) {
        GTensor *t = &g.tensors[i];
        uint64_t off      = (uint64_t)(uintptr_t)t->data;
        uint64_t nrows    = t->ne[1] * t->ne[2] * t->ne[3];
        uint64_t row_size = ggml_row_bytes(t->type, t->ne[0]);
        if (!row_size) goto bad; /* 不支持的量化类型, 错误信息已设置 */
        if (data_start + off + row_size * nrows > g.size) {
            fail("张量 %s 的数据越过文件末尾", t->name);
            goto bad;
        }
        t->data = g.base + data_start + off;
    }
    *out = g;
    return 0;

bad:
    gguf_close(&g);
    return -1;
}

/* 释放 gguf_open 占用的全部资源(映射/缓冲与目录表) */
static void gguf_close(Gguf *g) {
    if (!g->base) return;
#ifdef _WIN32
    free((void *)g->base);
#else
    munmap((void *)g->base, g->size);
#endif
    free(g->kv);
    free(g->tensors);
    memset(g, 0, sizeof(*g));
}

/* 按键名查找元数据(找不到返回 NULL) */
static const GgufKV *gguf_kv(const Gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

/* 取必需的无符号整数元数据。缺失时把 *err 置 -1 并记录错误信息;
 * *err 是"累积"语义, 调用方可连续取多个键后只检查一次 */
static uint64_t gguf_kv_uint(const Gguf *g, const char *key, int *err) {
    const GgufKV *kv = gguf_kv(g, key);
    if (!kv) { *err = fail("缺少元数据: %s", key); return 0; }
    if (kv->type == GGUF_I32 || kv->type == GGUF_I64) return (uint64_t)kv->v.i;
    return kv->v.u;
}

/* 取必需的浮点元数据(错误处理同上) */
static float gguf_kv_float(const Gguf *g, const char *key, int *err) {
    const GgufKV *kv = gguf_kv(g, key);
    if (!kv) { *err = fail("缺少元数据: %s", key); return 0.0f; }
    return (float)kv->v.f;
}

/* 按名字查找必需的张量, 缺失返回 NULL 并记录错误信息 */
static const GTensor *gguf_tensor(const Gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
    fail("缺少张量: %s", name);
    return NULL;
}

#ifndef TINYQWEN_LIB
/* --inspect: 打印全部元数据与张量目录, 便于核对模型文件内容 */
static void gguf_inspect(const Gguf *g) {
    printf("== 元数据 (%llu 项) ==\n", (unsigned long long)g->n_kv);
    for (uint64_t i = 0; i < g->n_kv; i++) {
        const GgufKV *kv = &g->kv[i];
        printf("  %-46s ", kv->key);
        switch (kv->type) {
        case GGUF_STRING: {
            int n = kv->str_len < 60 ? (int)kv->str_len : 60;
            printf("\"%.*s%s\"\n", n, kv->str, kv->str_len > 60 ? "..." : "");
            break;
        }
        case GGUF_ARRAY:
            printf("数组[%llu]\n", (unsigned long long)kv->arr_n);
            break;
        case GGUF_F32: case GGUF_F64:
            printf("%g\n", kv->v.f);
            break;
        case GGUF_I8: case GGUF_I16: case GGUF_I32: case GGUF_I64:
            printf("%lld\n", (long long)kv->v.i);
            break;
        default:
            printf("%llu\n", (unsigned long long)kv->v.u);
        }
    }
    printf("== 张量 (%llu 个) ==\n", (unsigned long long)g->n_tensors);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const GTensor *t = &g->tensors[i];
        printf("  %-34s %-5s [%llu, %llu]\n", t->name, ggml_type_name(t->type),
               (unsigned long long)t->ne[0], (unsigned long long)t->ne[1]);
    }
}
#endif /* TINYQWEN_LIB */

/* ===========================================================================
 * [2] 分词器 —— GPT-2 风格字节级 BPE (Byte-Pair Encoding)
 * ===========================================================================
 *
 * Qwen3 使用与 GPT-2 同源的“字节级 BPE”分词器, 词表与合并规则都存放在
 * GGUF 元数据里 (tokenizer.ggml.tokens / tokenizer.ggml.merges)。
 *
 * 字节级 BPE 的核心思想:
 *   1. 任意文本先看作原始字节序列 (天然支持所有语言和二进制内容);
 *   2. 为了让词表条目是“可打印字符串”, 每个字节被映射到一个 Unicode 码点:
 *      可打印的 188 个字节(如 'a'、'!')映射到自身, 其余 68 个字节
 *      (控制字符、空格等)按序映射到 U+0100 起的码点。
 *      例如空格(0x20) -> U+0120 'Ġ', 换行(0x0A) -> U+010A 'Ċ',
 *      这就是词表里 "Ġhello" 之类字符串的由来;
 *   3. 编码时从单字节符号出发, 反复把“合并优先级最高(规则表中最靠前)”
 *      的相邻符号对合并成更长的符号, 直到无规则可用, 每个符号即一个 token;
 *   4. 解码只需把 token 字符串的每个码点映射回原始字节。
 *
 * 简化说明(MVP): 标准实现在 BPE 之前还有一步正则预切分(按词/数字/标点
 * 分段, 段内独立 BPE)。合并规则本身是在预切分语料上学出来的, 跨词合并
 * 规则并不存在, 因此省略这一步对绝大多数文本产生完全相同的编码结果,
 * 少数边角(连续空格与标点混排等)可能与官方实现略有差异, 不影响模型使用。
 */

/* ---- 字节 <-> Unicode 码点映射表 ---- */
static uint16_t g_byte_to_cp[256];   /* 字节 -> 码点 (最大 0x143)          */
static int16_t  g_cp_to_byte[0x200]; /* 码点 -> 字节 (-1 表示无映射)       */

static void bytemap_init(void) {
    for (int i = 0; i < 0x200; i++) g_cp_to_byte[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int printable = (b >= 0x21 && b <= 0x7E) ||  /* ASCII 可见字符      */
                        (b >= 0xA1 && b <= 0xAC) ||  /* Latin-1 可见字符    */
                        (b >= 0xAE && b <= 0xFF);
        uint16_t cp = printable ? (uint16_t)b : (uint16_t)(256 + n++);
        g_byte_to_cp[b] = cp;
        g_cp_to_byte[cp] = (int16_t)b;
    }
}

/* 把码点写成 UTF-8 (本文件只需处理 <= 0x7FF 的码点, 最多 2 字节) */
static int cp_to_utf8(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    out[0] = (char)(0xC0 | (cp >> 6));
    out[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}

/* 读取一个 UTF-8 码点, 返回消费的字节数(遇到非法序列按单字节处理) */
static int utf8_to_cp(const unsigned char *s, uint32_t *cp) {
    if (s[0] < 0x80) { *cp = s[0]; return 1; }
    if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x1F) << 6) | (s[1] & 0x3F);
        return 2;
    }
    if ((s[0] & 0xF0) == 0xE0 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(s[0] & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return 3;
    }
    *cp = s[0];
    return 1;
}

/* ---- 简单的字符串 -> 整数哈希表 (开放寻址 + 线性探测) ----
 * 用于两张表: 词表字符串->id (15 万项), 合并规则"左 右"->优先级 (15 万项)。
 * 容量取 2 的幂且大于两倍元素数, 查询平均一两次探测即可命中。 */
typedef struct {
    const char **keys;  /* NULL 表示空槽; 键的内存由调用者持有 */
    int32_t     *vals;
    uint32_t     mask;  /* 容量 - 1 */
} StrMap;

static uint64_t str_hash(const char *s) {
    uint64_t h = 1469598103934665603ull;           /* FNV-1a */
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ull; }
    return h;
}

static int strmap_init(StrMap *m, uint32_t cap_pow2) {
    m->keys = calloc(cap_pow2, sizeof(char *));
    m->vals = malloc(cap_pow2 * sizeof(int32_t));
    m->mask = cap_pow2 - 1;
    if (!m->keys || !m->vals) {
        free((void *)m->keys); free(m->vals);
        m->keys = NULL; m->vals = NULL;
        return fail("内存不足(分词哈希表)");
    }
    return 0;
}

static void strmap_put(StrMap *m, const char *key, int32_t val) {
    uint32_t i = (uint32_t)str_hash(key) & m->mask;
    while (m->keys[i]) {
        if (!strcmp(m->keys[i], key)) return; /* 已存在: 保留先写入的(低 id) */
        i = (i + 1) & m->mask;
    }
    m->keys[i] = key;
    m->vals[i] = val;
}

static int32_t strmap_get(const StrMap *m, const char *key) {
    uint32_t i = (uint32_t)str_hash(key) & m->mask;
    while (m->keys[i]) {
        if (!strcmp(m->keys[i], key)) return m->vals[i];
        i = (i + 1) & m->mask;
    }
    return -1; /* 未找到 */
}

/* ---- 分词器 ---- */
typedef struct {
    int       n_vocab;
    char    **vocab;       /* id -> token 字符串(字节级 Unicode 表示)       */
    int32_t  *token_type;  /* 1=普通, 3=控制(如 <|im_start|>, 不参与显示)   */
    StrMap    vocab_map;   /* token 字符串 -> id                            */
    StrMap    merge_map;   /* "左 右"     -> 合并优先级(数字越小越优先)     */
    /* 常用特殊 token 的 id, 加载时从词表查出(而非硬编码):                  */
    int       eos_id;        /* <|im_end|>    对话轮次结束(也是生成停止符)  */
    int       im_start_id;   /* <|im_start|>                                */
    int       im_end_id;     /* <|im_end|>                                  */
    int       think_id;      /* <think>                                     */
    int       think_end_id;  /* </think>                                    */
    int       nl_id;         /* "\n" 换行                                   */
    char     *arena_vocab;   /* 词表字符串的整块内存(tok_free 释放用)       */
    char     *arena_merges;  /* 合并规则字符串的整块内存                    */
} Tokenizer;

#define TOKEN_TYPE_NORMAL  1
#define TOKEN_TYPE_CONTROL 3

static void tok_free(Tokenizer *t); /* 前向声明: 失败路径的清理用 */

/* 从 GGUF 元数据构建分词器: 拷贝词表字符串、建两张哈希表。
 * 结果填入 *t; 返回 0 成功, -1 失败(已释放全部中间资源)。 */
static int tok_init(Tokenizer *t, const Gguf *g) {
    memset(t, 0, sizeof(*t));
    bytemap_init();

    const GgufKV *tokens = gguf_kv(g, "tokenizer.ggml.tokens");
    const GgufKV *types  = gguf_kv(g, "tokenizer.ggml.token_type");
    const GgufKV *merges = gguf_kv(g, "tokenizer.ggml.merges");
    if (!tokens || tokens->arr_type != GGUF_STRING)
        return fail("模型缺少词表(tokenizer.ggml.tokens)");
    if (!merges || merges->arr_type != GGUF_STRING)
        return fail("模型缺少 BPE 合并规则(tokenizer.ggml.merges)");

    t->n_vocab = (int)tokens->arr_n;

    /* -- 词表: GGUF 里是长度前缀字符串, 拷贝为以 \0 结尾的 C 字符串 --
     * 全部字符串放进一个连续 arena, 只做一次大分配。 */
    uint64_t total = 0;
    {   /* 第一遍: 统计总字节数 */
        Cursor c = { tokens->arr_data, g->base + g->size, 0 };
        for (int i = 0; i < t->n_vocab; i++) { uint64_t l; cur_str(&c, &l); total += l + 1; }
    }
    t->arena_vocab = malloc(total);
    t->vocab = malloc(sizeof(char *) * t->n_vocab);
    if (!t->arena_vocab || !t->vocab) goto oom;
    {   /* 第二遍: 拷贝 */
        Cursor c = { tokens->arr_data, g->base + g->size, 0 };
        char *w = t->arena_vocab;
        for (int i = 0; i < t->n_vocab; i++) {
            uint64_t l;
            const char *s = cur_str(&c, &l);
            memcpy(w, s, l);
            w[l] = 0;
            t->vocab[i] = w;
            w += l + 1;
        }
    }

    /* -- token 类型表 (控制 token 解码时跳过显示) -- */
    t->token_type = malloc(sizeof(int32_t) * t->n_vocab);
    if (!t->token_type) goto oom;
    if (types && types->arr_type == GGUF_I32 && (int)types->arr_n == t->n_vocab) {
        memcpy(t->token_type, types->arr_data, sizeof(int32_t) * t->n_vocab);
    } else {
        for (int i = 0; i < t->n_vocab; i++) t->token_type[i] = TOKEN_TYPE_NORMAL;
    }

    /* -- 词表哈希: 字符串 -> id -- */
    if (strmap_init(&t->vocab_map, 1u << 19) < 0) goto bad; /* 52 万槽足够稀疏 */
    for (int i = 0; i < t->n_vocab; i++) strmap_put(&t->vocab_map, t->vocab[i], i);

    /* -- 合并规则哈希: "左 右" -> 优先级 --
     * GGUF 中每条规则形如 "Ġ Ġ" (左右两半用空格分隔; token 字符串本身
     * 不可能包含空格字节, 因为空格已被映射成 'Ġ', 所以分隔无歧义)。 */
    {
        int n_merges = (int)merges->arr_n;
        uint64_t mtotal = 0;
        Cursor c = { merges->arr_data, g->base + g->size, 0 };
        for (int i = 0; i < n_merges; i++) { uint64_t l; cur_str(&c, &l); mtotal += l + 1; }
        t->arena_merges = malloc(mtotal);
        if (!t->arena_merges) goto oom;
        if (strmap_init(&t->merge_map, 1u << 19) < 0) goto bad;
        c.p = merges->arr_data;
        char *w = t->arena_merges;
        for (int i = 0; i < n_merges; i++) {
            uint64_t l;
            const char *s = cur_str(&c, &l);
            memcpy(w, s, l);
            w[l] = 0;
            strmap_put(&t->merge_map, w, i); /* 越靠前的规则优先级越高(数值越小) */
            w += l + 1;
        }
    }

    /* -- 查出常用特殊 token 的 id -- */
    t->im_start_id  = strmap_get(&t->vocab_map, "<|im_start|>");
    t->im_end_id    = strmap_get(&t->vocab_map, "<|im_end|>");
    t->think_id     = strmap_get(&t->vocab_map, "<think>");
    t->think_end_id = strmap_get(&t->vocab_map, "</think>");
    t->nl_id        = strmap_get(&t->vocab_map, "\xC4\x8A"); /* 'Ċ' = 换行的映射 */
    if (t->im_start_id < 0 || t->im_end_id < 0 || t->nl_id < 0) {
        fail("词表中找不到 ChatML 特殊 token(不是 Qwen 系列模型?)");
        goto bad;
    }

    const GgufKV *eos = gguf_kv(g, "tokenizer.ggml.eos_token_id");
    t->eos_id = eos ? (int)eos->v.u : t->im_end_id;
    return 0;

oom:
    fail("内存不足(构建分词器)");
bad:
    tok_free(t);
    return -1;
}

/* 释放分词器全部内存(keys 的 const 强转: MSVC 对 free 有限定符警告) */
static void tok_free(Tokenizer *t) {
    free(t->vocab);
    free(t->token_type);
    free(t->arena_vocab);
    free(t->arena_merges);
    free((void *)t->vocab_map.keys);
    free(t->vocab_map.vals);
    free((void *)t->merge_map.keys);
    free(t->merge_map.vals);
    memset(t, 0, sizeof(*t));
}

/* 把 UTF-8 文本编码成 token id 序列, 返回 token 个数(失败返回 -1)。
 *
 * 算法: 经典的“贪心逐对合并” ——
 *   1. 每个输入字节先变成一个单字符符号(该字节映射码点的 UTF-8 串);
 *   2. 每轮扫描所有相邻符号对, 找到合并优先级最高的一对合并;
 *   3. 无可合并时结束, 每个符号查词表得到 id。
 * 复杂度 O(n^2) 次哈希查询, 对话输入(几 KB 以内)毫秒级完成。 */
static int tok_encode(const Tokenizer *t, const char *text, int *out, int max_out) {
    size_t n_bytes = strlen(text);
    if (n_bytes == 0) return 0;

    /* 符号数组: 指向 arena 中的 C 字符串 */
    typedef struct { const char *s; int len; } Sym;
    Sym *syms = malloc(sizeof(Sym) * n_bytes);
    /* arena 存放: 初始单字节符号(每个至多 3 字节) + 所有合并产生的新串。
     * 每次合并生成的串是词表中的合法 token(长度有限), 上界取充裕值。 */
    size_t arena_cap = n_bytes * 3 + 8 + (size_t)n_bytes * 96;
    char *arena = malloc(arena_cap), *aw = arena;
    if (!syms || !arena) {
        free(syms);
        free(arena);
        return fail("内存不足(分词缓冲)");
    }

    /* 1. 字节 -> 初始符号 */
    int n = 0;
    for (size_t i = 0; i < n_bytes; i++) {
        int len = cp_to_utf8(g_byte_to_cp[(unsigned char)text[i]], aw);
        aw[len] = 0;
        syms[n].s = aw;
        syms[n].len = len;
        n++;
        aw += len + 1;
    }

    /* 2. 反复合并优先级最高的相邻对 */
    char pair[512]; /* 拼接 "左 右" 作为规则表查询键 */
    while (n > 1) {
        int best_rank = INT32_MAX, best_i = -1;
        for (int i = 0; i + 1 < n; i++) {
            if (syms[i].len + syms[i + 1].len + 2 > (int)sizeof(pair)) continue;
            memcpy(pair, syms[i].s, syms[i].len);
            pair[syms[i].len] = ' ';
            memcpy(pair + syms[i].len + 1, syms[i + 1].s, syms[i + 1].len);
            pair[syms[i].len + 1 + syms[i + 1].len] = 0;
            int32_t rank = strmap_get(&t->merge_map, pair);
            if (rank >= 0 && rank < best_rank) { best_rank = rank; best_i = i; }
        }
        if (best_i < 0) break; /* 没有任何可用合并规则 */

        /* 合并 syms[best_i] 和 syms[best_i+1] -> arena 里的新串 */
        int newlen = syms[best_i].len + syms[best_i + 1].len;
        if ((size_t)(aw - arena) + newlen + 1 > arena_cap) {
            free(syms);
            free(arena);
            return fail("分词缓冲区溢出(输入过长)");
        }
        memcpy(aw, syms[best_i].s, syms[best_i].len);
        memcpy(aw + syms[best_i].len, syms[best_i + 1].s, syms[best_i + 1].len);
        aw[newlen] = 0;
        syms[best_i].s = aw;
        syms[best_i].len = newlen;
        aw += newlen + 1;
        memmove(&syms[best_i + 1], &syms[best_i + 2], sizeof(Sym) * (n - best_i - 2));
        n--;
    }

    /* 3. 符号 -> token id */
    int n_out = 0;
    for (int i = 0; i < n; i++) {
        int32_t id = strmap_get(&t->vocab_map, syms[i].s);
        if (id < 0) {
            /* 理论上不会发生(所有单字节都在词表中), 兜底: 拆回单字节查表 */
            for (int b = 0; b < syms[i].len; ) {
                uint32_t cp;
                b += utf8_to_cp((const unsigned char *)syms[i].s + b, &cp);
                char one[4];
                one[cp_to_utf8(cp, one)] = 0;
                int32_t bid = strmap_get(&t->vocab_map, one);
                if (bid >= 0 && n_out < max_out) out[n_out++] = bid;
            }
            continue;
        }
        if (n_out >= max_out) {
            free(syms);
            free(arena);
            return fail("输入过长(token 数超出缓冲)");
        }
        out[n_out++] = id;
    }
    free(syms);
    free(arena);
    return n_out;
}

/* 把单个 token 解码为原始字节, 写入 out(容量至少 256), 返回字节数。
 * 做法: 遍历 token 字符串的每个 Unicode 码点, 映射回原始字节。 */
static int tok_decode(const Tokenizer *t, int id, char *out) {
    if (id < 0 || id >= t->n_vocab) return 0;
    const unsigned char *s = (const unsigned char *)t->vocab[id];
    int n = 0;
    while (*s && n < 250) {
        uint32_t cp;
        int adv = utf8_to_cp(s, &cp);
        if (cp < 0x200 && g_cp_to_byte[cp] >= 0) {
            out[n++] = (char)g_cp_to_byte[cp];   /* 正常路径: 码点 -> 字节  */
        } else {
            memcpy(out + n, s, adv);             /* 无映射: 原样拷贝(罕见)  */
            n += adv;
        }
        s += adv;
    }
    return n;
}

/* ===========================================================================
 * [3] 模型定义与加载
 * ===========================================================================
 *
 * Qwen3 是标准的 decoder-only Transformer, 每层结构:
 *
 *     x ─┬─ RMSNorm ─ 注意力(QK-Norm + RoPE + GQA) ─┬─ x    (残差相加)
 *        └────────────────────────────────────────┘
 *     x ─┬─ RMSNorm ─ SwiGLU FFN (gate/up/down) ───┬─ x    (残差相加)
 *        └────────────────────────────────────────┘
 *
 * 权重张量直接引用 mmap 内存(不拷贝), 加载只是"按名字找指针"。
 */

/* 模型超参数(全部来自 GGUF 元数据, 对 Qwen3-0.6B 的取值见注释) */
typedef struct {
    int   dim;        /* 隐层维度                  1024                     */
    int   n_layers;   /* Transformer 层数          28                       */
    int   n_heads;    /* 查询头数                  16                       */
    int   n_kv_heads; /* KV 头数(GQA 分组)         8                        */
    int   head_dim;   /* 每头维度                  128                      */
    int   ffn_dim;    /* FFN 中间层维度            3072                     */
    int   n_vocab;    /* 词表大小                  151936                   */
    int   ctx_len;    /* 运行时上下文上限(KV 缓存长度, 用户可用 -c 调整)    */
    float rope_base;  /* RoPE 基频                 1e6                      */
    float rms_eps;    /* RMSNorm 的 epsilon        1e-6                     */
} Config;

/* 单层的全部权重(指向 mmap) */
typedef struct {
    const GTensor *attn_norm;              /* 注意力前的 RMSNorm  [dim]     */
    const GTensor *attn_q, *attn_k, *attn_v, *attn_out; /* 四个投影矩阵     */
    const GTensor *q_norm, *k_norm;        /* Qwen3 的 QK-RMSNorm [head_dim]*/
    const GTensor *ffn_norm;               /* FFN 前的 RMSNorm    [dim]     */
    const GTensor *ffn_gate, *ffn_up, *ffn_down;        /* SwiGLU 三矩阵    */
} Layer;

typedef struct {
    Config         cfg;
    const GTensor *token_embd;  /* 词嵌入 [dim, n_vocab]; 0.6B 与输出层共享 */
    const GTensor *output;      /* 独立输出投影(本模型无, 为 NULL 时用嵌入) */
    const GTensor *output_norm; /* 最后的 RMSNorm 权重                      */
    Layer         *layers;
    float         *rope_freq;   /* 预计算的 RoPE 频率表 [head_dim/2]        */
} Model;

/* 可选张量查找(不存在返回 NULL) */
static const GTensor *gguf_tensor_opt(const Gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}

static void model_free(Model *m); /* 前向声明: 失败路径的清理用 */

/* 加载模型: 读取超参数、定位全部权重张量, 结果填入 *out。
 * 返回 0 成功, -1 失败(已释放中间资源, 错误详情在 g_err_msg)。 */
static int model_load(Model *out, const Gguf *g, int ctx_limit) {
    Model m = {0};
    Config *c = &m.cfg;
    memset(out, 0, sizeof(*out));

    /* ---- 架构检查 ----
     * 同一套代码支持 qwen3 架构的全部稠密规格(0.6B/1.7B/4B/8B/14B/32B):
     * 层数、维度、头数等一切结构差异都由下面的元数据驱动。 */
    const GgufKV *arch = gguf_kv(g, "general.architecture");
    if (!arch || arch->str_len != 5 || memcmp(arch->str, "qwen3", 5))
        return fail("不支持的模型架构(目前只支持 qwen3 系列稠密模型)");

    /* ---- 超参数(err 为累积错误标志, 取完统一检查) ---- */
    int err = 0;
    c->dim        = (int)gguf_kv_uint(g, "qwen3.embedding_length", &err);
    c->n_layers   = (int)gguf_kv_uint(g, "qwen3.block_count", &err);
    c->n_heads    = (int)gguf_kv_uint(g, "qwen3.attention.head_count", &err);
    c->n_kv_heads = (int)gguf_kv_uint(g, "qwen3.attention.head_count_kv", &err);
    c->head_dim   = (int)gguf_kv_uint(g, "qwen3.attention.key_length", &err);
    c->ffn_dim    = (int)gguf_kv_uint(g, "qwen3.feed_forward_length", &err);
    c->rope_base  = gguf_kv_float(g, "qwen3.rope.freq_base", &err);
    c->rms_eps    = gguf_kv_float(g, "qwen3.attention.layer_norm_rms_epsilon", &err);
    int model_ctx = (int)gguf_kv_uint(g, "qwen3.context_length", &err);
    if (err) return -1;
    if ((int)gguf_kv_uint(g, "qwen3.attention.value_length", &err) != c->head_dim)
        return fail("暂不支持 K/V 头维度不一致的模型");
    if (c->dim <= 0 || c->n_layers <= 0 || c->n_heads <= 0 ||
        c->n_kv_heads <= 0 || c->head_dim <= 0 || c->ffn_dim <= 0 ||
        c->n_heads % c->n_kv_heads != 0)
        return fail("模型超参数取值异常");
    c->ctx_len = ctx_limit < model_ctx ? ctx_limit : model_ctx;

    /* ---- 全局张量 ---- */
    m.token_embd  = gguf_tensor(g, "token_embd.weight");
    m.output_norm = gguf_tensor(g, "output_norm.weight");
    m.output      = gguf_tensor_opt(g, "output.weight"); /* 小模型: 共享嵌入 */
    if (!m.token_embd || !m.output_norm) return -1;
    c->n_vocab = (int)m.token_embd->ne[1];

    /* ---- 每层张量: 名字形如 blk.<层号>.<用途>.weight ---- */
    m.layers = calloc(c->n_layers, sizeof(Layer));
    if (!m.layers) return fail("内存不足");
    char nm[64];
    for (int l = 0; l < c->n_layers; l++) {
        Layer *L = &m.layers[l];
#define GET(field, suffix) \
        snprintf(nm, sizeof(nm), "blk.%d." suffix ".weight", l); \
        if (!(L->field = gguf_tensor(g, nm))) goto bad
        GET(attn_norm, "attn_norm");
        GET(attn_q,    "attn_q");
        GET(attn_k,    "attn_k");
        GET(attn_v,    "attn_v");
        GET(attn_out,  "attn_output");
        GET(q_norm,    "attn_q_norm");
        GET(k_norm,    "attn_k_norm");
        GET(ffn_norm,  "ffn_norm");
        GET(ffn_gate,  "ffn_gate");
        GET(ffn_up,    "ffn_up");
        GET(ffn_down,  "ffn_down");
#undef GET
    }

    /* ---- 维度一致性检查(防呆: 确认文件与代码假设一致) ---- */
    if ((int)m.layers[0].attn_q->ne[0] != c->dim ||
        (int)m.layers[0].attn_q->ne[1] != c->n_heads * c->head_dim ||
        (int)m.layers[0].attn_k->ne[1] != c->n_kv_heads * c->head_dim ||
        (int)m.layers[0].ffn_gate->ne[1] != c->ffn_dim ||
        (int)m.token_embd->ne[0] != c->dim) {
        fail("张量维度与超参数不一致");
        goto bad;
    }

    /* ---- RoPE 频率表: freq[i] = base^(-2i/head_dim) ---- */
    int half = c->head_dim / 2;
    m.rope_freq = malloc(sizeof(float) * half);
    if (!m.rope_freq) { fail("内存不足"); goto bad; }
    for (int i = 0; i < half; i++)
        m.rope_freq[i] = powf(c->rope_base, -2.0f * i / c->head_dim);

    *out = m;
    return 0;

bad:
    model_free(&m);
    return -1;
}

/* 释放模型的目录性内存(权重本体在 mmap 中, 由 gguf_close 释放) */
static void model_free(Model *m) {
    free(m->layers);
    free(m->rope_freq);
    memset(m, 0, sizeof(*m));
}

/* ===========================================================================
 * [4] 数学内核 —— 反量化 + 标量矩阵乘, 以及 RMSNorm / Softmax
 * ===========================================================================
 *
 * 推理 99% 的时间花在"量化矩阵 × 浮点向量"上。GGUF 的量化矩阵按行存储,
 * 每行由若干"量化块"组成, 我们对每行做"逐块反量化并累加点积"(融合计算,
 * 不生成中间浮点行)。全部是朴素标量循环 —— 刻意不用 SIMD, 保持可读性。
 *
 * Q4_0 块(18 字节 = 32 个权重):       Q4_1 块(20 字节 = 32 个权重):
 *   fp16 d;        缩放因子             fp16 d, m;    缩放 + 偏移
 *   u8 qs[16];     32 个 4bit 值        u8 qs[16];
 *   权重 = (q - 8) * d                  权重 = q * d + m
 *   低半字节存元素 0..15,               (同左)
 *   高半字节存元素 16..31
 *
 * Q6_K 超块(210 字节 = 256 个权重, 仅词嵌入层使用):
 *   u8 ql[128];    低 4 位              权重 = d * scale[组] * (q - 32)
 *   u8 qh[64];     高 2 位              其中 q 由 ql 与 qh 拼成 6bit,
 *   i8 scales[16]; 16 组子缩放          每 16 个权重共享一个 scale
 *   fp16 d;        总缩放
 */

/* -- 一行 Q4_0 与向量 x 的点积 (n 是行长, 必为 32 的倍数) -- */
static float dot_q4_0(const uint8_t *row, const float *x, int n) {
    float acc = 0.0f;
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const uint8_t *qs = blk + 2;
        const float *xb = x + b * 32;
        float sum = 0.0f;
        for (int i = 0; i < 16; i++) {
            sum += (float)((int)(qs[i] & 0x0F) - 8) * xb[i];      /* 低半字节 */
            sum += (float)((int)(qs[i] >> 4)   - 8) * xb[i + 16]; /* 高半字节 */
        }
        acc += F16(dh) * sum;
    }
    return acc;
}

/* -- 一行 Q4_1 与向量 x 的点积 -- */
static float dot_q4_1(const uint8_t *row, const float *x, int n) {
    float acc = 0.0f;
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 20;
        uint16_t dh, mh;
        memcpy(&dh, blk, 2);
        memcpy(&mh, blk + 2, 2);
        const uint8_t *qs = blk + 4;
        const float *xb = x + b * 32;
        float qsum = 0.0f, xsum = 0.0f; /* Σq·x 与 Σx 分开累加: w=dq+m      */
        for (int i = 0; i < 16; i++) {
            qsum += (float)(qs[i] & 0x0F) * xb[i];
            qsum += (float)(qs[i] >> 4)   * xb[i + 16];
            xsum += xb[i] + xb[i + 16];
        }
        acc += F16(dh) * qsum + F16(mh) * xsum;
    }
    return acc;
}

/* -- 一行 Q8_0 与向量 x 的点积 (块 = fp16 缩放 + 32 个 int8) --
 * Q4_0 的 GGUF 里不出现, 但支持它之后同一程序也能加载 Q8_0 量化文件 */
static float dot_q8_0(const uint8_t *row, const float *x, int n) {
    float acc = 0.0f;
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float *xb = x + b * 32;
        float sum = 0.0f;
        for (int i = 0; i < 32; i++) sum += (float)qs[i] * xb[i];
        acc += F16(dh) * sum;
    }
    return acc;
}

/* -- 反量化一个 Q6_K 超块(256 个权重)到 out —— 布局见 ggml 参考实现 -- */
static void dequant_q6_k_block(const uint8_t *blk, float *out) {
    const uint8_t *ql = blk;                       /* 低 4 位, 128B         */
    const uint8_t *qh = blk + 128;                 /* 高 2 位, 64B          */
    const int8_t  *sc = (const int8_t *)(blk + 192); /* 16 组子缩放         */
    uint16_t dh;
    memcpy(&dh, blk + 208, 2);
    float d = F16(dh);

    for (int half = 0; half < 2; half++) {         /* 每半 128 个权重       */
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l]      = d * sc[is + 0] * q1;
            out[l + 32] = d * sc[is + 2] * q2;
            out[l + 64] = d * sc[is + 4] * q3;
            out[l + 96] = d * sc[is + 6] * q4;
        }
        out += 128; ql += 64; qh += 32; sc += 8;
    }
}

/* -- 一行 Q6_K 与向量 x 的点积(先反量化超块再点积; 仅输出层走此路径) -- */
static float dot_q6_k(const uint8_t *row, const float *x, int n) {
    float acc = 0.0f;
    float buf[256];
    for (int b = 0; b < n / 256; b++) {
        dequant_q6_k_block(row + b * 210, buf);
        const float *xb = x + b * 256;
        for (int i = 0; i < 256; i++) acc += buf[i] * xb[i];
    }
    return acc;
}

/* ===========================================================================
 * [4b] AVX2 SIMD 加速内核(可选, 运行时检测)
 * ===========================================================================
 *
 * 上面的标量内核在任何 CPU 上都能跑; 这里为 x86 的 AVX2+FMA 指令集提供
 * 等价的向量化版本, 一次算 8 个 float。设计要点:
 *
 *   - 运行时检测: 程序启动时用 CPUID 查询 CPU 是否支持 AVX2 和 FMA,
 *     支持才把函数指针切换到这里的实现 —— 同一个二进制文件在老 CPU 上
 *     自动退回标量路径, 不会崩溃;
 *   - 编译兼容: GCC/Clang 用 __attribute__((target("avx2,fma"))) 单独
 *     给这些函数开 AVX2 代码生成(整个文件仍按通用指令集编译), MSVC 则
 *     本来就允许直接使用 intrinsics; 非 x86 架构下本段代码不参与编译
 *     (aarch64 的 NEON 内核见下方 [4c] 分支, 其它架构只有标量内核);
 *   - 依然零依赖: intrinsics 头文件 immintrin.h 是编译器自带的。
 *
 * 注: FMA(乘加融合)与标量代码的舍入次序不同, 结果可能有 1ulp 级微小
 * 差异, 属正常现象(可用 --selftest 验证误差, --no-simd 强制标量)。
 */

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define TINYQWEN_X86 1
#include <immintrin.h>

#if defined(_MSC_VER)
#include <intrin.h>
#define TARGET_AVX2 /* MSVC 允许直接使用 AVX2 intrinsics, 无需目标属性 */

/* CPUID 检测 AVX2+FMA, 并用 XGETBV 确认操作系统保存 YMM 寄存器状态 */
static int cpu_has_avx2_fma(void) {
    int r[4];
    __cpuid(r, 1);
    int osxsave = (r[2] >> 27) & 1;   /* OS 启用了 XSAVE          */
    int avx     = (r[2] >> 28) & 1;
    int fma     = (r[2] >> 12) & 1;
    if (!osxsave || !avx || !fma) return 0;
    if ((_xgetbv(0) & 0x6) != 0x6) return 0; /* XMM+YMM 状态被 OS 保存 */
    __cpuidex(r, 7, 0);
    return (r[1] >> 5) & 1;           /* EBX bit5 = AVX2          */
}

#else /* GCC / Clang */
#define TARGET_AVX2 __attribute__((target("avx2,fma")))

/* GCC/Clang 内建的 CPU 特性检测(内部同样走 CPUID) */
static int cpu_has_avx2_fma(void) {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

/* 8 路水平求和: __m256 的 8 个 float 加成一个标量 */
TARGET_AVX2 static inline float hsum8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

/* 把 __m128i 的低 8 字节(无符号)转成 8 个 float */
#define CVT8(v128) _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v128))

/* -- Q4_0 点积的 AVX2 版: 与标量版逐块等价 -- */
TARGET_AVX2 static float dot_q4_0_avx2(const uint8_t *row, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    const __m256  off8  = _mm256_set1_ps(8.0f);
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        __m128i qs = _mm_loadu_si128((const __m128i *)(blk + 2));
        __m128i lo = _mm_and_si128(qs, mask4);                     /* 元素 0..15  */
        __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), mask4);  /* 元素 16..31 */
        const float *xb = x + b * 32;
        /* sum = Σ (q-8)·x, 8 个一组共 4 组 */
        __m256 sum = _mm256_setzero_ps();
        sum = _mm256_fmadd_ps(_mm256_sub_ps(CVT8(lo), off8),
                              _mm256_loadu_ps(xb), sum);
        sum = _mm256_fmadd_ps(_mm256_sub_ps(CVT8(_mm_srli_si128(lo, 8)), off8),
                              _mm256_loadu_ps(xb + 8), sum);
        sum = _mm256_fmadd_ps(_mm256_sub_ps(CVT8(hi), off8),
                              _mm256_loadu_ps(xb + 16), sum);
        sum = _mm256_fmadd_ps(_mm256_sub_ps(CVT8(_mm_srli_si128(hi, 8)), off8),
                              _mm256_loadu_ps(xb + 24), sum);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(F16(dh)), sum, acc);
    }
    return hsum8(acc);
}

/* -- Q4_1 点积的 AVX2 版: w = d·q + m, 分别累加 Σq·x 与 Σx -- */
TARGET_AVX2 static float dot_q4_1_avx2(const uint8_t *row, const float *x, int n) {
    __m256 accq = _mm256_setzero_ps(); /* Σ d·(q·x) */
    __m256 accx = _mm256_setzero_ps(); /* Σ m·x     */
    const __m128i mask4 = _mm_set1_epi8(0x0F);
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 20;
        uint16_t dh, mh;
        memcpy(&dh, blk, 2);
        memcpy(&mh, blk + 2, 2);
        __m128i qs = _mm_loadu_si128((const __m128i *)(blk + 4));
        __m128i lo = _mm_and_si128(qs, mask4);
        __m128i hi = _mm_and_si128(_mm_srli_epi16(qs, 4), mask4);
        const float *xb = x + b * 32;
        __m256 x0 = _mm256_loadu_ps(xb),      x1 = _mm256_loadu_ps(xb + 8);
        __m256 x2 = _mm256_loadu_ps(xb + 16), x3 = _mm256_loadu_ps(xb + 24);
        __m256 sumq = _mm256_setzero_ps();
        sumq = _mm256_fmadd_ps(CVT8(lo), x0, sumq);
        sumq = _mm256_fmadd_ps(CVT8(_mm_srli_si128(lo, 8)), x1, sumq);
        sumq = _mm256_fmadd_ps(CVT8(hi), x2, sumq);
        sumq = _mm256_fmadd_ps(CVT8(_mm_srli_si128(hi, 8)), x3, sumq);
        __m256 sumx = _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
        accq = _mm256_fmadd_ps(_mm256_set1_ps(F16(dh)), sumq, accq);
        accx = _mm256_fmadd_ps(_mm256_set1_ps(F16(mh)), sumx, accx);
    }
    return hsum8(accq) + hsum8(accx);
}

/* -- Q8_0 点积的 AVX2 版 -- */
TARGET_AVX2 static float dot_q8_0_avx2(const uint8_t *row, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        __m128i q0 = _mm_loadu_si128((const __m128i *)(blk + 2));      /* int8 0..15  */
        __m128i q1 = _mm_loadu_si128((const __m128i *)(blk + 18));     /* int8 16..31 */
        const float *xb = x + b * 32;
        __m256 sum = _mm256_setzero_ps();
        /* int8 是有符号的, 用 cvtepi8_epi32 做符号扩展(注意与 4bit 内核不同) */
        sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)),
                              _mm256_loadu_ps(xb), sum);
        sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8))),
                              _mm256_loadu_ps(xb + 8), sum);
        sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)),
                              _mm256_loadu_ps(xb + 16), sum);
        sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8))),
                              _mm256_loadu_ps(xb + 24), sum);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(F16(dh)), sum, acc);
    }
    return hsum8(acc);
}

/* Q6_K 辅助: 32 字节 6bit 值(0..63) q 与 x 的 32 个 float 做 (q-32)·x,
 * 前 16 个元素乘 dsc0、后 16 个乘 dsc1(d×子缩放), 累加进 acc */
TARGET_AVX2 static inline __m256 q6k_accum(__m256i q, const float *xp,
                                           float dsc0, float dsc1, __m256 acc) {
    const __m256 off32 = _mm256_set1_ps(32.0f);
    __m128i vlo = _mm256_castsi256_si128(q);        /* 字节 0..15  */
    __m128i vhi = _mm256_extracti128_si256(q, 1);   /* 字节 16..31 */
    __m256 s0 = _mm256_setzero_ps();
    s0 = _mm256_fmadd_ps(_mm256_sub_ps(CVT8(vlo), off32),
                         _mm256_loadu_ps(xp), s0);
    s0 = _mm256_fmadd_ps(_mm256_sub_ps(CVT8(_mm_srli_si128(vlo, 8)), off32),
                         _mm256_loadu_ps(xp + 8), s0);
    __m256 s1 = _mm256_setzero_ps();
    s1 = _mm256_fmadd_ps(_mm256_sub_ps(CVT8(vhi), off32),
                         _mm256_loadu_ps(xp + 16), s1);
    s1 = _mm256_fmadd_ps(_mm256_sub_ps(CVT8(_mm_srli_si128(vhi, 8)), off32),
                         _mm256_loadu_ps(xp + 24), s1);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(dsc0), s0, acc);
    acc = _mm256_fmadd_ps(_mm256_set1_ps(dsc1), s1, acc);
    return acc;
}

/* -- Q6_K 点积的 AVX2 版(输出层的大矩阵走这里, 收益最大) --
 * 6bit 值的拼装与标量版完全一致: 低 4 位来自 ql, 高 2 位来自 qh 的
 * 不同位段; 字节内位移用 epi16 移位 + 字节掩码实现(AVX2 无字节移位)。 */
TARGET_AVX2 static float dot_q6_k_avx2(const uint8_t *row, const float *x, int n) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i mf  = _mm256_set1_epi8(0x0F);
    const __m256i m30 = _mm256_set1_epi8(0x30);
    for (int b = 0; b < n / 256; b++) {
        const uint8_t *blk = row + b * 210;
        const int8_t  *sc  = (const int8_t *)(blk + 192);
        uint16_t dh;
        memcpy(&dh, blk + 208, 2);
        float d = F16(dh);
        const float *xb = x + b * 256;

        for (int half = 0; half < 2; half++) {     /* 每半 128 个权重 */
            const uint8_t *pql = blk + half * 64;        /* ql 低位表   */
            const uint8_t *pqh = blk + 128 + half * 32;  /* qh 高位表   */
            const int8_t  *ps  = sc + half * 8;          /* 8 组子缩放  */
            const float   *xh  = xb + half * 128;
            __m256i ql_lo = _mm256_loadu_si256((const __m256i *)pql);        /* ql[l]    */
            __m256i ql_hi = _mm256_loadu_si256((const __m256i *)(pql + 32)); /* ql[l+32] */
            __m256i qhv   = _mm256_loadu_si256((const __m256i *)pqh);        /* qh[l]    */
            /* 四组 6bit 值, 与标量版 q1..q4 一一对应:                      */
            __m256i q1 = _mm256_or_si256(_mm256_and_si256(ql_lo, mf),
                         _mm256_and_si256(_mm256_slli_epi16(qhv, 4), m30));
            __m256i q2 = _mm256_or_si256(_mm256_and_si256(ql_hi, mf),
                         _mm256_and_si256(_mm256_slli_epi16(qhv, 2), m30));
            __m256i q3 = _mm256_or_si256(
                         _mm256_and_si256(_mm256_srli_epi16(ql_lo, 4), mf),
                         _mm256_and_si256(qhv, m30));
            __m256i q4 = _mm256_or_si256(
                         _mm256_and_si256(_mm256_srli_epi16(ql_hi, 4), mf),
                         _mm256_and_si256(_mm256_srli_epi16(qhv, 2), m30));
            acc = q6k_accum(q1, xh,      d * ps[0], d * ps[1], acc);
            acc = q6k_accum(q2, xh + 32, d * ps[2], d * ps[3], acc);
            acc = q6k_accum(q3, xh + 64, d * ps[4], d * ps[5], acc);
            acc = q6k_accum(q4, xh + 96, d * ps[6], d * ps[7], acc);
        }
    }
    return hsum8(acc);
}

#elif defined(__aarch64__) || defined(_M_ARM64)
/* ===========================================================================
 * [4c] AArch64 NEON 加速内核(实验性)
 * ===========================================================================
 *
 * NEON(AdvSIMD) 是 armv8-a 的强制特性, 与 x86 的 AVX2 不同, 不需要
 * 按函数的目标属性, 基线编译即可使用 intrinsics; 这里的运行时检测
 * 主要是与 x86 路径保持同构(Linux 上真读 HWCAP, 其它系统按架构恒真),
 * 并保留 --no-simd 强制标量的能力。
 * NEON 是 128 位(AVX2 的一半宽): 一次算 4 个 float, 8bit 整数拓宽链是
 * int8x16 -> 2x int16x8 -> 4x int32x4 -> 4x float32x4。
 */
#define TINYQWEN_AARCH64 1
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>   /* getauxval    */
#include <asm/hwcap.h>  /* HWCAP_ASIMD  */
#endif

static int cpu_has_neon(void) {
#if defined(__linux__) && defined(HWCAP_ASIMD)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#else
    return 1; /* Apple Silicon / Windows ARM64: NEON 架构必备 */
#endif
}

/* 16 个 int8 与 16 个 float 的逐项乘积, 累加进 4 路求和向量 */
static inline float32x4_t neon_dot16(float32x4_t sum, int8x16_t q, const float *xp) {
    int16x8_t lo = vmovl_s8(vget_low_s8(q));   /* 低 8 字节拓宽到 16 位 */
    int16x8_t hi = vmovl_s8(vget_high_s8(q));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),  vld1q_f32(xp));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(xp + 4));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),  vld1q_f32(xp + 8));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(xp + 12));
    return sum;
}

/* -- Q4_0 点积的 NEON 版: 与标量版逐块等价 -- */
static float dot_q4_0_neon(const uint8_t *row, const float *x, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    const int8x16_t  off8  = vdupq_n_s8(8);
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 18;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        uint8x16_t qs = vld1q_u8(blk + 2);
        /* 低/高半字节各 16 个值, 减 8 得有符号权重 */
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(qs, mask4)), off8);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qs, 4)), off8);
        const float *xb = x + b * 32;
        float32x4_t sum = vdupq_n_f32(0.0f);
        sum = neon_dot16(sum, lo, xb);
        sum = neon_dot16(sum, hi, xb + 16);
        acc = vfmaq_f32(acc, sum, vdupq_n_f32(F16(dh)));
    }
    return vaddvq_f32(acc); /* 4 路横向求和 */
}

/* -- Q4_1 点积的 NEON 版: w = d·q + m, 分别累加 Σq·x 与 Σx -- */
static float dot_q4_1_neon(const uint8_t *row, const float *x, int n) {
    float32x4_t accq = vdupq_n_f32(0.0f); /* Σ d·(q·x) */
    float32x4_t accx = vdupq_n_f32(0.0f); /* Σ m·x     */
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 20;
        uint16_t dh, mh;
        memcpy(&dh, blk, 2);
        memcpy(&mh, blk + 2, 2);
        uint8x16_t qs = vld1q_u8(blk + 4);
        /* q 为无符号 0..15, 直接当 int8 用(不减偏移) */
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(qs, mask4));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(qs, 4));
        const float *xb = x + b * 32;
        float32x4_t sumq = vdupq_n_f32(0.0f);
        sumq = neon_dot16(sumq, lo, xb);
        sumq = neon_dot16(sumq, hi, xb + 16);
        float32x4_t sumx = vaddq_f32(
            vaddq_f32(vaddq_f32(vld1q_f32(xb),      vld1q_f32(xb + 4)),
                      vaddq_f32(vld1q_f32(xb + 8),  vld1q_f32(xb + 12))),
            vaddq_f32(vaddq_f32(vld1q_f32(xb + 16), vld1q_f32(xb + 20)),
                      vaddq_f32(vld1q_f32(xb + 24), vld1q_f32(xb + 28))));
        accq = vfmaq_f32(accq, sumq, vdupq_n_f32(F16(dh)));
        accx = vfmaq_f32(accx, sumx, vdupq_n_f32(F16(mh)));
    }
    return vaddvq_f32(accq) + vaddvq_f32(accx);
}

/* -- Q8_0 点积的 NEON 版 -- */
static float dot_q8_0_neon(const uint8_t *row, const float *x, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < n / 32; b++) {
        const uint8_t *blk = row + b * 34;
        uint16_t dh;
        memcpy(&dh, blk, 2);
        int8x16_t q0 = vld1q_s8((const int8_t *)(blk + 2));
        int8x16_t q1 = vld1q_s8((const int8_t *)(blk + 18));
        const float *xb = x + b * 32;
        float32x4_t sum = vdupq_n_f32(0.0f);
        sum = neon_dot16(sum, q0, xb);
        sum = neon_dot16(sum, q1, xb + 16);
        acc = vfmaq_f32(acc, sum, vdupq_n_f32(F16(dh)));
    }
    return vaddvq_f32(acc);
}

/* -- Q6_K 点积的 NEON 版 --
 * 6bit 值拼装与标量版一致; u8 的字节内移位天然干净(对比 AVX2 需要
 * epi16 移位加掩码), 只有跨位段拼装处需要 &0x30。
 * 每 16 个权重共享一个子缩放, 恰好一条 int8x16 一组。 */
static float dot_q6_k_neon(const uint8_t *row, const float *x, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    const uint8x16_t mf    = vdupq_n_u8(0x0F);
    const uint8x16_t m30   = vdupq_n_u8(0x30);
    const int8x16_t  off32 = vdupq_n_s8(32);
    for (int b = 0; b < n / 256; b++) {
        const uint8_t *blk = row + b * 210;
        const int8_t  *sc  = (const int8_t *)(blk + 192);
        uint16_t dh;
        memcpy(&dh, blk + 208, 2);
        float d = F16(dh);
        const float *xb = x + b * 256;

        for (int half = 0; half < 2; half++) {     /* 每半 128 个权重 */
            const uint8_t *pql = blk + half * 64;        /* ql 低位表  */
            const uint8_t *pqh = blk + 128 + half * 32;  /* qh 高位表  */
            const int8_t  *ps  = sc + half * 8;          /* 8 组子缩放 */
            const float   *xh  = xb + half * 128;
            for (int g = 0; g < 2; g++) {          /* 元素 l = g*16 .. g*16+15 */
                uint8x16_t ql_lo = vld1q_u8(pql + g * 16);      /* ql[l]    */
                uint8x16_t ql_hi = vld1q_u8(pql + 32 + g * 16); /* ql[l+32] */
                uint8x16_t qh    = vld1q_u8(pqh + g * 16);      /* qh[l]    */
                /* 四组 6bit 值, 与标量版 q1..q4 一一对应 */
                uint8x16_t q1 = vorrq_u8(vandq_u8(ql_lo, mf),
                                         vandq_u8(vshlq_n_u8(qh, 4), m30));
                uint8x16_t q2 = vorrq_u8(vandq_u8(ql_hi, mf),
                                         vandq_u8(vshlq_n_u8(qh, 2), m30));
                uint8x16_t q3 = vorrq_u8(vshrq_n_u8(ql_lo, 4),
                                         vandq_u8(qh, m30));
                uint8x16_t q4 = vorrq_u8(vshrq_n_u8(ql_hi, 4),
                                         vandq_u8(vshrq_n_u8(qh, 2), m30));
                float32x4_t t;
                t = neon_dot16(vdupq_n_f32(0.0f),
                               vsubq_s8(vreinterpretq_s8_u8(q1), off32), xh + g * 16);
                acc = vfmaq_f32(acc, t, vdupq_n_f32(d * ps[g]));
                t = neon_dot16(vdupq_n_f32(0.0f),
                               vsubq_s8(vreinterpretq_s8_u8(q2), off32), xh + 32 + g * 16);
                acc = vfmaq_f32(acc, t, vdupq_n_f32(d * ps[2 + g]));
                t = neon_dot16(vdupq_n_f32(0.0f),
                               vsubq_s8(vreinterpretq_s8_u8(q3), off32), xh + 64 + g * 16);
                acc = vfmaq_f32(acc, t, vdupq_n_f32(d * ps[4 + g]));
                t = neon_dot16(vdupq_n_f32(0.0f),
                               vsubq_s8(vreinterpretq_s8_u8(q4), off32), xh + 96 + g * 16);
                acc = vfmaq_f32(acc, t, vdupq_n_f32(d * ps[6 + g]));
            }
        }
    }
    return vaddvq_f32(acc);
}

#endif /* 架构分支(x86 / aarch64; 其它架构只有标量内核) */

/* ---- 点积内核的运行时分发 ----
 * 默认指向标量实现; simd_init() 按架构检测到可用指令集时切换。 */
typedef float (*DotFn)(const uint8_t *row, const float *x, int n);

static DotFn g_dot_q4_0 = dot_q4_0;
static DotFn g_dot_q4_1 = dot_q4_1;
static DotFn g_dot_q6_k = dot_q6_k;
static DotFn g_dot_q8_0 = dot_q8_0;
static int         g_use_simd  = 0;    /* 是否启用了 SIMD 内核    */
static const char *g_simd_name = "无"; /* 启用的指令集名(显示用)  */

/* 检测并启用 SIMD(allow=0 时强制标量, 对应 --no-simd) */
static void simd_init(int allow) {
#if defined(TINYQWEN_X86)
    if (allow && cpu_has_avx2_fma()) {
        g_use_simd  = 1;
        g_simd_name = "AVX2+FMA";
        g_dot_q4_0 = dot_q4_0_avx2;
        g_dot_q4_1 = dot_q4_1_avx2;
        g_dot_q6_k = dot_q6_k_avx2;
        g_dot_q8_0 = dot_q8_0_avx2;
    }
#elif defined(TINYQWEN_AARCH64)
    if (allow && cpu_has_neon()) {
        g_use_simd  = 1;
        g_simd_name = "NEON";
        g_dot_q4_0 = dot_q4_0_neon;
        g_dot_q4_1 = dot_q4_1_neon;
        g_dot_q6_k = dot_q6_k_neon;
        g_dot_q8_0 = dot_q8_0_neon;
    }
#else
    (void)allow; /* 其它架构: 恒用标量内核 */
#endif
}

/* -- 矩阵×向量: out[r] = W 第 r 行 · x  (W 是 GGUF 张量, [n_in, n_out]) --
 * 这是整个推理器的热点函数: 各行的点积互相独立, 用线程池按行并行。 */
typedef struct {
    float         *out;
    const GTensor *w;
    const float   *x;
} MatvecCtx;

static void matvec_task(int r0, int r1, const void *pc) {
    const MatvecCtx *c = (const MatvecCtx *)pc;
    const GTensor *w = c->w;
    int n_in = (int)w->ne[0];
    uint64_t row_bytes = ggml_row_bytes(w->type, w->ne[0]);
    const uint8_t *base = (const uint8_t *)w->data;

    for (int r = r0; r < r1; r++) {
        const uint8_t *row = base + (uint64_t)r * row_bytes;
        float v;
        switch (w->type) { /* 经函数指针分发: 标量或 AVX2(见 simd_init) */
        case GGML_Q4_0: v = g_dot_q4_0(row, c->x, n_in); break;
        case GGML_Q4_1: v = g_dot_q4_1(row, c->x, n_in); break;
        case GGML_Q6_K: v = g_dot_q6_k(row, c->x, n_in); break;
        case GGML_Q8_0: v = g_dot_q8_0(row, c->x, n_in); break;
        case GGML_F32: {
            const float *rw = (const float *)row;
            float s = 0.0f;
            for (int i = 0; i < n_in; i++) s += rw[i] * c->x[i];
            v = s;
            break;
        }
        case GGML_F16: {
            const uint16_t *rw = (const uint16_t *)row;
            float s = 0.0f;
            for (int i = 0; i < n_in; i++) s += F16(rw[i]) * c->x[i];
            v = s;
            break;
        }
        default: bug("matvec 遇到未知张量类型(加载期校验遗漏)"); v = 0;
        }
        c->out[r] = v;
    }
}

static void matvec(float *out, const GTensor *w, const float *x) {
    MatvecCtx c = { out, w, x };
    parallel_for(matvec_task, &c, (int)w->ne[1]);
}

/* -- 反量化张量的第 row 行到 out (用于词嵌入查表) -- */
static void dequant_row(const GTensor *t, int row, float *out) {
    int n = (int)t->ne[0];
    const uint8_t *r = (const uint8_t *)t->data + (uint64_t)row * ggml_row_bytes(t->type, t->ne[0]);
    switch (t->type) {
    case GGML_F32:
        memcpy(out, r, sizeof(float) * n);
        break;
    case GGML_F16:
        for (int i = 0; i < n; i++) out[i] = F16(((const uint16_t *)r)[i]);
        break;
    case GGML_Q6_K:
        for (int b = 0; b < n / 256; b++)
            dequant_q6_k_block(r + b * 210, out + b * 256);
        break;
    case GGML_Q4_0:
        for (int b = 0; b < n / 32; b++) {
            uint16_t dh;
            memcpy(&dh, r + b * 18, 2);
            float d = F16(dh);
            const uint8_t *qs = r + b * 18 + 2;
            for (int i = 0; i < 16; i++) {
                out[b * 32 + i]      = d * (float)((int)(qs[i] & 0x0F) - 8);
                out[b * 32 + i + 16] = d * (float)((int)(qs[i] >> 4)   - 8);
            }
        }
        break;
    case GGML_Q4_1:
        for (int b = 0; b < n / 32; b++) {
            uint16_t dh, mh;
            memcpy(&dh, r + b * 20, 2);
            memcpy(&mh, r + b * 20 + 2, 2);
            float d = F16(dh), mm = F16(mh);
            const uint8_t *qs = r + b * 20 + 4;
            for (int i = 0; i < 16; i++) {
                out[b * 32 + i]      = d * (float)(qs[i] & 0x0F) + mm;
                out[b * 32 + i + 16] = d * (float)(qs[i] >> 4)   + mm;
            }
        }
        break;
    case GGML_Q8_0:
        for (int b = 0; b < n / 32; b++) {
            uint16_t dh;
            memcpy(&dh, r + b * 34, 2);
            float d = F16(dh);
            const int8_t *qs = (const int8_t *)(r + b * 34 + 2);
            for (int i = 0; i < 32; i++) out[b * 32 + i] = d * (float)qs[i];
        }
        break;
    default: bug("dequant_row 遇到未知张量类型(加载期校验遗漏)");
    }
}

/* -- RMSNorm: out = x / rms(x) * w  (逐元素) -- */
static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* -- 数值稳定的 softmax(原地, 前 n 个元素) -- */
static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ===========================================================================
 * [5] 前向传播 —— 一次处理一个 token, 带 KV 缓存
 * ===========================================================================
 *
 * 自回归推理下, 每个新 token 只需与"历史所有位置的 K/V"做注意力,
 * 历史 K/V 保存在 KV 缓存里, 因此每步只做 O(层数×序列长)的注意力
 * 和 O(参数量) 的矩阵乘, 不必重算整个序列。
 */

/* 推理运行时状态: 激活缓冲 + KV 缓存 */
typedef struct {
    /* 激活缓冲(单 token) */
    float *x;      /* 残差流                [dim]                     */
    float *xb;     /* 通用缓冲              [max(dim, n_heads*head_dim)] */
    float *xb2;    /* 通用缓冲              [dim]                     */
    float *q;      /* 查询向量              [n_heads * head_dim]      */
    float *k, *v;  /* 当前位置的键/值       [n_kv_heads * head_dim]   */
    float *att;    /* 注意力分数, 每头一段  [n_heads * ctx_len]       */
    float *hb, *hb2; /* FFN 中间激活        [ffn_dim]                 */
    float *logits; /* 输出分布              [n_vocab]                 */
    float *cos_t, *sin_t; /* 当前位置的 RoPE cos/sin 表 [head_dim/2]  */
    /* KV 缓存: [n_layers][ctx_len][n_kv_heads*head_dim]              */
    float *k_cache, *v_cache;
    int    pos;    /* 已写入缓存的 token 数(= 下一个 token 的位置)    */
} RunState;

static void state_free(RunState *s); /* 前向声明: 失败路径的清理用 */

/* 分配全部激活缓冲与 KV 缓存, 结果填入 *s。返回 0 成功, -1 失败。 */
static int state_alloc(RunState *s, const Config *c) {
    memset(s, 0, sizeof(*s));
    int qdim  = c->n_heads * c->head_dim;
    int kvdim = c->n_kv_heads * c->head_dim;
    int xbdim = qdim > c->dim ? qdim : c->dim;
    size_t cache = (size_t)c->n_layers * c->ctx_len * kvdim;

    s->x      = malloc(sizeof(float) * c->dim);
    s->xb     = malloc(sizeof(float) * xbdim);
    s->xb2    = malloc(sizeof(float) * c->dim);
    s->q      = malloc(sizeof(float) * qdim);
    s->k      = malloc(sizeof(float) * kvdim);
    s->v      = malloc(sizeof(float) * kvdim);
    s->att    = malloc(sizeof(float) * c->n_heads * c->ctx_len);
    s->hb     = malloc(sizeof(float) * c->ffn_dim);
    s->hb2    = malloc(sizeof(float) * c->ffn_dim);
    s->logits = malloc(sizeof(float) * c->n_vocab);
    s->cos_t  = malloc(sizeof(float) * (c->head_dim / 2));
    s->sin_t  = malloc(sizeof(float) * (c->head_dim / 2));
    s->k_cache = malloc(sizeof(float) * cache);
    s->v_cache = malloc(sizeof(float) * cache);
    if (!s->x || !s->xb || !s->xb2 || !s->q || !s->k || !s->v || !s->att ||
        !s->hb || !s->hb2 || !s->logits || !s->cos_t || !s->sin_t ||
        !s->k_cache || !s->v_cache) {
        state_free(s);
        return fail("内存不足(KV 缓存过大? 可减小上下文长度)");
    }
    return 0;
}

/* 释放运行状态的全部缓冲 */
static void state_free(RunState *s) {
    free(s->x);   free(s->xb);  free(s->xb2);
    free(s->q);   free(s->k);   free(s->v);
    free(s->att); free(s->hb);  free(s->hb2);
    free(s->logits);
    free(s->cos_t); free(s->sin_t);
    free(s->k_cache); free(s->v_cache);
    memset(s, 0, sizeof(*s));
}

/* 多头因果注意力的线程池任务: 处理查询头 [h0, h1)。
 * 每个查询头对位置 0..pos 的历史做 "打分 -> softmax -> 加权求和",
 * 各头的分数区(att)与输出区(xb 的第 h 段)互不重叠, 可安全并行。
 * GQA: 查询头 h 使用第 h/gqa 个 KV 头的缓存。 */
typedef struct {
    const Config *c;
    RunState     *s;
    int           l;         /* 当前层号   */
    int           pos;       /* 当前位置   */
} AttnCtx;

static void attn_task(int h0, int h1, const void *pc) {
    const AttnCtx *a = (const AttnCtx *)pc;
    const Config *c = a->c;
    RunState *s = a->s;
    int hd    = c->head_dim;
    int kvdim = c->n_kv_heads * hd;
    int gqa   = c->n_heads / c->n_kv_heads;
    float att_scale = 1.0f / sqrtf((float)hd);

    for (int h = h0; h < h1; h++) {
        const float *q = s->q + h * hd;
        float *att     = s->att + (size_t)h * c->ctx_len;
        int kv_off     = (h / gqa) * hd;
        /* 1) 注意力分数 = q·k / sqrt(head_dim) */
        for (int t = 0; t <= a->pos; t++) {
            const float *kt = s->k_cache + ((size_t)a->l * c->ctx_len + t) * kvdim + kv_off;
            float score = 0.0f;
            for (int i = 0; i < hd; i++) score += q[i] * kt[i];
            att[t] = score * att_scale;
        }
        /* 2) softmax 归一化成权重 */
        softmax(att, a->pos + 1);
        /* 3) 用权重对 V 加权求和, 结果写入 xb 的第 h 段(拼接各头) */
        float *out = s->xb + h * hd;
        memset(out, 0, sizeof(float) * hd);
        for (int t = 0; t <= a->pos; t++) {
            const float *vt = s->v_cache + ((size_t)a->l * c->ctx_len + t) * kvdim + kv_off;
            float w = att[t];
            for (int i = 0; i < hd; i++) out[i] += w * vt[i];
        }
    }
}

/* RoPE 旋转位置编码(NeoX 风格: 第 i 维与第 i+head_dim/2 维配对旋转)。
 * cos/sin 依赖 (位置, 频率), 每个 token 算一次表, 各头共用。 */
static void rope_apply(float *vec, int head_dim, const float *cos_t, const float *sin_t) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float x0 = vec[i], x1 = vec[i + half];
        vec[i]        = x0 * cos_t[i] - x1 * sin_t[i];
        vec[i + half] = x0 * sin_t[i] + x1 * cos_t[i];
    }
}

/* 前向传播主体: 跑完全部 Transformer 层, 结果留在残差流 s->x 中
 * (不含最终归一化与输出投影)。同时把本位置的 K/V 写入缓存。
 * 拆出这一步是为了复用: 聊天模式在其后接 lm_head 得到 logits,
 * 嵌入模式只需最终归一化后的隐状态, 可完全跳过昂贵的词表投影。 */
static void forward_hidden(const Model *m, RunState *s, int token, int pos) {
    const Config *c = &m->cfg;
    int hd    = c->head_dim;
    int half  = hd / 2;
    int kvdim = c->n_kv_heads * hd;

    /* 词嵌入查表: x = embedding[token] */
    dequant_row(m->token_embd, token, s->x);

    /* 本位置的 RoPE cos/sin 表(所有层、所有头共用) */
    for (int i = 0; i < half; i++) {
        float ang = (float)pos * m->rope_freq[i];
        s->cos_t[i] = cosf(ang);
        s->sin_t[i] = sinf(ang);
    }

    for (int l = 0; l < c->n_layers; l++) {
        const Layer *L = &m->layers[l];

        /* ============ 注意力块 ============ */
        rmsnorm(s->xb, s->x, (const float *)L->attn_norm->data, c->dim, c->rms_eps);

        /* QKV 投影 */
        matvec(s->q, L->attn_q, s->xb);
        matvec(s->k, L->attn_k, s->xb);
        matvec(s->v, L->attn_v, s->xb);

        /* Qwen3 特色: 每个头先做 RMSNorm(QK-Norm), 再做 RoPE 旋转 */
        for (int h = 0; h < c->n_heads; h++) {
            float *q = s->q + h * hd;
            rmsnorm(q, q, (const float *)L->q_norm->data, hd, c->rms_eps);
            rope_apply(q, hd, s->cos_t, s->sin_t);
        }
        for (int h = 0; h < c->n_kv_heads; h++) {
            float *k = s->k + h * hd;
            rmsnorm(k, k, (const float *)L->k_norm->data, hd, c->rms_eps);
            rope_apply(k, hd, s->cos_t, s->sin_t);
        }

        /* 把本位置的 K/V 写入缓存 */
        float *kc = s->k_cache + ((size_t)l * c->ctx_len + pos) * kvdim;
        float *vc = s->v_cache + ((size_t)l * c->ctx_len + pos) * kvdim;
        memcpy(kc, s->k, sizeof(float) * kvdim);
        memcpy(vc, s->v, sizeof(float) * kvdim);

        /* 多头因果注意力(实现见 attn_task): 各头独立, 用线程池按头并行 */
        AttnCtx actx = { c, s, l, pos };
        parallel_for(attn_task, &actx, c->n_heads);

        /* 输出投影并加回残差流 */
        matvec(s->xb2, L->attn_out, s->xb);
        for (int i = 0; i < c->dim; i++) s->x[i] += s->xb2[i];

        /* ============ SwiGLU 前馈块 ============ */
        rmsnorm(s->xb, s->x, (const float *)L->ffn_norm->data, c->dim, c->rms_eps);
        matvec(s->hb,  L->ffn_gate, s->xb);   /* 门控分支 */
        matvec(s->hb2, L->ffn_up,   s->xb);   /* 数值分支 */
        /* SwiGLU: silu(gate) ⊙ up, 其中 silu(x) = x·sigmoid(x) */
        for (int i = 0; i < c->ffn_dim; i++) {
            float g = s->hb[i];
            s->hb[i] = g / (1.0f + expf(-g)) * s->hb2[i];
        }
        matvec(s->xb2, L->ffn_down, s->hb);
        for (int i = 0; i < c->dim; i++) s->x[i] += s->xb2[i];
    }
}

/* 完整前向: 输入 token id 与其位置 pos, 返回下一 token 的 logits。 */
static float *forward(const Model *m, RunState *s, int token, int pos) {
    const Config *c = &m->cfg;
    forward_hidden(m, s, token, pos);
    /* 最终归一化 + 输出投影(无独立输出层时与词嵌入共享权重) */
    rmsnorm(s->xb, s->x, (const float *)m->output_norm->data, c->dim, c->rms_eps);
    matvec(s->logits, m->output ? m->output : m->token_embd, s->xb);
    return s->logits;
}

/* ===========================================================================
 * [6] 采样器 —— temperature + top-p(核采样)
 * ===========================================================================
 *
 * temperature 缩放 logits 控制随机性(越低越保守, 0 = 贪心取最大);
 * top-p 只保留累计概率达到 p 的最高概率 token 集合, 去掉长尾杂讯。
 */

static uint64_t g_rng_state;

/* xorshift64* 伪随机数, 返回 [0,1) 均匀浮点 */
static float rand_f32(void) {
    g_rng_state ^= g_rng_state >> 12;
    g_rng_state ^= g_rng_state << 25;
    g_rng_state ^= g_rng_state >> 27;
    return (float)((g_rng_state * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

typedef struct { float prob; int idx; } ProbIdx;

static int probidx_cmp_desc(const void *a, const void *b) {
    float pa = ((const ProbIdx *)a)->prob, pb = ((const ProbIdx *)b)->prob;
    return (pa < pb) - (pa > pb);
}

/* 从 logits 采样出一个 token id。buf 是调用方提供的 n 项工作区。 */
static int sample(const float *logits, int n, float temp, float top_p, ProbIdx *buf) {
    /* 贪心: 直接取 logit 最大的 token */
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* softmax(logits / temp) —— 先减最大值保证数值稳定 */
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        buf[i].prob = expf((logits[i] - mx) / temp);
        buf[i].idx  = i;
        sum += buf[i].prob;
    }
    /* 预过滤(llama2.c 的启发式): 概率低于 (1-p)/(n-1) 的 token 几乎
     * 不可能进入 top-p 集合, 先滤掉它们, 把排序规模从 15 万降到几百。 */
    float cutoff = (1.0f - top_p) / (n - 1);
    int kept = 0;
    for (int i = 0; i < n; i++) {
        float p = buf[i].prob / sum;
        if (p >= cutoff) { buf[kept].prob = p; buf[kept].idx = i; kept++; }
    }

    /* top-p: 按概率降序累加, 到达阈值即截断 */
    qsort(buf, kept, sizeof(ProbIdx), probidx_cmp_desc);
    float cum = 0.0f;
    int cut = kept;
    for (int i = 0; i < kept; i++) {
        cum += buf[i].prob;
        if (cum >= top_p) { cut = i + 1; break; }
    }

    /* 在截断后的集合内按归一化概率掷骰子 */
    float r = rand_f32() * cum, acc = 0.0f;
    for (int i = 0; i < cut; i++) {
        acc += buf[i].prob;
        if (r < acc) return buf[i].idx;
    }
    return buf[cut - 1].idx; /* 浮点误差兜底 */
}

/* ===========================================================================
 * [7] 命令行参数与 main
 * ===========================================================================
 */

#ifndef TINYQWEN_LIB /* ---- 命令行专属: 选项结构与内核自检 ---- */

/* 命令行工具的错误策略: 打印后直接退出进程。
 * 只在命令行代码里使用 —— 库代码一律用 fail() 返回错误码。 */
static void die(const char *msg) {
    fprintf(stderr, "错误: %s\n", msg);
    exit(1);
}

/* 运行时可调参数（命令行覆盖默认值） */
typedef struct {
    const char *model_path;   /* GGUF 模型文件路径                         */
    float       temperature;  /* 采样温度，0 表示贪心                      */
    float       top_p;        /* 核采样阈值                                */
    uint64_t    seed;         /* 随机种子                                  */
    int         ctx_len;      /* KV 缓存长度上限（决定内存占用）           */
    int         max_gen;      /* 单轮回复最多生成多少 token                */
    int         n_threads;    /* 线程数, 0 = 自动(取 min(核数, 16))        */
    int         think;        /* 是否启用 Qwen3 思考模式（默认关闭）       */
    int         no_simd;      /* 强制标量内核(不用 AVX2)                   */
    int         selftest;     /* 对比标量/SIMD 内核输出误差后退出          */
    int         inspect;      /* 只打印模型信息后退出                      */
    const char *test_text;    /* 若非 NULL：只测试分词器（编码+解码）      */
    const char *prompt;       /* 若非 NULL：单次问答后退出(非交互模式)     */
    const char **embed_texts; /* --embed 的文本列表(嵌入模式)              */
    int          n_embed;
    const char **rerank_args; /* --rerank 的参数: [0]=查询, 其余=文档      */
    int          n_rerank;
    const char  *instruct;    /* 检索任务描述(嵌入查询模板/重排模板用)     */
} Options;

/* --selftest: 用固定伪随机输入对比标量内核与 AVX2 内核的输出误差。
 * FMA 舍入次序不同会带来 1ulp 级差异, 相对误差应远小于 1e-4。 */
static void kernel_selftest(const Gguf *g) {
    if (!g_use_simd) {
        printf("selftest: 本机未启用 SIMD(或 --no-simd), 无需对比\n");
        return;
    }
    /* 动态扫描: 文件中每种有 SIMD 实现的量化类型, 各挑第一个张量作代表 */
    uint32_t types[] = { GGML_Q4_0, GGML_Q4_1, GGML_Q6_K, GGML_Q8_0 };
    int all_ok = 1;
    for (int t = 0; t < 4; t++) {
        const GTensor *w = NULL;
        for (uint64_t i = 0; i < g->n_tensors; i++)
            if (g->tensors[i].type == types[t]) { w = &g->tensors[i]; break; }
        if (!w) continue;              /* 该类型在此文件中不存在 */
        DotFn scalar_fn, simd_fn;
        switch (w->type) {
        case GGML_Q4_0: scalar_fn = dot_q4_0; simd_fn = g_dot_q4_0; break;
        case GGML_Q4_1: scalar_fn = dot_q4_1; simd_fn = g_dot_q4_1; break;
        case GGML_Q6_K: scalar_fn = dot_q6_k; simd_fn = g_dot_q6_k; break;
        case GGML_Q8_0: scalar_fn = dot_q8_0; simd_fn = g_dot_q8_0; break;
        default: continue;
        }
        int n_in = (int)w->ne[0];
        float *x    = malloc(sizeof(float) * n_in);
        float *wrow = malloc(sizeof(float) * n_in);
        if (!x || !wrow) die("内存不足");
        uint64_t rng = 0x1234567ull;                 /* 固定种子, 结果可复现 */
        for (int i = 0; i < n_in; i++) {
            rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
            x[i] = (float)((rng * 0x2545F4914F6CDD1Dull >> 40) / 8388608.0) - 1.0f;
        }
        uint64_t row_bytes = ggml_row_bytes(w->type, w->ne[0]);
        int n_rows = w->ne[1] < 64 ? (int)w->ne[1] : 64;
        /* 参考值: 先精确反量化, 再用 double 累加 —— 两个 float 实现分别与
         * 它比较。误差分母用"逐项幅值和"而非点积本身, 避免点积因正负相消
         * 接近零时相对误差虚高。 */
        double max_scalar = 0.0, max_simd = 0.0;
        for (int r = 0; r < n_rows; r++) {
            const uint8_t *row = (const uint8_t *)w->data + (uint64_t)r * row_bytes;
            dequant_row(w, r, wrow);
            double ref = 0.0, denom = 0.0;
            for (int i = 0; i < n_in; i++) {
                double term = (double)wrow[i] * x[i];
                ref   += term;
                denom += fabs(term);
            }
            if (denom < 1e-12) denom = 1e-12;
            double ea = fabs(scalar_fn(row, x, n_in) - ref) / denom;
            double eb = fabs(simd_fn(row, x, n_in) - ref) / denom;
            if (ea > max_scalar) max_scalar = ea;
            if (eb > max_simd)   max_simd = eb;
        }
        /* float 顺序累加的理论误差量级 ~ sqrt(n)*eps ≈ 1e-6, 阈值放宽到 1e-5 */
        int ok = max_scalar < 1e-5 && max_simd < 1e-5;
        all_ok &= ok;
        printf("selftest %-22s %-5s %d 行, 对 double 参考误差: 标量 %.1e, %s %.1e %s\n",
               w->name, ggml_type_name(w->type), n_rows, max_scalar,
               g_simd_name, max_simd, ok ? "通过" : "失败!");
        free(x);
        free(wrow);
    }
    printf("selftest %s\n", all_ok ? "全部通过" : "存在失败项!");
    if (!all_ok) exit(1);
}

#endif /* TINYQWEN_LIB: 命令行专属段结束 */

/* 单调时钟毫秒数(计时统计用)。
 * clock_gettime 是 POSIX 接口, MSVC 没有, Windows 分支用性能计数器。 */
static double now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;              /* 计数器频率, 只查询一次 */
    LARGE_INTEGER t;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
#endif
}

/* 把字符串 BPE 编码后追加到 token 数组尾部, 返回新的元素个数。
 * 负值表示编码失败, 并且负值输入会原样透传 —— 调用方可以连续追加
 * 多段之后只在最后检查一次(append_id 同理)。 */
static int append_encoded(const Tokenizer *t, const char *s, int *toks, int n, int max) {
    if (n < 0) return n;
    int r = tok_encode(t, s, toks + n, max - n);
    return r < 0 ? r : n + r;
}

/* 追加单个 token id(特殊 token 直接按 id 插入, 不经过 BPE) */
static int append_id(int *toks, int n, int max, int id) {
    if (n < 0) return n;
    if (n >= max) return fail("输入过长(token 数超出缓冲)");
    toks[n] = id;
    return n + 1;
}

/* 生成配置与单轮统计(命令行与库 API 共用的内部结构) */
typedef struct {
    float temperature;          /* 0 = 贪心解码                  */
    float top_p;
    int   max_gen;
    int   think;
} GenCfg;

typedef struct {
    int    n_prompt, n_gen;     /* 本轮 prompt / 生成 token 数   */
    double prefill_ms, gen_ms;  /* 预填 / 生成耗时               */
    int    history_cleared;     /* 上下文满, 历史被自动清空      */
} TurnStats;

/* 解码一个 token 并交给回调(流式输出)。控制 token(如 <|im_end|>)跳过;
 * 回调返回非 0 时置中断标记。回调可为 NULL(丢弃输出)。 */
static void emit_token(const Tokenizer *t, int id, TqTokenFn sink, void *ud,
                       int *abort_flag) {
    if (id < 0 || id >= t->n_vocab || t->token_type[id] == TOKEN_TYPE_CONTROL)
        return;
    char buf[256];
    int n = tok_decode(t, id, buf);
    if (sink && n > 0 && sink(buf, n, ud)) *abort_flag = 1;
}

/* ===========================================================================
 * [7a] 嵌入与重排模式 —— Qwen3-Embedding / Qwen3-Reranker
 * ===========================================================================
 *
 * 这两类模型与聊天模型共用同一套 qwen3 网络结构, 只是"用法"不同:
 *
 * 嵌入(Embedding): 文本向量 = 末尾追加的 <|endoftext|> 位置经最终
 *   RMSNorm 后的隐状态(不过 lm_head), 再做 L2 归一化。检索场景下
 *   查询侧建议用指令模板包装(官方模板见 embed_run), 文档侧用原文。
 *
 * 重排(Reranker): 用一个固定的 ChatML 模板问模型 "该文档是否满足查询",
 *   相关性得分 = 末位 logits 上 "yes" 对 "no" 的二元 softmax 概率。
 */

#ifndef TINYQWEN_LIB
/* 求不超过 max_bytes 且不切断 UTF-8 多字节字符的前缀长度(预览显示用) */
static int utf8_prefix_len(const char *s, int max_bytes) {
    int len = (int)strlen(s);
    if (len <= max_bytes) return len;
    int n = max_bytes;
    /* 若停在续字节(10xxxxxx)中间, 回退到字符起始处 */
    while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80) n--;
    return n;
}
#endif

/* 官方模板中的固定文案(与 Qwen3-Reranker 模型卡一致, 必须逐字相同) */
static const char *RERANK_SYSTEM =
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".";
static const char *DEFAULT_TASK =
    "Given a web search query, retrieve relevant passages that answer the query";

/* 计算单个文本的嵌入向量, 写入 out[dim](已 L2 归一化)。
 * 返回 0 成功, -1 失败(文本过长等)。 */
static int embed_one(const Model *m, RunState *st, const Tokenizer *tok,
                     const char *text, float *out) {
    enum { MAX_TOK = 8192 };
    static int ids[MAX_TOK];
    int n = tok_encode(tok, text, ids, MAX_TOK - 1);
    if (n < 0) return -1;
    ids[n++] = tok->eos_id;  /* 嵌入取自末尾 EOS 位置(embedding 模型的
                                eos_token 是 <|endoftext|>, 来自元数据) */
    if (n > m->cfg.ctx_len)
        return fail("文本过长, 超出上下文长度(加载时增大 ctx_len / -c)");

    st->pos = 0;             /* 每个文本独立编码, 清空 KV 缓存 */
    for (int i = 0; i < n; i++) forward_hidden(m, st, ids[i], st->pos++);

    /* 最终 RMSNorm 后的隐状态即嵌入(跳过 lm_head, 词表投影完全不算) */
    rmsnorm(st->xb, st->x, (const float *)m->output_norm->data,
            m->cfg.dim, m->cfg.rms_eps);

    float ss = 0.0f;
    for (int i = 0; i < m->cfg.dim; i++) ss += st->xb[i] * st->xb[i];
    float inv = 1.0f / sqrtf(ss > 0 ? ss : 1.0f);
    for (int i = 0; i < m->cfg.dim; i++) out[i] = st->xb[i] * inv;
    return 0;
}

#ifndef TINYQWEN_LIB
/* --embed 入口: 单文本输出完整向量; 多文本输出两两余弦相似度矩阵。
 * 提供 --instruct 时, 第一个文本按官方查询模板包装(其余视为文档)。 */
static void embed_run(const Model *m, RunState *st, const Tokenizer *tok,
                      const char **texts, int n_texts, const char *instruct) {
    int dim = m->cfg.dim;
    float *embs = malloc(sizeof(float) * dim * n_texts);
    if (!embs) die("内存不足");

    for (int i = 0; i < n_texts; i++) {
        const char *t = texts[i];
        char *qbuf = NULL;
        if (i == 0 && instruct) {
            /* 官方查询模板: "Instruct: {task}\nQuery:{query}" */
            size_t need = strlen(instruct) + strlen(t) + 32;
            qbuf = malloc(need);
            if (!qbuf) die("内存不足");
            snprintf(qbuf, need, "Instruct: %s\nQuery:%s", instruct, t);
            t = qbuf;
        }
        double t0 = now_ms();
        if (embed_one(m, st, tok, t, embs + (size_t)i * dim) < 0) die(g_err_msg);
        int pl = utf8_prefix_len(texts[i], 40);
        printf("[%d] 已编码 (%.2fs): %.*s%s\n", i, (now_ms() - t0) / 1000.0,
               pl, texts[i], texts[i][pl] ? "..." : "");
        free(qbuf);
    }

    if (n_texts == 1) {
        printf("嵌入维度 %d (L2 已归一化):\n", dim);
        for (int i = 0; i < dim; i++)
            printf("%.6f%c", embs[i], i + 1 == dim ? '\n' : ' ');
    } else {
        /* 向量已归一化, 点积即余弦相似度 */
        printf("\n余弦相似度矩阵:\n      ");
        for (int j = 0; j < n_texts; j++) printf("  [%d]   ", j);
        printf("\n");
        for (int i = 0; i < n_texts; i++) {
            printf("  [%d] ", i);
            for (int j = 0; j < n_texts; j++) {
                float dot = 0.0f;
                for (int k = 0; k < dim; k++)
                    dot += embs[(size_t)i * dim + k] * embs[(size_t)j * dim + k];
                printf("%7.4f", dot);
            }
            printf("\n");
        }
    }
    free(embs);
}
#endif /* TINYQWEN_LIB */

/* 计算一对 (查询, 文档) 的重排得分: P(yes | 模板) ∈ (0,1)。
 * 失败返回负值(错误详情在 g_err_msg)。 */
static float rerank_score(const Model *m, RunState *st, const Tokenizer *tok,
                          const char *task, const char *query, const char *doc,
                          int yes_id, int no_id) {
    enum { MAX_TOK = 8192 };
    static int ids[MAX_TOK];
    int n = 0;

    /* 模板正文拼成一个字符串再整体编码(避免分段编码在空格边界产生
     * 与官方不同的切分), 特殊 token 用 id 直接插入 */
    size_t need = strlen(task) + strlen(query) + strlen(doc) + 64;
    char *content = malloc(need);
    if (!content) return (float)fail("内存不足");
    snprintf(content, need, "<Instruct>: %s\n<Query>: %s\n<Document>: %s",
             task, query, doc);

    n = append_id(ids, n, MAX_TOK, tok->im_start_id);        /* 系统消息   */
    n = append_encoded(tok, "system\n", ids, n, MAX_TOK);
    n = append_encoded(tok, RERANK_SYSTEM, ids, n, MAX_TOK);
    n = append_id(ids, n, MAX_TOK, tok->im_end_id);
    n = append_id(ids, n, MAX_TOK, tok->nl_id);
    n = append_id(ids, n, MAX_TOK, tok->im_start_id);        /* 用户消息   */
    n = append_encoded(tok, "user\n", ids, n, MAX_TOK);
    n = append_encoded(tok, content, ids, n, MAX_TOK - 32);
    n = append_id(ids, n, MAX_TOK, tok->im_end_id);
    n = append_id(ids, n, MAX_TOK, tok->nl_id);
    n = append_id(ids, n, MAX_TOK, tok->im_start_id);        /* 助手开头   */
    n = append_encoded(tok, "assistant\n", ids, n, MAX_TOK);
    n = append_id(ids, n, MAX_TOK, tok->think_id);           /* 空思考块   */
    n = append_encoded(tok, "\n\n", ids, n, MAX_TOK);
    n = append_id(ids, n, MAX_TOK, tok->think_end_id);
    n = append_encoded(tok, "\n\n", ids, n, MAX_TOK);
    free(content);
    if (n < 0) return -1.0f; /* 编码失败, 错误信息已设置 */
    if (n > m->cfg.ctx_len)
        return (float)fail("查询+文档过长, 超出上下文长度(加载时增大 ctx_len / -c)");

    /* 前 n-1 个 token 只需隐状态, 最后一个才需要 logits(省词表投影) */
    st->pos = 0;
    for (int i = 0; i < n - 1; i++) forward_hidden(m, st, ids[i], st->pos++);
    const float *logits = forward(m, st, ids[n - 1], st->pos++);

    /* 二元 softmax: P(yes) = 1 / (1 + e^(logit_no - logit_yes)) */
    return 1.0f / (1.0f + expf(logits[no_id] - logits[yes_id]));
}

#ifndef TINYQWEN_LIB
/* --rerank 入口: 逐文档打分, 按得分从高到低输出排名 */
static void rerank_run(const Model *m, RunState *st, const Tokenizer *tok,
                       const char **args, int n_args, const char *instruct) {
    const char *task  = instruct ? instruct : DEFAULT_TASK;
    const char *query = args[0];
    int n_docs = n_args - 1;

    int yes_id = strmap_get(&tok->vocab_map, "yes");
    int no_id  = strmap_get(&tok->vocab_map, "no");
    if (yes_id < 0 || no_id < 0) die("词表中找不到 yes/no token");

    printf("查询: %s\n", query);
    float *scores = malloc(sizeof(float) * n_docs);
    int   *order  = malloc(sizeof(int) * n_docs);
    if (!scores || !order) die("内存不足");
    for (int i = 0; i < n_docs; i++) {
        double t0 = now_ms();
        scores[i] = rerank_score(m, st, tok, task, query, args[1 + i], yes_id, no_id);
        if (scores[i] < 0) die(g_err_msg);
        order[i] = i;
        int pl = utf8_prefix_len(args[1 + i], 50);
        printf("  文档[%d] 得分 %.4f (%.2fs): %.*s%s\n", i, scores[i],
               (now_ms() - t0) / 1000.0, pl, args[1 + i],
               args[1 + i][pl] ? "..." : "");
    }
    /* 按得分降序输出名次(文档数很少, 冒泡即可) */
    for (int i = 0; i < n_docs; i++)
        for (int j = i + 1; j < n_docs; j++)
            if (scores[order[j]] > scores[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
    printf("排序结果(相关性从高到低):");
    for (int i = 0; i < n_docs; i++) printf(" 文档[%d]", order[i]);
    printf("\n");
    free(scores);
    free(order);
}
#endif /* TINYQWEN_LIB */

/* 执行一轮对话: 组装 ChatML prompt -> 预填 -> 逐 token 生成, 回复经
 * sink 回调流式送出(命令行模式的回调即"打印到终端")。
 *
 * ChatML 模板(与 Qwen3 官方 chat_template 一致):
 *     <|im_start|>user\n{用户输入}<|im_end|>\n<|im_start|>assistant\n
 * 关闭思考模式时, 再替模型补上空思考块:
 *     <think>\n\n</think>\n\n
 * 多轮对话依靠 KV 缓存天然延续: 每轮只把新增 token 喂进模型。
 * 注意上一轮生成在采样到 <|im_end|> 时即停止, 该 token 并未写入缓存,
 * 所以非首轮的 prompt 以 [<|im_end|>, "\n"] 开头补全模板。
 *
 * 返回 0 成功, -1 输入过长(错误描述在 g_err_msg); 统计写入 *stats。 */
static int chat_turn(const Model *m, RunState *st, const Tokenizer *tok,
                     const GenCfg *gc, ProbIdx *sbuf, const char *user_text,
                     TqTokenFn sink, void *ud, TurnStats *stats) {
    const Config *c = &m->cfg;
    enum { MAX_PROMPT = 8192 };
    static int ptoks[MAX_PROMPT];
    int np = 0;

    /* ---- 1. 组装本轮的 prompt token 序列 ----
     * append_* 系列对负值(错误)透传, 最后统一检查一次即可 */
    if (st->pos > 0) {                       /* 非首轮: 补上一轮的收尾      */
        np = append_id(ptoks, np, MAX_PROMPT, tok->im_end_id);
        np = append_id(ptoks, np, MAX_PROMPT, tok->nl_id);
    }
    np = append_id(ptoks, np, MAX_PROMPT, tok->im_start_id);
    np = append_encoded(tok, "user\n", ptoks, np, MAX_PROMPT);
    np = append_encoded(tok, user_text, ptoks, np, MAX_PROMPT - 32);
    np = append_id(ptoks, np, MAX_PROMPT, tok->im_end_id);
    np = append_id(ptoks, np, MAX_PROMPT, tok->nl_id);
    np = append_id(ptoks, np, MAX_PROMPT, tok->im_start_id);
    np = append_encoded(tok, "assistant\n", ptoks, np, MAX_PROMPT);
    if (!gc->think && tok->think_id >= 0 && tok->think_end_id >= 0) {
        /* 空思考块 = 关闭思考模式(Qwen3 约定) */
        np = append_id(ptoks, np, MAX_PROMPT, tok->think_id);
        np = append_encoded(tok, "\n\n", ptoks, np, MAX_PROMPT);
        np = append_id(ptoks, np, MAX_PROMPT, tok->think_end_id);
        np = append_encoded(tok, "\n\n", ptoks, np, MAX_PROMPT);
    }
    if (np < 0) return -1; /* 编码失败(输入过长等), 错误信息已设置 */

    /* ---- 2. 上下文容量检查 ---- */
    if (np + 8 > c->ctx_len)
        return fail("输入过长, 超出上下文长度上限(加载时增大 ctx_len / -c)");
    if (st->pos + np + 8 > c->ctx_len) {
        /* 历史 + 新输入放不下: 清空对话历史重新开始(仅在有历史时发生) */
        st->pos = 0;
        int rc = chat_turn(m, st, tok, gc, sbuf, user_text, sink, ud, stats);
        stats->history_cleared = 1;
        return rc;
    }

    /* ---- 3. 预填: 把 prompt token 逐个喂给模型 ---- */
    double t0 = now_ms();
    float *logits = NULL;
    for (int i = 0; i < np; i++) logits = forward(m, st, ptoks[i], st->pos++);
    double t1 = now_ms();

    /* ---- 4. 自回归生成 ---- */
    int n_gen = 0, aborted = 0;
    while (n_gen < gc->max_gen && st->pos < c->ctx_len && !aborted) {
        int id = sample(logits, c->n_vocab, gc->temperature, gc->top_p, sbuf);
        /* 任何控制 token(<|im_end|>/<|endoftext|> 等)都视为回复结束 */
        if (tok->token_type[id] == TOKEN_TYPE_CONTROL) break;
        emit_token(tok, id, sink, ud, &aborted);
        /* 中断时也先喂进缓存, 保证多轮模板状态一致 */
        logits = forward(m, st, id, st->pos++);
        n_gen++;
    }
    double t2 = now_ms();

    stats->n_prompt   = np;
    stats->n_gen      = n_gen;
    stats->prefill_ms = t1 - t0;
    stats->gen_ms     = t2 - t1;
    return 0;
}

/* ===========================================================================
 * [8] 库 API 实现 —— 公开接口声明与文档见 tinyqwen.h
 * ===========================================================================
 *
 * 这些函数没有 static, 是编译成 libtinyqwen.so / libtinyqwen.a 时对外
 * 导出的全部符号。命令行工具与库共用同一份内部实现, 差别只是:
 * 定义 TINYQWEN_LIB 宏时不编译 main() 等命令行部分。
 */

/* 不透明句柄的真身: 打包一个模型的全部状态 */
struct TqModel {
    Gguf      gguf;   /* mmap 的模型文件与元数据  */
    Tokenizer tok;    /* 分词器                   */
    Model     model;  /* 权重指针与超参数         */
    RunState  st;     /* 激活缓冲与 KV 缓存       */
    ProbIdx  *sbuf;   /* 采样工作区               */
};

/* 进程级运行时(f16 查找表 / 线程池 / SIMD 检测)只初始化一次;
 * 线程池规模以第一次调用为准(见 tinyqwen.h 的线程安全说明)。
 * 返回 0 成功, -1 失败(线程池创建失败)。 */
static int g_runtime_ready = 0;

static int runtime_init(int n_threads, int no_simd) {
    if (g_runtime_ready) return 0;
    if (n_threads <= 0) {
        n_threads = cpu_count();
        if (n_threads > 32) n_threads = 32; /* 收益递减, 更高并行度需显式指定 */
    }
    if (pool_init(n_threads) < 0) return -1;
    atexit(pool_shutdown);   /* 进程退出时回收工作线程 */
    simd_init(!no_simd);
    f16_table_init();
    g_rng_state = (uint64_t)time(NULL) | 1; /* 默认种子(xorshift 状态不可为 0) */
    g_runtime_ready = 1;
    return 0;
}

const char *tq_last_error(void) { return g_err_msg; }

TqModel *tq_load(const char *gguf_path, const TqLoadParams *params) {
    TqLoadParams p = {0};
    if (params) p = *params;

    if (!gguf_path) {
        fail("模型路径为空");
        return NULL;
    }
    if (runtime_init(p.n_threads, p.no_simd) < 0) return NULL;

    /* 逐步构建, 任何一步失败都释放已成形的部分(calloc 保证可安全释放) */
    TqModel *h = calloc(1, sizeof(TqModel));
    if (!h) { fail("内存不足"); return NULL; }
    if (gguf_open(&h->gguf, gguf_path) < 0)                             goto bad;
    if (tok_init(&h->tok, &h->gguf) < 0)                                goto bad;
    if (model_load(&h->model, &h->gguf, p.ctx_len > 0 ? p.ctx_len : 4096) < 0) goto bad;
    if (state_alloc(&h->st, &h->model.cfg) < 0)                         goto bad;
    h->sbuf = malloc(sizeof(ProbIdx) * h->model.cfg.n_vocab);
    if (!h->sbuf) { fail("内存不足"); goto bad; }
    return h;

bad:
    tq_free(h);
    return NULL;
}

void tq_free(TqModel *m) {
    if (!m) return;
    state_free(&m->st);
    model_free(&m->model);
    tok_free(&m->tok);
    gguf_close(&m->gguf);
    free(m->sbuf);
    free(m);
}

int tq_dim(const TqModel *m) { return m ? m->model.cfg.dim : 0; }

void tq_reset(TqModel *m) { if (m) m->st.pos = 0; }

int tq_chat(TqModel *m, const char *user_text, const TqGenParams *gen,
            TqTokenFn on_token, void *userdata) {
    if (!m || !user_text) return fail("参数为空");
    TqGenParams def = {0};
    const TqGenParams *g = gen ? gen : &def;
    GenCfg gc;
    gc.temperature = g->temperature < 0 ? 0.0f                       /* 负数 = 贪心 */
                   : (g->temperature > 0 ? g->temperature : 0.7f);   /* 0 = 默认    */
    gc.top_p   = g->top_p > 0 ? g->top_p : 0.8f;
    gc.max_gen = g->max_tokens > 0 ? g->max_tokens : 1024;
    gc.think   = g->think;
    if (g->seed) g_rng_state = g->seed;

    TurnStats ts = {0};
    if (chat_turn(&m->model, &m->st, &m->tok, &gc, m->sbuf, user_text,
                  on_token, userdata, &ts) < 0)
        return -1;
    return ts.history_cleared ? 1 : 0;
}

int tq_embed(TqModel *m, const char *text, const char *instruct, float *out) {
    if (!m || !text || !out) return fail("参数为空");
    if (!instruct)
        return embed_one(&m->model, &m->st, &m->tok, text, out);

    /* 官方查询模板包装 */
    size_t need = strlen(instruct) + strlen(text) + 32;
    char *buf = malloc(need);
    if (!buf) return fail("内存不足");
    snprintf(buf, need, "Instruct: %s\nQuery:%s", instruct, text);
    int rc = embed_one(&m->model, &m->st, &m->tok, buf, out);
    free(buf);
    return rc;
}

float tq_rerank(TqModel *m, const char *query, const char *document,
                const char *instruct) {
    if (!m || !query || !document) return (float)fail("参数为空");
    int yes_id = strmap_get(&m->tok.vocab_map, "yes");
    int no_id  = strmap_get(&m->tok.vocab_map, "no");
    if (yes_id < 0 || no_id < 0)
        return (float)fail("词表中找不到 yes/no token");
    return rerank_score(&m->model, &m->st, &m->tok,
                        instruct ? instruct : DEFAULT_TASK,
                        query, document, yes_id, no_id);
}

/* ===========================================================================
 * [9] 命令行工具 —— 定义 TINYQWEN_LIB 编译库时, 以下全部不参与编译
 * ===========================================================================
 */
#ifndef TINYQWEN_LIB

/* 读取一行输入(任意长度), 返回 malloc 的字符串(已去掉行尾换行/回车),
 * EOF 且无内容时返回 NULL。
 * 不用 POSIX 的 getline —— Windows(MSVC) 没有该函数和 ssize_t 类型。 */
static char *read_line(FILE *f) {
    size_t cap = 256, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    int ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n') {
        if (len + 2 > cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        buf[len++] = (char)ch;
    }
    if (ch == EOF && len == 0) { free(buf); return NULL; }
    while (len > 0 && buf[len - 1] == '\r') len--;  /* Windows 的 \r\n */
    buf[len] = 0;
    return buf;
}

/* 命令行的流式输出回调: 原样写到终端 */
static int cli_on_token(const char *piece, int len, void *ud) {
    (void)ud;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return 0;
}

/* 命令行的一轮对话: 调共享的 chat_turn 并打印统计信息 */
static void cli_chat_turn(const Model *m, RunState *st, const Tokenizer *tok,
                          const Options *opt, ProbIdx *sbuf, const char *text) {
    GenCfg gc = { opt->temperature, opt->top_p, opt->max_gen, opt->think };
    TurnStats ts = {0};
    if (chat_turn(m, st, tok, &gc, sbuf, text, cli_on_token, NULL, &ts) < 0) {
        printf("[%s]\n", g_err_msg);
        return;
    }
    if (ts.history_cleared) printf("\n[提示: 上下文已满, 本轮回答前已自动清空对话历史]");
    printf("\n[预填 %d tok / %.1fs · 生成 %d tok / %.1fs · %.2f tok/s]\n",
           ts.n_prompt, ts.prefill_ms / 1000.0, ts.n_gen, ts.gen_ms / 1000.0,
           ts.n_gen > 0 ? ts.n_gen / (ts.gen_ms / 1000.0) : 0.0);
}

static void usage(const char *prog) {
    fprintf(stderr,
        "用法: %s <模型.gguf> [选项]\n"
        "选项:\n"
        "  -t <float>   采样温度 (默认 0.7, 0 = 贪心解码)\n"
        "  -p <float>   top-p 核采样阈值 (默认 0.8)\n"
        "  -s <int>     随机种子 (默认取当前时间)\n"
        "  -c <int>     上下文长度上限 (默认 4096, 影响 KV 缓存内存)\n"
        "  -n <int>     单轮最大生成 token 数 (默认 1024)\n"
        "  -j <int>     线程数 (默认 0 = 自动: min(CPU核数, 16); 1 = 单线程)\n"
        "  --think      启用思考模式 (生成 <think>...</think>, 默认关闭)\n"
        "  --no-simd    强制使用标量内核 (不检测/不使用 AVX2)\n"
        "  --prompt <文本>  非交互模式: 回答一个问题后退出\n"
        "  --embed <文本>...   嵌入模式(需 Embedding 模型): 单文本输出向量,\n"
        "               多文本输出两两余弦相似度; 必须是最后一个选项\n"
        "  --rerank <查询> <文档>...  重排模式(需 Reranker 模型): 输出每个\n"
        "               文档的相关性得分与排名; 必须是最后一个选项\n"
        "  --instruct <指令>  检索任务描述(配合 --embed/--rerank, 有默认值)\n"
        "  --selftest   对比标量与 AVX2 内核的数值误差后退出\n"
        "  --inspect    打印模型元数据与张量列表后退出\n"
        "  --test-tokenizer <文本>  对文本做编码+解码测试后退出\n",
        prog);
    exit(1);
}

/* 模型名(general.name)是否包含某子串(不分大小写), 用于模式适配提示 */
static int model_name_has(const Gguf *g, const char *needle) {
    const GgufKV *kv = gguf_kv(g, "general.name");
    if (!kv || kv->type != GGUF_STRING) return 0;
    char buf[128];
    size_t n = kv->str_len < sizeof(buf) - 1 ? kv->str_len : sizeof(buf) - 1;
    for (size_t i = 0; i < n; i++) {
        char ch = kv->str[i];
        buf[i] = (ch >= 'A' && ch <= 'Z') ? (char)(ch + 32) : ch;
    }
    buf[n] = 0;
    return strstr(buf, needle) != NULL;
}

static Options parse_args(int argc, char **argv) {
    Options o = {0};
    o.temperature = 0.7f;
    o.top_p       = 0.8f;
    o.seed        = (uint64_t)time(NULL);
    o.ctx_len     = 4096;
    o.max_gen     = 1024;

    if (argc < 2) usage(argv[0]);
    o.model_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "-t") && i + 1 < argc) o.temperature = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) o.top_p       = strtof(argv[++i], NULL);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) o.seed        = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) o.ctx_len     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) o.max_gen     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-j") && i + 1 < argc) o.n_threads   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--think"))            o.think       = 1;
        else if (!strcmp(argv[i], "--no-simd"))          o.no_simd     = 1;
        else if (!strcmp(argv[i], "--selftest"))         o.selftest    = 1;
        else if (!strcmp(argv[i], "--inspect"))          o.inspect     = 1;
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc)         o.prompt    = argv[++i];
        else if (!strcmp(argv[i], "--instruct") && i + 1 < argc)       o.instruct  = argv[++i];
        else if (!strcmp(argv[i], "--test-tokenizer") && i + 1 < argc) o.test_text = argv[++i];
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) {
            /* 其后全部参数都是待嵌入文本 */
            o.embed_texts = (const char **)&argv[i + 1];
            o.n_embed = argc - i - 1;
            break;
        }
        else if (!strcmp(argv[i], "--rerank") && i + 2 < argc) {
            /* 其后第一个参数是查询, 其余是文档(至少一个) */
            o.rerank_args = (const char **)&argv[i + 1];
            o.n_rerank = argc - i - 1;
            break;
        }
        else usage(argv[0]);
    }
    if (o.ctx_len < 16) die("-c 上下文长度太小");
    return o;
}

int main(int argc, char **argv) {
    Options opt = parse_args(argc, argv);

#ifdef _WIN32
    /* Windows 控制台默认代码页不是 UTF-8, 中文输出会乱码 */
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    /* 进程级运行时初始化(线程池/SIMD 检测/f16 表), 与库 API 共用。
     * 线程数: -j 显式指定, 或自动取 min(核数, 32) —— matvec 行数只有
     * 1~3 千, 线程更多时每份工作变小、同步开销占比上升, 收益递减。 */
    if (runtime_init(opt.n_threads, opt.no_simd) < 0) die(g_err_msg);
    int n_threads = g_pool.n_threads;

    /* ---- 打开并解析 GGUF 模型文件 ----
     * 内部函数一律返回错误码; 命令行工具的策略是"检查到失败即退出" */
    Gguf gguf;
    if (gguf_open(&gguf, opt.model_path) < 0) die(g_err_msg);
    const GgufKV *name_kv = gguf_kv(&gguf, "general.name");
    printf("已加载 %s: %.*s, 架构 %s, %llu 个张量\n",
           opt.model_path,
           name_kv ? (int)name_kv->str_len : 1, name_kv ? name_kv->str : "?",
           "qwen3", (unsigned long long)gguf.n_tensors);

    if (opt.inspect)  { gguf_inspect(&gguf); return 0; }
    if (opt.selftest) { kernel_selftest(&gguf); return 0; }

    /* ---- 构建分词器 ---- */
    Tokenizer tok;
    if (tok_init(&tok, &gguf) < 0) die(g_err_msg);
    printf("分词器: %d 词表项, eos=%d, im_start=%d, im_end=%d\n",
           tok.n_vocab, tok.eos_id, tok.im_start_id, tok.im_end_id);

    if (opt.test_text) {
        /* 分词器自检: 编码 -> 打印 token -> 解码 -> 与原文比对 */
        int ids[4096];
        int n = tok_encode(&tok, opt.test_text, ids, 4096);
        if (n < 0) die(g_err_msg);
        printf("编码为 %d 个 token:\n", n);
        char buf[256];
        char *recon = calloc(1, strlen(opt.test_text) * 2 + 16);
        for (int i = 0; i < n; i++) {
            int len = tok_decode(&tok, ids[i], buf);
            printf("  [%6d] \"%.*s\"\n", ids[i], len, buf);
            strncat(recon, buf, len);
        }
        printf("解码回原文: %s\n",
               strcmp(recon, opt.test_text) == 0 ? "一致 ✔" : "不一致 ✘");
        free(recon);
        return 0;
    }

    /* ---- 加载模型权重、分配运行状态 ---- */
    Model model;
    if (model_load(&model, &gguf, opt.ctx_len) < 0) die(g_err_msg);
    const Config *cfg = &model.cfg;
    size_t kv_bytes = 2ull * cfg->n_layers * cfg->ctx_len
                          * cfg->n_kv_heads * cfg->head_dim * sizeof(float);
    printf("模型: %d 层, 隐层 %d, %d/%d 头(每头 %d), FFN %d, 词表 %d\n"
           "上下文: %d token (KV 缓存 %.0f MB)\n",
           cfg->n_layers, cfg->dim, cfg->n_heads, cfg->n_kv_heads,
           cfg->head_dim, cfg->ffn_dim, cfg->n_vocab,
           cfg->ctx_len, kv_bytes / 1048576.0);
    printf("线程: %d (%s, -j 可调) · SIMD: %s%s\n", n_threads,
#ifdef _WIN32
           "Win32 原生线程",
#else
           "POSIX pthread",
#endif
           g_use_simd ? g_simd_name
                      : (opt.no_simd ? "已禁用(--no-simd)" : "不支持, 用标量内核"),
           g_use_simd ? "(运行时检测)" : "");

    RunState st;
    if (state_alloc(&st, cfg) < 0) die(g_err_msg);
    ProbIdx *sbuf = malloc(sizeof(ProbIdx) * cfg->n_vocab); /* 采样工作区 */
    if (!sbuf) die("内存不足");
    g_rng_state = opt.seed ? opt.seed : 0x9E3779B97F4A7C15ull; /* 种子不可为 0 */

    /* ---- 模式与模型类型的适配提示(仅提醒, 不阻止) ---- */
    int is_embed_model  = model_name_has(&gguf, "embedding");
    int is_rerank_model = model_name_has(&gguf, "rerank");
    if (opt.n_embed && !is_embed_model)
        fprintf(stderr, "提示: 该模型名不含 Embedding, 建议用 Qwen3-Embedding 系列做嵌入\n");
    if (opt.n_rerank && !is_rerank_model)
        fprintf(stderr, "提示: 该模型名不含 Reranker, 建议用 Qwen3-Reranker 系列做重排\n");
    if (!opt.n_embed && !opt.n_rerank && (is_embed_model || is_rerank_model))
        fprintf(stderr, "提示: 这是%s模型, 聊天输出无意义, 请使用 %s 选项\n",
                is_embed_model ? "嵌入" : "重排", is_embed_model ? "--embed" : "--rerank");

    /* ---- 嵌入 / 重排模式: 处理完即退出 ---- */
    if (opt.n_embed) {
        embed_run(&model, &st, &tok, opt.embed_texts, opt.n_embed, opt.instruct);
        return 0;
    }
    if (opt.n_rerank) {
        rerank_run(&model, &st, &tok, opt.rerank_args, opt.n_rerank, opt.instruct);
        return 0;
    }

    /* ---- 非交互模式: 回答单个问题后退出 ---- */
    if (opt.prompt) {
        cli_chat_turn(&model, &st, &tok, &opt, sbuf, opt.prompt);
        return 0;
    }

    /* ---- 交互式问答主循环 ----
     * 多轮对话依靠 KV 缓存延续: 每轮只把新增的 token 喂给模型,
     * 历史内容的注意力状态都保存在缓存里, 模型能"记住"前文。 */
    printf("\n进入交互模式: 输入问题回车提问; /clear 清空对话; /exit 或 Ctrl-D 退出\n");
    for (;;) {
        printf("\n用户> ");
        fflush(stdout);
        char *line = read_line(stdin);
        if (!line) { printf("\n再见!\n"); break; }             /* Ctrl-D / EOF */
        if (!line[0]) { free(line); continue; }                /* 空行忽略 */

        if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) {
            free(line);
            printf("再见!\n");
            break;
        }
        if (!strcmp(line, "/clear")) {
            st.pos = 0; /* 清空 KV 缓存即忘掉全部历史 */
            printf("[对话历史已清空]\n");
            free(line);
            continue;
        }

        printf("助手> ");
        fflush(stdout);
        cli_chat_turn(&model, &st, &tok, &opt, sbuf, line);
        free(line);
    }
    return 0;
}

#endif /* TINYQWEN_LIB: 命令行工具段结束 */
