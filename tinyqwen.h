/* ===========================================================================
 * tinyqwen.h — TinyQwen 公共 C API
 * ===========================================================================
 *
 * TinyQwen 是零依赖、单文件实现的 Qwen3 系列 GGUF 模型 CPU 推理引擎,
 * 既可编译成命令行工具, 也可编译成静态/动态库嵌入宿主程序。
 * 支持三类模型: 聊天(Qwen3)、嵌入(Qwen3-Embedding)、重排(Qwen3-Reranker)。
 *
 * 最小用法(聊天):
 *
 *     #include "tinyqwen.h"
 *
 *     static int on_token(const char *piece, int len, void *ud) {
 *         fwrite(piece, 1, len, stdout);   // 流式打印回复
 *         return 0;                        // 返回非 0 可中断生成
 *     }
 *
 *     TqModel *m = tq_load("Qwen3-0.6B-Q4_0.gguf", NULL);
 *     if (!m) { fprintf(stderr, "%s\n", tq_last_error()); return 1; }
 *     tq_chat(m, "你好", NULL, on_token, NULL);   // 多轮状态保存在句柄内
 *     tq_free(m);
 *
 * 构建库:
 *     动态库: gcc -O2 -fPIC -shared -DTINYQWEN_LIB -o libtinyqwen.so \
 *             tinyqwen.c -lm -pthread
 *     静态库: gcc -O2 -DTINYQWEN_LIB -c tinyqwen.c && ar rcs libtinyqwen.a tinyqwen.o
 *     (TINYQWEN_LIB 只是去掉命令行工具的 main 函数, API 始终存在)
 *
 * 线程安全: 单个句柄不可多线程并发调用; 进程内的多个句柄共享同一个
 * 计算线程池, 池的线程数以第一次 tq_load 传入的值为准。
 *
 * 错误处理: 失败的调用返回 NULL / 负值, 错误描述用 tq_last_error() 获取
 * (进程级缓冲区, 与调用线程无关)。
 * ===========================================================================
 */
#ifndef TINYQWEN_H
#define TINYQWEN_H

#include <stdint.h>

/* Windows 动态库的符号导入/导出(其它平台为空) */
#if defined(_WIN32) && defined(TINYQWEN_SHARED)
#  ifdef TINYQWEN_BUILD
#    define TQ_API __declspec(dllexport)
#  else
#    define TQ_API __declspec(dllimport)
#  endif
#else
#  define TQ_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TINYQWEN_VERSION "0.2.0"

/* 不透明句柄: 内部包含 mmap 的权重、分词器、KV 缓存等全部推理状态 */
typedef struct TqModel TqModel;

/* 加载参数。传 NULL 或把字段置 0 表示使用默认值。 */
typedef struct {
    int ctx_len;    /* 上下文长度上限(决定 KV 缓存内存), 0 = 4096          */
    int n_threads;  /* 计算线程数, 0 = 自动 min(CPU 核数, 32), 1 = 单线程  */
    int no_simd;    /* 1 = 强制标量内核; 0 = 自动检测并使用 AVX2           */
} TqLoadParams;

/* 生成参数。传 NULL 或把字段置 0 表示使用默认值。 */
typedef struct {
    float    temperature; /* 采样温度; 0 = 默认 0.7; 负数 = 贪心解码       */
    float    top_p;       /* 核采样阈值; 0 = 默认 0.8                      */
    int      max_tokens;  /* 单轮生成 token 上限; 0 = 默认 1024            */
    uint64_t seed;        /* 非 0 时重置随机种子(可复现输出)               */
    int      think;       /* 1 = 启用 Qwen3 思考模式(<think>...</think>)   */
} TqGenParams;

/* 流式 token 回调: piece 是解码后的 UTF-8 字节片段(不含结尾 \0, 单个
 * 多字节字符可能跨片段)。返回 0 继续生成, 返回非 0 立即中断本轮生成。 */
typedef int (*TqTokenFn)(const char *piece, int len, void *userdata);

/* ---- 生命周期 ---- */

/* 加载 GGUF 模型文件。失败返回 NULL(原因见 tq_last_error)。 */
TQ_API TqModel *tq_load(const char *gguf_path, const TqLoadParams *params);

/* 释放句柄与全部相关内存(权重映射、KV 缓存、词表)。可传 NULL。 */
TQ_API void tq_free(TqModel *m);

/* 最近一次失败的错误描述(UTF-8 中文)。 */
TQ_API const char *tq_last_error(void);

/* ---- 查询与控制 ---- */

/* 模型隐层维度 —— 即 tq_embed 输出向量的长度。 */
TQ_API int tq_dim(const TqModel *m);

/* 清空多轮对话历史(重置 KV 缓存)。 */
TQ_API void tq_reset(TqModel *m);

/* ---- 推理 ---- */

/* 聊天一轮: 输入用户文本, 回复经 on_token 流式回调(可为 NULL 表示丢弃)。
 * 多轮对话状态保留在句柄内, 用 tq_reset 开启新对话。
 * 返回 0 = 成功; 1 = 成功但上下文已满、历史被自动清空后重答;
 *     -1 = 失败(tq_last_error)。 */
TQ_API int tq_chat(TqModel *m, const char *user_text, const TqGenParams *gen,
                   TqTokenFn on_token, void *userdata);

/* 文本嵌入(需 Qwen3-Embedding 系列模型)。
 * out 容量至少 tq_dim(m) 个 float, 结果已 L2 归一化(点积即余弦相似度)。
 * instruct 非 NULL 时按官方检索模板 "Instruct: {instruct}\nQuery:{text}"
 * 包装(查询侧建议提供, 文档侧传 NULL)。返回 0 成功, -1 失败。 */
TQ_API int tq_embed(TqModel *m, const char *text, const char *instruct, float *out);

/* 查询-文档相关性打分(需 Qwen3-Reranker 系列模型)。
 * 返回 (0,1) 内的概率, 越大越相关; 失败返回负值。
 * instruct 传 NULL 使用官方默认任务描述。 */
TQ_API float tq_rerank(TqModel *m, const char *query, const char *document,
                       const char *instruct);

#ifdef __cplusplus
}
#endif
#endif /* TINYQWEN_H */
