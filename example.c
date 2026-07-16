/* ===========================================================================
 * example.c — TinyQwen 库用法示例
 * ===========================================================================
 *
 * 只包含公共头文件 tinyqwen.h, 演示三类模型的 API 用法:
 *
 *     ./example chat   <聊天模型.gguf>   "问题"
 *     ./example embed  <嵌入模型.gguf>   "文本A" ["文本B"]
 *     ./example rerank <重排模型.gguf>   "查询" "文档"
 *
 * 构建(见 Makefile 的 examples 目标):
 *     静态链接: gcc -O2 -o example example.c libtinyqwen.a -lm -pthread
 *     动态链接: gcc -O2 -o example example.c -L. -ltinyqwen -lm -pthread
 * ===========================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyqwen.h"

/* 流式回调: 把回复片段原样打印(返回非 0 可中断生成) */
static int on_token(const char *piece, int len, void *userdata) {
    (void)userdata;
    fwrite(piece, 1, len, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "用法: %s chat   <模型.gguf> <问题>\n"
                "      %s embed  <模型.gguf> <文本A> [文本B]\n"
                "      %s rerank <模型.gguf> <查询> <文档>\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    const char *mode = argv[1], *path = argv[2];

    /* 全部字段置 0 = 默认值(上下文 4096, 线程自动, 自动检测 AVX2) */
    TqLoadParams lp = {0};
    TqModel *m = tq_load(path, &lp);
    if (!m) {
        fprintf(stderr, "加载失败: %s\n", tq_last_error());
        return 1;
    }
    printf("模型已加载 (TinyQwen %s, 隐层维度 %d)\n", TINYQWEN_VERSION, tq_dim(m));

    int rc = 0;
    if (!strcmp(mode, "chat")) {
        /* ---- 聊天: 两轮对话演示多轮状态保存在句柄内 ---- */
        TqGenParams gp = {0};
        gp.seed = 42; /* 固定种子, 结果可复现 */
        printf("用户> %s\n助手> ", argv[3]);
        if (tq_chat(m, argv[3], &gp, on_token, NULL) < 0) {
            fprintf(stderr, "\n生成失败: %s\n", tq_last_error());
            rc = 1;
        } else {
            printf("\n用户> 请用一句话总结你刚才的回答\n助手> ");
            tq_chat(m, "请用一句话总结你刚才的回答", &gp, on_token, NULL);
            printf("\n");
        }
    } else if (!strcmp(mode, "embed")) {
        /* ---- 嵌入: 输出向量前 8 维; 有两个文本时再算余弦相似度 ---- */
        int dim = tq_dim(m);
        float *a = malloc(sizeof(float) * dim);
        float *b = malloc(sizeof(float) * dim);
        if (!a || !b) { fprintf(stderr, "内存不足\n"); return 1; }
        if (tq_embed(m, argv[3], NULL, a) < 0) {
            fprintf(stderr, "嵌入失败: %s\n", tq_last_error());
            rc = 1;
        } else {
            printf("《%s》 向量前 8 维:", argv[3]);
            for (int i = 0; i < 8 && i < dim; i++) printf(" %.4f", a[i]);
            printf(" ...\n");
            if (argc >= 5 && tq_embed(m, argv[4], NULL, b) == 0) {
                float cos = 0.0f;
                for (int i = 0; i < dim; i++) cos += a[i] * b[i];
                printf("与《%s》的余弦相似度: %.4f\n", argv[4], cos);
            }
        }
        free(a);
        free(b);
    } else if (!strcmp(mode, "rerank")) {
        /* ---- 重排: 查询-文档相关性概率 ---- */
        if (argc < 5) { fprintf(stderr, "rerank 需要 <查询> <文档>\n"); return 1; }
        float score = tq_rerank(m, argv[3], argv[4], NULL);
        if (score < 0) {
            fprintf(stderr, "打分失败: %s\n", tq_last_error());
            rc = 1;
        } else {
            printf("查询《%s》vs 文档《%s》 相关性: %.4f\n", argv[3], argv[4], score);
        }
    } else {
        fprintf(stderr, "未知模式: %s\n", mode);
        rc = 1;
    }

    tq_free(m); /* 释放权重映射、KV 缓存等全部资源 */
    return rc;
}
