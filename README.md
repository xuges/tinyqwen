# TinyQwen — 零依赖单文件 Qwen3 系列 CPU 推理引擎

> **AI 创作说明**：本项目的全部代码、注释与文档均由 Anthropic 的 AI 编程助手
> Claude Code（模型 Claude Opus 4.8）编写，人类作者负责提出需求、把控方向与
> 验收结果。项目定位为教学与学习用途。

用**纯 C 语言（C99）**编写的大语言模型推理引擎，既是命令行工具，也可编译成
**静态/动态库**嵌入宿主程序（公共 API 见 `tinyqwen.h`）。

**⚡ 冷启动极快**：得益于 GGUF 权重 `mmap` 零拷贝映射（页由内核按需加载，
不做任何预读或反量化预处理），从进程启动到能开始回答几乎是**瞬时**的——
不像多数框架那样要先把几百 MB 到数 GB 的权重读进内存、构建运行时。
配合零依赖单文件，`gcc 一行编译 → 立即运行`。

**✅ 已在 Windows、Linux、macOS 三大平台测试通过**（x86-64 与 aarch64 均验证）。

特点：

- **零依赖**：只用 C 标准库 + 各平台原生 API（`mmap` / Win32），不链接任何第三方库
- **单文件**：全部逻辑都在 `tinyqwen.c` 一个文件里（注释丰富）
- **直接读 GGUF**：解析 [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) 模型文件格式，
  权重、超参数、分词器词表全部来自一个 `.gguf` 文件，无需额外的 tokenizer 文件
- **量化推理**：支持 Q4_0 / Q4_1 的 4bit 块量化、Q8_0 的 8bit 块量化、
  Q6_K 超块量化以及 F16/F32，推理时逐块反量化做矩阵乘法，
  覆盖官方 Q4_0 与 Q8_0 量化文件的全部张量组合
- **原生多线程**：自带跨平台线程池（POSIX pthread / Win32 线程双实现，不依赖 OpenMP），
  线程数用 `-j` 指定，默认自动取 min(核数, 32)
- **SIMD 运行时检测**：x86 用 CPUID 检测 AVX2+FMA，aarch64 用 HWCAP 检测
  NEON，支持则切换到向量化点积内核（x86 实测约 3~11 倍提速），
  不支持自动退回标量路径——同一个二进制到处能跑
- **命令行问答交互**：启动后加载模型，进入多轮对话 REPL（ChatML 模板 + KV 缓存记忆）
- **嵌入与重排**：同一程序还支持 Qwen3-Embedding（文本向量/相似度）与
  Qwen3-Reranker（查询-文档相关性打分）模型

支持 **Qwen3 全系列稠密模型**（0.6B / 1.7B / 4B / 8B / 14B / 32B，含 Embedding 与
Reranker 变体）——层数、维度、头数等结构差异全部由 GGUF 元数据驱动，已实测
0.6B / 4B / Embedding-0.6B / Reranker-0.6B。

## 构建

```bash
make            # 命令行工具 ./tinyqwen
make libs       # 库: libtinyqwen.so (动态) + libtinyqwen.a (静态)
make examples   # 库用法示例: example-static / example-shared
```

或者手动编译（真的只有一条命令）：

```bash
gcc -O2 -std=c99 -o tinyqwen tinyqwen.c -lm -pthread     # Linux / macOS
gcc -O2 -o tinyqwen.exe tinyqwen.c                      # Windows (MinGW)
cl /O2 /utf-8 tinyqwen.c                                # Windows (MSVC)
```

（MSVC 的 `/utf-8` 不能省：源码为 UTF-8 编码，MSVC 默认按系统代码页解析。）

SIMD 无需编译开关：AVX2 内核用按函数的目标属性编译，运行时检测启用。

## 作为库使用

公共 API 只有 8 个函数（详见 `tinyqwen.h` 的文档注释，用法示例见 `example.c`）：

```c
#include "tinyqwen.h"

static int on_token(const char *piece, int len, void *ud) {
    fwrite(piece, 1, len, stdout);   /* 流式接收回复 */
    return 0;                        /* 返回非 0 中断生成 */
}

int main(void) {
    TqModel *m = tq_load("models/Qwen3-0.6B-Q4_0.gguf", NULL);
    if (!m) { fprintf(stderr, "%s\n", tq_last_error()); return 1; }

    tq_chat(m, "你好", NULL, on_token, NULL);  /* 多轮状态在句柄内 */
    tq_reset(m);                               /* 开启新对话 */

    float vec[1024];                           /* tq_dim(m) 维 */
    tq_embed(m, "文本", NULL, vec);            /* Embedding 模型 */
    float s = tq_rerank(m, "查询", "文档", NULL); /* Reranker 模型 */

    tq_free(m);
    return 0;
}
```

```bash
# 静态链接
gcc -O2 -o app app.c libtinyqwen.a -lm -pthread
# 动态链接
gcc -O2 -o app app.c -L. -ltinyqwen -lm -pthread
```

要点：内部函数统一返回错误码并逐层向上传递——库调用失败返回 NULL/负值
（描述见 `tq_last_error()`），永远不会杀死宿主进程；只有命令行工具在检查到
错误后才打印并退出。单个句柄不可并发调用；进程内多个句柄共享线程池。
`-DTINYQWEN_LIB` 只是去掉 `main()`，库与命令行工具共用同一份实现。

## 获取模型

```bash
mkdir -p models
# 0.6B (382MB, 内存占用小, 速度快)
curl -L -o models/Qwen3-0.6B-Q4_0.gguf \
  "https://modelscope.cn/models/unsloth/Qwen3-0.6B-GGUF/resolve/master/Qwen3-0.6B-Q4_0.gguf"
# 4B (2.4GB, 回答质量明显更好)
curl -L -o models/Qwen3-4B-Q4_0.gguf \
  "https://modelscope.cn/models/unsloth/Qwen3-4B-GGUF/resolve/master/Qwen3-4B-Q4_0.gguf"
```

其它规格同理：把 URL 中的 `0.6B` 换成 `1.7B`/`8B`/`14B` 即可
（也可以用 HuggingFace 上 `unsloth/Qwen3-*-GGUF` 仓库的同名文件）。

## 运行

```bash
./tinyqwen models/Qwen3-0.6B-Q4_0.gguf     # 或 models/Qwen3-4B-Q4_0.gguf
```

示例会话：

```text
模型: 28 层, 隐层 1024, 16/8 头(每头 128), FFN 3072, 词表 151936
上下文: 4096 token (KV 缓存 896 MB)
线程: 32 (POSIX pthread, -j 可调) · SIMD: AVX2+FMA(运行时检测)

进入交互模式: 输入问题回车提问; /clear 清空对话; /exit 或 Ctrl-D 退出

用户> 我叫小明，请记住我的名字。
助手> 你好，小明！记得我的名字。
用户> 我叫什么名字？
助手> 我叫小明。
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `-t <float>` | 采样温度，默认 0.7；`-t 0` 为贪心解码（确定性输出） |
| `-p <float>` | top-p 核采样阈值，默认 0.8 |
| `-s <int>`   | 随机种子，默认取当前时间 |
| `-c <int>`   | 上下文长度上限，默认 4096（KV 缓存约 896MB，内存紧张可调小） |
| `-n <int>`   | 单轮最大生成 token 数，默认 1024 |
| `-j <int>`   | 线程数，默认 0 = 自动（min(CPU 核数, 32)），`-j 1` 纯单线程 |
| `--think`    | 启用 Qwen3 思考模式（回复前先生成 `<think>...</think>` 推理过程） |
| `--no-simd`  | 强制标量内核（不检测/不使用 AVX2） |
| `--prompt <文本>` | 非交互模式：回答一个问题后退出 |
| `--embed <文本>...` | 嵌入模式：单文本输出 L2 归一化向量，多文本输出余弦相似度矩阵（须放最后） |
| `--rerank <查询> <文档>...` | 重排模式：输出各文档相关性得分与排名（须放最后） |
| `--instruct <指令>` | 检索任务描述，配合 `--embed`（包装第一个文本为查询）或 `--rerank` |
| `--selftest` | 对比标量与 AVX2 内核的数值误差后退出 |
| `--inspect`  | 打印模型元数据与全部张量列表后退出 |
| `--test-tokenizer <文本>` | 分词器自检：编码→打印 token→解码→比对 |

REPL 内命令：`/clear` 清空对话历史，`/exit`（或 Ctrl-D）退出。

## 嵌入与重排模型

```bash
# 下载(ModelScope)
curl -L -o models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  "https://modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/master/Qwen3-Embedding-0.6B-Q8_0.gguf"
curl -L -o models/Qwen3-Reranker-0.6B-q4_0.gguf \
  "https://modelscope.cn/models/Mungert/Qwen3-Reranker-0.6B-GGUF/resolve/master/Qwen3-Reranker-0.6B-q4_0.gguf"

# 文本相似度(向量已 L2 归一化, 点积即余弦)
./tinyqwen models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --embed "北京是中国的首都" "The capital of China is Beijing" "我想吃苹果"
#   [0]-[1] 0.86 (中英互译)   [0]-[2] 0.27 (无关)

# 检索式嵌入: --instruct 把第一个文本按官方模板包装为查询
./tinyqwen models/Qwen3-Embedding-0.6B-Q8_0.gguf \
  --instruct "Given a web search query, retrieve relevant passages that answer the query" \
  --embed "什么是光合作用" "光合作用是植物利用光能合成有机物的过程" "股市今天上涨"

# 重排: 相关性得分 = 模型对"该文档是否满足查询"回答 yes 的概率
./tinyqwen models/Qwen3-Reranker-0.6B-q4_0.gguf \
  --rerank "什么是光合作用" "光合作用是植物利用光能合成有机物的过程" "股市今天上涨"
#   文档[0] 0.9990   文档[1] 0.0001
```

实现说明：嵌入 = 末尾 `<|endoftext|>` 位置经最终 RMSNorm 的隐状态（完全跳过
lm_head 词表投影），重排 = 官方 ChatML 模板下末位 logits 的 yes/no 二元 softmax。

## 代码结构（tinyqwen.c 内部分区）

| 分区 | 内容 |
|------|------|
| [0] 通用工具 | 错误处理、fp16→f32 转换（含 65536 项查找表） |
| [0b] 跨平台线程池 | pthread / Win32 双实现的 parallel_for（常驻线程 + 条件变量同步） |
| [1] GGUF 解析 | mmap 打开文件（零拷贝，Windows 用 fread），解析头部/元数据/张量目录 |
| [2] 分词器 | GPT-2 风格字节级 BPE：字节↔Unicode 映射、哈希表、贪心合并编码、解码 |
| [3] 模型加载 | 读取超参数、按名字定位 28 层权重、RoPE 频率表 |
| [4] 数学内核 | Q4_0/Q4_1/Q6_K 反量化点积（热点）、RMSNorm、Softmax |
| [4b] AVX2 内核 | x86：CPUID 运行时检测；四种量化点积的 SIMD 版本，函数指针分发 |
| [4c] NEON 内核 | aarch64：HWCAP 运行时检测；同一组点积的 128 位 NEON 版本 |
| [5] 前向传播 | QK-Norm、RoPE(NeoX)、GQA 因果注意力、KV 缓存、SwiGLU FFN |
| [6] 采样器 | temperature + top-p 核采样，xorshift64* 随机数 |
| [7] 聊天主循环 | ChatML 模板拼接、逐 token 流式回调生成 |
| [7a] 嵌入与重排 | 最后 token 池化 + L2 归一化；重排模板 + yes/no 概率 |
| [8] 库 API | `tq_load/tq_chat/tq_embed/tq_rerank` 等 8 个导出函数 |
| [9] 命令行工具 | 参数解析、REPL（`-DTINYQWEN_LIB` 时不编译） |

## 实现要点

- **Q4_0 量化**：每 32 个权重一个块（18 字节 = fp16 缩放 + 16 字节半字节），
  `权重 = (4bit值 - 8) × 缩放`。矩阵乘时逐块"反量化+点积"融合计算，不生成中间浮点矩阵。
- **Qwen3 结构特点**：GQA（16 查询头共享 8 个 KV 头）、每头 128 维、
  注意力前对每头做 QK-RMSNorm、RoPE 采用 NeoX 风格半维配对、SwiGLU 激活、
  输出层与词嵌入共享权重（从 Q6_K 张量读取）。
- **多轮对话**：KV 缓存跨轮保留，每轮只把新增 token 喂给模型；
  上下文写满时自动清空历史重新开始。
- **思考模式**：Qwen3 约定在 assistant 开头插入空 `<think>\n\n</think>\n\n` 块即关闭思考；
  `--think` 则让模型自由生成思考过程。

## 性能参考（96 核 Xeon 类服务器）

| 配置 | Qwen3-0.6B | Qwen3-4B |
|------|-----------|----------|
| AVX2 + 32 线程（默认） | ~40 tok/s | ~15 tok/s |
| 标量 + 32 线程（`--no-simd`） | ~13 tok/s | — |
| AVX2 单线程（`-j 1`） | ~12 tok/s | — |
| 标量单线程（`-j 1 --no-simd`） | ~1 tok/s | — |

标量内核保留为教学参考与未知平台的兜底路径。`--selftest` 会用模型真实权重
把标量与 SIMD 内核分别对照 double 精度参考值验证（误差均在 1e-7 量级）。

### aarch64 / NEON

NEON 是 armv8-a 的强制特性，无需编译开关，运行时经 Linux HWCAP 检测启用
（Apple Silicon / Windows ARM64 按架构恒真），`--no-simd` 可强制标量。
四种量化内核（Q4_0/Q4_1/Q8_0/Q6_K）均有 128 位 NEON 实现，`--selftest`
对 double 参考误差在 1e-8 量级，贪心生成输出与标量/x86 逐字一致。

在没有 ARM 硬件时，可用交叉编译 + qemu 做数值校验：

```bash
aarch64-linux-gnu-gcc -O2 -std=c99 -static -o tinyqwen-arm64 tinyqwen.c -lm -pthread
qemu-aarch64 ./tinyqwen-arm64 models/Qwen3-0.6B-Q4_0.gguf --selftest   # NEON 内核数值校验
```

## 功能清单

- [x] GGUF 文件解析（mmap 零拷贝，冷启动瞬时）
- [x] 字节级 BPE 分词器（词表/合并规则来自 GGUF 元数据）
- [x] Transformer 前向传播（QK-Norm + RoPE + GQA + KV 缓存 + SwiGLU）
- [x] 量化内核 Q4_0 / Q4_1 / Q8_0 / Q6_K / F16 / F32
- [x] 三类模型：聊天（多轮 REPL）、Qwen3-Embedding、Qwen3-Reranker
- [x] Qwen3 全系列稠密规格（0.6B ~ 32B，结构由元数据驱动）
- [x] 跨平台原生线程池（POSIX pthread / Win32 线程），`-j` 指定线程数
- [x] SIMD 运行时检测：x86 AVX2+FMA、aarch64 NEON
- [x] 可编译为静态 / 动态库（`tinyqwen.h` 公共 API + 示例程序）
- [x] 三平台构建通过：Windows、Linux、macOS（x86-64 与 aarch64）

## 许可证

[MIT](LICENSE)。注意：模型权重文件本身的许可以其发布方为准
（Qwen3 系列模型采用 Apache-2.0 许可发布）。
