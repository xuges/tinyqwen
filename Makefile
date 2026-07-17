# TinyQwen 构建脚本
#
# 目标一览:
#   make                  命令行工具 ./tinyqwen
#   make libs             动态库 libtinyqwen.so + 静态库 libtinyqwen.a
#   make examples         库用法示例(分别链接静态库与动态库)
#   make clean            清理全部产物
#
# 零依赖说明: 只链接 libm(数学库)和 pthread(POSIX 线程), 都是系统自带的
# 基础库。多线程用平台原生 API(POSIX pthread / Win32 线程), SIMD 用运行时
# 检测(同一产物在无 AVX2 的机器上也能跑)。
#
# macOS: 直接 make 即可(gcc 通常是 clang 的别名; -lm/-pthread 都由 libSystem
#        提供, 可正常链接)。make libs 生成的 libtinyqwen.so 在 macOS 上是有效
#        的动态库, 惯例扩展名虽是 .dylib, 但 .so 一样能被链接和加载。
# Windows 下用 MinGW: gcc -O2 -o tinyqwen.exe tinyqwen.c
#          或 MSVC:  cl /O2 /utf-8 tinyqwen.c
#                    (/utf-8 必须加: 源码是 UTF-8 编码, MSVC 默认按系统
#                     代码页解析, 中文注释与字符串会乱码甚至报错)
# Windows 构建 DLL:   gcc -O2 -shared -DTINYQWEN_LIB -DTINYQWEN_SHARED \
#                         -o tinyqwen.dll tinyqwen.c

CC      ?= gcc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra
LDLIBS   = -lm -pthread

# ---- 命令行工具 ----
tinyqwen: tinyqwen.c tinyqwen.h
	$(CC) $(CFLAGS) -o $@ tinyqwen.c $(LDLIBS)

# ---- 库(TINYQWEN_LIB 宏去掉 main, 其余完全相同) ----
libtinyqwen.so: tinyqwen.c tinyqwen.h
	$(CC) $(CFLAGS) -fPIC -shared -DTINYQWEN_LIB -o $@ tinyqwen.c $(LDLIBS)

libtinyqwen.a: tinyqwen.c tinyqwen.h
	$(CC) $(CFLAGS) -DTINYQWEN_LIB -c -o tinyqwen.o tinyqwen.c
	ar rcs $@ tinyqwen.o

.PHONY: libs
libs: libtinyqwen.so libtinyqwen.a

# ---- 库用法示例(example.c 只 include tinyqwen.h, 证明 API 自洽) ----
example-static: example.c libtinyqwen.a
	$(CC) $(CFLAGS) -o $@ example.c libtinyqwen.a $(LDLIBS)

example-shared: example.c libtinyqwen.so
	$(CC) $(CFLAGS) -o $@ example.c -L. -ltinyqwen $(LDLIBS)
	@echo "运行动态链接版: LD_LIBRARY_PATH=. ./example-shared ..."

.PHONY: examples
examples: example-static example-shared

.PHONY: clean
clean:
	rm -f tinyqwen tinyqwen.o libtinyqwen.so libtinyqwen.a example-static example-shared
