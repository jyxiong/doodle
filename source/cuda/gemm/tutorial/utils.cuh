#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ─────────────────────────────────────────────
//  CUDA 错误检查宏
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d  %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ─────────────────────────────────────────────
//  矩阵大小（行优先，M×K × K×N → M×N）
// ─────────────────────────────────────────────
struct GemmParams {
    int M, N, K;    // C(M×N) = A(M×K) * B(K×N)
    float alpha;    // C = alpha * A*B + beta * C
    float beta;
};

// ─────────────────────────────────────────────
//  数据初始化：随机浮点数
// ─────────────────────────────────────────────
inline void init_matrix(float* mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 9 + 1) * 0.1f;  // 0.1 ~ 0.9
    }
}

// ─────────────────────────────────────────────
//  CPU 参考实现（三重循环，用于正确性验证）
// ─────────────────────────────────────────────
inline void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     float alpha = 1.0f, float beta = 0.0f)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = alpha * sum + beta * C[m * N + n];
        }
    }
}

// ─────────────────────────────────────────────
//  结果验证（逐元素相对误差 < 0.1%）
// ─────────────────────────────────────────────
inline bool verify_gemm(const float* gpu_C, const float* cpu_C,
                         int M, int N, float tol = 1e-3f)
{
    int err_count = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(gpu_C[i] - cpu_C[i]);
        float ref  = fabsf(cpu_C[i]) + 1e-6f;
        if (diff / ref > tol) {
            if (err_count < 5) {
                fprintf(stderr, "  Mismatch at [%d]: gpu=%.6f  cpu=%.6f\n",
                        i, gpu_C[i], cpu_C[i]);
            }
            err_count++;
        }
    }
    return err_count == 0;
}

// ─────────────────────────────────────────────
//  GPU 计时器（基于 CUDA Event）
// ─────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t start_, stop_;

    GpuTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop()  { CUDA_CHECK(cudaEventRecord(stop_)); }

    float elapsed_ms()
    {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ─────────────────────────────────────────────
//  打印性能摘要
//  FLOP = 2 * M * N * K（每个输出元素：K 次乘 + K 次加）
// ─────────────────────────────────────────────
inline void print_result(const char* name,
                          int M, int N, int K,
                          float elapsed_ms,
                          bool correct)
{
    double flops     = 2.0 * M * N * K;
    double gflops    = flops / (elapsed_ms * 1e-3) / 1e9;
    printf("%-45s  time=%7.3f ms  GFLOPS=%7.2f  [%s]\n",
           name, elapsed_ms, gflops, correct ? "PASS" : "FAIL");
}
