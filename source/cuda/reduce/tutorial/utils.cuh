#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
//  数据初始化：全 1，方便验证（sum == N）
// ─────────────────────────────────────────────
inline void init_data(float* data, int n)
{
    for (int i = 0; i < n; i++) data[i] = 1.0f;
}

// ─────────────────────────────────────────────
//  CPU 参考实现（double 累加，避免 float 精度损失）
// ─────────────────────────────────────────────
inline double cpu_reduce(const float* data, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)data[i];
    return sum;
}

// ─────────────────────────────────────────────
//  结果验证（相对误差 < 0.1%）
// ─────────────────────────────────────────────
inline bool verify(float gpu_result, double cpu_result, float tol = 1e-3f)
{
    double diff = fabs((double)gpu_result - cpu_result) / fabs(cpu_result);
    return diff < (double)tol;
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

    void stop() { CUDA_CHECK(cudaEventRecord(stop_)); }

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
// ─────────────────────────────────────────────
inline void print_result(const char* label, int n, float elapsed_ms,
                         float gpu_result, double cpu_result)
{
    // 有效带宽：只读一遍输入
    float bw = (float)(n * sizeof(float)) / (elapsed_ms * 1e-3f) /
               (1024.0f * 1024.0f * 1024.0f);
    printf("%-50s  time=%6.3f ms  BW=%6.2f GB/s  result=%.1f  [%s]\n",
           label, elapsed_ms, bw, gpu_result,
           verify(gpu_result, cpu_result) ? "PASS" : "FAIL");
}
