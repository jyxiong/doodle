# 五、并发优化

CUDA 的并发优化通过重叠计算与数据传输、重叠多个 kernel 的执行，来隐藏各类延迟，最大化 GPU 利用率。

---

## 5.1 CUDA Stream 基础

### Stream 是什么

Stream 是 GPU 上有序的任务队列。提交到同一 Stream 的操作按顺序串行执行；不同 Stream 的操作可以并发执行。

```
默认流（Default Stream / Stream 0）：
  与所有其他 Stream 隐式同步，无法实现真正并发

自定义流（Custom Stream）：
  操作异步提交，不同流之间独立并行
```

### Stream 操作

```cuda
// 创建/销毁
cudaStream_t stream;
cudaStreamCreate(&stream);
// ...
cudaStreamDestroy(stream);

// 提交操作到 Stream
kernel<<<grid, block, smem, stream>>>(args);
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);

// 同步
cudaStreamSynchronize(stream);  // 等待指定 stream 完成
cudaDeviceSynchronize();        // 等待所有 stream 完成
```

---

## 5.2 计算与数据传输重叠

### 前提条件

主机内存必须使用 **Pinned Memory（页锁定内存）**，才能支持 DMA 传输（绕过 CPU，直接 PCIe 传输），从而实现真正的计算与传输重叠。

```cuda
// ❌ 普通内存（Pageable），不能异步传输
float* h_data = new float[N];

// ✅ Pinned Memory，支持异步传输
float* h_data;
cudaMallocHost(&h_data, N * sizeof(float));
// ...
cudaFreeHost(h_data);
```

### 流水线模式

将数据分成多个 Chunk，在不同 Stream 中交错执行传输和计算：

```
                 Stream 0        Stream 1        Stream 2
                 ─────────────   ─────────────   ─────────────
时间轴：
 t0         [H2D Chunk0]
 t1                         [H2D Chunk1]
 t2         [Kernel Chunk0]                 [H2D Chunk2]
 t3                         [Kernel Chunk1]
 t4         [D2H Chunk0]                    [Kernel Chunk2]
 t5                         [D2H Chunk1]
 t6                                         [D2H Chunk2]
```

**代码示例**：
```cuda
const int NUM_STREAMS = 4;
const int CHUNK = N / NUM_STREAMS;

cudaStream_t streams[NUM_STREAMS];
for (int i = 0; i < NUM_STREAMS; ++i)
    cudaStreamCreate(&streams[i]);

for (int i = 0; i < NUM_STREAMS; ++i) {
    int offset = i * CHUNK;
    // H2D
    cudaMemcpyAsync(d_in + offset, h_in + offset,
                    CHUNK * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    // Kernel
    process<<<CHUNK/256, 256, 0, streams[i]>>>(d_in + offset, d_out + offset, CHUNK);
    // D2H
    cudaMemcpyAsync(h_out + offset, d_out + offset,
                    CHUNK * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
}

cudaDeviceSynchronize();

for (int i = 0; i < NUM_STREAMS; ++i)
    cudaStreamDestroy(streams[i]);
```

### 传输带宽上限

PCIe 4.0 x16 双向带宽约 32 GB/s（单向 ~16 GB/s），是 CPU-GPU 数据传输的硬上限。优化传输效率的手段：

| 手段 | 说明 |
|------|------|
| Pinned Memory | 必须，否则带宽减半 |
| 批量传输 | 多次小传输合并为一次大传输，摊薄启动开销 |
| NVLink（GPU-GPU） | A100 NVLink 带宽 600 GB/s，远超 PCIe |
| 减少传输量 | 在 GPU 上计算，只传输最终结果 |

---

## 5.3 CUDA Event（事件）

### 主要用途

**① 精确计时**：CPU 时间戳无法准确测量 GPU 执行时间
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);        // 在 stream 中插入开始标记
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);         // 在 stream 中插入结束标记

cudaEventSynchronize(stop);            // 等待 stop 事件完成
float ms = 0.0f;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel time: %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

**② 跨 Stream 依赖同步**：让 Stream B 等待 Stream A 的某个事件：
```cuda
cudaEventRecord(event, streamA);               // 标记 streamA 的中间点
cudaStreamWaitEvent(streamB, event, 0);        // streamB 等待该事件
// streamB 后续的任务 在 event 完成后才会执行
```

---

## 5.4 多 GPU 编程

### 设备管理

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

cudaSetDevice(0);           // 后续操作绑定到 GPU 0
cudaSetDevice(1);           // 切换到 GPU 1
```

### P2P（Peer-to-Peer）通信

GPU 间如果有 NVLink 或支持 P2P 的 PCIe 通道，可以直接互访显存：
```cuda
// 检查 P2P 支持
int canAccess;
cudaDeviceCanAccessPeer(&canAccess, 0, 1); // GPU 0 能否直接访问 GPU 1

// 开启 P2P
cudaSetDevice(0);
cudaDeviceEnablePeerAccess(1, 0);

// GPU 0 上的 kernel 可以直接读写 GPU 1 上的内存
```

### 多 GPU 数据并行模式

```
数据分片 → 各 GPU 独立计算 → 汇总结果（AllReduce 或 Gather）
```

生产级多 GPU 通信推荐使用 **NCCL（NVIDIA Collective Communications Library）**：
```cuda
ncclAllReduce(sendbuff, recvbuff, count, ncclFloat, ncclSum, comm, stream);
```

---

## 5.5 CUDA Graph（计算图）

### 问题

每次调用 `kernel<<<...>>>`、`cudaMemcpyAsync` 都有**驱动层 API 开销**（~5-20 μs/次）。对于由大量小操作组成的任务（如深度学习 inference），这些开销累积显著。

### 解决方案

CUDA Graph 将一系列操作**预先录制为静态图**，之后重复执行时只需提交一次图实例，驱动开销降低 **10~50x**：

```cuda
// 录制阶段
cudaGraph_t graph;
cudaGraphExec_t instance;

cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// 提交一系列操作（这些操作被录制，不立即执行）
kernel_a<<<grid, block, 0, stream>>>(args);
kernel_b<<<grid, block, 0, stream>>>(args);
cudaMemcpyAsync(d_out, d_tmp, size, cudaMemcpyDeviceToDevice, stream);
kernel_c<<<grid, block, 0, stream>>>(args);
cudaStreamEndCapture(stream, &graph);

// 实例化图
cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);

// 重复执行（API 开销极低）
for (int iter = 0; iter < 1000; ++iter) {
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
}

// 清理
cudaGraphExecDestroy(instance);
cudaGraphDestroy(graph);
```

### 适用场景

- 深度学习 inference（固定拓扑，重复执行）
- 科学计算的时间步迭代
- 任何固定流程、重复执行多次的 GPU 任务

---

## 5.6 并发 Kernel 执行

**多 Kernel 并发**（MPS / Multi-Process Service）：

- 单 GPU 可以同时执行来自不同 Stream 的多个 Kernel（前提是各 Kernel 的资源总量未超过 SM 上限）
- 使用多 Stream 时，硬件 Scheduler 自动将不同 Block 交错分配到空闲 SM

**Persistent Kernel（持久化 Kernel）**：
- 一个 Kernel 长期驻留 SM，通过全局内存队列接收工作，避免反复启动开销
- 适用于任务粒度极小、提交频率极高的场景

---

## 5.7 并发优化总结

| 优化手段 | 解决的问题 | 适用场景 |
|----------|-----------|----------|
| Pinned Memory + 多 Stream | 计算与 H2D/D2H 传输串行 | 数据流水线处理 |
| Stream Event 同步 | 流间精确依赖控制 | 有数据依赖的多 Stream 任务 |
| CUDA Graph | 高频小 Kernel 的驱动开销 | Inference / 迭代计算 |
| P2P / NVLink | GPU 间通信瓶颈 | 多 GPU 大模型训练 |
| NCCL | 多 GPU 集合通信 | 分布式训练 |
| Persistent Kernel | 细粒度任务的启动开销 | 图遍历、BVH 遍历 |
