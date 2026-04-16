# 四、CUDA Stream 与事件

## 4.1 CUDA Stream 概述

- **Stream** 是 GPU 的任务队列，用于管理核函数、数据拷贝（CPU-GPU、GPU-GPU）等任务
- **核心作用：** 组织任务执行顺序，实现任务并发，提升 GPU 利用率

### 执行规则

| 场景 | 规则 |
|------|------|
| 同一 Stream 内 | 任务按提交顺序**串行执行** |
| 不同 Stream 间 | 若无依赖关系，任务可**并行执行**（由 GPU 硬件调度） |

---

## 4.2 默认流 vs 自定义流

| 对比 | 默认流（Default Stream） | 自定义流（Custom Stream） |
|------|--------------------------|--------------------------|
| 同步行为 | 阻塞同步，与所有流自动同步 | 异步，不与其他流自动同步 |
| 并发能力 | 无法实现并发 | 不同自定义流的任务可并行 |
| 灵活性 | 低 | 高 |

---

## 4.3 多 Stream 并发

**为什么使用多 Stream：**
- 单 Stream 下，数据拷贝（CPU→GPU）与核计算无法并行，SM 会空闲等待
- 多 Stream 可让**数据拷贝与核计算并行**，隐藏数据传输延迟

```
Stream 1: [拷贝 A] -------> [计算 A] -------> [拷贝结果 A]
Stream 2:          [拷贝 B] -------> [计算 B] -------> ...
```

---

## 4.4 Stream 创建与销毁

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);          // 创建自定义 Stream

// 提交任务到 Stream
kernel<<<grid, block, 0, stream>>>(args);
cudaMemcpyAsync(dst, src, size, kind, stream);

cudaStreamDestroy(stream);          // 销毁 Stream（使用完后必须释放）
```

**注意事项：**
- 用完后必须调用 `cudaStreamDestroy()` 释放资源
- 需要同步时，必须手动调用同步函数
- 避免流过多导致调度开销增加

---

## 4.5 Stream 同步方式

| 方式 | 说明 |
|------|------|
| 流内自动串行 | 同一 Stream 内任务自动按顺序执行 |
| `cudaStreamSynchronize(stream)` | 主机等待指定 Stream 执行完成 |
| `cudaStreamWaitEvent(stream, event)` | 让一个 Stream 等待另一个 Stream 的某个 Event |
| `cudaDeviceSynchronize()` | 主机等待所有 Stream 上的所有任务完成 |

---

## 4.6 CUDA Event

**定义：** 用于标记 GPU 任务执行节点的对象。

**核心应用场景：**
1. **精准性能计时**
2. **流间同步**（让一个流等待另一个流的某个任务完成）
3. 检测核函数、数据拷贝的执行状态

### 用 Event 实现精准计时

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);             // 记录开始点
kernel<<<grid, block>>>(args);      // 被计时的操作
cudaEventRecord(stop);              // 记录结束点

cudaEventSynchronize(stop);         // 等待结束点记录完成（必须！）

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);  // 计算时间差

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

**注意事项：**
- 计时前确保 Event 被正确初始化和记录
- **必须调用 `cudaEventSynchronize(stop)`**，否则可能获取到错误时间
- 用完后必须调用 `cudaEventDestroy()` 释放资源

---

## 4.7 多 Stream 数据竞争预防

1. 确保不同 Stream 访问的内存区域**不重叠**
2. 若需共享数据，使用流同步（Event）保证数据访问顺序
3. 用原子操作保护共享变量的读写

---

## 4.8 Stream 异步执行原理

主机提交任务到 Stream 后，**无需等待任务执行完成**，可继续提交下一个任务。GPU 硬件异步执行 Stream 内的任务，主机与 GPU 并行工作，提升整体效率。
