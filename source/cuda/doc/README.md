# CUDA 知识体系索引

> 本目录整理自 CUDA 核心知识点，按主题拆分为 5 个文档。

## 文档列表

| 文件 | 内容 |
|------|------|
| [01_thread_model.md](01_thread_model.md) | 线程层级、Warp、Divergence、同步机制 |
| [02_memory_model.md](02_memory_model.md) | 7 种存储器、访存合并、Bank Conflict、缓存策略 |
| [03_scheduling_sm.md](03_scheduling_sm.md) | SM 结构、Block/Warp 调度、占用率、延迟隐藏 |
| [04_stream_event.md](04_stream_event.md) | Stream 并发、Event 计时与同步 |
| [05_primitives_optimization.md](05_primitives_optimization.md) | 原子操作、Kernel Fusion、Block Reduce、双缓冲、循环展开 |

---

## 快速参考

### 关键数字

| 概念 | 数值 |
|------|------|
| Warp 大小 | 32 threads |
| Block 最大线程数 | 1024 threads |
| A100 SM 最大 Warp 数 | 64 |
| A100 SM 最大线程数 | 2048 |
| 共享内存 Bank 数量 | 32 |
| 全局内存典型延迟 | ~数百 ns |
| 共享内存典型延迟 | ~几十 ns |
| 寄存器典型延迟 | ~几 ns |

### 性能优化核心原则

1. **访存合并** — 保证 Warp 内线程访问连续对齐的内存地址
2. **消除 Warp Divergence** — 避免同一 Warp 内线程走不同分支
3. **提升 SM 占用率** — 驻留足够多的 Warp 以隐藏访存延迟
4. **利用共享内存** — 减少全局内存的重复读取
5. **Kernel Fusion** — 合并小核减少全局内存读写和启动开销
6. **多 Stream 并发** — 计算与数据传输重叠，隐藏传输延迟
