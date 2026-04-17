# CUDA 优化技术体系

> 本目录系统梳理 CUDA 性能优化的核心技术，按优化层次分类组织。

## 文档列表

| 文件 | 内容 |
|------|------|
| [01_memory_optimization.md](01_memory_optimization.md) | 内存层优化：访存合并、共享内存、Bank Conflict、寄存器优化 |
| [02_compute_optimization.md](02_compute_optimization.md) | 计算层优化：Warp Divergence、指令级优化、数值精度 |
| [03_occupancy_latency.md](03_occupancy_latency.md) | 占用率与延迟隐藏：SM 调度、Occupancy 调优 |
| [04_algorithmic_optimization.md](04_algorithmic_optimization.md) | 算法层优化：Kernel Fusion、Reduce、Tiling、双缓冲 |
| [05_concurrency.md](05_concurrency.md) | 并发优化：Stream、Async、流水线 |

## 优化层次速览

```
性能优化
├── 内存层        → 访存合并、共享内存、减少全局内存访问
├── 计算层        → 消除 Warp Divergence、循环展开、指令吞吐
├── 调度层        → 提升 SM 占用率、延迟隐藏
├── 算法层        → Kernel Fusion、高效归约、Tiling
└── 并发层        → 多 Stream、计算与传输重叠
```

## 关键数字参考

| 概念 | 数值 |
|------|------|
| Warp 大小 | 32 threads |
| Block 最大线程数 | 1024 threads |
| 共享内存 Bank 数 | 32 |
| 全局内存典型延迟 | ~300-800 ns |
| 共享内存典型延迟 | ~20-40 ns |
| L1 Cache 典型延迟 | ~30 ns |
| 寄存器典型延迟 | ~1 ns |
| A100 SM 数量 | 108 |
| A100 Warp/SM 上限 | 64 |

## 核心优化原则

1. **最大化内存带宽利用率** — 访存合并是全局内存优化的首要目标
2. **减少全局内存访问次数** — 用共享内存/寄存器缓存复用数据
3. **消除串行化** — Warp Divergence、Bank Conflict、原子操作竞争都会导致串行
4. **保持 SM 持续工作** — 足够的驻留 Warp 数隐藏访存延迟
5. **减少 Kernel 启动开销** — 合并小核，降低调度与驱动开销
6. **重叠计算与传输** — 多 Stream 流水线隐藏 PCIe 传输延迟
