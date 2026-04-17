# 三、占用率与延迟隐藏

SM 占用率（Occupancy）和延迟隐藏（Latency Hiding）是 CUDA 调度层优化的核心，两者密切相关。

---

## 3.1 SM 硬件结构回顾

```
GPU
├── SM 0
│   ├── Warp 调度器 × 4
│   ├── CUDA Core（INT/FP32 单元）
│   ├── 寄存器文件（Register File）
│   ├── 共享内存 / L1 Cache（合并，可配比例）
│   └── LD/ST 单元、SFU 单元
├── SM 1
│   └── ...
└── L2 Cache（所有 SM 共享）
```

关键资源约束（以 A100 为例）：

| 资源 | A100 SM 上限 |
|------|-------------|
| Block / SM | 32 |
| Warp / SM | 64 |
| Thread / SM | 2048 |
| 寄存器 / SM | 65536（每线程最多 255） |
| 共享内存 / SM | 最大 164 KB（可配置） |

---

## 3.2 SM 占用率（Occupancy）

### 定义

$$\text{占用率} = \frac{\text{SM 实际驻留 Warp 数}}{\text{SM 最大支持 Warp 数}} \times 100\%$$

占用率反映 SM 上 **并发 Warp 数量**，是延迟隐藏能力的基础。

### 三大限制因子

**① 寄存器用量**

每个线程使用的寄存器越多，SM 能驻留的线程数越少：

```
SM 寄存器总量 / (每线程寄存器数 × Block 线程数) = SM 能驻留的 Block 数
```

示例（A100，65536 寄存器/SM）：
- 每线程 32 寄存器，Block=256 线程 → 每 Block 消耗 8192 → 最多 8 Block → 64 Warp → 100% 占用率
- 每线程 64 寄存器，Block=256 线程 → 每 Block 消耗 16384 → 最多 4 Block → 32 Warp → 50% 占用率

**② 共享内存用量**

```
SM 共享内存总量 / 每 Block 共享内存消耗 = SM 能驻留的 Block 数
```

示例（48KB 共享内存/SM）：
- 每 Block 用 4KB → 最多 12 Block
- 每 Block 用 24KB → 最多 2 Block

**③ Block 线程数**

Block 线程数必须是 32 的整数倍（否则末尾 Warp 有空线程）。且需足够大以让 SM 驻留足够多 Warp：
- Block=32（1 Warp）：SM 需驻留 64 个 Block 才能填满，但 SM 最多 32 Block → 最多 32 Warp → 50% 占用率
- Block=128（4 Warp）：SM 需驻留 16 Block → 可行，64 Warp → 100% 占用率

### 占用率计算工具

```bash
# 方法一：nvcc 编译时输出寄存器用量
nvcc --ptxas-options=-v kernel.cu

# 方法二：CUDA Occupancy Calculator（Excel 表格，NVIDIA 官网下载）

# 方法三：运行时 API 查询最优 Block Size
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
```

### 占用率的局限性

> 高占用率不等于高性能！

- 计算密集型 kernel：占用率 25% 即可充分利用 CUDA Core，更多 Warp 反而增加资源竞争
- 内存密集型 kernel：需要高占用率来隐藏内存延迟
- 最优占用率需通过 **实际性能测试** 确定，而非单纯追求 100%

---

## 3.3 延迟隐藏（Latency Hiding）

### 原理

GPU 通过**驻留大量 Warp** 来隐藏各种延迟：当一个 Warp 等待内存数据时，调度器**零开销**切换到其他就绪 Warp 继续执行计算。

```
时间轴：
Warp A: [计算] [发出全局内存请求] .........等待 300ns......... [计算]
Warp B:        [计算] [计算] [计算] [发出请求] ......等待...... [计算]
Warp C:               [计算] [计算] [计算] [计算] [发出请求] ...
─────────────────────────────────────────────────────────────────────
SM :   WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW  ← 几乎没有空闲
```

### 需要隐藏多少 Warp？

理论所需 Warp 数 = 延迟（cycle） / 发射间隔（cycle）

- 全局内存延迟 ~300-800 ns，时钟 ~1 GHz → ~300-800 cycle
- 算术指令延迟 ~4-8 cycle，但吞吐受限于发射带宽
- 典型需要 **32-64 个 Warp/SM** 才能充分隐藏全局内存延迟

### 影响延迟隐藏的因素

| 因素 | 影响 |
|------|------|
| SM 驻留 Warp 数（占用率） | 越多越好，隐藏越充分 |
| 指令级并行度（ILP） | 每 Warp 内的独立指令越多，单 Warp 更能自我隐藏 |
| 计算访存比（Arithmetic Intensity） | 比值高时较少 Warp 就能充分隐藏 |
| 访存模式 | 合并访问减少访存次数，也减少需要隐藏的请求数量 |

### 计算密集型 vs 内存密集型

**Roofline Model** 是判断 kernel 瓶颈的核心工具：

$$\text{Arithmetic Intensity (AI)} = \frac{\text{FLOP}}{\text{Bytes}}$$

```
性能 (FLOP/s)
    │
    │              ╱ 计算上限（峰值算力）
    │            ╱
    │          ╱  ← 计算密集区域（AI > 拐点）
    │        ╱
    │      ╱
    │____╱___ ← 内存密集区域（AI < 拐点）
    │
    └──────────────── AI (FLOP/Byte)
```

- **拐点** = 峰值算力 / 峰值内存带宽
- AI < 拐点：内存受限，优化内存带宽利用率
- AI > 拐点：计算受限，优化指令吞吐

---

## 3.4 Block 大小调优实践

### 调优步骤

1. **确定理论限制**：通过 `--ptxas-options=-v` 获取寄存器/共享内存用量
2. **计算理论占用率**：代入公式或使用 CUDA Occupancy Calculator
3. **实测不同 Block 大小的性能**：一般在 128~512 之间测试
4. **用 `ncu` 验证实际瓶颈**：确认是否真正受占用率影响

### 常用 Block 大小建议

| 场景 | Block 大小建议 |
|------|---------------|
| 一维向量化计算 | 256 或 512 |
| 二维矩阵计算 | 16×16=256 或 32×8=256 |
| 归约（Reduce） | 256 或 512 |
| 寄存器消耗极大的 kernel | 128 或更小 |
| 共享内存消耗大的 kernel | 按共享内存上限计算 |

### 共享内存配置（Ampere+）

A100 上共享内存与 L1 Cache 共享 228KB 空间，可按需配置：
```cuda
// 设置 kernel 的共享内存偏向（大共享内存模式）
cudaFuncSetAttribute(myKernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared); // 最大化共享内存
```

---

## 3.5 指令流水线与 ILP

**ILP（指令级并行）**：单个线程内的多条独立指令可以在硬件流水线中并行执行，减少对 TLP（线程级并行）的依赖：

```cuda
// ❌ 链式依赖，无 ILP
float a = x * 2.0f;
float b = a + 1.0f;  // 依赖 a
float c = b * 3.0f;  // 依赖 b

// ✅ 独立计算，编译器可以流水线执行
float a = x * 2.0f;
float b = y + 1.0f;  // 独立于 a
float c = z * 3.0f;  // 独立于 a, b
float result = a + b + c;
```

在寄存器充足时，**适度 ILP（处理多个元素）** 可以在不增加线程数的情况下提升 SM 利用率：
```cuda
// 每线程处理 4 个元素（提升 ILP 和内存带宽利用率）
float4 v = reinterpret_cast<float4*>(data)[tid];
v.x = v.x * scale;
v.y = v.y * scale;
v.z = v.z * scale;
v.w = v.w * scale;
reinterpret_cast<float4*>(output)[tid] = v;
```
