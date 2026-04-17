# 一、内存层优化

内存访问效率是 CUDA 性能的核心瓶颈。GPU 的峰值算力远超内存带宽，大多数 kernel 是内存带宽受限（Memory Bound）而非计算受限。

---

## 1.1 GPU 内存层次结构

```
寄存器（Register）
    ↕ ~1 ns
共享内存 / L1 Cache（片上）
    ↕ ~20-40 ns
L2 Cache（片上）
    ↕ ~100-200 ns
全局内存（HBM / GDDR，板上）
    ↕ ~300-800 ns
系统内存（PCIe）
    ↕ 几 μs ~ 几十 μs
```

| 内存类型 | 位置 | 作用域 | 读写 | 延迟 | 带宽 | 典型用途 |
|----------|------|--------|------|------|------|----------|
| 寄存器 | 片上 | 线程私有 | 读写 | ~1 ns | 极高 | 局部变量、中间计算结果 |
| 共享内存 | 片上 | Block 共享 | 读写 | ~20-40 ns | 极高 | Block 内数据交换、缓存复用数据 |
| L1 Cache | 片上 | SM 内 | 只读（自动） | ~30 ns | 高 | 全局内存访问的自动缓存 |
| L2 Cache | 片上 | 全 GPU | 只读（自动） | ~100 ns | 中 | 跨 SM 数据共享缓存 |
| 本地内存 | 板上 | 线程私有 | 读写 | ~300-800 ns | 低 | 寄存器溢出的变量 |
| 全局内存 | 板上 | 所有线程 | 读写 | ~300-800 ns | 低-中 | 输入/输出数据 |
| 常量内存 | 板上 | 所有线程 | 只读 | ~300 ns (miss) / ~1 ns (hit) | — | 广播常量 |
| 纹理内存 | 板上 | 所有线程 | 只读 | 命中时快 | — | 二维/空间局部性访问 |
| 统一内存 | CPU+GPU | 所有线程 | 读写 | 依迁移 | 低 | 简化编程，非高性能路径 |

---

## 1.2 全局内存访问合并（Memory Coalescing）

### 原理

硬件以 **Cache Line（32 或 128 字节）** 为最小数据传输单元。当 Warp 内的 32 个线程发出内存请求时：
- **合并访问**：32 个线程的地址连续且对齐 → 合并为 1 次（或少数几次）事务，带宽利用率 ~100%
- **非合并访问**：地址散乱 → 最多 32 次独立事务，有效带宽降至 1/32

### 合并条件

1. Warp 内线程按线程 ID 顺序访问**连续内存地址**
2. 起始地址对齐到 32 字节（float: 128B cache line 须 128B 对齐）
3. 访问步长为 1（stride-1 access）

### 常见违反模式及修复

**① Stride 访问（跨步访问）**
```cuda
// ❌ stride-2，带宽利用率 ~50%
float val = data[2 * tid];

// ✅ AoS → SoA 数据重排
float val = data[tid];
```

**② 矩阵转置——需要 Shared Memory 中转**
```cuda
// ❌ 写转置矩阵时列访问，非合并
out[col * N + row] = in[row * N + col];

// ✅ 借助共享内存：合并读 → 共享内存转置 → 合并写
__shared__ float tile[TILE][TILE + 1]; // +1 避免 Bank Conflict

int r = blockIdx.y * TILE + threadIdx.y;
int c = blockIdx.x * TILE + threadIdx.x;
tile[threadIdx.y][threadIdx.x] = in[r * N + c];  // 合并读
__syncthreads();
int tr = blockIdx.x * TILE + threadIdx.y;
int tc = blockIdx.y * TILE + threadIdx.x;
out[tr * N + tc] = tile[threadIdx.x][threadIdx.y]; // 合并写
```

**③ 结构体数组（AoS → SoA）**
```cuda
// ❌ AoS：访问 x 字段需跨步
struct Particle { float x, y, z, w; };
Particle particles[N];
float xi = particles[tid].x; // stride-4

// ✅ SoA：每个字段连续存储
float xs[N], ys[N], zs[N], ws[N];
float xi = xs[tid]; // stride-1，完全合并
```

---

## 1.3 共享内存（Shared Memory）

### 使用时机

- 同一 Block 内不同线程访问**相同或相邻**的全局内存数据
- 数据被**多次复用**（如 GEMM 的 Tile、卷积的输入 patch）
- 需要 Block 内线程间**数据交换**（如归约、前缀和）

### 声明方式

```cuda
// 静态分配（编译期确定大小）
__shared__ float smem[BLOCK_SIZE];

// 动态分配（运行时确定，在 kernel 启动参数中指定字节数）
extern __shared__ float smem[];
kernel<<<grid, block, shared_bytes>>>(args);
```

### Tiling 模式（以 GEMM 为例）

```cuda
__global__ void gemm(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < N / TILE; ++k) {
        tileA[threadIdx.y][threadIdx.x] = A[row * N + k * TILE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(k * TILE + threadIdx.y) * N + col];
        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

---

## 1.4 Bank Conflict

### 原理

共享内存被划分为 **32 个独立 Bank**（每个 Bank 宽 4 字节）。Warp 内若多个线程访问**同一 Bank 的不同地址**，访问会串行化（k 路冲突 = k 倍延迟）。

**例外：广播（Broadcast）** — 多线程访问同一 Bank 的**同一地址**，硬件会广播，无冲突。

### Bank 映射规则

```
Bank 编号 = (地址 / 4字节) % 32
```

### 常见冲突场景

**① 列访问二维数组（步长为行宽）**
```cuda
// 若 WIDTH = 32，tile[i][0] 和 tile[j][0] 映射到同一 Bank
__shared__ float tile[32][32];
float val = tile[threadIdx.x][threadIdx.y]; // ❌ 32 路冲突

// ✅ Padding：打破步长规律
__shared__ float tile[32][33]; // 每行多一个 padding 元素
float val = tile[threadIdx.x][threadIdx.y];
```

**② Reduce 时折半步长为 32 的倍数**
```cuda
// ❌ 步长 = 32 时，所有线程访问 Bank 0
for (int s = 32; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
}

// ✅ 改用交错（interleaved）方式或 warp shuffle
```

### 检测工具
- `ncu`（Nsight Compute）→ `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` 指标

---

## 1.5 常量内存与纹理内存

### 常量内存

```cuda
__constant__ float coef[MAX_COEF]; // 声明在文件作用域

// 主机端赋值
cudaMemcpyToSymbol(coef, h_coef, sizeof(h_coef));

// Kernel 内访问（所有线程读相同值时触发广播，效率极高）
float c = coef[i];
```

**适用场景**：卷积滤波器系数、查找表、模型超参数

### 纹理内存

现代 CUDA（Compute Capability 3.5+）推荐使用 **`__ldg()`** 或将指针声明为 `const __restrict__` 触发只读缓存，而非传统纹理 API：

```cuda
__global__ void kernel(const float* __restrict__ input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 编译器会利用只读缓存路径
    output[i] = __ldg(&input[i]) * 2.0f;
}
```

---

## 1.6 寄存器优化

### 寄存器溢出（Register Spilling）

当线程使用的寄存器超过硬件单线程上限时，多余变量溢出到**本地内存**（物理上就是全局内存），性能剧降。

**诊断**：
```bash
nvcc --ptxas-options=-v kernel.cu
# 输出示例：
# ptxas info: Used 64 registers, 256 bytes smem, ...
```

**优化方法**：
1. 减少单线程局部变量数量，合并中间计算
2. 降低 Block 线程数，让每线程可用寄存器增多
3. `--maxrregcount=N` 强制限制寄存器数（可能导致寄存器溢出，需权衡）
4. 将不常用的大数组放共享内存而非局部变量

### 寄存器复用

```cuda
// ❌ 声明过多变量
float a = ..., b = ..., c = ..., d = ...;

// ✅ 计算完立即复用同一变量
float tmp = compute_a();
use(tmp);
tmp = compute_b(); // 复用 tmp
use(tmp);
```

---

## 1.7 内存访问模式总结

| 访问模式 | 建议 |
|----------|------|
| 相邻线程访问相邻地址 | ✅ 理想，充分合并 |
| 相邻线程访问步长 > 1 | ⚠️ 带宽浪费，考虑 SoA 重排 |
| 随机散乱访问全局内存 | ❌ 性能极差，考虑纹理缓存或数据预处理 |
| 多次访问同一全局内存块 | 必须用共享内存缓存 |
| 所有线程读同一常量 | 使用常量内存（Warp 广播） |
| 二维局部性访问 | 使用纹理内存或 `__ldg()` |
