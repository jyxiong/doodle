# 3D Gaussian Splatting — 从零开始的分步 Tutorial

3D Gaussian Splatting（3DGS）是 2023 年 SIGGRAPH Best Paper。
它用一组 **3D 高斯球** 表示场景，通过 **可微分光栅化** 从多视角照片优化，
最终实现实时、高质量的新视角合成。

本 Tutorial 参照 **CUDA Reduce** 的组织方式，逐步拆解 3DGS 的每个核心模块：
每个版本文件独立可运行，可逐个编译验证，概念由浅入深。

---

## 学习路线

```
3dgs_v0   3D 高斯数学基础        ← 从这里开始
3dgs_v1   相机投影 & EWA Splatting
3dgs_v2   球谐函数颜色
3dgs_v3   CPU Alpha 合成渲染器   ← 端到端打通，输出 PPM 图像
3dgs_v4   CUDA Tile-based 光栅化 ← GPU 并行加速
```

---

## 各步骤概览

### V0 — 3D 高斯数学基础

**核心问题：一个 "高斯球" 在数学上长什么样？**

```
G(x) = exp( -½ · (x-μ)ᵀ Σ⁻¹ (x-μ) )
```

- `μ ∈ ℝ³`：高斯球的中心位置
- `Σ ∈ ℝ³×³`：协方差矩阵，决定 **大小、方向、椭球形状**

**关键推导：**
```
Σ = R · S · Sᵀ · Rᵀ
```
其中 `R ∈ SO(3)` 由四元数生成，`S = diag(sx, sy, sz)` 是缩放。

**本文件演示：**
- 从四元数 + 缩放 → 构建 Σ
- 对 Σ 求逆
- 沿 x/y/z 轴采样 G(x)，打印剖面图

---

### V1 — 相机投影 & EWA Splatting

**核心问题：3D 高斯如何映射到 2D 屏幕？**

透视投影是非线性的，对高斯做精确映射需要积分。
3DGS 采用 **EWA（Elliptical Weighted Average）** 近似：

```
Σ' = J · W · Σ · Wᵀ · Jᵀ  （取 2×2 左上角）
```

- `W`：相机旋转矩阵（世界 → 相机）
- `J`：透视投影的局部 Jacobian（在高斯中心处线性化）

**本文件演示：**
- 定义相机内参 `(fx, fy, cx, cy)` 和外参 `[R|t]`
- 将 3D 高斯中心投影到像素坐标
- 计算 2D 协方差 Σ'

---

### V2 — 球谐函数颜色

**核心问题：高斯球的颜色如何随视角变化？**

3DGS 用 **球谐函数（Spherical Harmonics, SH）** 编码依赖视角的颜色：

```
C(d) = Σ_{l=0}^{L} Σ_{m=-l}^{l}  c_{lm} · Y_l^m(d)
```

- `d`：单位视角方向向量
- `Y_l^m(d)`：SH 基函数（实数形式）
- `c_{lm}`：每个基函数对应 RGB 三个通道的系数

常用 **0~3 阶**（共 16 个基函数），每个高斯存 48 个系数（16 × RGB 3 通道）。

**本文件演示：**
- 0~3 阶 SH 基函数实现
- 从 SH 系数 + 视角方向计算 RGB 颜色
- 演示颜色随视角旋转的变化

---

### V3 — CPU Alpha 合成渲染器

**核心问题：如何将所有高斯混合成一幅图像？**

3DGS 使用 **前向 Alpha Blending（front-to-back）**：

```
C_final = Σ_i  c_i · α_i · T_i

T_i = Π_{j<i} (1 - α_j)         （累积透明度）
```

其中每个高斯的贡献 `α_i` 由 2D 高斯在像素处的响应决定：

```
α_i = opacity_i · exp( -½ · Δxᵀ (Σ')⁻¹ Δx )
```

**本文件演示（端到端 CPU 版本）：**
- 构建包含若干高斯的简单场景
- 对每个像素按深度排序高斯，依次 Alpha Blend
- 输出 **PPM 图像文件**，用图像查看器验证

---

### V4 — CUDA Tile-based 光栅化

**核心问题：如何在 GPU 上高效并行渲染？**

原始 3DGS 论文的 GPU 管线分 **三个 Pass**：

```
Pass 1 — Preprocess（每个 Gaussian 一个线程）
  ├─ 世界 → 相机坐标变换
  ├─ 透视投影，得到 2D 中心 & 深度
  ├─ EWA 计算 2D 协方差
  ├─ 计算 2D AABB，找出覆盖的 Tile 范围
  ├─ 求 SH 颜色
  └─ 输出：(tile_id, depth) 对，供排序

Pass 2 — Radix Sort（按 tile_id | depth 排序）
  └─ 使 同一 Tile 内的 Gaussian 按深度连续

Pass 3 — Forward Rendering（每个 Tile 一个 Block）
  ├─ blockDim = (16, 16)，每个线程处理一个像素
  ├─ 所有线程协作地从共享内存批量加载 Gaussian 数据
  └─ 每个像素 front-to-back alpha blend，直到不透明度饱和
```

**本文件演示：**
- Pass 1 Preprocess Kernel
- Pass 2 Key-Value Sort（使用 `thrust::sort_by_key`）
- Pass 3 Forward Rendering Kernel
- 输出 **PPM 图像文件**

---

## 目录结构

```
source/3dgs/
├── README.md           ← 本文件
├── CMakeLists.txt
└── tutorial/
    ├── utils.cuh            公共数学工具（Mat3、四元数、相机…）
    ├── 3dgs_v0.cu           3D 高斯数学基础
    ├── 3dgs_v1.cu           相机投影 & EWA Splatting
    ├── 3dgs_v2.cu           球谐函数颜色
    ├── 3dgs_v3.cu           CPU 端到端渲染器
    └── 3dgs_v4.cu           CUDA Tile-based 光栅化
```

---

## 推荐阅读

- 原始论文：[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- EWA Splatting：Zwicker et al., 2001
- 球谐函数入门：[Spherical Harmonic Lighting](https://3dvar.com/Green2003Spherical.pdf)
- CUDA 优化：参考本项目 `source/cuda/reduce/` tutorial
