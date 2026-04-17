# C++ 容器 Tutorial

## 学习路线

| Level | 文件 | 主题 |
|-------|------|------|
| 1 | level1.cpp | `std::vector` —— 动态数组 |
| 2 | level2.cpp | `std::deque` / `std::list` / `std::forward_list` —— 链式与双端 |
| 3 | level3.cpp | `std::stack` / `std::queue` / `std::priority_queue` —— 容器适配器 |
| 4 | level4.cpp | `std::set` / `std::multiset` —— 有序集合 |
| 5 | level5.cpp | `std::map` / `std::multimap` —— 有序映射 |
| 6 | level6.cpp | `std::unordered_set` / `std::unordered_map` —— 哈希表 |
| 7 | level7.cpp | `std::array` / `std::span` + 综合选型 |

---

## 各容器特点速查

### std::vector
- **内部结构**：连续堆内存，自动扩容（通常翻倍）
- **随机访问**：O(1)
- **尾部插删**：均摊 O(1)；头/中间插删 O(n)
- **迭代器**：随机迭代器；插入/扩容后迭代器**可能失效**
- **内存**：连续，cache 友好，可用 `data()` 传递给 C API
- **典型用途**：绝大多数场景的默认选择

### std::deque
- **内部结构**：多段固定大小缓冲块 + 中控索引
- **随机访问**：O(1)
- **头尾插删**：O(1)；中间插删 O(n)
- **内存**：不连续，不能用原始指针遍历
- **典型用途**：需要频繁从两端操作（滑动窗口、`stack`/`queue` 的默认底层容器）

### std::list
- **内部结构**：双向链表，每节点独立堆分配
- **随机访问**：O(n)（不支持）
- **任意位置插删**：O(1)（已有迭代器时）；迭代器**永不失效**（除被删节点）
- **额外接口**：`splice`、`sort`（自带）、`merge`、`unique`、`remove_if`
- **内存**：碎片化，cache 不友好
- **典型用途**：需要频繁在中间插删且已持有迭代器，或需要无失效迭代器

### std::forward_list（C++11）
- **内部结构**：单向链表
- **插删**：只能 `insert_after` / `erase_after`；无 `size()`
- **内存**：比 `list` 少一个指针/节点，最省内存的链表
- **典型用途**：内存极度敏感、只需单向遍历

### std::stack
- **语义**：LIFO（后进先出）
- **接口**：`push` / `pop` / `top` / `empty` / `size`
- **底层容器**：默认 `deque`，可指定 `vector` / `list`
- **典型用途**：DFS、表达式求值、括号匹配、函数调用模拟

### std::queue
- **语义**：FIFO（先进先出）
- **接口**：`push` / `pop` / `front` / `back` / `empty` / `size`
- **底层容器**：默认 `deque`，可指定 `list`
- **典型用途**：BFS、任务调度、生产者-消费者缓冲

### std::priority_queue
- **语义**：堆，每次取出优先级最高的元素
- **内部结构**：堆（底层默认 `vector` + `make_heap`）
- **接口**：`push` / `pop` / `top` / `empty` / `size`
- **默认**：最大堆；传 `std::greater<T>` 变最小堆
- **操作复杂度**：`push`/`pop` O(log n)，`top` O(1)
- **典型用途**：Top-K 问题、Dijkstra 最短路、任务优先调度

### std::set / std::multiset
- **内部结构**：红黑树（自平衡 BST）
- **元素**：`set` 唯一；`multiset` 允许重复
- **所有操作**：O(log n)
- **迭代器**：双向迭代器；遍历有序；元素值**不可修改**（修改需先删后插）
- **额外接口**：`lower_bound` / `upper_bound` / `equal_range`（范围查询）
- **典型用途**：维护有序唯一集合、频繁范围查找

### std::map / std::multimap
- **内部结构**：红黑树，按 key 排序
- **key**：`map` 唯一；`multimap` 允许重复
- **所有操作**：O(log n)
- **注意**：`operator[]` 在 key 不存在时会**插入默认值**，建议用 `find` 或 `at()`
- **C++17 新增**：`try_emplace`（避免不必要构造）、`insert_or_assign`（无条件设置）
- **遍历**：结构化绑定 `for (auto& [k, v] : m)` 按 key 升序
- **典型用途**：词频统计、需要按 key 有序遍历或范围查询的键值存储

### std::unordered_set / std::unordered_map
- **内部结构**：哈希表（拉链法）
- **元素/key**：唯一
- **操作复杂度**：平均 O(1)；哈希冲突严重时最坏 O(n)
- **自定义类型**：需要提供 `operator==` + 特化 `std::hash<T>`（或传入函数对象）
- **容量控制**：`reserve` 预分配；`rehash` 手动设置桶数；`load_factor` 监控冲突
- **遍历**：无序
- **典型用途**：O(1) 去重/计数/查找，不关心顺序（两数之和、字符频率等）

### std::array（C++11）
- **内部结构**：固定大小，栈上连续内存（大小是编译期常量）
- **随机访问**：O(1)
- **不支持**：动态增删，无 `push_back` / `resize`
- **优势**：无堆分配开销；支持拷贝赋值（C 风格数组不支持）；`std::get<I>` 编译期索引
- **典型用途**：大小固定的小型数组（颜色 RGBA、矩阵行列等）

### std::span（C++20）
- **语义**：非拥有的连续序列视图（指针 + 长度），不分配/管理内存
- **作用**：统一接收 `vector` / `array` / 原始数组的函数参数，零拷贝
- **接口**：`data` / `size` / `subspan` / 迭代器
- **典型用途**：函数参数类型，替代 `const vector<T>&` 以支持所有连续容器

---

## 性能对比

| 容器 | 随机访问 | 头部插删 | 尾部插删 | 中间插删 | 查找（无序） | 查找（有序/哈希） | 内存连续 |
|------|:--------:|:--------:|:--------:|:--------:|:------------:|:-----------------:|:--------:|
| `vector` | O(1) | O(n) | O(1)* | O(n) | O(n) | — | ✓ |
| `deque` | O(1) | O(1) | O(1) | O(n) | O(n) | — | ✗ |
| `list` | O(n) | O(1) | O(1) | O(1)** | O(n) | — | ✗ |
| `forward_list` | O(n) | O(1) | O(n) | O(1)** | O(n) | — | ✗ |
| `set/map` | — | — | — | O(log n) | — | O(log n) | ✗ |
| `unordered_set/map` | — | — | — | O(1)* | — | O(1)* | ✗ |
| `array` | O(1) | — | — | — | O(n) | — | ✓ |

\* 均摊 O(1)  
\** 前提：已持有该位置的迭代器

---

## 选型决策树

```
需要键值对？
├── 是 → 需要按 key 有序遍历/范围查询？
│         ├── 是 → std::map
│         └── 否 → std::unordered_map   ← 优先（更快）
└── 否 → 只存储值？
          ├── 需要去重？
          │   ├── 需要有序 → std::set
          │   └── 不需要有序 → std::unordered_set
          └── 不需要去重？
              ├── 大小编译期确定 → std::array
              ├── 需要 LIFO → std::stack
              ├── 需要 FIFO → std::queue
              ├── 需要优先级出队 → std::priority_queue
              ├── 频繁头部插删 → std::deque
              ├── 频繁中间插删（已有迭代器）→ std::list
              └── 其他 → std::vector   ← 默认首选
```

---

## 通用最佳实践

1. **默认选 `vector`**，只在有明确性能需求时换其他容器
2. **预分配容量**：知道大致大小时用 `reserve(n)`，避免反复扩容
3. **优先 `emplace_back`** / `emplace`，避免创建临时对象
4. **不要用 `operator[]` 做存在性探测**（会插入默认值），用 `find` 或 `count`
5. **`map::at()`** 比 `operator[]` 更安全（不存在时抛异常而非插入）
6. **C++17 结构化绑定**让 map 遍历更清晰：`for (auto& [k, v] : m)`
7. **`erase-remove` 惯用法**删除 `vector` 中满足条件的元素：
   ```cpp
   v.erase(std::remove_if(v.begin(), v.end(), pred), v.end());
   ```
8. **迭代器安全删除**：`it = container.erase(it)` 而非 `erase(it); ++it`
