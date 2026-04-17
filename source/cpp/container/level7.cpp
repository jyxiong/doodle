// Level 7: std::array & std::span & 容器综合选型
// 涵盖：std::array 固定大小数组、std::span 非拥有视图（C++20）、
//        容器适配场景总结、move 语义对容器性能的影响

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <span>      // C++20
#include <string>
#include <tuple>
#include <vector>

// ============================================================
// 7.1 std::array —— 固定大小数组（大小是编译期常量）
//   内部：在栈上分配（或作为类成员），无堆分配开销
//   优点：大小固定、不会扩容、支持聚合初始化、可以拷贝
//   缺点：大小必须编译期确定
// ============================================================
void demo_array() {
    std::cout << "=== 7.1 std::array ===\n";

    std::array<int, 5> arr = {3, 1, 4, 1, 5};

    // 全部 std::vector 风格的接口
    std::cout << "size=" << arr.size() << "\n";
    std::cout << "front=" << arr.front() << "  back=" << arr.back() << "\n";
    std::cout << "arr[2]=" << arr[2] << "  arr.at(2)=" << arr.at(2) << "\n";

    // 排序（支持随机迭代器，可以用 std::sort）
    std::sort(arr.begin(), arr.end());
    std::cout << "排序后: ";
    for (int x : arr) std::cout << x << " ";
    std::cout << "\n";

    // fill：全部填充同一值
    arr.fill(0);
    std::cout << "fill(0): ";
    for (int x : arr) std::cout << x << " ";
    std::cout << "\n";

    // data()：获得原始指针（内存连续，可传给 C API）
    int* ptr = arr.data();
    ptr[0] = 99;
    std::cout << "arr[0] via data pointer=" << arr[0] << "\n";

    // 拷贝赋值（C 风格数组不支持，std::array 支持）
    std::array<int, 5> arr2 = arr;
    arr2[0] = 0;
    std::cout << "arr[0]=" << arr[0] << "  arr2[0]=" << arr2[0] << "\n";

    // std::get<I>：编译期常量索引
    std::cout << "std::get<3>(arr)=" << std::get<3>(arr) << "\n";
}

// ============================================================
// 7.2 多维 std::array
// ============================================================
void demo_array_2d() {
    std::cout << "\n=== 7.2 二维 std::array ===\n";

    // 3x4 的矩阵，完全在栈上
    std::array<std::array<int, 4>, 3> mat = {{{1,2,3,4},{5,6,7,8},{9,10,11,12}}};

    for (const auto& row : mat) {
        for (int x : row) std::cout << x << "\t";
        std::cout << "\n";
    }

    // 用 iota 填充
    std::array<int, 10> a;
    std::iota(a.begin(), a.end(), 0);
    std::cout << "iota: ";
    for (int x : a) std::cout << x << " ";
    std::cout << "\n";
}

// ============================================================
// 7.3 std::span —— 非拥有的连续序列视图（C++20）
//   std::span<T> 只持有指针+长度，不拥有内存，不做拷贝
//   适用于：写接受"任意连续容器"的函数，避免重载 vector/array/原始数组
// ============================================================
void sum_and_print(std::span<const int> s, const std::string& label) {
    int total = 0;
    for (int x : s) total += x;
    std::cout << label << " sum=" << total << " data=[";
    for (size_t i = 0; i < s.size(); i++) {
        std::cout << s[i];
        if (i + 1 < s.size()) std::cout << ",";
    }
    std::cout << "]\n";
}

void demo_span() {
    std::cout << "\n=== 7.3 std::span (C++20) ===\n";

    std::vector<int> v = {1, 2, 3, 4, 5};
    std::array<int, 4> a = {6, 7, 8, 9};
    int raw[] = {10, 11, 12};

    // 同一个函数接收三种来源
    sum_and_print(v, "vector");
    sum_and_print(a, "array");
    sum_and_print(raw, "raw array");

    // 子视图（subspan）：取一部分，零拷贝
    std::span<const int> sv(v);
    sum_and_print(sv.subspan(1, 3), "vector[1,4)");

    // 固定大小 span：std::span<T, N>，N 是编译期常量
    std::span<int, 3> fixed_s(raw);
    std::cout << "fixed span size=" << fixed_s.size() << "\n";
}

// ============================================================
// 7.4 移动语义对容器的影响
//   容器元素 move 而非 copy，可大幅减少拷贝开销
// ============================================================
struct Heavy {
    std::vector<int> data;
    std::string name;

    Heavy(std::string n, int size) : name(std::move(n)), data(size, 42) {}

    // 打印构造/拷贝/移动情况
    Heavy(const Heavy& o) : data(o.data), name(o.name) {
        std::cout << "  [COPY] " << name << "\n";
    }
    Heavy(Heavy&& o) noexcept : data(std::move(o.data)), name(std::move(o.name)) {
        std::cout << "  [MOVE] " << name << "\n";
    }
};

void demo_move() {
    std::cout << "\n=== 7.4 移动语义对容器的影响 ===\n";

    std::vector<Heavy> vec;
    vec.reserve(3);  // 预留空间，避免 push_back 触发扩容拷贝

    std::cout << "emplace_back（就地构造，无 copy/move）:\n";
    vec.emplace_back("A", 100);

    std::cout << "push_back lvalue（触发 copy）:\n";
    Heavy h("B", 100);
    vec.push_back(h);

    std::cout << "push_back rvalue（触发 move）:\n";
    vec.push_back(std::move(h));

    std::cout << "vec size=" << vec.size() << "\n";
}

// ============================================================
// 7.5 综合容器选型指南
// ============================================================
void demo_selection_guide() {
    std::cout << "\n=== 7.5 容器选型指南 ===\n";
    std::cout << R"(
  需求                              推荐容器
  ─────────────────────────────────────────────────────────────
  大小固定，存在栈上               std::array<T, N>
  大小动态，大多数情况下            std::vector<T>           ← 首选
  需要频繁头部插删                  std::deque<T>
  在任意位置频繁插删，已有迭代器    std::list<T>
  LIFO（后进先出）                  std::stack<T>
  FIFO（先进先出）                  std::queue<T>
  按优先级出队                      std::priority_queue<T>
  唯一元素，需要有序                std::set<T>
  键值对，需要有序，范围查询        std::map<K,V>
  唯一元素，O(1) 查找，不需要顺序  std::unordered_set<T>
  键值对，O(1) 查找，不需要顺序    std::unordered_map<K,V>
  非拥有连续视图（函数参数）        std::span<T>
  ─────────────────────────────────────────────────────────────
  通用原则：
  1. 默认选 vector，只在有具体性能需求时换
  2. 优先使用 emplace_back / emplace，避免临时对象
  3. 已知大小时用 reserve() 预分配
  4. 用 at() 代替 [] 做边界安全访问
  5. 结构化绑定（C++17）让 map 遍历更清晰：for (auto& [k,v] : m)
)";
}

int main() {
    demo_array();
    demo_array_2d();
    demo_span();
    demo_move();
    demo_selection_guide();
    return 0;
}
