// Level 6: 集合操作与堆操作 (Set Operations & Heap)
// 涵盖：merge, set_union, set_intersection, set_difference,
//        set_symmetric_difference, includes,
//        push_heap, pop_heap, make_heap, sort_heap, min_element, max_element, minmax_element

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

void print(const std::vector<int>& v, const std::string& label = "") {
    if (!label.empty()) std::cout << label << ": ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;
}

// ============================================================
// 6.1 std::merge —— 合并两个有序范围
// ============================================================
void demo_merge() {
    std::cout << "=== 6.1 std::merge ===" << std::endl;

    std::vector<int> a = {1, 3, 5, 7};
    std::vector<int> b = {2, 4, 6, 8};
    std::vector<int> result(a.size() + b.size());

    std::merge(a.begin(), a.end(), b.begin(), b.end(), result.begin());
    print(result, "merge(a,b)");
}

// ============================================================
// 6.2 集合运算（要求两个范围均有序）
// ============================================================
void demo_set_operations() {
    std::cout << "\n=== 6.2 集合运算 ===" << std::endl;

    std::vector<int> A = {1, 2, 3, 4, 5};
    std::vector<int> B = {3, 4, 5, 6, 7};
    std::vector<int> out;

    // set_union：A ∪ B
    out.clear();
    std::set_union(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(out));
    print(out, "A∪B");

    // set_intersection：A ∩ B
    out.clear();
    std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(out));
    print(out, "A∩B");

    // set_difference：A - B（在 A 中但不在 B 中）
    out.clear();
    std::set_difference(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(out));
    print(out, "A-B");

    // set_symmetric_difference：A Δ B（只在其中一个集合中）
    out.clear();
    std::set_symmetric_difference(A.begin(), A.end(), B.begin(), B.end(), std::back_inserter(out));
    print(out, "A△B");
}

// ============================================================
// 6.3 std::includes —— 判断子集关系
// ============================================================
void demo_includes() {
    std::cout << "\n=== 6.3 std::includes ===" << std::endl;

    std::vector<int> A = {1, 2, 3, 4, 5, 6};
    std::vector<int> B = {2, 4, 6};
    std::vector<int> C = {2, 4, 7};

    std::cout << "B ⊆ A: " << std::boolalpha
              << std::includes(A.begin(), A.end(), B.begin(), B.end()) << std::endl;
    std::cout << "C ⊆ A: "
              << std::includes(A.begin(), A.end(), C.begin(), C.end()) << std::endl;
}

// ============================================================
// 6.4 堆操作 —— 手动管理最大堆
// ============================================================
void demo_heap() {
    std::cout << "\n=== 6.4 堆操作 ===" << std::endl;

    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // make_heap：将向量变为最大堆（堆顶最大）
    std::make_heap(v.begin(), v.end());
    std::cout << "堆顶(最大): " << v.front() << std::endl;

    // push_heap：先 push_back，再 push_heap
    v.push_back(10);
    std::push_heap(v.begin(), v.end());
    std::cout << "插入 10 后堆顶: " << v.front() << std::endl;

    // pop_heap：将堆顶移到末尾，再 pop_back 删除
    std::pop_heap(v.begin(), v.end());
    int top = v.back();
    v.pop_back();
    std::cout << "弹出堆顶: " << top << std::endl;

    // sort_heap：对堆进行排序（此后不再是合法堆）
    std::sort_heap(v.begin(), v.end());
    print(v, "sort_heap 后");
}

// ============================================================
// 6.5 min/max 系列 —— 求最值
// ============================================================
void demo_minmax() {
    std::cout << "\n=== 6.5 min/max 系列 ===" << std::endl;

    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 单个范围的最小/最大值
    auto it_min = std::min_element(v.begin(), v.end());
    auto it_max = std::max_element(v.begin(), v.end());
    std::cout << "最小值: " << *it_min << "，下标: " << std::distance(v.begin(), it_min) << std::endl;
    std::cout << "最大值: " << *it_max << "，下标: " << std::distance(v.begin(), it_max) << std::endl;

    // minmax_element：同时返回最小和最大（C++11）
    auto [mn, mx] = std::minmax_element(v.begin(), v.end());
    std::cout << "minmax: [" << *mn << ", " << *mx << "]" << std::endl;

    // 标量版 std::clamp（C++17）：将值约束在 [lo, hi]
    int x = 15;
    std::cout << "clamp(15, 1, 10) = " << std::clamp(x, 1, 10) << std::endl;
    std::cout << "clamp(5, 1, 10)  = " << std::clamp(5, 1, 10) << std::endl;
}

int main() {
    demo_merge();
    demo_set_operations();
    demo_includes();
    demo_heap();
    demo_minmax();
    return 0;
}
