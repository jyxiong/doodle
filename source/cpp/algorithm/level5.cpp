// Level 5: 二分查找 (Binary Search on Sorted Range)
// 涵盖：lower_bound, upper_bound, binary_search, equal_range
// 前提：所有操作要求范围已按同一比较器有序！

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
// 5.1 std::lower_bound —— 第一个 >= value 的位置
// ============================================================
void demo_lower_bound() {
    std::cout << "=== 5.1 std::lower_bound ===" << std::endl;

    std::vector<int> v = {1, 2, 4, 4, 4, 7, 9};
    print(v, "有序数组");

    // 查找第一个 >= 4 的位置
    auto it = std::lower_bound(v.begin(), v.end(), 4);
    std::cout << "lower_bound(4) 下标: " << std::distance(v.begin(), it)
              << "，值: " << *it << std::endl;

    // 查找不存在的值：第一个 >= 5 的位置（即 7 的位置）
    auto it2 = std::lower_bound(v.begin(), v.end(), 5);
    std::cout << "lower_bound(5) 下标: " << std::distance(v.begin(), it2)
              << "，值: " << *it2 << std::endl;

    // 应用：在有序数组中插入元素并保持有序
    auto pos = std::lower_bound(v.begin(), v.end(), 6);
    v.insert(pos, 6);
    print(v, "插入 6 后");
}

// ============================================================
// 5.2 std::upper_bound —— 第一个 > value 的位置
// ============================================================
void demo_upper_bound() {
    std::cout << "\n=== 5.2 std::upper_bound ===" << std::endl;

    std::vector<int> v = {1, 2, 4, 4, 4, 7, 9};

    // 查找第一个 > 4 的位置
    auto it = std::upper_bound(v.begin(), v.end(), 4);
    std::cout << "upper_bound(4) 下标: " << std::distance(v.begin(), it)
              << "，值: " << *it << std::endl;

    // lower_bound 和 upper_bound 结合可得到等值范围
    auto lo = std::lower_bound(v.begin(), v.end(), 4);
    auto hi = std::upper_bound(v.begin(), v.end(), 4);
    std::cout << "值为 4 的元素个数: " << std::distance(lo, hi) << std::endl;
}

// ============================================================
// 5.3 std::binary_search —— 判断值是否存在（只返回 bool）
// ============================================================
void demo_binary_search() {
    std::cout << "\n=== 5.3 std::binary_search ===" << std::endl;

    std::vector<int> v = {1, 2, 4, 4, 7, 9};

    std::cout << "4 存在: " << std::boolalpha << std::binary_search(v.begin(), v.end(), 4) << std::endl;
    std::cout << "5 存在: " << std::binary_search(v.begin(), v.end(), 5) << std::endl;

    // 使用自定义比较器（数组必须按相同比较器有序）
    std::vector<int> desc = {9, 7, 4, 4, 2, 1};
    bool found = std::binary_search(desc.begin(), desc.end(), 4, std::greater<int>());
    std::cout << "降序数组中查找 4: " << found << std::endl;
}

// ============================================================
// 5.4 std::equal_range —— 同时返回 lower_bound 和 upper_bound
// ============================================================
void demo_equal_range() {
    std::cout << "\n=== 5.4 std::equal_range ===" << std::endl;

    std::vector<int> v = {1, 2, 4, 4, 4, 7, 9};

    auto [lo, hi] = std::equal_range(v.begin(), v.end(), 4);
    std::cout << "等值范围: [" << std::distance(v.begin(), lo)
              << ", " << std::distance(v.begin(), hi) << ")" << std::endl;
    std::cout << "4 出现次数: " << std::distance(lo, hi) << std::endl;

    // 未找到时：lo == hi（空区间）
    auto [lo2, hi2] = std::equal_range(v.begin(), v.end(), 5);
    std::cout << "5 出现次数: " << std::distance(lo2, hi2) << std::endl;
}

// ============================================================
// 5.5 综合示例：在结构体有序向量上使用二分查找
// ============================================================
void demo_struct_binary_search() {
    std::cout << "\n=== 5.5 结构体二分查找 ===" << std::endl;

    struct Record {
        int id;
        std::string name;
    };

    std::vector<Record> records = {
        {1, "Alice"}, {3, "Bob"}, {5, "Carol"}, {7, "Dave"}, {9, "Eve"}
    };
    // 已按 id 升序排列

    int target_id = 5;

    // 使用 lower_bound 配合 lambda 比较器
    auto it = std::lower_bound(records.begin(), records.end(), target_id,
                               [](const Record& r, int id) { return r.id < id; });

    if (it != records.end() && it->id == target_id) {
        std::cout << "找到 id=" << target_id << "，name=" << it->name << std::endl;
    } else {
        std::cout << "未找到 id=" << target_id << std::endl;
    }
}

int main() {
    demo_lower_bound();
    demo_upper_bound();
    demo_binary_search();
    demo_equal_range();
    demo_struct_binary_search();
    return 0;
}
