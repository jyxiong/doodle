// Level 6: std::unordered_set & std::unordered_map —— 无序哈希容器
// 涵盖：哈希表原理、平均 O(1) 操作、负载因子与 rehash、
//        为自定义类型提供哈希函数、与有序容器的对比

#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>

// ============================================================
// 6.1 std::unordered_set 基础
//   内部：哈希表（拉链法或开放寻址）
//   特性：元素唯一；插入/查找/删除平均 O(1)，最坏 O(n)；元素无序
// ============================================================
void demo_unordered_set() {
    std::cout << "=== 6.1 std::unordered_set ===\n";

    std::unordered_set<int> us = {3, 1, 4, 1, 5, 9, 2, 6};

    // 重复元素被忽略，遍历顺序不确定
    std::cout << "遍历（无序）: ";
    for (int x : us) std::cout << x << " ";
    std::cout << "\n";

    // insert / count / find / erase 接口与 set 相同
    us.insert(7);
    std::cout << "count(9)=" << us.count(9) << "\n";

    auto it = us.find(5);
    if (it != us.end()) std::cout << "找到 5\n";

    us.erase(1);
    std::cout << "erase(1) 后 count(1)=" << us.count(1) << "\n";
}

// ============================================================
// 6.2 容量与哈希统计
// ============================================================
void demo_hash_stats() {
    std::cout << "\n=== 6.2 哈希桶统计 ===\n";

    std::unordered_set<int> us;
    us.reserve(16);  // 预分配桶数，减少 rehash

    for (int i = 0; i < 10; i++) us.insert(i);

    std::cout << "size=" << us.size()
              << "  bucket_count=" << us.bucket_count()
              << "  load_factor=" << us.load_factor()
              << "  max_load_factor=" << us.max_load_factor() << "\n";

    // 查看每个桶里有多少元素（调试哈希冲突用）
    std::cout << "各桶元素数 (非空桶):\n";
    for (size_t i = 0; i < us.bucket_count(); i++) {
        if (us.bucket_size(i) > 0)
            std::cout << "  桶[" << i << "]=" << us.bucket_size(i) << "\n";
    }

    // rehash：手动设置桶数（可能触发重新哈希所有元素）
    us.rehash(64);
    std::cout << "rehash(64) 后 bucket_count=" << us.bucket_count() << "\n";
}

// ============================================================
// 6.3 std::unordered_map 基础
// ============================================================
void demo_unordered_map() {
    std::cout << "\n=== 6.3 std::unordered_map ===\n";

    std::unordered_map<std::string, int> um;

    // 与 map 接口基本相同
    um["apple"]  = 3;
    um["banana"] = 5;
    um.emplace("cherry", 2);
    um.insert({"date", 8});

    // 遍历（无序）
    std::cout << "遍历（无序）:\n";
    for (const auto& [k, v] : um)
        std::cout << "  " << k << "=" << v << "\n";

    // try_emplace / insert_or_assign（C++17）
    um.try_emplace("apple", 999);    // apple 已存在，不修改
    um.insert_or_assign("apple", 100);  // 强制覆盖
    std::cout << "apple=" << um["apple"] << "\n";

    // find 避免不必要的插入
    if (auto it = um.find("banana"); it != um.end())
        std::cout << "banana=" << it->second << "\n";
}

// ============================================================
// 6.4 为自定义类型提供哈希函数
//   方法一：特化 std::hash
//   方法二：传入自定义哈希函数对象
// ============================================================

// 方法一：特化 std::hash
struct Point {
    int x, y;
    bool operator==(const Point& o) const { return x == o.x && y == o.y; }
};

namespace std {
template <>
struct hash<Point> {
    size_t operator()(const Point& p) const {
        // 常用的哈希组合技巧
        size_t hx = std::hash<int>{}(p.x);
        size_t hy = std::hash<int>{}(p.y);
        return hx ^ (hy << 32 | hy >> 32);  // XOR + 位旋转
    }
};
}  // namespace std

void demo_custom_hash() {
    std::cout << "\n=== 6.4 自定义哈希（特化 std::hash） ===\n";

    std::unordered_set<Point> ps;
    ps.insert({0, 0});
    ps.insert({1, 2});
    ps.insert({3, 4});
    ps.insert({1, 2});  // 重复，不插入
    std::cout << "size=" << ps.size() << "\n";

    if (ps.count({1, 2})) std::cout << "找到 (1,2)\n";

    // 方法二：用 lambda 作比较器
    auto hash_fn = [](const Point& p) {
        return std::hash<long long>{}((long long)p.x << 32 | (unsigned)p.y);
    };
    auto eq_fn = [](const Point& a, const Point& b) {
        return a.x == b.x && a.y == b.y;
    };
    std::unordered_set<Point, decltype(hash_fn), decltype(eq_fn)>
        ps2(0, hash_fn, eq_fn);
    ps2.insert({5, 6});
    std::cout << "(5,6) count=" << ps2.count({5, 6}) << "\n";
}

// ============================================================
// 6.5 有序 vs 无序容器选型
// ============================================================
void demo_compare() {
    std::cout << "\n=== 6.5 有序 vs 无序 ===\n";
    std::cout << R"(
  容器              插入      查找      空间  元素顺序  范围查询
  set / map         O(log n)  O(log n)  中    有序      支持 lower/upper_bound
  unordered_set/map O(1)*     O(1)*     高    无序      不支持范围查询
  * 平均；哈希冲突严重时退化为 O(n)

  选 map/set：需要按序遍历、范围查询、key 需要定义有意义的排序
  选 unordered：只需O(1)查找、不关心顺序、key 可以高效哈希
)";
}

// ============================================================
// 6.6 应用：两数之和（哈希表经典题）
// ============================================================
void demo_two_sum() {
    std::cout << "\n=== 6.6 应用：两数之和 ===\n";

    std::vector<int> nums = {2, 7, 11, 15};
    int target = 9;

    std::unordered_map<int, int> seen;  // value -> index
    for (int i = 0; i < (int)nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            std::cout << "结果: [" << seen[complement] << ", " << i << "]\n";
            return;
        }
        seen[nums[i]] = i;
    }
}

int main() {
    demo_unordered_set();
    demo_hash_stats();
    demo_unordered_map();
    demo_custom_hash();
    demo_compare();
    demo_two_sum();
    return 0;
}
