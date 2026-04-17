// Level 4: std::set & std::multiset —— 有序集合
// 涵盖：红黑树原理、插入/查找/删除、迭代器有序性、lower_bound/upper_bound、自定义比较器

#include <iostream>
#include <set>
#include <string>

// ============================================================
// 4.1 std::set 基础
//   内部：红黑树（自平衡 BST）
//   特性：元素自动排序（默认升序）、元素唯一、所有操作 O(log n)
// ============================================================
void demo_set_basic() {
    std::cout << "=== 4.1 std::set 基础 ===\n";

    std::set<int> s;

    // insert：插入元素，返回 pair<iterator, bool>
    auto [it1, ok1] = s.insert(5);
    auto [it2, ok2] = s.insert(3);
    auto [it3, ok3] = s.insert(5);  // 重复，插入失败
    std::cout << "插入 5: " << ok1 << "  插入 3: " << ok2
              << "  重复插入 5: " << ok3 << "\n";

    // 批量插入
    s.insert({8, 1, 7, 2, 6});

    // 遍历：始终有序
    std::cout << "有序遍历: ";
    for (int x : s) std::cout << x << " ";
    std::cout << "\n";

    // count：存在返回 1，不存在返回 0（set 中元素唯一）
    std::cout << "count(5)=" << s.count(5) << "  count(99)=" << s.count(99) << "\n";

    // find：返回迭代器，未找到返回 end()
    auto it = s.find(7);
    if (it != s.end())
        std::cout << "找到 7\n";

    // contains（C++20）：更直观的存在性检查
    // std::cout << s.contains(7) << "\n";

    // erase：三种重载
    s.erase(3);                           // 按值删除，O(log n)
    s.erase(s.find(8));                   // 按迭代器删除，O(1)
    s.erase(s.find(1), s.find(7));        // 按范围删除 [1,7)
    std::cout << "删除后: ";
    for (int x : s) std::cout << x << " ";
    std::cout << "\n";

    // size / empty / clear
    std::cout << "size=" << s.size() << "\n";
}

// ============================================================
// 4.2 lower_bound / upper_bound / equal_range
//   这是 set 相比 vector 的核心优势：O(log n) 的范围查询
// ============================================================
void demo_set_bounds() {
    std::cout << "\n=== 4.2 边界查询 ===\n";

    std::set<int> s = {1, 3, 5, 7, 9, 11};
    std::cout << "集合: ";
    for (int x : s) std::cout << x << " ";
    std::cout << "\n";

    // lower_bound(k)：第一个 >= k 的迭代器
    auto lb = s.lower_bound(5);
    std::cout << "lower_bound(5)=" << *lb << "\n";   // 5

    auto lb2 = s.lower_bound(4);
    std::cout << "lower_bound(4)=" << *lb2 << "\n";  // 5

    // upper_bound(k)：第一个 > k 的迭代器
    auto ub = s.upper_bound(5);
    std::cout << "upper_bound(5)=" << *ub << "\n";   // 7

    // 利用两者求 [3, 8) 范围内的元素
    std::cout << "范围 [3,8): ";
    for (auto it = s.lower_bound(3); it != s.upper_bound(7); ++it)
        std::cout << *it << " ";
    std::cout << "\n";

    // equal_range(k)：返回 pair{lower_bound, upper_bound}
    auto [lo, hi] = s.equal_range(5);
    std::cout << "equal_range(5): [" << *lo << ", " << *hi << ")\n";
}

// ============================================================
// 4.3 std::multiset —— 允许重复元素的有序集合
// ============================================================
void demo_multiset() {
    std::cout << "\n=== 4.3 std::multiset ===\n";

    std::multiset<int> ms = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3};

    std::cout << "有序（含重复）: ";
    for (int x : ms) std::cout << x << " ";
    std::cout << "\n";

    // count：返回值出现次数
    std::cout << "count(1)=" << ms.count(1) << "  count(5)=" << ms.count(5) << "\n";

    // erase(val)：删除所有等于 val 的元素
    ms.erase(1);
    std::cout << "erase(1) 后: ";
    for (int x : ms) std::cout << x << " ";
    std::cout << "\n";

    // 只删除一个：先 find 再 erase 迭代器
    auto it = ms.find(5);
    if (it != ms.end()) ms.erase(it);
    std::cout << "只删一个5后: ";
    for (int x : ms) std::cout << x << " ";
    std::cout << "\n";

    // equal_range 取得所有等值元素
    ms.insert({3, 3});
    auto [lo, hi] = ms.equal_range(3);
    std::cout << "所有3的位置个数=" << std::distance(lo, hi) << "\n";
}

// ============================================================
// 4.4 自定义比较器
// ============================================================
struct CaseInsensitiveLess {
    bool operator()(const std::string& a, const std::string& b) const {
        std::string la = a, lb = b;
        for (auto& c : la) c = (char)tolower(c);
        for (auto& c : lb) c = (char)tolower(c);
        return la < lb;
    }
};

void demo_custom_compare() {
    std::cout << "\n=== 4.4 自定义比较器 ===\n";

    // 大小写不敏感的 set
    std::set<std::string, CaseInsensitiveLess> s;
    s.insert("Banana");
    s.insert("apple");
    s.insert("CHERRY");
    s.insert("Apple");   // 与 "apple" 相同，不插入

    std::cout << "大小写不敏感: ";
    for (const auto& x : s) std::cout << x << " ";
    std::cout << "\n";

    // 也可用 lambda（需要 decltype）
    auto cmp = [](int a, int b) { return a > b; };  // 降序
    std::set<int, decltype(cmp)> desc_set(cmp);
    desc_set.insert({3, 1, 4, 1, 5, 9});
    std::cout << "降序 set: ";
    for (int x : desc_set) std::cout << x << " ";
    std::cout << "\n";
}

// ============================================================
// 4.5 set 的迭代器特性
//   - set 的迭代器是双向迭代器（支持 ++ 和 --）
//   - 不支持随机访问（不能 it + 3）
//   - 迭代器指向的值是 const（不能直接修改，否则破坏排序）
// ============================================================
void demo_iterator() {
    std::cout << "\n=== 4.5 迭代器特性 ===\n";

    std::set<int> s = {2, 4, 6, 8, 10};

    // 反向迭代（降序）
    std::cout << "反向遍历: ";
    for (auto it = s.rbegin(); it != s.rend(); ++it)
        std::cout << *it << " ";
    std::cout << "\n";

    // ⚠️ 不能修改元素值（*it = 99 会编译报错）
    // 正确做法：先删除旧值，再插入新值
    auto it = s.find(4);
    if (it != s.end()) {
        int new_val = *it + 1;
        s.erase(it);
        s.insert(new_val);
    }
    std::cout << "把4改为5后: ";
    for (int x : s) std::cout << x << " ";
    std::cout << "\n";
}

int main() {
    demo_set_basic();
    demo_set_bounds();
    demo_multiset();
    demo_custom_compare();
    demo_iterator();
    return 0;
}
