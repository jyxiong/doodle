// Level 5: std::map & std::multimap —— 有序键值映射
// 涵盖：map 的增删改查、operator[]、at()、insert/emplace 的区别、
//        结构化绑定遍历、multimap 多值映射、自定义 key 比较

#include <iostream>
#include <map>
#include <string>
#include <vector>

// ============================================================
// 5.1 std::map 基础
//   内部：红黑树，按 key 排序
//   key 唯一；所有操作 O(log n)
// ============================================================
void demo_map_basic() {
    std::cout << "=== 5.1 std::map 基础 ===\n";

    std::map<std::string, int> scores;

    // 三种插入方式
    scores["Alice"] = 90;                            // operator[]：不存在则默认构造后赋值
    scores.insert({"Bob", 85});                      // insert pair
    scores.insert(std::make_pair("Charlie", 92));    // make_pair
    scores.emplace("Diana", 88);                     // emplace：就地构造，最高效

    // operator[] 的副作用：若 key 不存在，会插入值初始化（int 为 0）的元素
    std::cout << "访问不存在的 Eve: " << scores["Eve"] << "\n";
    std::cout << "size 变为: " << scores.size() << "\n";  // 会多出 Eve:0

    // ⚠️ 推荐用 at() 代替 []（不存在时抛出异常，不插入）
    try {
        std::cout << scores.at("Alice") << "\n";
        std::cout << scores.at("Zoe") << "\n";     // 抛出 std::out_of_range
    } catch (const std::out_of_range& e) {
        std::cout << "at() 异常: " << e.what() << "\n";
    }

    // 遍历（C++17 结构化绑定）
    std::cout << "按 key 有序遍历:\n";
    for (const auto& [name, score] : scores)
        std::cout << "  " << name << " -> " << score << "\n";
}

// ============================================================
// 5.2 查找与修改
// ============================================================
void demo_map_find() {
    std::cout << "\n=== 5.2 查找与修改 ===\n";

    std::map<std::string, int> m = {{"a", 1}, {"b", 2}, {"c", 3}};

    // find：推荐方式（不会插入新元素）
    auto it = m.find("b");
    if (it != m.end()) {
        std::cout << "找到 b=" << it->second << "\n";
        it->second = 99;    // 可以修改 value（但不能修改 key）
    }

    // count：0 或 1（map 中 key 唯一）
    std::cout << "count(\"a\")=" << m.count("a") << "\n";

    // contains（C++20）
    // std::cout << m.contains("a") << "\n";

    // 安全的"查找或默认值"模式（不插入）
    auto it2 = m.find("z");
    int val = (it2 != m.end()) ? it2->second : -1;
    std::cout << "查找 z（默认-1）=" << val << "\n";

    // lower_bound / upper_bound：与 set 相同语义
    auto lb = m.lower_bound("b");
    std::cout << "lower_bound(\"b\")=" << lb->first << "\n";
}

// ============================================================
// 5.3 insert vs operator[] vs emplace 的区别
// ============================================================
void demo_insert_vs_emplace() {
    std::cout << "\n=== 5.3 insert / [] / emplace 对比 ===\n";

    std::map<std::string, int> m;

    // insert：只在 key 不存在时插入，返回 pair<iter, bool>
    auto [it1, ok1] = m.insert({"x", 10});
    auto [it2, ok2] = m.insert({"x", 99});  // key 已存在，不覆盖
    std::cout << "insert x=10: " << ok1 << "  再insert x=99: " << ok2
              << "  x=" << m["x"] << "\n";

    // operator[]：若存在则修改，若不存在则插入
    m["y"] = 20;
    m["y"] = 30;  // 覆盖
    std::cout << "[]设置 y=30: " << m["y"] << "\n";

    // insert_or_assign（C++17）：无论是否存在都设置值，返回 pair<iter, bool>
    auto [it3, ok3] = m.insert_or_assign("x", 100);
    std::cout << "insert_or_assign x=100: ok=" << ok3 << " x=" << m["x"] << "\n";

    // try_emplace（C++17）：只在 key 不存在时构造 value，避免不必要的构造
    auto [it4, ok4] = m.try_emplace("z", 40);
    auto [it5, ok5] = m.try_emplace("z", 999);  // 不会覆盖
    std::cout << "try_emplace z=40: " << ok4 << "  再try_emplace z=999: "
              << ok5 << "  z=" << m["z"] << "\n";
}

// ============================================================
// 5.4 删除元素
// ============================================================
void demo_map_erase() {
    std::cout << "\n=== 5.4 删除 ===\n";

    std::map<int, std::string> m = {{1,"a"},{2,"b"},{3,"c"},{4,"d"},{5,"e"}};

    // erase by key
    m.erase(3);

    // erase by iterator
    m.erase(m.find(5));

    // erase by range
    m.erase(m.lower_bound(1), m.upper_bound(2));

    std::cout << "删除后: ";
    for (const auto& [k, v] : m) std::cout << k << ":" << v << " ";
    std::cout << "\n";
}

// ============================================================
// 5.5 std::multimap —— key 可重复的有序映射
// ============================================================
void demo_multimap() {
    std::cout << "\n=== 5.5 std::multimap ===\n";

    std::multimap<std::string, int> mm;

    // 同一 key 对应多个 value
    mm.insert({"apple", 1});
    mm.insert({"apple", 2});
    mm.insert({"banana", 5});
    mm.insert({"apple", 3});

    // ⚠️ multimap 不支持 operator[]（key 不唯一，无法确定取哪个）

    // 遍历所有 apple 的值
    std::cout << "所有 apple: ";
    auto [lo, hi] = mm.equal_range("apple");
    for (auto it = lo; it != hi; ++it)
        std::cout << it->second << " ";
    std::cout << "\n";

    // count
    std::cout << "apple 个数=" << mm.count("apple") << "\n";

    // 按 key 有序遍历全部
    std::cout << "全部:\n";
    for (const auto& [k, v] : mm)
        std::cout << "  " << k << " -> " << v << "\n";
}

// ============================================================
// 5.6 应用：词频统计
// ============================================================
void demo_word_count() {
    std::cout << "\n=== 5.6 应用：词频统计 ===\n";

    std::vector<std::string> words = {
        "the", "fox", "the", "quick", "fox", "the", "brown"
    };

    std::map<std::string, int> freq;
    for (const auto& w : words)
        freq[w]++;  // 不存在时 operator[] 初始化为 0，然后 ++

    for (const auto& [word, cnt] : freq)
        std::cout << word << ": " << cnt << "\n";
}

// ============================================================
// 5.7 自定义 key 类型
//   条件：key 类型需要支持 operator< （或提供自定义比较器）
// ============================================================
struct Point {
    int x, y;
    bool operator<(const Point& o) const {
        return x != o.x ? x < o.x : y < o.y;  // 先按 x，再按 y
    }
};

void demo_custom_key() {
    std::cout << "\n=== 5.7 自定义 key ===\n";

    std::map<Point, std::string> grid;
    grid[{0, 0}] = "origin";
    grid[{1, 2}] = "A";
    grid[{0, 1}] = "B";

    for (const auto& [p, label] : grid)
        std::cout << "(" << p.x << "," << p.y << ")=" << label << "\n";
}

int main() {
    demo_map_basic();
    demo_map_find();
    demo_insert_vs_emplace();
    demo_map_erase();
    demo_multimap();
    demo_word_count();
    demo_custom_key();
    return 0;
}
