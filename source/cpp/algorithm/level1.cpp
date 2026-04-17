// Level 1: 查找与统计 (Find & Count)
// 涵盖：find, find_if, find_if_not, count, count_if, any_of, all_of, none_of

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// ============================================================
// 1.1 std::find —— 线性查找第一个匹配值
// ============================================================
void demo_find() {
    std::cout << "=== 1.1 std::find ===" << std::endl;

    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};

    // 查找值 5，返回迭代器；未找到则返回 v.end()
    auto it = std::find(v.begin(), v.end(), 5);
    if (it != v.end()) {
        std::cout << "找到 5，位置: " << std::distance(v.begin(), it) << std::endl;
    }

    // 查找不存在的值
    auto it2 = std::find(v.begin(), v.end(), 7);
    std::cout << "查找 7: " << (it2 == v.end() ? "未找到" : "找到") << std::endl;
}

// ============================================================
// 1.2 std::find_if / std::find_if_not —— 按条件查找
// ============================================================
void demo_find_if() {
    std::cout << "\n=== 1.2 std::find_if / find_if_not ===" << std::endl;

    std::vector<int> v = {1, 3, 5, 8, 9, 11};

    // find_if：查找第一个偶数
    auto it = std::find_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
    if (it != v.end()) {
        std::cout << "第一个偶数: " << *it << std::endl;
    }

    // find_if_not：查找第一个不是奇数的元素（等价于 find_if 查找偶数）
    auto it2 = std::find_if_not(v.begin(), v.end(), [](int x) { return x % 2 != 0; });
    if (it2 != v.end()) {
        std::cout << "find_if_not 找到第一个非奇数: " << *it2 << std::endl;
    }
}

// ============================================================
// 1.3 std::count / std::count_if —— 统计元素个数
// ============================================================
void demo_count() {
    std::cout << "\n=== 1.3 std::count / count_if ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 2, 5, 2, 7};

    // count：统计值为 2 的个数
    int n = std::count(v.begin(), v.end(), 2);
    std::cout << "值为 2 的元素个数: " << n << std::endl;

    // count_if：统计大于 3 的元素个数
    int m = std::count_if(v.begin(), v.end(), [](int x) { return x > 3; });
    std::cout << "大于 3 的元素个数: " << m << std::endl;
}

// ============================================================
// 1.4 std::any_of / all_of / none_of —— 范围判断
// ============================================================
void demo_predicates() {
    std::cout << "\n=== 1.4 any_of / all_of / none_of ===" << std::endl;

    std::vector<int> v = {2, 4, 6, 8, 10};

    // any_of：是否存在奇数？
    bool hasOdd = std::any_of(v.begin(), v.end(), [](int x) { return x % 2 != 0; });
    std::cout << "存在奇数: " << std::boolalpha << hasOdd << std::endl;

    // all_of：是否全为偶数？
    bool allEven = std::all_of(v.begin(), v.end(), [](int x) { return x % 2 == 0; });
    std::cout << "全为偶数: " << allEven << std::endl;

    // none_of：是否不含负数？
    bool noNeg = std::none_of(v.begin(), v.end(), [](int x) { return x < 0; });
    std::cout << "不含负数: " << noNeg << std::endl;
}

// ============================================================
// 1.5 std::search —— 查找子序列
// ============================================================
void demo_search() {
    std::cout << "\n=== 1.5 std::search ===" << std::endl;

    std::vector<int> haystack = {1, 2, 3, 4, 5, 6, 7};
    std::vector<int> needle   = {3, 4, 5};

    // search：在 haystack 中查找 needle 子序列
    auto it = std::search(haystack.begin(), haystack.end(),
                          needle.begin(), needle.end());
    if (it != haystack.end()) {
        std::cout << "子序列起始位置: " << std::distance(haystack.begin(), it) << std::endl;
    }

    // 字符串查找
    std::string text = "hello world";
    std::string word = "world";
    auto sit = std::search(text.begin(), text.end(), word.begin(), word.end());
    std::cout << "\"world\" 起始位置: " << std::distance(text.begin(), sit) << std::endl;
}

// ============================================================
// 1.6 std::adjacent_find —— 查找相邻重复元素
// ============================================================
void demo_adjacent_find() {
    std::cout << "\n=== 1.6 std::adjacent_find ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 3, 5, 6};

    auto it = std::adjacent_find(v.begin(), v.end());
    if (it != v.end()) {
        std::cout << "首个相邻重复值: " << *it
                  << "，位置: " << std::distance(v.begin(), it) << std::endl;
    }

    // 自定义判断：查找相邻两个元素之差大于 3 的位置
    auto it2 = std::adjacent_find(v.begin(), v.end(),
                                  [](int a, int b) { return (b - a) > 3; });
    std::cout << "相邻差>3: " << (it2 == v.end() ? "未找到" : "找到") << std::endl;
}

int main() {
    demo_find();
    demo_find_if();
    demo_count();
    demo_predicates();
    demo_search();
    demo_adjacent_find();
    return 0;
}
