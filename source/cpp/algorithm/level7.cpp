// Level 7: 数值算法 (Numeric Algorithms)
// 涵盖：accumulate, reduce, inner_product, partial_sum,
//        adjacent_difference, exclusive_scan, inclusive_scan (C++17),
//        以及 shuffle、sample、permutation
// 注意：本文件依赖 <numeric>，C++17 特性需 -std=c++17

#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <string>

void print(const std::vector<int>& v, const std::string& label = "") {
    if (!label.empty()) std::cout << label << ": ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;
}

// ============================================================
// 7.1 std::accumulate —— 折叠/归约（顺序执行）
// ============================================================
void demo_accumulate() {
    std::cout << "=== 7.1 std::accumulate ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5};

    // 默认求和
    int sum = std::accumulate(v.begin(), v.end(), 0);
    std::cout << "sum = " << sum << std::endl;

    // 自定义二元操作：求积
    int product = std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>());
    std::cout << "product = " << product << std::endl;

    // 字符串拼接
    std::vector<std::string> words = {"Hello", " ", "World", "!"};
    std::string sentence = std::accumulate(words.begin(), words.end(), std::string());
    std::cout << "sentence = " << sentence << std::endl;
}

// ============================================================
// 7.2 std::reduce (C++17) —— 可并行的归约
// ============================================================
void demo_reduce() {
    std::cout << "\n=== 7.2 std::reduce (C++17) ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5};

    // reduce 与 accumulate 类似，但允许任意执行顺序（适合并行化）
    // 因此操作必须满足交换律和结合律
    int sum = std::reduce(v.begin(), v.end(), 0);
    std::cout << "reduce sum = " << sum << std::endl;

    int product = std::reduce(v.begin(), v.end(), 1, std::multiplies<int>());
    std::cout << "reduce product = " << product << std::endl;
}

// ============================================================
// 7.3 std::inner_product —— 内积（点积）
// ============================================================
void demo_inner_product() {
    std::cout << "\n=== 7.3 std::inner_product ===" << std::endl;

    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};

    // 标准点积：1*4 + 2*5 + 3*6 = 32
    int dot = std::inner_product(a.begin(), a.end(), b.begin(), 0);
    std::cout << "dot(a,b) = " << dot << std::endl;

    // 自定义操作：用加法 + 最大值代替乘积 + 求和
    int custom = std::inner_product(a.begin(), a.end(), b.begin(), 0,
                                    std::plus<int>(),
                                    [](int x, int y) { return std::max(x, y); });
    std::cout << "sum-of-max pairs = " << custom << std::endl;  // max(1,4)+max(2,5)+max(3,6)=15
}

// ============================================================
// 7.4 std::partial_sum —— 前缀和
// ============================================================
void demo_partial_sum() {
    std::cout << "\n=== 7.4 std::partial_sum ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5};
    std::vector<int> prefix(v.size());

    std::partial_sum(v.begin(), v.end(), prefix.begin());
    print(prefix, "前缀和");  // 1 3 6 10 15

    // 自定义操作：前缀积
    std::vector<int> prefix_prod(v.size());
    std::partial_sum(v.begin(), v.end(), prefix_prod.begin(), std::multiplies<int>());
    print(prefix_prod, "前缀积");  // 1 2 6 24 120
}

// ============================================================
// 7.5 std::adjacent_difference —— 相邻差分
// ============================================================
void demo_adjacent_difference() {
    std::cout << "\n=== 7.5 std::adjacent_difference ===" << std::endl;

    std::vector<int> v = {1, 3, 6, 10, 15};
    std::vector<int> diff(v.size());

    std::adjacent_difference(v.begin(), v.end(), diff.begin());
    print(diff, "差分（导数）");  // 1 2 3 4 5

    // adjacent_difference 与 partial_sum 互为逆运算
    std::vector<int> restored(v.size());
    std::partial_sum(diff.begin(), diff.end(), restored.begin());
    print(restored, "还原后");
}

// ============================================================
// 7.6 std::exclusive_scan / inclusive_scan (C++17) —— 扫描
// ============================================================
void demo_scan() {
    std::cout << "\n=== 7.6 exclusive_scan / inclusive_scan (C++17) ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5};
    std::vector<int> out(v.size());

    // inclusive_scan：包含当前元素（等同于 partial_sum）
    std::inclusive_scan(v.begin(), v.end(), out.begin());
    print(out, "inclusive_scan");  // 1 3 6 10 15

    // exclusive_scan：不包含当前元素（错开一位）
    std::exclusive_scan(v.begin(), v.end(), out.begin(), 0);
    print(out, "exclusive_scan");  // 0 1 3 6 10
}

// ============================================================
// 7.7 std::shuffle —— 随机打乱
// ============================================================
void demo_shuffle() {
    std::cout << "\n=== 7.7 std::shuffle ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};
    std::mt19937 rng(42);  // 固定种子，结果可复现

    std::shuffle(v.begin(), v.end(), rng);
    print(v, "shuffle 后");
}

// ============================================================
// 7.8 std::sample (C++17) —— 随机抽样
// ============================================================
void demo_sample() {
    std::cout << "\n=== 7.8 std::sample (C++17) ===" << std::endl;

    std::vector<int> pool = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> sampled;
    std::mt19937 rng(42);

    // 从 pool 中随机抽取 4 个元素，相对顺序保持不变
    std::sample(pool.begin(), pool.end(), std::back_inserter(sampled), 4, rng);
    print(sampled, "sample(4)");
}

// ============================================================
// 7.9 排列 (permutation)
// ============================================================
void demo_permutation() {
    std::cout << "\n=== 7.9 next_permutation / prev_permutation ===" << std::endl;

    std::vector<int> v = {1, 2, 3};

    std::cout << "所有排列:\n";
    do {
        for (int x : v) std::cout << x << " ";
        std::cout << "\n";
    } while (std::next_permutation(v.begin(), v.end()));

    // is_permutation：判断两个序列是否互为排列
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {3, 1, 2};
    std::cout << "a 和 b 互为排列: " << std::boolalpha
              << std::is_permutation(a.begin(), a.end(), b.begin()) << std::endl;
}

int main() {
    demo_accumulate();
    demo_reduce();
    demo_inner_product();
    demo_partial_sum();
    demo_adjacent_difference();
    demo_scan();
    demo_shuffle();
    demo_sample();
    demo_permutation();
    return 0;
}
