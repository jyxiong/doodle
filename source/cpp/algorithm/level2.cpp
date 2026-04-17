// Level 2: 变换与填充 (Transform & Fill)
// 涵盖：for_each, transform, fill, fill_n, generate, generate_n, iota

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

void print(const std::vector<int>& v, const std::string& label = "") {
    if (!label.empty()) std::cout << label << ": ";
    for (int x : v) std::cout << x << " ";
    std::cout << std::endl;
}

// ============================================================
// 2.1 std::for_each —— 对每个元素执行操作（不修改容器结构）
// ============================================================
void demo_for_each() {
    std::cout << "=== 2.1 std::for_each ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5};

    // 遍历打印
    std::for_each(v.begin(), v.end(), [](int x) {
        std::cout << x << " ";
    });
    std::cout << std::endl;

    // 通过引用修改元素（就地乘以 2）
    std::for_each(v.begin(), v.end(), [](int& x) { x *= 2; });
    print(v, "乘以 2 后");
}

// ============================================================
// 2.2 std::transform —— 将元素映射到新值（写入目标范围）
// ============================================================
void demo_transform() {
    std::cout << "\n=== 2.2 std::transform ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5};
    std::vector<int> result(v.size());

    // 一元变换：每个元素平方
    std::transform(v.begin(), v.end(), result.begin(),
                   [](int x) { return x * x; });
    print(result, "平方");

    // 二元变换：两个向量对应元素相加
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {10, 20, 30};
    std::vector<int> sum(3);
    std::transform(a.begin(), a.end(), b.begin(), sum.begin(),
                   [](int x, int y) { return x + y; });
    print(sum, "a+b");

    // 原地变换（输出迭代器 == 输入迭代器）
    std::transform(v.begin(), v.end(), v.begin(),
                   [](int x) { return x * 10; });
    print(v, "原地 *10");
}

// ============================================================
// 2.3 std::fill / std::fill_n —— 填充固定值
// ============================================================
void demo_fill() {
    std::cout << "\n=== 2.3 std::fill / fill_n ===" << std::endl;

    std::vector<int> v(6);

    // fill：把整个范围设为同一值
    std::fill(v.begin(), v.end(), 7);
    print(v, "fill(7)");

    // fill_n：从起点填充 n 个元素
    std::fill_n(v.begin(), 3, 0);
    print(v, "fill_n(3, 0)");
}

// ============================================================
// 2.4 std::generate / std::generate_n —— 用函数生成值
// ============================================================
void demo_generate() {
    std::cout << "\n=== 2.4 std::generate / generate_n ===" << std::endl;

    std::vector<int> v(6);

    // generate：每次调用生成器填充
    int counter = 0;
    std::generate(v.begin(), v.end(), [&counter]() { return counter++; });
    print(v, "generate (0..5)");

    // generate_n：只填充前 4 个
    std::generate_n(v.begin(), 4, []() { return 42; });
    print(v, "generate_n(4, 42)");
}

// ============================================================
// 2.5 std::iota —— 填充连续递增序列（在 <numeric> 中）
// ============================================================
void demo_iota() {
    std::cout << "\n=== 2.5 std::iota ===" << std::endl;

    std::vector<int> v(8);

    // 从 1 开始递增填充
    std::iota(v.begin(), v.end(), 1);
    print(v, "iota(1)");

    // 常见用法：生成索引数组后按规则排序
    std::vector<int> data = {50, 10, 40, 20, 30};
    std::vector<int> idx(data.size());
    std::iota(idx.begin(), idx.end(), 0);                       // 0,1,2,3,4
    std::sort(idx.begin(), idx.end(),
              [&data](int a, int b) { return data[a] < data[b]; });
    std::cout << "data 排序后的索引: ";
    for (int i : idx) std::cout << i << " ";
    std::cout << std::endl;
}

int main() {
    demo_for_each();
    demo_transform();
    demo_fill();
    demo_generate();
    demo_iota();
    return 0;
}
