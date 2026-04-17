// Level 3: 复制、移除与替换 (Copy, Remove & Replace)
// 涵盖：copy, copy_if, copy_n, move, remove, remove_if,
//        unique, replace, replace_if, reverse, rotate

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
// 3.1 std::copy / copy_if / copy_n —— 复制元素到目标范围
// ============================================================
void demo_copy() {
    std::cout << "=== 3.1 std::copy / copy_if / copy_n ===" << std::endl;

    std::vector<int> src = {1, 2, 3, 4, 5, 6};
    std::vector<int> dst(src.size());

    // copy：全量复制
    std::copy(src.begin(), src.end(), dst.begin());
    print(dst, "copy");

    // copy_if：只复制偶数
    std::vector<int> evens;
    std::copy_if(src.begin(), src.end(), std::back_inserter(evens),
                 [](int x) { return x % 2 == 0; });
    print(evens, "copy_if(偶数)");

    // copy_n：只复制前 3 个
    std::vector<int> first3(3);
    std::copy_n(src.begin(), 3, first3.begin());
    print(first3, "copy_n(3)");

    // copy_backward：从后向前复制（通常用于区间重叠向右移动）
    std::vector<int> v = {1, 2, 3, 4, 5};
    std::copy_backward(v.begin(), v.begin() + 3, v.end());
    print(v, "copy_backward 右移后");
}

// ============================================================
// 3.2 std::move（范围版）—— 移动语义批量转移
// ============================================================
void demo_move_range() {
    std::cout << "\n=== 3.2 范围 std::move ===" << std::endl;

    std::vector<std::string> src = {"hello", "world", "foo"};
    std::vector<std::string> dst(3);

    // 移动后 src 中元素处于"有效但未指定"状态
    std::move(src.begin(), src.end(), dst.begin());
    std::cout << "dst: ";
    for (auto& s : dst) std::cout << "\"" << s << "\" ";
    std::cout << std::endl;
    std::cout << "src[0] 移动后: \"" << src[0] << "\"" << std::endl;
}

// ============================================================
// 3.3 std::remove / remove_if —— 逻辑删除（erase-remove 惯用法）
// ============================================================
void demo_remove() {
    std::cout << "\n=== 3.3 std::remove / remove_if ===" << std::endl;

    // remove 不实际缩短容器，返回新逻辑末尾
    std::vector<int> v = {1, 3, 2, 3, 4, 3, 5};

    // 移除所有值为 3 的元素
    auto new_end = std::remove(v.begin(), v.end(), 3);
    v.erase(new_end, v.end());   // 配合 erase 真正删除
    print(v, "remove(3)后");

    // remove_if：移除所有负数
    std::vector<int> v2 = {1, -2, 3, -4, 5};
    auto ne2 = std::remove_if(v2.begin(), v2.end(), [](int x) { return x < 0; });
    v2.erase(ne2, v2.end());
    print(v2, "remove_if(负数)后");
}

// ============================================================
// 3.4 std::unique —— 去除相邻重复（需先排序才能去全部重复）
// ============================================================
void demo_unique() {
    std::cout << "\n=== 3.4 std::unique ===" << std::endl;

    std::vector<int> v = {1, 1, 2, 3, 3, 3, 4, 5, 5};

    // unique 将相邻重复元素"覆盖"，返回新逻辑末尾
    auto new_end = std::unique(v.begin(), v.end());
    v.erase(new_end, v.end());
    print(v, "unique 后");

    // 先排序再 unique，可去掉所有重复值
    std::vector<int> v2 = {3, 1, 4, 1, 5, 9, 2, 6, 5};
    std::sort(v2.begin(), v2.end());
    v2.erase(std::unique(v2.begin(), v2.end()), v2.end());
    print(v2, "sort+unique 去重后");
}

// ============================================================
// 3.5 std::replace / replace_if —— 替换元素
// ============================================================
void demo_replace() {
    std::cout << "\n=== 3.5 std::replace / replace_if ===" << std::endl;

    std::vector<int> v = {1, 0, 3, 0, 5, 0};

    // replace：把所有 0 换成 -1
    std::replace(v.begin(), v.end(), 0, -1);
    print(v, "replace(0→-1)");

    // replace_if：把所有大于 3 的值替换为 99
    std::replace_if(v.begin(), v.end(), [](int x) { return x > 3; }, 99);
    print(v, "replace_if(>3→99)");
}

// ============================================================
// 3.6 std::reverse / rotate —— 翻转与旋转
// ============================================================
void demo_reverse_rotate() {
    std::cout << "\n=== 3.6 std::reverse / rotate ===" << std::endl;

    std::vector<int> v = {1, 2, 3, 4, 5};

    // reverse：就地翻转
    std::reverse(v.begin(), v.end());
    print(v, "reverse");

    // rotate：将 [first, middle) 和 [middle, last) 互换
    // 把 {5,4,3,2,1} 向左旋转 2 位 → {3,2,1,5,4}
    std::rotate(v.begin(), v.begin() + 2, v.end());
    print(v, "rotate(左移2)");
}

int main() {
    demo_copy();
    demo_move_range();
    demo_remove();
    demo_unique();
    demo_replace();
    demo_reverse_rotate();
    return 0;
}
