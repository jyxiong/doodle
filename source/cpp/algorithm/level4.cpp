// Level 4: 排序算法 (Sorting)
// 涵盖：sort, stable_sort, partial_sort, nth_element,
//        is_sorted, is_sorted_until, 自定义比较器

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
// 4.1 std::sort —— 不稳定快速排序，O(N log N)
// ============================================================
void demo_sort() {
    std::cout << "=== 4.1 std::sort ===" << std::endl;

    std::vector<int> v = {5, 3, 1, 4, 2};

    // 默认升序
    std::sort(v.begin(), v.end());
    print(v, "升序");

    // 使用 greater<> 降序
    std::sort(v.begin(), v.end(), std::greater<int>());
    print(v, "降序");

    // 自定义比较器（按绝对值排序）
    std::vector<int> v2 = {-4, 1, -3, 2, -5};
    std::sort(v2.begin(), v2.end(), [](int a, int b) {
        return std::abs(a) < std::abs(b);
    });
    print(v2, "按绝对值升序");
}

// ============================================================
// 4.2 std::stable_sort —— 稳定排序（相等元素保持原有相对顺序）
// ============================================================
void demo_stable_sort() {
    std::cout << "\n=== 4.2 std::stable_sort ===" << std::endl;

    struct Item {
        int key;
        std::string name;
    };

    std::vector<Item> items = {{2, "B"}, {1, "A"}, {2, "C"}, {1, "D"}};

    // stable_sort 按 key 排序，相同 key 保持原始顺序
    std::stable_sort(items.begin(), items.end(),
                     [](const Item& a, const Item& b) { return a.key < b.key; });

    std::cout << "stable_sort 后: ";
    for (auto& item : items)
        std::cout << "(" << item.key << "," << item.name << ") ";
    std::cout << std::endl;
    // 预期: (1,A) (1,D) (2,B) (2,C)  —— 相同 key 保持原始相对顺序
}

// ============================================================
// 4.3 std::partial_sort —— 只排序前 k 个最小元素
// ============================================================
void demo_partial_sort() {
    std::cout << "\n=== 4.3 std::partial_sort ===" << std::endl;

    std::vector<int> v = {5, 7, 1, 3, 9, 2, 8, 4, 6};

    // 只将最小的 4 个元素排到前面
    std::partial_sort(v.begin(), v.begin() + 4, v.end());
    print(v, "最小4个已排序（后半部分顺序不定）");
}

// ============================================================
// 4.4 std::nth_element —— 让第 n 个位置放正确元素（快速选择）
// ============================================================
void demo_nth_element() {
    std::cout << "\n=== 4.4 std::nth_element ===" << std::endl;

    std::vector<int> v = {5, 7, 1, 3, 9, 2, 8, 4, 6};

    // 让下标 4 的位置（即第 5 小的元素）归位
    // 左侧元素 <= v[4] <= 右侧元素，但两侧内部无序
    std::nth_element(v.begin(), v.begin() + 4, v.end());
    std::cout << "下标4处的元素（第5小）: " << v[4] << std::endl;
    print(v, "nth_element 后");

    // 常用于求中位数
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());
    std::cout << "中位数: " << data[data.size() / 2] << std::endl;
}

// ============================================================
// 4.5 std::is_sorted / is_sorted_until —— 检查是否有序
// ============================================================
void demo_is_sorted() {
    std::cout << "\n=== 4.5 is_sorted / is_sorted_until ===" << std::endl;

    std::vector<int> v1 = {1, 2, 3, 4, 5};
    std::vector<int> v2 = {1, 2, 5, 3, 4};

    std::cout << "v1 有序: " << std::boolalpha << std::is_sorted(v1.begin(), v1.end()) << std::endl;
    std::cout << "v2 有序: " << std::is_sorted(v2.begin(), v2.end()) << std::endl;

    // is_sorted_until：返回第一个破坏有序性的位置
    auto it = std::is_sorted_until(v2.begin(), v2.end());
    std::cout << "v2 有序前缀长度: " << std::distance(v2.begin(), it) << std::endl;
}

// ============================================================
// 4.6 综合示例：对结构体按多字段排序
// ============================================================
void demo_struct_sort() {
    std::cout << "\n=== 4.6 结构体多字段排序 ===" << std::endl;

    struct Student {
        std::string name;
        int score;
        int age;
    };

    std::vector<Student> students = {
        {"Alice", 90, 20},
        {"Bob",   85, 22},
        {"Carol", 90, 19},
        {"Dave",  85, 21},
    };

    // 先按分数降序，分数相同按年龄升序
    std::stable_sort(students.begin(), students.end(),
        [](const Student& a, const Student& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.age < b.age;
        });

    std::cout << "排序结果:\n";
    for (auto& s : students)
        std::cout << "  " << s.name << " score=" << s.score << " age=" << s.age << "\n";
}

int main() {
    demo_sort();
    demo_stable_sort();
    demo_partial_sort();
    demo_nth_element();
    demo_is_sorted();
    demo_struct_sort();
    return 0;
}
