// Level 1: std::vector —— 动态数组
// 涵盖：构造方式、增删改查、迭代器、容量管理、常见陷阱

#include <algorithm>
#include <iostream>
#include <vector>

// ============================================================
// 辅助：打印 vector
// ============================================================
void print(const std::vector<int>& v, const std::string& label = "") {
    if (!label.empty()) std::cout << label << ": ";
    for (int x : v) std::cout << x << " ";
    std::cout << "\n";
}

// ============================================================
// 1.1 构造方式
// ============================================================
void demo_construct() {
    std::cout << "=== 1.1 构造方式 ===\n";

    std::vector<int> v1;                        // 空 vector
    std::vector<int> v2(5, 0);                  // 5 个 0
    std::vector<int> v3 = {1, 2, 3, 4, 5};     // 初始化列表
    std::vector<int> v4(v3.begin(), v3.end());  // 迭代器范围
    std::vector<int> v5(v3);                    // 拷贝构造
    std::vector<int> v6(std::move(v5));         // 移动构造（v5 之后为空）

    print(v2, "v2(5,0)");
    print(v3, "v3 列表");
    print(v4, "v4 从迭代器");
    print(v6, "v6 移动自v5");
    std::cout << "v5 移动后 size=" << v5.size() << "\n";
}

// ============================================================
// 1.2 增加元素
// ============================================================
void demo_insert() {
    std::cout << "\n=== 1.2 增加元素 ===\n";

    std::vector<int> v;

    // push_back：尾部追加（最常用，均摊 O(1)）
    v.push_back(10);
    v.push_back(20);
    v.push_back(30);
    print(v, "push_back 后");

    // emplace_back：就地构造，避免临时对象（C++11）
    v.emplace_back(40);
    print(v, "emplace_back 后");

    // insert：在指定位置插入（O(n)，插入后该位置之后的元素后移）
    auto it = v.begin() + 1;          // 指向第二个元素
    v.insert(it, 99);                 // 在位置 1 插入 99
    print(v, "insert(pos=1, 99)");

    // insert 范围
    std::vector<int> extra = {7, 8};
    v.insert(v.end(), extra.begin(), extra.end());
    print(v, "insert 范围后");
}

// ============================================================
// 1.3 删除元素
// ============================================================
void demo_erase() {
    std::cout << "\n=== 1.3 删除元素 ===\n";

    std::vector<int> v = {1, 2, 3, 4, 5, 6};

    // pop_back：删除末尾元素（O(1)）
    v.pop_back();
    print(v, "pop_back 后");

    // erase(pos)：删除指定位置（O(n)，之后元素前移）
    v.erase(v.begin() + 1);  // 删除索引 1 的元素（值 2）
    print(v, "erase(pos=1)");

    // erase(first, last)：删除区间 [first, last)
    v.erase(v.begin() + 1, v.begin() + 3);
    print(v, "erase([1,3)) 后");

    // 惯用法：erase-remove 删除所有满足条件的元素
    std::vector<int> v2 = {1, 2, 3, 4, 5, 6};
    v2.erase(std::remove_if(v2.begin(), v2.end(),
                            [](int x) { return x % 2 == 0; }),
             v2.end());
    print(v2, "删除所有偶数");
}

// ============================================================
// 1.4 访问元素
// ============================================================
void demo_access() {
    std::cout << "\n=== 1.4 访问元素 ===\n";

    std::vector<int> v = {10, 20, 30, 40, 50};

    // operator[]：不做边界检查，越界是未定义行为
    std::cout << "v[2]=" << v[2] << "\n";

    // at()：做边界检查，越界抛出 std::out_of_range
    std::cout << "v.at(2)=" << v.at(2) << "\n";

    // front / back
    std::cout << "front=" << v.front() << "  back=" << v.back() << "\n";

    // data()：获取底层原始指针（可传给 C API）
    int* ptr = v.data();
    std::cout << "data[0]=" << ptr[0] << "\n";

    // 范围 for（推荐）
    std::cout << "范围for: ";
    for (const auto& x : v) std::cout << x << " ";
    std::cout << "\n";
}

// ============================================================
// 1.5 迭代器
// ============================================================
void demo_iterator() {
    std::cout << "\n=== 1.5 迭代器 ===\n";

    std::vector<int> v = {1, 2, 3, 4, 5};

    // 正向迭代
    std::cout << "正向: ";
    for (auto it = v.begin(); it != v.end(); ++it)
        std::cout << *it << " ";
    std::cout << "\n";

    // 反向迭代
    std::cout << "反向: ";
    for (auto it = v.rbegin(); it != v.rend(); ++it)
        std::cout << *it << " ";
    std::cout << "\n";

    // const 迭代器（只读）
    std::cout << "const迭代器: ";
    for (auto it = v.cbegin(); it != v.cend(); ++it)
        std::cout << *it << " ";
    std::cout << "\n";

    // ⚠️ 陷阱：插入/删除操作会使迭代器失效
    // 在循环中删除元素时，必须更新迭代器
    std::vector<int> v2 = {1, 2, 3, 4, 5};
    for (auto it = v2.begin(); it != v2.end(); ) {
        if (*it % 2 == 0)
            it = v2.erase(it);  // erase 返回下一个有效迭代器
        else
            ++it;
    }
    print(v2, "迭代器安全删除偶数");
}

// ============================================================
// 1.6 容量管理
// ============================================================
void demo_capacity() {
    std::cout << "\n=== 1.6 容量管理 ===\n";

    std::vector<int> v;
    std::cout << "初始: size=" << v.size() << " capacity=" << v.capacity() << "\n";

    // push_back 触发自动扩容（通常每次翻倍）
    for (int i = 0; i < 10; i++) {
        v.push_back(i);
        std::cout << "push_back(" << i << ")  size=" << v.size()
                  << " capacity=" << v.capacity() << "\n";
    }

    // reserve：预分配内存，避免反复扩容（知道大约大小时推荐）
    std::vector<int> v2;
    v2.reserve(100);
    std::cout << "\nreserve(100): size=" << v2.size()
              << " capacity=" << v2.capacity() << "\n";

    // resize：改变 size（新增元素值初始化；shrink 不释放 capacity）
    v2.resize(5, 42);
    std::cout << "resize(5, 42): size=" << v2.size() << "\n";
    print(v2, "v2");

    // shrink_to_fit：请求释放多余 capacity（非强制，实现相关）
    v2.shrink_to_fit();
    std::cout << "shrink_to_fit: capacity=" << v2.capacity() << "\n";

    // clear：清空所有元素，但不释放内存
    v2.clear();
    std::cout << "clear: size=" << v2.size()
              << " capacity=" << v2.capacity() << "\n";
}

// ============================================================
// 1.7 二维 vector
// ============================================================
void demo_2d() {
    std::cout << "\n=== 1.7 二维 vector ===\n";

    int rows = 3, cols = 4;
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols, 0));

    // 填入数据
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            matrix[r][c] = r * cols + c;

    // 打印
    for (const auto& row : matrix) {
        for (int x : row) std::cout << x << "\t";
        std::cout << "\n";
    }
}

int main() {
    demo_construct();
    demo_insert();
    demo_erase();
    demo_access();
    demo_iterator();
    demo_capacity();
    demo_2d();
    return 0;
}
