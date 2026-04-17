// Level 2: std::deque & std::list —— 双端队列与双向链表
// 涵盖：deque 双端操作、list 的高效插删、forward_list 单向链表、与 vector 的对比

#include <deque>
#include <forward_list>
#include <iostream>
#include <list>

// ============================================================
// 辅助打印
// ============================================================
template <typename Container>
void print(const Container& c, const std::string& label = "") {
    if (!label.empty()) std::cout << label << ": ";
    for (const auto& x : c) std::cout << x << " ";
    std::cout << "\n";
}

// ============================================================
// 2.1 std::deque —— 双端队列
//   内部：分段连续内存（多个固定大小缓冲块）
//   优点：头尾 O(1) 插删；随机访问 O(1)
//   缺点：中间插删 O(n)；内存不连续（不能用 data() 指针）
// ============================================================
void demo_deque() {
    std::cout << "=== 2.1 std::deque ===\n";

    std::deque<int> dq = {3, 4, 5};

    // 头尾均可 O(1) 插入
    dq.push_front(2);
    dq.push_front(1);
    dq.push_back(6);
    dq.push_back(7);
    print(dq, "push_front/back");

    // 头尾均可 O(1) 删除
    dq.pop_front();
    dq.pop_back();
    print(dq, "pop_front/back");

    // 支持下标随机访问（与 vector 相同接口）
    std::cout << "dq[2]=" << dq[2] << "\n";

    // 在中间插入（O(n)，会选择移动较少的一侧）
    auto it = dq.begin() + 2;
    dq.insert(it, 99);
    print(dq, "insert(pos=2, 99)");

    // 常用场景：需要频繁从两端操作的滑动窗口
    std::cout << "size=" << dq.size() << "\n";
}

// ============================================================
// 2.2 std::list —— 双向链表
//   内部：每个节点独立堆分配，含前后指针
//   优点：任意位置 O(1) 插删（已有迭代器情况下）；迭代器永不失效（除被删节点）
//   缺点：不支持随机访问；内存碎片多；cache 不友好
// ============================================================
void demo_list() {
    std::cout << "\n=== 2.2 std::list ===\n";

    std::list<int> lst = {1, 3, 5, 7, 9};
    print(lst, "初始");

    // push_front / push_back / pop_front / pop_back（同 deque）
    lst.push_front(0);
    lst.push_back(10);
    print(lst, "push_front(0) push_back(10)");

    // insert：在迭代器前插入（O(1)）
    auto it = lst.begin();
    std::advance(it, 3);           // 移到第 4 个元素
    lst.insert(it, 99);
    print(lst, "insert 在第4位前");

    // erase：删除指定迭代器（O(1)，不需要移动其他元素）
    it = lst.begin();
    std::advance(it, 2);
    lst.erase(it);
    print(lst, "erase 第3位");

    // remove：删除所有等于某值的节点（O(n)）
    lst.remove(99);
    print(lst, "remove(99)");

    // remove_if：按条件删除
    lst.remove_if([](int x) { return x % 2 == 0; });
    print(lst, "remove_if 偶数");

    // sort：list 自带 sort（不能用 std::sort，因为不是随机迭代器）
    std::list<int> lst2 = {5, 1, 4, 2, 3};
    lst2.sort();
    print(lst2, "sort 后");

    lst2.sort(std::greater<int>());  // 降序
    print(lst2, "sort 降序");

    // reverse：就地翻转（O(n)）
    lst2.reverse();
    print(lst2, "reverse");

    // unique：删除连续重复元素（常配合 sort 做去重）
    std::list<int> lst3 = {1, 1, 2, 3, 3, 3, 4};
    lst3.unique();
    print(lst3, "unique");

    // splice：高效地将另一个 list 的元素移动进来（O(1) 或 O(n)）
    std::list<int> src = {100, 200, 300};
    lst2.splice(lst2.end(), src);      // 整个 src 移入 lst2 末尾
    print(lst2, "splice 后 lst2");
    std::cout << "src size after splice=" << src.size() << "\n";
}

// ============================================================
// 2.3 std::forward_list —— 单向链表（C++11）
//   内存更省（每节点少一个指针），但只能单向遍历、只有 insert_after
// ============================================================
void demo_forward_list() {
    std::cout << "\n=== 2.3 std::forward_list ===\n";

    std::forward_list<int> fl = {1, 2, 3, 4, 5};
    print(fl, "初始");

    // 只能在某个元素之后插入（before_begin() 指向头部的哑节点）
    fl.insert_after(fl.before_begin(), 0);
    print(fl, "insert_after(before_begin, 0)");

    // erase_after：删除某迭代器之后的元素
    auto it = fl.begin();           // 指向 0
    fl.erase_after(it);             // 删除 0 之后的 1
    print(fl, "erase_after(begin)");

    // 没有 size()（O(1) 无法计算），需要 std::distance
    std::cout << "size=" << std::distance(fl.begin(), fl.end()) << "\n";
}

// ============================================================
// 2.4 容器对比小结
// ============================================================
void demo_comparison() {
    std::cout << "\n=== 2.4 容器特性对比 ===\n";
    std::cout << R"(
  容器              随机访问  头部插删  尾部插删  中间插删  内存连续
  vector            O(1)      O(n)      O(1)*    O(n)      是
  deque             O(1)      O(1)      O(1)     O(n)      否
  list              O(n)      O(1)      O(1)     O(1)**    否
  forward_list      O(n)      O(1)      O(n)     O(1)**    否
  * 均摊 O(1)，偶尔触发扩容
  ** 前提：已有指向该位置的迭代器
)";
}

int main() {
    demo_deque();
    demo_list();
    demo_forward_list();
    demo_comparison();
    return 0;
}
