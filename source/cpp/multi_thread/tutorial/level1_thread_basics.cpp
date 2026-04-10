/*
 * ============================================================
 * Level 1 — 线程创建、传参、等待
 * ============================================================
 *
 * 目标：掌握 std::thread 最基础的用法
 *
 * 知识点：
 *   - std::thread t(func, arg1, arg2...)  创建线程
 *   - t.join()    等待线程结束（必须在 joinable 的 thread 析构前调用）
 *   - t.detach()  分离线程，不再等待（析构时不 terminate）
 *   - t.joinable() 判断是否可 join
 *
 * 注意：
 *   - std::thread 析构时如果 joinable() == true 且既没有 join 也没有 detach，
 *     会调用 std::terminate() 直接崩溃！
 *   - 传引用参数时必须用 std::ref()，否则按值复制
 */

#include <thread>
#include <iostream>
#include <vector>
#include <cassert>

// ==================== 示例 1.1：最简单的线程 ====================

void hello(int id) {
    std::cout << "Hello from thread " << id << "\n";
}

void demo_basic() {
    std::cout << "=== 1.1 基础线程 ===\n";

    std::thread t1(hello, 1);
    std::thread t2(hello, 2);

    t1.join();  // 等待 t1 结束
    t2.join();  // 等待 t2 结束

    std::cout << "All threads done.\n\n";
}

// ==================== 示例 1.2：传引用参数 ====================

void accumulate(const std::vector<int>& data, int start, int end, int& result) {
    result = 0;
    for (int i = start; i < end; ++i) {
        result += data[i];
    }
}

void demo_ref_arg() {
    std::cout << "=== 1.2 传引用参数（std::ref）===\n";

    std::vector<int> data(100, 1);  // 100 个 1
    int r1 = 0, r2 = 0;

    // 前 50 个元素由 t1 计算，后 50 个由 t2 计算
    std::thread t1(accumulate, std::cref(data), 0,  50, std::ref(r1));
    std::thread t2(accumulate, std::cref(data), 50, 100, std::ref(r2));

    t1.join();
    t2.join();

    std::cout << "r1=" << r1 << " r2=" << r2 << " sum=" << r1 + r2 << "\n\n";
    assert(r1 + r2 == 100);
}

// ==================== 示例 1.3：lambda 线程 ====================

void demo_lambda() {
    std::cout << "=== 1.3 Lambda 线程 ===\n";

    int sharedVal = 0;  // 注意：此处为演示，实际并发修改是 UB！
                        // 这里只有一个线程修改，所以安全

    std::thread t([&sharedVal] {
        sharedVal = 42;
        std::cout << "Thread set value to " << sharedVal << "\n";
    });

    t.join();
    std::cout << "Main sees value: " << sharedVal << "\n\n";
}

// ==================== 示例 1.4：线程数组 ====================

void demo_thread_array() {
    std::cout << "=== 1.4 线程数组 ===\n";

    const int N = 5;
    std::vector<std::thread> threads;
    threads.reserve(N);

    for (int i = 0; i < N; ++i) {
        threads.emplace_back([i] {
            std::cout << "Worker " << i << " running\n";
        });
    }

    // 依次等待所有线程
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "All workers done.\n\n";
}

// ==================== 示例 1.5：joinable 检查的重要性 ====================

void demo_joinable() {
    std::cout << "=== 1.5 joinable 安全检查 ===\n";

    std::thread t(hello, 99);

    // 模拟某个条件下决定是否 join
    bool shouldJoin = true;

    if (shouldJoin && t.joinable()) {
        t.join();
    } else if (t.joinable()) {
        t.detach();  // 不等待，让线程在后台自行结束
    }
    // 此时 t.joinable() == false，析构安全

    std::cout << "joinable after join: " << t.joinable() << "\n\n";
}

// ==================== 练习题 ====================
/*
 * [练习 1] 实现一个函数 parallelFill，将 vector<int> 分成 N 段，
 *          每段由一个线程填充其段的索引值（第 i 段所有元素填 i）。
 *          要求：不使用任何同步原语（每个线程写不同位置，天然安全）。
 *
 * [练习 2] 实现 parallelSum，将 vector<int> 分成 4 段并行求和，
 *          最终汇总所有段的结果。
 *
 * [练习 3] 以下代码有什么问题？
 *
 *   void broken() {
 *       int x = 0;
 *       std::thread t([&x]{ x = 1; });
 *       // 忘记 join 了！
 *   }  // t 析构 → std::terminate()
 */

int main() {
    demo_basic();
    demo_ref_arg();
    demo_lambda();
    demo_thread_array();
    demo_joinable();

    std::cout << "Level 1 Complete!\n";
    std::cout << "下一步 → level2_race_condition.cpp\n";
    return 0;
}

/*
 * 编译运行：
 *   g++ -std=c++17 -pthread level1_thread_basics.cpp -o level1 && ./level1
 *
 * 关键结论：
 *   1. 每个 joinable 的线程析构前必须 join() 或 detach()。
 *   2. 传引用给线程函数必须用 std::ref() / std::cref()。
 *   3. Lambda 捕获引用时，需确保被捕获对象的生命周期长于线程。
 */
