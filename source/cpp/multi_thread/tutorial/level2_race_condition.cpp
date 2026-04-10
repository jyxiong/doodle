/*
 * ============================================================
 * Level 2 — 数据竞争（Race Condition）：认识问题
 * ============================================================
 *
 * 目标：亲眼看到并发 bug，理解为什么需要同步
 *
 * 数据竞争（Data Race）：
 *   两个或更多线程同时访问同一内存，至少有一个是写操作，
 *   且没有同步机制保护 → 未定义行为（UB）
 *
 * 本文件故意写出有 bug 的代码，观察非预期结果。
 * 请先思考每段代码的预期输出，再运行对比实际输出。
 */

#include <thread>
#include <iostream>
#include <vector>
#include <cassert>

// ==================== Bug 示例 2.1：计数器竞争 ====================
//
// 预期：1000000
// 实际：每次运行结果不同，通常 < 1000000
//
// 原因：counter++ 不是原子操作，实际是 3 条指令：
//   1. LOAD  counter → reg
//   2. ADD   reg, 1
//   3. STORE reg → counter
// 两个线程可能同时 LOAD 到相同值，导致其中一次 +1 被覆盖（"丢失更新"）

int gCounter = 0;  // 共享变量，无保护

void incrementUnsafe(int n) {
    for (int i = 0; i < n; ++i) {
        gCounter++;  // ← 非原子！数据竞争！
    }
}

void demo_counter_race() {
    std::cout << "=== 2.1 计数器竞争 ===\n";
    gCounter = 0;

    std::thread t1(incrementUnsafe, 500000);
    std::thread t2(incrementUnsafe, 500000);
    t1.join();
    t2.join();

    std::cout << "Expected: 1000000\n";
    std::cout << "Actual:   " << gCounter << "\n";
    std::cout << (gCounter == 1000000 ? "✓ 幸运！（不总是这样）" : "✗ 丢失更新！") << "\n\n";
}

// ==================== Bug 示例 2.2：先检查后操作（Check-Then-Act）====================
//
// 经典 TOCTOU（Time of Check to Time of Use）bug：
// 线程 A 检查条件后，条件在执行操作前被线程 B 改变

std::vector<int> gQueue;  // 无保护的全局队列

void unsafePop() {
    // 检查非空后再 pop，看似安全，实则不然
    if (!gQueue.empty()) {
        // ← 此处被另一个线程抢先清空了！
        std::this_thread::yield();  // 模拟调度切换
        // 以下 pop_back 可能在 empty 时调用 → UB！
        // gQueue.pop_back();  // 故意注释掉，否则真崩溃
        std::cout << "Thread " << std::this_thread::get_id()
                  << " saw non-empty but now size=" << gQueue.size() << "\n";
    }
}

void demo_toctou() {
    std::cout << "=== 2.2 TOCTOU 竞争 ===\n";
    gQueue = {1, 2, 3};

    std::thread t1(unsafePop);
    std::thread t2(unsafePop);
    t1.join();
    t2.join();
    std::cout << "\n";
}

// ==================== Bug 示例 2.3：可见性问题（指令重排）====================
//
// 没有内存屏障的情况下，编译器/CPU 可能重排指令。
// 线程 A 写入的值，线程 B 不一定能及时看到。
//
// 以下示例在某些平台/优化级别下，线程2可能永远看不到 ready=true

bool gReady = false;   // 无 volatile，无 atomic
int  gData  = 0;

void writer() {
    gData  = 42;       // ← 可能被重排到 gReady=true 之后
    gReady = true;
}

void reader() {
    while (!gReady) {  // ← 可能被优化为只读一次寄存器
        std::this_thread::yield();
    }
    // 此时 gData 不一定是 42！（内存可见性问题）
    std::cout << "Reader got data=" << gData << " (may not be 42 without sync!)\n";
}

void demo_visibility() {
    std::cout << "=== 2.3 可见性问题 ===\n";
    gReady = false;
    gData  = 0;

    std::thread t1(writer);
    std::thread t2(reader);
    t1.join();
    t2.join();
    std::cout << "(在 -O2 优化下此问题更容易复现)\n\n";
}

// ==================== 竞争条件的三种根本原因 ====================
/*
 * 1. 原子性破坏（Atomicity Violation）
 *    操作在中途被打断（如 counter++）
 *    → 解决：std::mutex 或 std::atomic
 *
 * 2. 顺序破坏（Order Violation）
 *    操作执行顺序不符合预期（线程 A 的操作应在 B 之前）
 *    → 解决：std::condition_variable 或 std::promise/future
 *
 * 3. 可见性问题（Visibility）
 *    一个线程的写对另一个线程不可见（编译器/CPU 缓存）
 *    → 解决：std::atomic（提供 happens-before 语义）
 */

// ==================== 思考题 ====================
/*
 * [思考 1] 以下代码的 counter 最终可能是哪些值？
 *
 *   int counter = 0;
 *   std::thread t1([&]{ counter++; counter++; });  // 加 2
 *   std::thread t2([&]{ counter++; });              // 加 1
 *   t1.join(); t2.join();
 *   // 预期 3，可能出现哪些值？(1, 2, 3 都可能)
 *
 * [思考 2] 为什么 double-checked locking（双重检查锁）
 *          在没有 atomic 的情况下是不安全的？
 *
 *   Singleton* p = sInstance;       // 看到非空
 *   if (p == nullptr) { ... }
 *   return p;
 *   // 问题：看到非空不代表对象已完全构造完毕（可见性问题）
 *
 * [思考 3] volatile 能解决数据竞争吗？
 *   答：不能。volatile 只禁止编译器缓存变量到寄存器，
 *       不提供原子性，也不提供内存屏障。是 C 时代的遗留。
 *       C++ 中并发应使用 std::atomic。
 */

int main() {
    demo_counter_race();
    demo_toctou();
    demo_visibility();

    std::cout << "Level 2 Complete! 看到了 bug 了吗？\n";
    std::cout << "下一步 → level3_mutex_basics.cpp：用 mutex 修复这些问题\n";
    return 0;
}

/*
 * 编译运行：
 *   # 不加优化，竞争可能不明显
 *   g++ -std=c++17 -pthread level2_race_condition.cpp -o level2 && ./level2
 *
 *   # 加 -O2 后竞争更容易出现（编译器会做更多假设）
 *   g++ -std=c++17 -pthread -O2 level2_race_condition.cpp -o level2 && ./level2
 *
 *   # 用 ThreadSanitizer 检测数据竞争（强烈推荐！）
 *   g++ -std=c++17 -pthread -fsanitize=thread -g level2_race_condition.cpp -o level2_tsan && ./level2_tsan
 */
