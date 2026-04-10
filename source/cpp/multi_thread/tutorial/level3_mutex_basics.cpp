/*
 * ============================================================
 * Level 3 — std::mutex：修复数据竞争
 * ============================================================
 *
 * 目标：学会用 mutex 保护共享数据，理解临界区的概念
 *
 * 核心思想：
 *   互斥锁（mutex）保证同一时刻只有一个线程能进入"临界区"。
 *   临界区 = 访问共享数据的代码段。
 *
 * 原语：
 *   std::mutex m;
 *   m.lock();    // 进入临界区，若已锁则阻塞等待
 *   m.unlock();  // 离开临界区
 *   m.try_lock() // 非阻塞尝试，失败返回 false
 *
 * 警告：本文件故意展示裸 lock/unlock，下一级会学习更安全的 RAII 包装。
 */

#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <cassert>

// ==================== 3.1：修复计数器竞争 ====================

std::mutex gMutex;
int gCounter = 0;

void incrementSafe(int n) {
    for (int i = 0; i < n; ++i) {
        gMutex.lock();
        gCounter++;          // 临界区：同一时刻只有一个线程执行
        gMutex.unlock();
    }
}

void demo_fixed_counter() {
    std::cout << "=== 3.1 修复后的计数器 ===\n";
    gCounter = 0;

    std::thread t1(incrementSafe, 500000);
    std::thread t2(incrementSafe, 500000);
    t1.join();
    t2.join();

    std::cout << "Expected: 1000000\n";
    std::cout << "Actual:   " << gCounter << "\n";
    assert(gCounter == 1000000);
    std::cout << "✓ 正确！\n\n";
}

// ==================== 3.2：临界区粒度的影响 ====================
//
// 锁的粒度（granularity）：
//   细粒度（临界区小）→ 高并发，但锁开销高
//   粗粒度（临界区大）→ 低并发，但简单安全
//
// 下面对比"每次加锁"和"批量加锁"的性能

#include <chrono>

void incrementBatch(int n) {
    // 粗粒度：一次锁住整个循环
    gMutex.lock();
    for (int i = 0; i < n; ++i) {
        gCounter++;
    }
    gMutex.unlock();
}

void demo_granularity() {
    std::cout << "=== 3.2 粒度对比 ===\n";
    const int N = 1000000;

    // 细粒度
    gCounter = 0;
    auto t0 = std::chrono::steady_clock::now();
    std::thread t1(incrementSafe, N / 2);
    std::thread t2(incrementSafe, N / 2);
    t1.join(); t2.join();
    auto fine = std::chrono::steady_clock::now() - t0;

    // 粗粒度
    gCounter = 0;
    t0 = std::chrono::steady_clock::now();
    std::thread t3(incrementBatch, N / 2);
    std::thread t4(incrementBatch, N / 2);
    t3.join(); t4.join();
    auto coarse = std::chrono::steady_clock::now() - t0;

    std::cout << "细粒度 (lock per op):  "
              << std::chrono::duration_cast<std::chrono::milliseconds>(fine).count() << "ms\n";
    std::cout << "粗粒度 (lock per batch): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(coarse).count() << "ms\n";
    std::cout << "(粗粒度更快，但并发度更低)\n\n";
}

// ==================== 3.3：异常导致忘记 unlock — 死锁！====================
//
// 裸 lock/unlock 的最大危险：
//   如果临界区内抛出异常，unlock 永远不会被调用 → 死锁！

std::mutex gDangerMutex;

void dangerous() {
    gDangerMutex.lock();
    // 如果下面抛出异常
    // throw std::runtime_error("oops");  // ← 取消注释会导致死锁！
    gDangerMutex.unlock();
}

void demo_exception_danger() {
    std::cout << "=== 3.3 异常安全问题（裸 lock 的危险）===\n";
    std::cout << "裸 lock/unlock 在异常路径下会忘记 unlock → 死锁\n";
    std::cout << "解决方案 → Level 4 的 RAII 锁（lock_guard）\n\n";
}

// ==================== 3.4：死锁示例 ====================
//
// 死锁（Deadlock）：两个线程互相等待对方持有的锁
//
// 线程 A: lock(m1) → 等待 lock(m2)
// 线程 B: lock(m2) → 等待 lock(m1)
// → 永久阻塞

std::mutex gM1, gM2;

void threadA_buggy() {
    gM1.lock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));  // 模拟工作
    gM2.lock();   // ← 等待 m2，但 m2 被 B 持有
    // ... 临界区
    gM2.unlock();
    gM1.unlock();
}

void threadB_buggy() {
    gM2.lock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    gM1.lock();   // ← 等待 m1，但 m1 被 A 持有
    // ... 临界区
    gM1.unlock();
    gM2.unlock();
}

// 修复方案：统一加锁顺序（总是先锁 m1 再锁 m2）
void threadA_fixed() {
    gM1.lock();
    gM2.lock();   // 顺序与 B 相同
    gM2.unlock();
    gM1.unlock();
}

void threadB_fixed() {
    gM1.lock();   // 顺序与 A 相同
    gM2.lock();
    gM2.unlock();
    gM1.unlock();
}

void demo_deadlock() {
    std::cout << "=== 3.4 死锁与修复 ===\n";
    std::cout << "Buggy 版本（不运行，否则永久挂起）\n";

    // 运行修复版本
    std::thread a(threadA_fixed);
    std::thread b(threadB_fixed);
    a.join();
    b.join();
    std::cout << "Fixed 版本运行成功（统一加锁顺序）\n\n";
}

// ==================== 3.5：try_lock 非阻塞获取 ====================

void demo_try_lock() {
    std::cout << "=== 3.5 try_lock ===\n";
    std::mutex m;

    m.lock();  // 主线程先持有锁

    std::thread t([&m] {
        if (m.try_lock()) {
            std::cout << "Thread: got lock\n";
            m.unlock();
        } else {
            std::cout << "Thread: lock busy, skip\n";
        }
    });

    t.join();
    m.unlock();
    std::cout << "\n";
}

// ==================== 练习题 ====================
/*
 * [练习 1] 修复以下代码，使结果始终为 100：
 *
 *   int count = 0;
 *   std::mutex m;
 *   auto work = [&]{ for(int i=0;i<10;++i) count++; };
 *   std::vector<std::thread> ts;
 *   for(int i=0;i<10;++i) ts.emplace_back(work);
 *   for(auto& t:ts) t.join();
 *   assert(count == 100);
 *
 * [练习 2] 实现线程安全的 Stack<int>：
 *   - push(int)    压栈
 *   - int pop()    出栈（空时 throw）
 *   - bool empty() 判空
 *   每个方法内部用 lock/unlock 保护。
 *
 * [练习 3] 以下代码会死锁吗？为什么？
 *
 *   std::mutex m;
 *   void foo() {
 *       m.lock();
 *       bar();      // bar 内部也调用 m.lock()
 *       m.unlock();
 *   }
 *   // 答：会死锁！std::mutex 不可重入。
 *   // 解决：用 std::recursive_mutex，或重构避免嵌套加锁。
 */

int main() {
    demo_fixed_counter();
    demo_granularity();
    demo_exception_danger();
    demo_deadlock();
    demo_try_lock();

    std::cout << "Level 3 Complete!\n";
    std::cout << "下一步 → level4_raii_locks.cpp：告别裸 lock/unlock\n";
    return 0;
}

/*
 * 编译运行：
 *   g++ -std=c++17 -pthread level3_mutex_basics.cpp -o level3 && ./level3
 *
 * 本关总结：
 *   ✓ mutex 保证临界区互斥执行
 *   ✗ 裸 lock/unlock 有三大危险：
 *     1. 忘记 unlock（→ 死锁）
 *     2. 异常路径跳过 unlock（→ 死锁）
 *     3. 提前 return 忘记 unlock（→ 死锁）
 *   → Level 4 用 RAII 彻底解决
 */
