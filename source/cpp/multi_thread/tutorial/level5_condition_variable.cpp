/*
 * ============================================================
 * Level 5 — condition_variable：线程间通知与等待
 * ============================================================
 *
 * 目标：理解为什么需要条件变量，掌握正确用法
 *
 * 问题背景：
 *   mutex 解决了"互斥"问题，但无法解决"等待某个条件成立"。
 *   例如：消费者线程需要等待队列非空，但不应该一直占着 CPU 自旋。
 *
 *   错误方案（自旋等待）：
 *     while (queue.empty()) {}   // 浪费 CPU，持锁时其他线程无法 push
 *
 *   正确方案（条件变量）：
 *     cv.wait(lock, predicate)   // 挂起线程，释放锁，等待通知后重新加锁
 *
 * 核心 API：
 *   std::condition_variable cv;
 *   cv.wait(unique_lock, predicate)          // 等待，predicate 为假则继续等
 *   cv.wait_for(unique_lock, duration, pred) // 带超时
 *   cv.notify_one()                          // 唤醒一个等待线程
 *   cv.notify_all()                          // 唤醒所有等待线程
 */

#include <mutex>
#include <condition_variable>
#include <thread>
#include <iostream>
#include <queue>
#include <cassert>
#include <chrono>

using namespace std::chrono_literals;

// ==================== 5.1：最简单的一次性通知 ====================
//
// 场景：主线程做完准备工作，通知工作线程开始

std::mutex gMutex;
std::condition_variable gCV;
bool gReady = false;

void worker() {
    std::unique_lock<std::mutex> lock(gMutex);

    // wait 的内部等价于：
    //   while (!gReady) {
    //     lock.unlock();      // 释放锁（让其他线程能修改 gReady）
    //     /* 挂起，等待 notify */
    //     lock.lock();        // 被唤醒后重新持锁
    //   }
    gCV.wait(lock, [] { return gReady; });

    std::cout << "Worker: got signal, starting work!\n";
}

void demo_one_shot() {
    std::cout << "=== 5.1 一次性通知 ===\n";
    gReady = false;

    std::thread t(worker);

    // 主线程做准备工作
    std::this_thread::sleep_for(50ms);
    std::cout << "Main: preparation done, sending signal...\n";

    {
        std::lock_guard<std::mutex> lock(gMutex);
        gReady = true;
    }  // 先释放锁，再 notify（避免 worker 被唤醒后立刻阻塞在锁上）
    gCV.notify_one();

    t.join();
    std::cout << "\n";
}

// ==================== 5.2：为什么必须有谓词（防虚假唤醒）====================
//
// 虚假唤醒（Spurious Wakeup）：
//   操作系统可能在没有 notify 的情况下唤醒线程（这是 POSIX 规范允许的）。
//   如果不检查条件，线程会在条件未满足时就继续执行 → bug！
//
// 错误写法：
//   cv.wait(lock);           // 无谓词，虚假唤醒后直接继续
//
// 正确写法：
//   cv.wait(lock, pred);     // 等价于 while(!pred()) cv.wait(lock);

void demo_spurious_wakeup() {
    std::cout << "=== 5.2 虚假唤醒示意 ===\n";
    std::cout << R"(
  错误写法（有虚假唤醒风险）：
    cv.wait(lock);                          // 直接等待，无条件检查

  正确写法（谓词形式）：
    cv.wait(lock, []{ return ready; });     // 等价于：
                                            // while (!ready) cv.wait(lock);

  ✓ 谓词保证：即使虚假唤醒，条件不满足也会继续等待。
)" << "\n";
}

// ==================== 5.3：带超时的等待 ====================
//
// wait_for 返回：
//   true  → 条件满足（正常 notify 唤醒）
//   false → 超时（条件仍不满足）

std::mutex gTimedMutex;
std::condition_variable gTimedCV;
bool gTimedReady = false;

void demo_wait_for() {
    std::cout << "=== 5.3 带超时的等待 ===\n";
    gTimedReady = false;

    std::thread t([] {
        std::unique_lock<std::mutex> lock(gTimedMutex);
        bool ok = gTimedCV.wait_for(lock, 100ms, [] { return gTimedReady; });
        if (ok) {
            std::cout << "  Thread: got signal in time\n";
        } else {
            std::cout << "  Thread: timed out!\n";
        }
    });

    // 故意不发送通知，让线程超时
    t.join();
    std::cout << "\n";
}

// ==================== 5.4：notify_one vs notify_all ====================
//
// notify_one：唤醒一个等待线程（随机选择）
//             适用于：多个线程等同种任务，只需一个来处理
//
// notify_all：唤醒所有等待线程
//             适用于：条件对所有等待者都有意义（如停止信号）

std::mutex gBroadcastMutex;
std::condition_variable gBroadcastCV;
bool gGo = false;

void racer(int id) {
    std::unique_lock<std::mutex> lock(gBroadcastMutex);
    gBroadcastCV.wait(lock, [] { return gGo; });
    std::cout << "  Racer " << id << " started!\n";
}

void demo_notify_all() {
    std::cout << "=== 5.4 notify_all（发令枪）===\n";
    gGo = false;

    std::vector<std::thread> racers;
    for (int i = 0; i < 4; ++i) {
        racers.emplace_back(racer, i);
    }

    std::this_thread::sleep_for(50ms);
    std::cout << "  GO!\n";
    {
        std::lock_guard<std::mutex> lock(gBroadcastMutex);
        gGo = true;
    }
    gBroadcastCV.notify_all();  // 同时唤醒所有选手

    for (auto& t : racers) t.join();
    std::cout << "\n";
}

// ==================== 5.5：notify 是否需要在锁内？====================

void demo_notify_timing() {
    std::cout << "=== 5.5 notify 在锁内外的区别 ===\n";
    std::cout << R"(
  方式A（锁内 notify，可能导致"立刻再竞争"）：
    {
        lock_guard lock(m);
        ready = true;
        cv.notify_one();    // worker 唤醒后立刻要抢锁，但锁还在主线程手里
    }

  方式B（锁外 notify，推荐）：
    {
        lock_guard lock(m);
        ready = true;
    }                       // 先释放锁
    cv.notify_one();        // 再通知，worker 唤醒时锁已释放，无竞争

  ✓ 两种方式都正确，但方式B性能略好（减少一次不必要的锁争用）。
)" << "\n";
}

// ==================== 5.6：综合练习——两个线程交替打印 ====================
//
// 题目：线程 A 打印奇数，线程 B 打印偶数，交替输出 1 2 3 4 5 ...

std::mutex gAltMutex;
std::condition_variable gAltCV;
int gNum = 1;
const int MAX = 10;

void printOdd() {
    while (true) {
        std::unique_lock<std::mutex> lock(gAltMutex);
        gAltCV.wait(lock, [] { return gNum % 2 == 1 || gNum > MAX; });
        if (gNum > MAX) break;
        std::cout << "  A: " << gNum++ << "\n";
        gAltCV.notify_one();
    }
}

void printEven() {
    while (true) {
        std::unique_lock<std::mutex> lock(gAltMutex);
        gAltCV.wait(lock, [] { return gNum % 2 == 0 || gNum > MAX; });
        if (gNum > MAX) break;
        std::cout << "  B: " << gNum++ << "\n";
        gAltCV.notify_one();
    }
}

void demo_alternating() {
    std::cout << "=== 5.6 交替打印（经典考题）===\n";
    gNum = 1;
    std::thread a(printOdd);
    std::thread b(printEven);
    a.join();
    b.join();
    std::cout << "\n";
}

// ==================== 练习题 ====================
/*
 * [练习 1] 实现一个 Flag 类：
 *   - void set()            设置标志
 *   - void wait()           阻塞直到标志被设置
 *   - void reset()          重置标志
 *   - bool waitFor(ms)      带超时等待，超时返回 false
 *
 * [练习 2] 三个线程 A/B/C 打印 ABCABCABC...（循环 5 次）
 *   提示：用一个 int turn 变量（0=A, 1=B, 2=C），每次打印后 turn=(turn+1)%3
 *
 * [练习 3] 理解以下代码为什么有 bug：
 *
 *   bool ready = false;
 *   std::mutex m;
 *   std::condition_variable cv;
 *
 *   // 线程 1（通知方）
 *   ready = true;            // ❌ 没有加锁就修改
 *   cv.notify_one();
 *
 *   // 线程 2（等待方）
 *   std::unique_lock lock(m);
 *   cv.wait(lock, [&]{ return ready; });
 *
 *   // 问题：notify 可能在 wait 之前发送，然后线程 2 永久等待（通知丢失）
 *   // 修复：通知方也必须持锁修改 ready
 */

int main() {
    demo_one_shot();
    demo_spurious_wakeup();
    demo_wait_for();
    demo_notify_all();
    demo_notify_timing();
    demo_alternating();

    std::cout << "Level 5 Complete!\n";
    std::cout << "下一步 → level6_producer_consumer.cpp：构建完整的生产者消费者\n";
    return 0;
}

/*
 * 编译运行：
 *   g++ -std=c++17 -pthread level5_condition_variable.cpp -o level5 && ./level5
 *
 * 条件变量三步口诀：
 *   等待方：unique_lock → wait(lock, pred) → 处理
 *   通知方：修改条件（持锁）→ 释放锁 → notify
 */
