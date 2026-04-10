/*
 * ============================================================
 * Level 7 — std::atomic：无锁原子操作
 * ============================================================
 *
 * 目标：理解 atomic 与 mutex 的区别，掌握原子操作的常用场景
 *
 * 为什么需要 atomic？
 *   mutex 有开销：系统调用、上下文切换、线程挂起/唤醒。
 *   对于简单的计数器、标志位，atomic 提供更轻量级的方案：
 *   硬件层面保证原子性，不需要挂起线程。
 *
 * atomic 适用场景：
 *   ✓ 计数器 (fetch_add/fetch_sub)
 *   ✓ 标志位 (load/store, atomic<bool>)
 *   ✓ 状态指示 (exchange, compare_exchange)
 *   ✗ 保护多个变量之间的一致性 → 还是需要 mutex
 */

#include <atomic>
#include <thread>
#include <vector>
#include <iostream>
#include <cassert>
#include <chrono>
#include <mutex>

using namespace std::chrono_literals;

// ============================================================
// 7.1：基础操作对比（mutex vs atomic）
// ============================================================

// 方案 A：用 mutex 保护的计数器
struct MutexCounter {
    std::mutex m;
    int value = 0;
    void inc() { std::lock_guard lock(m); ++value; }
    int get() { std::lock_guard lock(m); return value; }
};

// 方案 B：atomic 计数器
struct AtomicCounter {
    std::atomic<int> value{0};
    void inc() { value.fetch_add(1); }  // 原子加
    int get()  { return value.load(); } // 原子读
};

void demo_counter_compare() {
    std::cout << "=== 7.1 mutex vs atomic 计数器 ===\n";
    const int N = 500000;

    // Mutex 版本
    MutexCounter mc;
    auto t0 = std::chrono::steady_clock::now();
    {
        std::thread t1([&mc, N]{ for(int i=0;i<N;++i) mc.inc(); });
        std::thread t2([&mc, N]{ for(int i=0;i<N;++i) mc.inc(); });
        t1.join(); t2.join();
    }
    auto mutexTime = std::chrono::steady_clock::now() - t0;
    assert(mc.get() == 2 * N);

    // Atomic 版本
    AtomicCounter ac;
    t0 = std::chrono::steady_clock::now();
    {
        std::thread t1([&ac, N]{ for(int i=0;i<N;++i) ac.inc(); });
        std::thread t2([&ac, N]{ for(int i=0;i<N;++i) ac.inc(); });
        t1.join(); t2.join();
    }
    auto atomicTime = std::chrono::steady_clock::now() - t0;
    assert(ac.get() == 2 * N);

    std::cout << "  mutex:  " << std::chrono::duration_cast<std::chrono::milliseconds>(mutexTime).count() << "ms\n";
    std::cout << "  atomic: " << std::chrono::duration_cast<std::chrono::milliseconds>(atomicTime).count() << "ms\n";
    std::cout << "  (atomic 通常 3~10x 更快)\n\n";
}

// ============================================================
// 7.2：常用操作详解
// ============================================================

void demo_atomic_ops() {
    std::cout << "=== 7.2 atomic 操作详解 ===\n";

    std::atomic<int> a{10};

    // load / store
    int v = a.load();                   // 原子读
    a.store(20);                        // 原子写
    std::cout << "  store(20), load() = " << a.load() << "\n";

    // fetch_add / fetch_sub（返回操作前的值）
    int old = a.fetch_add(5);           // a += 5，返回旧值 20
    std::cout << "  fetch_add(5): old=" << old << " new=" << a.load() << "\n";

    old = a.fetch_sub(3);               // a -= 3，返回旧值 25
    std::cout << "  fetch_sub(3): old=" << old << " new=" << a.load() << "\n";

    // exchange（原子地写入新值，返回旧值）
    old = a.exchange(0);                // a = 0，返回旧值 22
    std::cout << "  exchange(0):  old=" << old << " new=" << a.load() << "\n";

    // compare_exchange_strong（CAS：Compare-And-Swap）
    // 若 a == expected，则 a = desired，返回 true
    // 否则 expected = a（最新值），返回 false
    int expected = 0;
    bool ok = a.compare_exchange_strong(expected, 100);
    std::cout << "  CAS(0→100): " << (ok ? "success" : "fail") << " a=" << a.load() << "\n";

    expected = 0;                       // 此时 a=100，expected 不匹配
    ok = a.compare_exchange_strong(expected, 200);
    std::cout << "  CAS(0→200): " << (ok ? "success" : "fail")
              << " a=" << a.load() << " expected now=" << expected << "\n\n";
}

// ============================================================
// 7.3：用 atomic<bool> 作为停止标志
// ============================================================
// 这是 atomic 最常见的用法之一：控制线程退出

void demo_stop_flag() {
    std::cout << "=== 7.3 atomic 停止标志 ===\n";
    std::atomic<bool> stop{false};

    std::thread worker([&stop] {
        int workDone = 0;
        while (!stop.load()) {  // 原子读，不需要 mutex
            // 模拟工作
            std::this_thread::sleep_for(10ms);
            ++workDone;
        }
        std::cout << "  Worker stopped after " << workDone << " iterations\n";
    });

    std::this_thread::sleep_for(50ms);
    stop.store(true);   // 原子写，通知 worker 停止
    worker.join();
    std::cout << "\n";
}

// ============================================================
// 7.4：getAndReset — 采样统计
// ============================================================
// exchange(0) 原子地读取并重置，常用于周期性采样

void demo_sample_reset() {
    std::cout << "=== 7.4 采样统计（getAndReset）===\n";
    std::atomic<int> eventCount{0};
    std::atomic<bool> done{false};

    // 事件产生线程
    std::thread producer([&] {
        for (int i = 0; i < 100; ++i) {
            eventCount.fetch_add(1);
            std::this_thread::sleep_for(1ms);
        }
        done.store(true);
    });

    // 每 20ms 采样一次
    std::thread sampler([&] {
        while (!done.load() || eventCount.load() > 0) {
            std::this_thread::sleep_for(20ms);
            int sample = eventCount.exchange(0);  // 原子读并重置
            if (sample > 0) {
                std::cout << "  Sample: " << sample << " events in last 20ms\n";
            }
        }
    });

    producer.join();
    sampler.join();
    std::cout << "\n";
}

// ============================================================
// 7.5：CAS 自旋锁（手写 mutex）
// ============================================================
// 理解 atomic 如何实现锁原语（面试常考原理题）

class SpinLock {
public:
    void lock() {
        // test_and_set：将 flag 设为 true，返回旧值
        // 若旧值是 false（锁未被持有），则成功获取
        // 若旧值是 true（锁已被持有），则自旋等待
        while (mFlag.test_and_set(std::memory_order_acquire)) {
            // 让出 CPU，避免白白消耗（x86 上用 pause 指令降低功耗）
            std::this_thread::yield();
        }
    }

    void unlock() {
        mFlag.clear(std::memory_order_release);
    }

private:
    std::atomic_flag mFlag = ATOMIC_FLAG_INIT;
};

void demo_spinlock() {
    std::cout << "=== 7.5 自旋锁（手写 mutex）===\n";
    SpinLock spin;
    int counter = 0;

    auto work = [&spin, &counter] {
        for (int i = 0; i < 10000; ++i) {
            spin.lock();
            ++counter;   // 临界区
            spin.unlock();
        }
    };

    std::thread t1(work), t2(work);
    t1.join(); t2.join();

    assert(counter == 20000);
    std::cout << "  counter = " << counter << " ✓\n";
    std::cout << "  (自旋锁适合临界区极短、竞争极低的场景)\n\n";
}

// ============================================================
// 7.6：atomic 不能解决的问题
// ============================================================
/*
 * atomic 只能保证单个变量的原子性。
 * 若需要保证多个变量之间的一致性，仍然需要 mutex。
 *
 * 错误示例：
 *   std::atomic<int> x{0}, y{0};
 *
 *   // 线程 A
 *   x.store(1);
 *   y.store(1);
 *
 *   // 线程 B
 *   if (y.load() == 1) {
 *       assert(x.load() == 1);  // ← 不保证！x 和 y 的更新不是原子的
 *   }
 *
 * 正确做法：用 mutex 保护 x 和 y 的一致性。
 */

// ==================== 练习题 ====================
/*
 * [练习 1] 实现一个原子的 ClampedCounter：
 *   - increment()：+1，但不超过 maxValue
 *   - 不使用 mutex，只用 compare_exchange_strong 实现
 *
 * [练习 2] 用 atomic<int> 实现一个简单的信号量（Semaphore）：
 *   - acquire()：若计数 > 0 则 -1，否则忙等
 *   - release()：+1
 *   注意：这是忙等（自旋）信号量，效率低，与 cv 版对比
 *
 * [练习 3] 以下代码是否有问题？
 *
 *   std::atomic<int> count{0};
 *   if (count.load() > 0) {
 *       count.fetch_sub(1);   // ← TOCTOU！load 和 sub 不是原子的
 *   }
 *   // 修复：用 CAS 循环，或改用 mutex
 */

int main() {
    demo_counter_compare();
    demo_atomic_ops();
    demo_stop_flag();
    demo_sample_reset();
    demo_spinlock();

    std::cout << "Level 7 Complete!\n";
    std::cout << "下一步 → level8_thread_pool.cpp：综合题——实现线程池\n";
    return 0;
}

/*
 * 编译运行：
 *   g++ -std=c++17 -pthread -O2 level7_atomic.cpp -o level7 && ./level7
 *
 * atomic 选择指南：
 *   std::atomic<bool>      — 标志位、停止信号
 *   std::atomic<int>       — 计数器、版本号
 *   std::atomic<T*>        — 无锁链表/栈的节点指针
 *   std::atomic_flag       — 最轻量的自旋锁原语
 */
