/*
 * ============================================================
 * Level 4 — RAII 锁：lock_guard / unique_lock / scoped_lock
 * ============================================================
 *
 * 目标：告别裸 lock/unlock，用 RAII 保证异常安全
 *
 * RAII（Resource Acquisition Is Initialization）：
 *   资源在构造时获取，在析构时释放。
 *   无论正常退出还是异常退出，析构函数必然被调用 → 资源必然释放。
 *
 * C++ 提供三种锁的 RAII 包装：
 *   lock_guard   — 最简单，不能手动解锁
 *   unique_lock  — 最灵活，可手动控制，配合 condition_variable 必须用它
 *   scoped_lock  — C++17，同时锁多个 mutex，防死锁
 */

#include <mutex>
#include <thread>
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <chrono>

// ==================== 4.1：lock_guard — 最简单 ====================
//
// 构造时 lock()，析构时 unlock()。
// 不能手动解锁，不能转移所有权。

std::mutex gMutex;
int gCounter = 0;

void incrementWithGuard(int n) {
    for (int i = 0; i < n; ++i) {
        std::lock_guard<std::mutex> lock(gMutex);  // 【关键】RAII 加锁
        gCounter++;
        // lock 析构 → 自动 unlock，即使此处抛出异常也安全
    }
}

void demo_lock_guard() {
    std::cout << "=== 4.1 lock_guard ===\n";
    gCounter = 0;

    std::thread t1(incrementWithGuard, 500000);
    std::thread t2(incrementWithGuard, 500000);
    t1.join(); t2.join();

    assert(gCounter == 1000000);
    std::cout << "counter = " << gCounter << " ✓\n\n";
}

// ==================== 4.2：异常安全验证 ====================
//
// lock_guard 在异常路径下也会正确解锁

std::mutex gSafeMutex;
int gValue = 0;

void mayThrow(bool doThrow) {
    std::lock_guard<std::mutex> lock(gSafeMutex);
    gValue = 100;
    if (doThrow) {
        throw std::runtime_error("something went wrong");
        // lock 析构时仍然会 unlock！
    }
    gValue = 200;
}

void demo_exception_safety() {
    std::cout << "=== 4.2 异常安全 ===\n";

    try {
        mayThrow(true);   // 抛出异常
    } catch (const std::exception& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }

    // 验证锁已被释放（能成功获取锁）
    {
        std::lock_guard<std::mutex> lock(gSafeMutex);
        std::cout << "Lock released properly after exception. value=" << gValue << "\n\n";
    }
}

// ==================== 4.3：unique_lock — 灵活版 ====================
//
// 相比 lock_guard，unique_lock 额外支持：
//   1. 手动 lock() / unlock()（在锁外做耗时操作，减少持锁时间）
//   2. defer_lock（延迟加锁）
//   3. try_to_lock（非阻塞尝试）
//   4. 可移动（std::move 转移所有权）
//   5. 配合 condition_variable（必须用 unique_lock）

void demo_unique_lock() {
    std::cout << "=== 4.3 unique_lock ===\n";
    std::mutex m;

    // 4.3a：手动 unlock（减少持锁时间）
    {
        std::unique_lock<std::mutex> lock(m);
        std::cout << "  Holding lock\n";
        lock.unlock();   // 手动提前解锁
        // 在锁外做耗时操作（不阻塞其他线程）
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        lock.lock();     // 重新加锁
        std::cout << "  Holding lock again\n";
    }   // 析构时解锁

    // 4.3b：defer_lock（先不加锁）
    {
        std::unique_lock<std::mutex> lock(m, std::defer_lock);
        std::cout << "  owns_lock before manually locking: " << lock.owns_lock() << "\n";
        lock.lock();
        std::cout << "  owns_lock after manually locking:  " << lock.owns_lock() << "\n";
    }

    // 4.3c：try_to_lock（非阻塞）
    {
        std::unique_lock<std::mutex> lock(m, std::try_to_lock);
        if (lock.owns_lock()) {
            std::cout << "  Got lock immediately\n";
        } else {
            std::cout << "  Lock busy, gave up\n";
        }
    }
    std::cout << "\n";
}

// ==================== 4.4：scoped_lock — 同时锁多个 mutex（C++17）====================
//
// 当需要同时持有多个 mutex 时，有死锁风险：
//   线程 A: lock(m1) → lock(m2)
//   线程 B: lock(m2) → lock(m1)  → 死锁！
//
// scoped_lock 内部使用 std::lock() 的死锁避免算法，
// 原子地锁住所有 mutex，顺序由算法决定，不需要调用者操心。

std::mutex gM1, gM2;
int gX = 0, gY = 0;

void swapValues_buggy() {
    // ❌ 可能死锁（若另一个线程以相反顺序加锁）
    std::lock_guard<std::mutex> l1(gM1);
    std::lock_guard<std::mutex> l2(gM2);
    std::swap(gX, gY);
}

void swapValues_safe() {
    // ✓ scoped_lock 原子锁住两个
    std::scoped_lock lock(gM1, gM2);
    std::swap(gX, gY);
}

void demo_scoped_lock() {
    std::cout << "=== 4.4 scoped_lock（多锁防死锁）===\n";
    gX = 1; gY = 2;

    std::thread t1(swapValues_safe);
    std::thread t2(swapValues_safe);
    t1.join(); t2.join();

    std::cout << "gX=" << gX << " gY=" << gY << " (偶数次 swap 结果)\n\n";
}

// ==================== 4.5：使用规则总结 ====================
/*
 * ┌──────────────┬────────┬────────┬──────────┬─────────────────────────┐
 * │ 包装类        │ 手动解锁│ 多mutex│ 可移动   │ 适用场景                │
 * ├──────────────┼────────┼────────┼──────────┼─────────────────────────┤
 * │ lock_guard   │ ✗      │ ✗(1个) │ ✗        │ 简单临界区（最常用）    │
 * │ unique_lock  │ ✓      │ ✗(1个) │ ✓        │ 配合 cv，手动控制       │
 * │ scoped_lock  │ ✗      │ ✓(N个) │ ✗        │ 同时锁多个 mutex        │
 * └──────────────┴────────┴────────┴──────────┴─────────────────────────┘
 *
 * 口诀：
 *   简单场景 → lock_guard
 *   配合条件变量 → unique_lock（必须，因为 wait 需要临时解锁）
 *   多个锁 → scoped_lock
 */

// ==================== 4.6：线程安全 Stack（综合练习答案）====================

template <typename T>
class SafeStack {
public:
    void push(const T& val) {
        std::lock_guard<std::mutex> lock(mMutex);
        mData.push_back(val);
    }

    T pop() {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mData.empty()) throw std::runtime_error("stack is empty");
        T val = mData.back();
        mData.pop_back();
        return val;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mMutex);
        return mData.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mMutex);
        return mData.size();
    }

private:
    mutable std::mutex mMutex;  // mutable 允许在 const 方法中加锁
    std::vector<T> mData;
};

void demo_safe_stack() {
    std::cout << "=== 4.6 线程安全 Stack ===\n";
    SafeStack<int> stack;

    std::thread t1([&stack] {
        for (int i = 0; i < 100; ++i) stack.push(i);
    });
    std::thread t2([&stack] {
        for (int i = 100; i < 200; ++i) stack.push(i);
    });
    t1.join(); t2.join();

    assert(stack.size() == 200);
    std::cout << "Stack size = " << stack.size() << " ✓\n\n";
}

// ==================== 练习题 ====================
/*
 * [练习 1] 用 lock_guard 封装以下代码，使其异常安全：
 *
 *   std::mutex m;
 *   std::vector<int> data;
 *   void addItem(int x) {
 *       m.lock();
 *       if (x < 0) throw std::invalid_argument("negative");  // 泄漏锁！
 *       data.push_back(x);
 *       m.unlock();
 *   }
 *
 * [练习 2] 解释为什么 condition_variable::wait() 只接受 unique_lock？
 *   提示：wait() 内部需要临时解锁 mutex，让其他线程能 notify。
 *         lock_guard 没有 unlock() 方法，所以不能用。
 *
 * [练习 3] 用 scoped_lock 修复以下死锁：
 *
 *   void transfer(Account& from, Account& to, int amount) {
 *       from.mutex.lock();    // 线程A锁from，线程B锁to
 *       to.mutex.lock();      // 互相等待 → 死锁
 *       from.balance -= amount;
 *       to.balance   += amount;
 *       to.mutex.unlock();
 *       from.mutex.unlock();
 *   }
 */

int main() {
    demo_lock_guard();
    demo_exception_safety();
    demo_unique_lock();
    demo_scoped_lock();
    demo_safe_stack();

    std::cout << "Level 4 Complete!\n";
    std::cout << "下一步 → level5_condition_variable.cpp：线程间通知机制\n";
    return 0;
}

/*
 * 编译运行：
 *   g++ -std=c++17 -pthread level4_raii_locks.cpp -o level4 && ./level4
 */
