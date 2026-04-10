/*
 * ============================================================
 * Level 6 — 生产者-消费者：从无界到有界，一步步构建
 * ============================================================
 *
 * 目标：综合运用 mutex + condition_variable，掌握经典并发模式
 *
 * 本文件分四个版本逐步演进：
 *   V1：无界队列（最简单）
 *   V2：无界队列 + 优雅停止
 *   V3：有界队列（capacity 限制）
 *   V4：多生产者多消费者（MPMC）
 */

#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <iostream>
#include <vector>
#include <atomic>
#include <cassert>
#include <chrono>
#include <optional>

using namespace std::chrono_literals;

// ============================================================
// V1：无界阻塞队列（最简版）
// ============================================================
// 只有一个 condition_variable：not_empty
// push 永不阻塞，pop 在空时阻塞

class UnboundedQueue {
public:
    void push(int val) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mQueue.push(val);
        }
        mNotEmpty.notify_one();  // 通知消费者
    }

    int pop() {
        std::unique_lock<std::mutex> lock(mMutex);
        mNotEmpty.wait(lock, [this] { return !mQueue.empty(); });
        int val = mQueue.front();
        mQueue.pop();
        return val;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mMutex);
        return mQueue.empty();
    }

private:
    std::queue<int> mQueue;
    mutable std::mutex mMutex;
    std::condition_variable mNotEmpty;
};

void demo_v1() {
    std::cout << "=== V1：无界队列 ===\n";
    UnboundedQueue q;
    std::atomic<int> consumed{0};

    std::thread producer([&q] {
        for (int i = 0; i < 5; ++i) {
            q.push(i);
            std::cout << "  Push " << i << "\n";
            std::this_thread::sleep_for(10ms);
        }
    });

    std::thread consumer([&q, &consumed] {
        for (int i = 0; i < 5; ++i) {
            int val = q.pop();
            std::cout << "  Pop  " << val << "\n";
            consumed++;
        }
    });

    producer.join();
    consumer.join();
    assert(consumed == 5);
    std::cout << "\n";
}

// ============================================================
// V2：无界队列 + 优雅停止
// ============================================================
// 问题：V1 的消费者在生产完成后不知道何时退出
// 方案：增加 close() 方法，pop 在关闭且空时返回 nullopt

class ClosableQueue {
public:
    void push(int val) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mClosed) return;   // 关闭后忽略新数据
            mQueue.push(val);
        }
        mCV.notify_one();
    }

    // 返回 nullopt 表示队列已关闭且为空
    std::optional<int> pop() {
        std::unique_lock<std::mutex> lock(mMutex);
        mCV.wait(lock, [this] {
            return !mQueue.empty() || mClosed;
        });
        if (mQueue.empty()) return std::nullopt;  // 关闭信号
        int val = mQueue.front();
        mQueue.pop();
        return val;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mClosed = true;
        }
        mCV.notify_all();  // 唤醒所有消费者，让它们检测到关闭
    }

private:
    std::queue<int> mQueue;
    std::mutex mMutex;
    std::condition_variable mCV;
    bool mClosed = false;
};

void demo_v2() {
    std::cout << "=== V2：可关闭队列 ===\n";
    ClosableQueue q;

    std::thread producer([&q] {
        for (int i = 0; i < 5; ++i) {
            q.push(i);
            std::this_thread::sleep_for(5ms);
        }
        q.close();  // 生产完成，关闭队列
        std::cout << "  Producer: closed queue\n";
    });

    std::thread consumer([&q] {
        while (auto val = q.pop()) {  // 直到返回 nullopt 时退出
            std::cout << "  Consumer got: " << *val << "\n";
        }
        std::cout << "  Consumer: queue closed, exiting\n";
    });

    producer.join();
    consumer.join();
    std::cout << "\n";
}

// ============================================================
// V3：有界阻塞队列（Bounded Queue）
// ============================================================
// 增加容量限制：队满时生产者阻塞（背压机制）
// 需要两个 cv：not_empty（消费者等待）和 not_full（生产者等待）
//
// 这是最常被考到的版本！

class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : mCapacity(capacity), mClosed(false) {}

    // 入队：队满时阻塞
    bool push(int val) {
        std::unique_lock<std::mutex> lock(mMutex);
        // 等待条件：未满 OR 已关闭
        mNotFull.wait(lock, [this] {
            return mQueue.size() < mCapacity || mClosed;
        });
        if (mClosed) return false;
        mQueue.push(val);
        lock.unlock();
        mNotEmpty.notify_one();  // 通知消费者
        return true;
    }

    // 出队：队空时阻塞
    std::optional<int> pop() {
        std::unique_lock<std::mutex> lock(mMutex);
        // 等待条件：非空 OR 已关闭
        mNotEmpty.wait(lock, [this] {
            return !mQueue.empty() || mClosed;
        });
        if (mQueue.empty()) return std::nullopt;
        int val = mQueue.front();
        mQueue.pop();
        lock.unlock();
        mNotFull.notify_one();   // 通知生产者
        return val;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mClosed = true;
        }
        mNotFull.notify_all();   // 唤醒所有等待中的生产者
        mNotEmpty.notify_all();  // 唤醒所有等待中的消费者
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mMutex);
        return mQueue.size();
    }

private:
    const size_t mCapacity;
    std::queue<int> mQueue;
    mutable std::mutex mMutex;
    std::condition_variable mNotFull;   // 生产者等待这个
    std::condition_variable mNotEmpty;  // 消费者等待这个
    bool mClosed;
};

void demo_v3() {
    std::cout << "=== V3：有界队列（capacity=3）===\n";
    BoundedQueue q(3);
    std::atomic<int> count{0};

    // 生产者：快速推入 10 个
    std::thread producer([&q] {
        for (int i = 0; i < 10; ++i) {
            if (q.push(i)) {
                std::cout << "  Push " << i << " (size=" << q.size() << ")\n";
            }
            // 不 sleep，尽量快推，会被 not_full 阻塞
        }
        q.close();
    });

    // 消费者：慢慢消费
    std::thread consumer([&q, &count] {
        while (auto val = q.pop()) {
            std::cout << "  Pop  " << *val << "\n";
            count++;
            std::this_thread::sleep_for(20ms);  // 消费慢，背压生效
        }
    });

    producer.join();
    consumer.join();
    assert(count == 10);
    std::cout << "\n";
}

// ============================================================
// V4：多生产者多消费者（MPMC）
// ============================================================
// V3 的队列本身就支持 MPMC！因为 mutex 保证了对队列的互斥访问。
// 只需要将多个线程指向同一个 BoundedQueue 即可。

void demo_v4() {
    std::cout << "=== V4：多生产者多消费者 ===\n";
    BoundedQueue q(5);
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};

    const int PRODUCERS = 3;
    const int CONSUMERS = 2;
    const int ITEMS_EACH = 4;

    // 3 个生产者
    std::vector<std::thread> producers;
    for (int i = 0; i < PRODUCERS; ++i) {
        producers.emplace_back([&q, &produced, i] {
            for (int j = 0; j < ITEMS_EACH; ++j) {
                int val = i * 100 + j;
                q.push(val);
                produced++;
            }
        });
    }

    // 等所有生产者完成后关闭
    std::thread closer([&q, &producers] {
        for (auto& t : producers) t.join();
        q.close();
    });

    // 2 个消费者
    std::vector<std::thread> consumers;
    for (int i = 0; i < CONSUMERS; ++i) {
        consumers.emplace_back([&q, &consumed] {
            while (auto val = q.pop()) {
                consumed++;
            }
        });
    }

    closer.join();
    for (auto& t : consumers) t.join();

    std::cout << "  Produced: " << produced << "\n";
    std::cout << "  Consumed: " << consumed << "\n";
    assert(produced == PRODUCERS * ITEMS_EACH);
    assert(consumed == produced);
    std::cout << "  ✓ 全部处理完毕\n\n";
}

// ==================== 练习题 ====================
/*
 * [练习 1] 改造 BoundedQueue，为 push 增加超时版本：
 *   bool push(int val, std::chrono::milliseconds timeout);
 *   // 超时返回 false
 *   提示：将 mNotFull.wait 改为 mNotFull.wait_for
 *
 * [练习 2] 实现泛型版本 BoundedQueue<T> 支持任意类型。
 *
 * [练习 3] 为什么有界队列需要两个 condition_variable，
 *          而无界队列只需要一个？
 *   答：无界队列 push 永不阻塞，不需要等待"未满"条件。
 *       有界队列的生产者在队满时需要等待，需要额外的 not_full cv。
 *
 * [练习 4]（进阶）实现一个双端阻塞队列 BlockingDeque<T>：
 *   支持 pushFront/pushBack/popFront/popBack，容量有限。
 */

int main() {
    demo_v1();
    demo_v2();
    demo_v3();
    demo_v4();

    std::cout << "Level 6 Complete!\n";
    std::cout << "下一步 → level7_atomic.cpp：无锁编程基础\n";
    return 0;
}

/*
 * 编译运行：
 *   g++ -std=c++17 -pthread level6_producer_consumer.cpp -o level6 && ./level6
 *
 * 核心设计模式总结：
 *
 *  无界队列（1个cv）：
 *    push: lock → enqueue → unlock → notify_one(not_empty)
 *    pop:  lock → wait(not_empty) → dequeue → unlock
 *
 *  有界队列（2个cv）：
 *    push: lock → wait(not_full) → enqueue → unlock → notify_one(not_empty)
 *    pop:  lock → wait(not_empty) → dequeue → unlock → notify_one(not_full)
 *
 *  关闭队列：
 *    close: lock → closed=true → unlock → notify_all(所有cv)
 */
