/*
 * ============================================================
 * Level 8 — 线程池：综合题，从零构建
 * ============================================================
 *
 * 目标：把 Level 1~7 的所有知识综合起来，实现一个生产级线程池
 *
 * 本文件分 4 步逐步构建，每步都是可运行的完整代码：
 *   Step 1：最简单的线程池（固定线程，无超时，无返回值）
 *   Step 2：优雅停止（等待所有任务完成）
 *   Step 3：支持返回值（std::future）
 *   Step 4：完整版（异常安全、边界检查）
 *
 * 用到的知识：
 *   Level 1：std::thread 创建和 join
 *   Level 3：std::mutex
 *   Level 4：lock_guard / unique_lock
 *   Level 5：condition_variable
 *   Level 6：任务队列（生产者消费者）
 *   Level 7：atomic<bool> 停止标志
 */

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <atomic>
#include <future>
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <chrono>
#include <type_traits>

using namespace std::chrono_literals;

// ============================================================
// Step 1：最骨干的线程池
// ============================================================
// 只实现核心：N 个工作线程 + 无界任务队列

class ThreadPoolV1 {
public:
    explicit ThreadPoolV1(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            mWorkers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mMutex);
                        // 等待：有任务 OR 停止
                        mCV.wait(lock, [this] {
                            return !mTasks.empty() || mStop;
                        });
                        if (mStop && mTasks.empty()) return;  // 退出
                        task = std::move(mTasks.front());
                        mTasks.pop();
                    }
                    task();  // 在锁外执行任务
                }
            });
        }
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mTasks.push(std::move(task));
        }
        mCV.notify_one();
    }

    ~ThreadPoolV1() {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStop = true;
        }
        mCV.notify_all();
        for (auto& t : mWorkers) t.join();
    }

private:
    std::vector<std::thread> mWorkers;
    std::queue<std::function<void()>> mTasks;
    std::mutex mMutex;
    std::condition_variable mCV;
    bool mStop = false;
};

void demo_v1() {
    std::cout << "=== Step 1：最骨干线程池 ===\n";
    ThreadPoolV1 pool(4);
    std::atomic<int> count{0};

    for (int i = 0; i < 8; ++i) {
        pool.submit([&count, i] {
            std::this_thread::sleep_for(10ms);
            count++;
            std::cout << "  Task " << i << " done\n";
        });
    }
    // 析构时等待所有任务完成
}  // pool 在此析构，join 所有线程

void run_v1() {
    demo_v1();
    std::cout << "\n";
}

// ============================================================
// Step 2：加入 waitForDone()（等待队列清空）
// ============================================================

class ThreadPoolV2 {
public:
    explicit ThreadPoolV2(size_t numThreads) : mStop(false), mActiveTasks(0) {
        for (size_t i = 0; i < numThreads; ++i) {
            mWorkers.emplace_back([this] { workerLoop(); });
        }
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mStop) throw std::runtime_error("Pool is stopped");
            mTasks.push(std::move(task));
            mActiveTasks++;  // 提交即计为活跃
        }
        mCV.notify_one();
    }

    // 阻塞直到所有提交的任务全部完成
    void waitForDone() {
        std::unique_lock<std::mutex> lock(mMutex);
        mDoneCV.wait(lock, [this] {
            return mTasks.empty() && mActiveTasks == 0;
        });
    }

    ~ThreadPoolV2() {
        waitForDone();  // 先等所有任务完成
        {
            std::lock_guard<std::mutex> lock(mMutex);
            mStop = true;
        }
        mCV.notify_all();
        for (auto& t : mWorkers) t.join();
    }

private:
    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCV.wait(lock, [this] {
                    return !mTasks.empty() || mStop;
                });
                if (mStop && mTasks.empty()) return;
                task = std::move(mTasks.front());
                mTasks.pop();
            }
            task();
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mActiveTasks--;
                if (mTasks.empty() && mActiveTasks == 0) {
                    mDoneCV.notify_all();  // 通知 waitForDone
                }
            }
        }
    }

    std::vector<std::thread> mWorkers;
    std::queue<std::function<void()>> mTasks;
    std::mutex mMutex;
    std::condition_variable mCV;
    std::condition_variable mDoneCV;
    bool mStop;
    int mActiveTasks;
};

void demo_v2() {
    std::cout << "=== Step 2：支持 waitForDone ===\n";
    ThreadPoolV2 pool(3);
    std::atomic<int> count{0};

    for (int i = 0; i < 6; ++i) {
        pool.submit([&count, i] {
            std::this_thread::sleep_for(20ms);
            count++;
        });
    }

    pool.waitForDone();
    assert(count == 6);
    std::cout << "  All 6 tasks completed: count=" << count << " ✓\n\n";
}

// ============================================================
// Step 3：支持返回值（std::future）
// ============================================================

class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads) : mStop(false) {
        if (numThreads == 0) throw std::invalid_argument("numThreads must be > 0");
        for (size_t i = 0; i < numThreads; ++i) {
            mWorkers.emplace_back([this] { workerLoop(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        shutdown();
    }

    // 无返回值版本
    void submit(std::function<void()> task) {
        enqueue(std::move(task));
    }

    // 有返回值版本：返回 future，可异步获取结果
    template <typename F>
    auto submitWithResult(F&& f) -> std::future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;
        auto promise = std::make_shared<std::promise<R>>();
        auto future  = promise->get_future();

        enqueue([promise, f = std::forward<F>(f)]() mutable {
            try {
                if constexpr (std::is_void_v<R>) {
                    f();
                    promise->set_value();
                } else {
                    promise->set_value(f());
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });

        return future;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mStop) return;
            mStop = true;
        }
        mCV.notify_all();
        for (auto& t : mWorkers) {
            if (t.joinable()) t.join();
        }
    }

    size_t pendingTasks() const {
        std::lock_guard<std::mutex> lock(mMutex);
        return mTasks.size();
    }

private:
    void enqueue(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            if (mStop) throw std::runtime_error("ThreadPool is shut down");
            mTasks.push(std::move(task));
        }
        mCV.notify_one();
    }

    void workerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCV.wait(lock, [this] {
                    return !mTasks.empty() || mStop;
                });
                if (mStop && mTasks.empty()) return;
                task = std::move(mTasks.front());
                mTasks.pop();
            }
            // 在锁外执行，任务异常由 promise 捕获
            try {
                task();
            } catch (const std::exception& e) {
                std::cerr << "[Pool] uncaught exception: " << e.what() << "\n";
            } catch (...) {
                std::cerr << "[Pool] unknown exception\n";
            }
        }
    }

    std::vector<std::thread> mWorkers;
    std::queue<std::function<void()>> mTasks;
    mutable std::mutex mMutex;
    std::condition_variable mCV;
    bool mStop;
};

void demo_final() {
    std::cout << "=== Step 3：完整版线程池 ===\n";
    ThreadPool pool(4);

    // 提交无返回值任务
    std::atomic<int> count{0};
    for (int i = 0; i < 5; ++i) {
        pool.submit([&count] { count++; });
    }

    // 提交有返回值任务
    std::vector<std::future<int>> futures;
    for (int i = 1; i <= 5; ++i) {
        futures.push_back(pool.submitWithResult([i] { return i * i; }));
    }

    // 收集结果
    int sumOfSquares = 0;
    for (auto& f : futures) {
        sumOfSquares += f.get();  // 阻塞等待每个结果
    }

    pool.shutdown();  // 等待所有任务完成

    assert(count == 5);
    assert(sumOfSquares == 1 + 4 + 9 + 16 + 25);  // 55
    std::cout << "  Fire-and-forget count=" << count << " ✓\n";
    std::cout << "  Sum of squares 1²+..+5²=" << sumOfSquares << " ✓\n\n";
}

// ==================== 设计要点速记 ====================
/*
 * 线程池三个核心设计决策：
 *
 * 1. 工作线程的退出条件：
 *    if (mStop && mTasks.empty()) return;
 *    → 要先处理完剩余任务再退出（graceful shutdown）
 *    → 若要立刻停止，改为 if (mStop) return;
 *
 * 2. 任务在锁内还是锁外执行？
 *    → 锁外！在锁内执行任务会导致其他线程无法提交新任务。
 *
 * 3. 析构顺序：
 *    mStop=true → notify_all → join 所有 worker
 *    → 顺序不能错：先通知，再等待
 *
 * 练习：
 * [练习 1] 为 ThreadPool 添加 submit(task, delay) 方法：
 *   提交一个延迟执行的任务。
 *
 * [练习 2] 实现有界任务队列版本：任务队列满时 submit 阻塞。
 *
 * [练习 3] 添加 size_t activeWorkers() 方法，返回正在执行任务的线程数。
 */

int main() {
    run_v1();
    demo_v2();
    demo_final();

    std::cout << "Level 8 Complete! 恭喜完成全部 8 关！\n\n";
    std::cout << "学习路径回顾:\n";
    std::cout << "  L1: 线程创建/join\n";
    std::cout << "  L2: 认识数据竞争\n";
    std::cout << "  L3: mutex 修复竞争\n";
    std::cout << "  L4: RAII 锁（lock_guard/unique_lock/scoped_lock）\n";
    std::cout << "  L5: 条件变量（wait/notify）\n";
    std::cout << "  L6: 生产者消费者（无界→有界→MPMC）\n";
    std::cout << "  L7: atomic（无锁计数/标志/CAS）\n";
    std::cout << "  L8: 线程池（综合应用）\n";
    return 0;
}

/*
 * 编译运行：
 *   g++ -std=c++17 -pthread -O2 level8_thread_pool.cpp -o level8 && ./level8
 */
