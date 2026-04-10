# Tesla Codility 笔试 — C++ 并发编程知识点

> 岗位：Sr/Staff C++ Software Engineer, Infotainment
> 笔试时长：80 分钟，1 题，C++17

---

## 一、互斥锁（Mutex）体系

### 1.1 基础互斥 `std::mutex`

```cpp
#include <mutex>
std::mutex m;
m.lock();    // 阻塞直到获取锁
m.unlock();  // 释放锁
m.try_lock(); // 非阻塞尝试，失败返回 false
```

**注意**：`lock()` 后必须 `unlock()`，强烈建议用 RAII 包装。

---

### 1.2 超时互斥 `std::timed_mutex`

```cpp
std::timed_mutex tm;
if (tm.try_lock_for(std::chrono::milliseconds(2))) {
    // 获取成功
    tm.unlock();
} else {
    // 超时失败
}
// 也可用 try_lock_until(time_point)
```

**应用场景**：避免永久阻塞，实现带超时的资源获取。

---

### 1.3 可重入互斥 `std::recursive_mutex`

```cpp
std::recursive_mutex rm;
rm.lock();
rm.lock();   // 同一线程可再次加锁，不会死锁
rm.unlock();
rm.unlock(); // 加几次就要解几次
```

**应用场景**：递归函数内部需要加锁时。

---

### 1.4 读写锁 `std::shared_mutex`（C++17）

```cpp
#include <shared_mutex>
std::shared_mutex sm;

// 读者（共享锁）：多个线程可同时持有
std::shared_lock<std::shared_mutex> rLock(sm);

// 写者（独占锁）：排他，阻塞所有读写
std::unique_lock<std::shared_mutex> wLock(sm);
```

**应用场景**：读多写少，如缓存、配置表。

---

## 二、锁的 RAII 包装

### 2.1 `std::lock_guard` — 最简单

```cpp
{
    std::lock_guard<std::mutex> lock(m);  // 构造时加锁
    // ... 临界区
}  // 析构时自动解锁
```

**局限**：不能手动解锁，不能转移所有权。

---

### 2.2 `std::unique_lock` — 最灵活

```cpp
std::unique_lock<std::mutex> lock(m);             // 立即加锁
std::unique_lock<std::mutex> lock(m, std::defer_lock); // 延迟加锁
std::unique_lock<std::mutex> lock(m, std::try_to_lock); // 尝试加锁

lock.lock();    // 手动加锁
lock.unlock();  // 手动解锁（之后析构不会重复解锁）
lock.owns_lock(); // 是否持有锁
```

**必须配合 condition_variable 使用**（`cv.wait()` 需要 `unique_lock`）。

---

### 2.3 `std::scoped_lock` — 同时锁多个（C++17）

```cpp
std::scoped_lock lock(m1, m2);  // 原子地锁住两个，不会死锁
```

**原理**：内部使用死锁避免算法（等价于 `std::lock`）。

---

### 2.4 `std::lock()` + `defer_lock` — 手动多锁

```cpp
std::unique_lock lock1{m1, std::defer_lock};
std::unique_lock lock2{m2, std::defer_lock};
std::lock(lock1, lock2);  // 原子地锁住两个
```

---

## 三、原子操作 `std::atomic`

```cpp
#include <atomic>
std::atomic<int> counter{0};

counter.fetch_add(1);       // 原子加，返回旧值
counter.fetch_sub(1);       // 原子减，返回旧值
counter.load();             // 原子读
counter.store(10);          // 原子写
counter++;                  // 等价于 fetch_add(1)
counter.exchange(5);        // 原子交换，返回旧值

// CAS（Compare-And-Swap）
int expected = 0;
counter.compare_exchange_strong(expected, 1); // 若当前==expected，则写入1
```

**内存序（了解即可）**：
- `memory_order_relaxed`：最宽松，仅保证原子性
- `memory_order_acquire` / `memory_order_release`：同步语义
- `memory_order_seq_cst`（默认）：最强，全序一致

---

## 四、条件变量 `std::condition_variable`

```cpp
#include <condition_variable>
std::mutex m;
std::condition_variable cv;
bool ready = false;

// 等待方（消费者）
void consumer() {
    std::unique_lock lock(m);
    cv.wait(lock, []{ return ready; });  // 防虚假唤醒
    // 或带超时：
    cv.wait_for(lock, 200ms, []{ return ready; });
    cv.wait_until(lock, deadline, []{ return ready; });
}

// 通知方（生产者）
void producer() {
    {
        std::lock_guard lock(m);
        ready = true;
    }
    cv.notify_one();   // 唤醒一个等待线程
    // cv.notify_all(); // 唤醒所有等待线程
}
```

**关键点**：
1. `wait()` 必须传 `unique_lock`，不能传 `lock_guard`
2. 第二个参数（谓词 lambda）用于**防止虚假唤醒**，等价于 `while(!ready) cv.wait(lock);`
3. `notify_*` 不需要持有锁（但修改共享状态时要加锁）

---

## 五、经典并发模式

### 5.1 生产者-消费者（有界缓冲区）

- 两个条件变量：`not_full`（生产者等待）、`not_empty`（消费者等待）
- 或一个 `shared_mutex` + `size` 计数控制

### 5.2 读者-写者

- `std::shared_mutex` + `shared_lock`（读）+ `unique_lock`（写）

### 5.3 线程池

- 任务队列（`std::queue<std::function<void()>>`）
- 工作线程循环取任务
- `stop` 标志位（`std::atomic<bool>`）
- 析构时 `notify_all()` 唤醒所有线程并 `join()`

### 5.4 单例（线程安全）

- `std::call_once` + `std::once_flag`（推荐）
- 或 C++11 static 局部变量（自动线程安全）

---

## 六、C++17 相关特性（补充）

### 6.1 `std::thread`

```cpp
#include <thread>
std::thread t(func, arg1, arg2);
t.join();    // 等待线程结束
t.detach();  // 分离线程（析构不 join）
t.joinable(); // 是否可 join
```

### 6.2 `std::future` / `std::async`

```cpp
#include <future>
auto fut = std::async(std::launch::async, []{ return 42; });
int result = fut.get();  // 阻塞等待结果
```

### 6.3 `std::promise`

```cpp
std::promise<int> p;
std::future<int> f = p.get_future();
// 另一个线程：
p.set_value(100);
// 主线程：
int v = f.get();
```

### 6.4 `std::chrono` 时间工具

```cpp
using namespace std::chrono_literals;
std::this_thread::sleep_for(100ms);
auto deadline = std::chrono::steady_clock::now() + 500ms;
```

---

## 七、异常安全与边界条件

- **RAII 保证**：锁的 RAII 包装在异常情况下也能正确释放
- **析构安全**：析构函数中设置 `stop=true` → `notify_all()` → `join()` 所有线程
- **空队列/满队列**：边界条件，用条件变量等待
- **零线程池**：构造时线程数为 0 的处理
- **多次 notify**：`notify_all` vs `notify_one` 的选择（多消费者用 `notify_all`）

---

## 八、常见错误清单

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 忘记 `unlock()` | 死锁 | 用 RAII 包装 |
| `condition_variable::wait()` 不用谓词 | 虚假唤醒导致逻辑错误 | 传 lambda 谓词 |
| 两个线程按不同顺序锁 m1, m2 | 死锁 | `std::scoped_lock` 或 `std::lock` |
| 析构时不 `join` 线程 | `std::terminate` 崩溃 | 析构时先 `join()` |
| 在锁内 `notify_*` | 无错误，但会导致立即竞争 | notify 移到锁外 |
| `shared_ptr` 本体不线程安全 | 数据竞争 | 对对象本身加锁 |
