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

class UnboundedQueue {
public:
  void push(int val) {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_quque.push(val);
    }
    m_notEmpty.notify_one();
  }

  int pop() {
    std::unique_lock<std::mutex> lock(m_mutex);

    m_notEmpty.wait(lock, [&]() {
      return !empty();
    });

    auto val = m_quque.back();
    m_quque.pop();
    return val;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_quque.empty();
  }

private:
  mutable std::mutex m_mutex;
  std::condition_variable m_notEmpty;

  std::queue<int> m_quque;
};

void demo1() {
  UnboundedQueue q;

  std::atomic<int> consumed{0};

  std::thread producer([&]() {
    for (int i = 0; i < 5; ++i) {
      q.push(i);
      std::cout << "  Push " << i << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  });

  std::thread comsumer([&]() {
    for (int i = 0;i< 5;++i) {
      int val = q.pop();
      std::cout << "  Pop  " << val << "\n";
      consumed++;
    }
  });

  producer.join();
  comsumer.join();
}

class ClosableQueue {
public:

void push(int val){
  {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_closed) {
      return;
    }
    m_queue.push(val);
  }

  m_cv.notify_one();
}

std::optional<int> pop() {
  std::unique_lock<std::mutex> lock(m_mutex);

  m_cv.wait(lock, [&]() {
    return !empty() || m_closed;
  });

  if (empty()) {
    return std::nullopt;
  }

  int val = m_queue.front();
  m_queue.pop();
  return val;
}

bool empty() {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_queue.empty();
}

void close() {
  
}

private:
  std::queue<int> m_queue;

  std::mutex m_mutex;
  std::condition_variable m_cv;
  bool m_closed = false;
};

int main() {
  demo1();

  return 0;
}