#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

std::mutex gMutex;
int gCounter = 0;

void incrementSafe(int n) {
  for (int i = 0; i < n; ++i) {
    gMutex.lock();
    gCounter++;
    gMutex.unlock();
  }
}

void demo_counter() {
  std::thread t1(incrementSafe, 500000);
  std::thread t2(incrementSafe, 500000);

  t1.join();
  t2.join();

  std::cout << "gCounter = " << gCounter << std::endl;
}

void incrementBatch(int n) {
  gMutex.lock();

  for (int i = 0; i < n; ++i) {
    gCounter++;
  }

  gMutex.unlock();
}

void demo_counter_batch() {
  const int N = 1000000;

  //
  gCounter = 0;
  auto t0 = std::chrono::high_resolution_clock::now();
  std::thread t1(incrementSafe, N / 2);
  std::thread t2(incrementSafe, N / 2);
  t1.join();
  t2.join();
  auto delta = std::chrono::high_resolution_clock::now() - t0;
  std::cout
      << "cost time "
      << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
      << std::endl;
  //
  gCounter = 1;
  t0 = std::chrono::high_resolution_clock::now();
  std::thread t3(incrementSafe, N / 2);
  std::thread t4(incrementSafe, N / 2);
  t3.join();
  t4.join();
  delta = std::chrono::high_resolution_clock::now() - t0;
  std::cout
      << "cost time "
      << std::chrono::duration_cast<std::chrono::milliseconds>(delta).count()
      << std::endl;
}

void exercise1() {
  int count = 0;
  std::mutex m;

  auto work = [&] {
    for (int i = 0; i < 10; ++i) {
      m.lock();
      count++;
      m.unlock();
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < 10; ++i)
    ts.emplace_back(work);

  for (auto &t : ts)
    t.join();

  assert(count == 100);
}

class StackSafe {
public:
  StackSafe() = default;
  ~StackSafe() = default;

  void push(int val) {
    m_mutex.lock();
    m_data.push_back(val);
    m_mutex.unlock();
  }

  int pop() {
    m_mutex.lock();
    if (!empty()) {
      auto val = m_data.back();
      m_data.pop_back();
      m_mutex.unlock();

      return val;
    } else {
      m_mutex.unlock();
      throw(std::runtime_error("empty"));
    }
  }

  bool empty() {
    m_mutex.lock();
    auto val = m_data.empty();
    m_mutex.unlock();
    return val;
  }

private:
  std::mutex m_mutex;
  std::vector<int> m_data;
};

int main() {
  demo_counter();
  demo_counter_batch();
  exercise1();
}