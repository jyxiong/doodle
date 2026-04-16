#include <exception>
#include <iostream>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <thread>
#include <vector>

std::mutex gMutex;
int gCounter = 0;

void incrementGuard(int n) {
  for (int i = 0; i < n; ++i) {
    std::lock_guard<std::mutex> lock(gMutex);
    gCounter++;
  }
}

void demo_guard() {
  std::thread t1(incrementGuard, 500000);
  std::thread t2(incrementGuard, 500000);

  t1.join();
  t2.join();

  std::cout << "gCounter = " << gCounter << std::endl;
}

void mayThrow() {
  std::lock_guard<std::mutex> lock(gMutex);
  gCounter = 100;

  throw std::runtime_error("throw");

  gCounter = 200;
}

void demo_throw() {
  try {
    mayThrow();
  } catch (const std::exception &e) {
    std::cout << "catch " << e.what() << std::endl;
  }

  std::lock_guard<std::mutex> lock2(gMutex);
}

void demo_unique() {
  {
    std::unique_lock<std::mutex> lock(gMutex);
    std::cout << "manuel unlock" << std::endl;
    lock.unlock();
    std::cout << "lock again" << std::endl;
    lock.lock();
  }

  {
    std::unique_lock<std::mutex> lock(gMutex, std::defer_lock);
    std::cout << "mauel lock" << std::endl;
    lock.lock();
  }

  {
    gMutex.lock();
    std::unique_lock<std::mutex> lock(gMutex, std::try_to_lock);
    if (lock.owns_lock()) {
      lock.lock();
      std::cout << "own" << std::endl;
    } else {
      std::cout << "busy" << std::endl;
    }
    gMutex.unlock();
  }

  {
    std::unique_lock<std::mutex> lock(gMutex, std::try_to_lock);
    if (lock.owns_lock()) {
      lock.unlock();
      std::cout << "own" << std::endl;
    } else {
      lock.lock();
      std::cout << "busy" << std::endl;
    }
  }
}

std::mutex gMutex1;
std::mutex gMutex2;
int gX, gY;

void swapValueSafe() {
  std::scoped_lock lock(gMutex1, gMutex2);
  std::swap(gX, gY);
}

void demo_scoped() {
  gX = 1;
  gY = 2;
  std::thread t1(swapValueSafe);
  std::thread t2(swapValueSafe);

  t1.join();
  t2.join();

  std::cout << "gX = " << gX << ", "
            << "gY = " << gY << std::endl;
}

template <typename T>
class StackSafe {
public:
  void push(const T& val) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_data.push_back(val);
  }

  T pop() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (empty()) throw std::runtime_error("empty");
    auto val = m_data.back();
    m_data.pop_back();
    return val;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_data.empty();
  }

private:
  mutable std::mutex m_mutex;
  std::vector<T> m_data;
};

int main() {
  demo_guard();

  demo_throw();

  demo_unique();

  demo_scoped();

  return 0;
}