#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

std::mutex gMutex;
std::condition_variable gCV;
bool gReady;

void worker() {
  // hold mutex
  std::unique_lock<std::mutex> lock(gMutex);

  // wait notice and gReady
  gCV.wait(lock, []() { return gReady; });

  // woking
  std::cout << "working" << std::endl;
}

void demo_cv() {
  // init
  gReady = false;
  std::thread t(worker);

  // prepare
  std::cout << "do something" << std::endl;

  // prepare done;
  {
    std::lock_guard<std::mutex> lock(gMutex);
    gReady = true;
  }
  // notify worker
  gCV.notify_one();

  // wait worker;
  t.join();
}

void demo_wait_for() {
  bool ready = false;
  std::condition_variable cv;
  std::mutex mutex;

  std::thread t([&]() {
    std::unique_lock<std::mutex> lock(mutex);

    cv.wait_for(lock, std::chrono::milliseconds(100), [&]() { return ready; });

    std::cout << "working" << std::endl;
  });

  std::cout << "prepare" << std::endl;

  {
    std::lock_guard<std::mutex> lock(mutex);
    ready = true;
  }

  cv.notify_one();

  t.join();

  std::cout << "work done" << std::endl;
}

std::mutex gMutexAll;
std::condition_variable gCVAll;
bool gReadyAll = false;

void racer(int n) {
  std::unique_lock<std::mutex> lock(gMutexAll);

  gCVAll.wait(lock, []() { return gReadyAll; });

  std::cout << "running " << n << std::endl;
}

void demo_notify_all() {
  gReadyAll = false;

  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back(racer, i);
  }

  std::cout << "prepare" << std::endl;

  {
    std::lock_guard<std::mutex> lock(gMutexAll);
    gReadyAll = true;
  }

  gCVAll.notify_all();

  for (auto &t : threads) {
    t.join();
  }
}

void demo_print() {
  std::mutex mutex;
  std::condition_variable cv;
  int number = 1;
  const int Count = 10;

  std::thread t1([&]() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);

      cv.wait(lock, [&] { return number % 2 == 1 || number > Count; });

      if (number > Count) {
        break;
      }

      std::cout << "number = " << number++ << std::endl;

      cv.notify_one();
    }
  });

  std::thread t2([&]() {
    while (true) {

      std::unique_lock<std::mutex> lock(mutex);

      cv.wait(lock, [&] { return number % 2 == 0 || number > Count; });

      if (number > Count) {
        break;
      }

      std::cout << "number = " << number++ << std::endl;

      cv.notify_one();
    }
  });

  t1.join();
  t2.join();
}

class Flag {
public:
  void set() {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_flag = true;
    }
    m_cv.notify_all();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cv.wait(lock, [&] { return m_flag; });
  }

  void reset() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_flag = false;
  }

  bool waitFor(const std::chrono::milliseconds &time) {
    std::unique_lock<std::mutex> lock(m_mutex);
    return m_cv.wait_for(lock, time, [&] { return m_flag; });
  }

private:
  std::mutex m_mutex;
  std::condition_variable m_cv;
  bool m_flag = false;
};

void exercise2() {
  std::mutex mutex;
  std::condition_variable cv;
  int number = 1;
  const int N = 15;

  std::thread t1([&]() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);

      cv.wait(lock, [&] { return number % 3 == 1 || number > N; });

      if (number > N) {
        break;
      }

      std::cout << "A";

      number++;

      cv.notify_all();
    }
  });

  std::thread t2([&]() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);

      cv.wait(lock, [&] { return number % 3 == 2 || number > N; });

      if (number > N) {
        break;
      }

      std::cout << "B";

      number++;

      cv.notify_all();
    }
  });

  std::thread t3([&]() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);

      cv.wait(lock, [&] { return number % 3 == 0 || number > N; });

      if (number > N) {
        break;
      }

      std::cout << "C";

      number++;

      cv.notify_all();
    }
  });

  {
    std::lock_guard<std::mutex> lock(mutex);
    number = 1;
  }
  cv.notify_all();

  t1.join();
  t2.join();
  t3.join();

  std::cout << std::endl;
}

int main() {
  demo_cv();

  demo_wait_for();

  demo_notify_all();

  demo_print();

  exercise2();

  return 0;
}