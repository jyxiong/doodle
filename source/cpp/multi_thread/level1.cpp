#include <functional>
#include <iostream>
#include <ostream>
#include <system_error>
#include <thread>
#include <vector>

void hello(int id) { std::cout << "Hello from thread " << id << std::endl; }

void demo_basic() {
  std::cout << "=== 1.1 demo_basic ===" << std::endl;

  std::thread th1(hello, 1);
  std::thread th2(hello, 2);
  std::thread th3(hello, 3);

  th1.join();
  th3.join();
  th2.join();
}

void accumulate(const std::vector<int> &data, int start, int end, int &res) {
  res = 0;
  for (int i = start; i < end; ++i) {
    res += data[i];
  }
}

void demo_ref() {
  std::cout << "=== 1.2 demo_ref ===" << std::endl;

  std::vector<int> data(100, 1);
  int res1, res2;

  std::thread th1(accumulate, std::cref(data), 0, 50, std::ref(res1));
  std::thread th2(accumulate, std::cref(data), 50, 100, std::ref(res2));

  th1.join();
  th2.join();

  std::cout << "res1 = " << res1 << " res2 = " << res2 << std::endl;
}

void demo_lambda() {
  std::cout << "=== 1.3 demo_lambda ===" << std::endl;

  int sharedVal = 0;
  auto func = [&sharedVal]() {
    sharedVal = 42;
    std::cout << "set value to " << sharedVal << std::endl;
  };

  std::thread th(func);

  th.join();

  std::cout << "Main thread value " << sharedVal << std::endl;
}

void demo_thread_array() {
  std::cout << "=== 1.4 demo_thread_array ===" << std::endl;

  const int N = 5;
  std::vector<std::thread> threads;
  threads.reserve(N);

  for (int i = 0; i < N; ++i) {
    threads.emplace_back(
        [i]() { std::cout << "Thread " << i << " is running" << std::endl; });
  }

  for (auto &thread : threads) {
    thread.join();
  }
}

void demo_joinable() {
  std::cout << "=== 1.5 demo_joinable ===" << std::endl;

  std::thread th(hello, 99);

  std::cout << "is thread joinable before join " << th.joinable() << std::endl;

  bool shouldJoin = false;

  if (shouldJoin && th.joinable()) {
    th.join();
  } else if (th.joinable()) {
    th.detach();
  }

  std::cout << "is thread joinable after join " << th.joinable() << std::endl;
}

void fill(std::vector<int> &data, int start, int end, int val) {
  for (int i = start; i < end; ++i) {
    data[i] = val;
  }
}

void exercise1() {
  std::cout << "=== exercise1 ===" << std::endl;

  const int Count = 10;
  const int N = 3;

  int numPerThread = Count / N;
  int delta = Count % N;

  std::vector<int> data(Count);

  std::vector<std::thread> threads;
  threads.reserve(N);

  for (int i = 0; i < N; ++i) {
    // 更均衡的分法：前 delta 个线程各多分 1 个元素
    int start, end;
    if (i < delta) {
      start = i * (numPerThread + 1);
      end = start + (numPerThread + 1);
    } else {
      start = i * numPerThread + delta;
      end = start + numPerThread;
    }

    threads.emplace_back(fill, std::ref(data), start, end, i);
  }

  for (auto &thread : threads) {
    thread.join();
  }

  std::cout << "data has been set to ";
  for (int i = 0; i < Count; ++i) {
    std::cout << data[i] << " ";
  }
  std::cout << std::endl;
}

void exercise2() {
  std::cout << "=== exercise2 ===" << std::endl;

  const int Count = 101;
  const int N = 4;

  int numPerThread = Count / N;
  int delta = Count % N;

  std::vector<int> data(Count, 1);

  std::vector<std::thread> threads;
  threads.reserve(N);
  std::vector<int> results;
  results.resize(N);  // 必须 resize，reserve 只预留容量不创建元素，results[i] 会 UB

  for (int i = 0; i < N; ++i) {
    int start, end;
    if (i < delta) {
      start = i * (numPerThread + 1);
      end = start + (numPerThread + 1);
    } else {
      start = i * numPerThread + delta;
      end = start + numPerThread;
    }

    threads.emplace_back(accumulate, std::cref(data), start, end,
                         std::ref(results[i]));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int res = 0;
  for (int i = 0; i < N; ++i) {
    res += results[i];
  }

  std::cout << "sum of data " << res << std::endl;
}

int main() {
  demo_basic();

  demo_ref();

  demo_lambda();

  demo_thread_array();

  demo_joinable();

  exercise1();

  exercise2();

  return 0;
}