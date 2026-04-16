#include <iostream>
#include <ostream>
#include <thread>
#include <vector>

int gCounter = 0;
void incrementUnsafe(int n) {
  for (int i = 0; i < n; ++i) {
    gCounter++;
  }
}

void demo_counter_race() {
  std::cout << "=== 2.1 demo_counter_race ===" << std::endl;

  std::thread t1(incrementUnsafe, 500000);
  std::thread t2(incrementUnsafe, 500000);

  t1.join();
  t2.join();

  std::cout << "gCounter = " << gCounter << std::endl;
  std::cout << (gCounter == 1000000 ? "True" : "False") << std::endl;
}

std::vector<int> gQueue;
void unsafePop() {
  if (!gQueue.empty()) {
    std::this_thread::yield();

    gQueue.pop_back();

    std::cout << "Thread " << std::this_thread::get_id()
              << " saw non-empty but now size=" << gQueue.size() << std::endl;
  }
}

void demo_toctou() {
  std::cout << "=== 2.2 demo_toctou ===" << std::endl;

  gQueue = {1, 2, 3};

  std::thread t1(unsafePop);
  std::thread t2(unsafePop);

  t1.join();
  t2.join();
}

bool gReady = false;
int gData = 0;

void writer() {
  gData = 42;
  gReady = true;
}

void reader() {
  while (!gReady) {
    std::this_thread::yield();
  }

  std::cout << "Reader got data=" << gData << " (may not be 42 without sync!)"
            << std::endl;
}

void demo_visibility() {
  std::cout << "=== 2.3 visibility ===" << std::endl;

  gReady = false;
  gData = 0;

  std::thread t1(writer);
  std::thread t2(reader);

  t1.join();
  t2.join();
}

void exercise1() {
  int counter = 0;
  
  std::thread t1([&]() {
    counter++;
    counter++;
  });

  std::thread t2([&]() {
    counter++;
  });

  t1.join();
  t2.join();
}

void exercise2() {

}

void exercise3() {
  
}

int main() {

  demo_counter_race();

  demo_toctou();

  demo_visibility();

  return 0;
}