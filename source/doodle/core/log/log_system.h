#pragma once

#include <memory>

#include "spdlog/spdlog.h"

class LogSystem {
public:
  static void init();
  static std::shared_ptr<spdlog::logger> &getLogger();

private:
  static std::shared_ptr<spdlog::logger> sLogger;
};
