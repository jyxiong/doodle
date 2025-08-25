#include "log_system.h"
#include <spdlog/sinks/stdout_color_sinks.h>

std::shared_ptr<spdlog::logger> LogSystem::sLogger = nullptr;

void LogSystem::init() {
  sLogger = spdlog::stdout_color_mt("console");
  sLogger->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
  sLogger->set_level(spdlog::level::debug);
}

std::shared_ptr<spdlog::logger> &LogSystem::getLogger() { return sLogger; }