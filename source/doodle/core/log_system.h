#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#include <spdlog/spdlog.h>

namespace doodle {

class LogSystem {
public:
  enum class LogLevel : uint8_t { Debug, Info, Warn, Error, Fatal };

public:
  LogSystem();
  ~LogSystem();

  static LogSystem &getInstance();

  template <typename... TARGS> void log(LogLevel level, TARGS &&...args) {
    switch (level) {
    case LogLevel::Debug:
      mLogger->debug(std::forward<TARGS>(args)...);
      break;
    case LogLevel::Info:
      mLogger->info(std::forward<TARGS>(args)...);
      break;
    case LogLevel::Warn:
      mLogger->warn(std::forward<TARGS>(args)...);
      break;
    case LogLevel::Error:
      mLogger->error(std::forward<TARGS>(args)...);
      break;
    case LogLevel::Fatal:
      mLogger->critical(std::forward<TARGS>(args)...);
      fatalCallback(std::forward<TARGS>(args)...);
      break;
    default:
      break;
    }
  }

  template <typename... TARGS>
  void fatalCallback(std::string_view fmt, TARGS &&...args) {
    auto formatStr =
        std::vformat(fmt, std::make_format_args(std::forward<TARGS>(args)...));
    throw std::runtime_error(formatStr);
  }

private:
  std::shared_ptr<spdlog::logger> mLogger;
};

} // namespace doodle

#define LOG_HELPER(LOG_LEVEL, ...)                                             \
  doodle::LogSystem::getInstance().log(                                        \
      LOG_LEVEL, "[" + std::string(__FUNCTION__) + "] " + __VA_ARGS__)

#define LOG_DEBUG(...)                                                         \
  LOG_HELPER(doodle::LogSystem::LogLevel::Debug, __VA_ARGS__)

#define LOG_INFO(...)                                                          \
  LOG_HELPER(doodle::LogSystem::LogLevel::Info, __VA_ARGS__)

#define LOG_WARN(...)                                                          \
  LOG_HELPER(doodle::LogSystem::LogLevel::Warn, __VA_ARGS__)

#define LOG_ERROR(...)                                                         \
  LOG_HELPER(doodle::LogSystem::LogLevel::Error, __VA_ARGS__)

#define LOG_FATAL(...)                                                         \
  LOG_HELPER(doodle::LogSystem::LogLevel::Fatal, __VA_ARGS__)
