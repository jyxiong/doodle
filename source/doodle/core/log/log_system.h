#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "spdlog/spdlog.h"

namespace doodle {

class LogSystem {
public:
  enum class LogLevel : uint8_t { debug, info, warn, error, fatal };

public:
  LogSystem();
  ~LogSystem();

  static LogSystem &getInstance();

  template <typename... TARGS> void log(LogLevel level, TARGS &&...args) {
    switch (level) {
    case LogLevel::debug:
      mLogger->debug(std::forward<TARGS>(args)...);
      break;
    case LogLevel::info:
      mLogger->info(std::forward<TARGS>(args)...);
      break;
    case LogLevel::warn:
      mLogger->warn(std::forward<TARGS>(args)...);
      break;
    case LogLevel::error:
      mLogger->error(std::forward<TARGS>(args)...);
      break;
    case LogLevel::fatal:
      mLogger->critical(std::forward<TARGS>(args)...);
      fatalCallback(std::forward<TARGS>(args)...);
      break;
    default:
      break;
    }
  }

  template <typename... TARGS> 
  void fatalCallback(std::string_view fmt, TARGS &&...args) {
    auto formatStr = std::vformat(fmt, std::make_format_args(std::forward<TARGS>(args)...));
    throw std::runtime_error(formatStr);
  }

private:
  std::shared_ptr<spdlog::logger> mLogger;
};

} // namespace doodle