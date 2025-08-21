#pragma once

#include "doodle/core/log/log_system.h"

#define LOG_HELPER(LOG_LEVEL, ...)                                             \
  doodle::LogSystem::getInstance().log(LOG_LEVEL, "[" + std::string(__FUNCTION__) +    \
                                              "] " + __VA_ARGS__)

#define LOG_DEBUG(...) LOG_HELPER(doodle::LogSystem::LogLevel::debug, __VA_ARGS__)

#define LOG_INFO(...) LOG_HELPER(doodle::LogSystem::LogLevel::info, __VA_ARGS__)

#define LOG_WARN(...) LOG_HELPER(doodle::LogSystem::LogLevel::warn, __VA_ARGS__)

#define LOG_ERROR(...) LOG_HELPER(doodle::LogSystem::LogLevel::error, __VA_ARGS__)

#define LOG_FATAL(...) LOG_HELPER(doodle::LogSystem::LogLevel::fatal, __VA_ARGS__)