#pragma once

#include "doodle/core/log/log_system.h"

#define LOG_INFO(...) ::LogSystem::getLogger()->info(__VA_ARGS__)
#define LOG_ERROR(...) ::LogSystem::getLogger()->error(__VA_ARGS__)
#define LOG_WARN(...) ::LogSystem::getLogger()->warn(__VA_ARGS__)
#define LOG_DEBUG(...) ::LogSystem::getLogger()->debug(__VA_ARGS__)