#include "doodle/core/log/log_system.h"

#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

using namespace doodle;

LogSystem::LogSystem() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::trace);
  console_sink->set_pattern("[%^%l%$] %v");

  const spdlog::sinks_init_list sink_list = {console_sink};

  spdlog::init_thread_pool(8192, 1);

  mLogger = std::make_shared<spdlog::async_logger>(
      "muggle_logger", sink_list.begin(), sink_list.end(),
      spdlog::thread_pool(), spdlog::async_overflow_policy::block);
  mLogger->set_level(spdlog::level::trace);

  spdlog::register_logger(mLogger);
}

LogSystem::~LogSystem() {
  mLogger->flush();
  spdlog::drop_all();
}

LogSystem &LogSystem::getInstance() {
  static LogSystem logSystem;
  return logSystem;
}
