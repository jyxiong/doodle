#include "doodle/core/base/macro.h"

#include "volk.h"

int main() {
  LogSystem::init();
  LOG_INFO("Hello world!");

  VkResult r;
  uint32_t version;
  void *ptr;

//   /* This won't compile if the appropriate Vulkan platform define isn't set. */
//   ptr =
// #if defined(_WIN32)
//       &vkCreateWin32SurfaceKHR;
// #elif defined(__linux__) || defined(__unix__)
//       &vkCreateXlibSurfaceKHR;
// #elif defined(__APPLE__)
//       &vkCreateMacOSSurfaceMVK;
// #else
//       /* Platform not recogized for testing. */
//       NULL;
// #endif

  /* Try to initialize volk. This might not work on CI builds, but the
   * above should have compiled at least. */
  r = volkInitialize();
  if (r != VK_SUCCESS) {
    LOG_ERROR("volkInitialize failed!\n");
    return -1;
  }

  version = volkGetInstanceVersion();
  auto major = VK_VERSION_MAJOR(version);
  auto minor = VK_VERSION_MINOR(version);
  auto patch = VK_VERSION_PATCH(version);
  LOG_INFO("Vulkan version {}.{}.{} initialized.\n", major, minor, patch);

  return 0;
}