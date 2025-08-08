#include <cstdint>
#include <stdexcept>

#include "frame_graph.h"

#define CHECK(condition)                                                       \
  if (!(condition)) {                                                          \
    throw std::runtime_error("Check failed: " #condition);                     \
  }

#define REQUIRE(condition)                                                     \
  if (!(condition)) {                                                          \
    throw std::runtime_error("Requirement failed: " #condition);               \
  }

#define REQUIRE_FALSE(condition)                                               \
  if ((condition)) {                                                           \
    throw std::runtime_error("Requirement failed: " #condition);               \
  }

struct FrameGraphTexture {
  struct Desc {
    uint32_t width;
    uint32_t height;
  };

  FrameGraphTexture() = default;
  explicit FrameGraphTexture(int32_t id_) : id{id_} {}
  FrameGraphTexture(FrameGraphTexture &&) noexcept = default;

  void create(const Desc &, void *) {
    static auto lastId = 0;
    id = ++lastId;
  }
  void destroy(const Desc &, void *) {}

  void preRead(const Desc &, uint32_t, void *) const {}
  void preWrite() const {
    // Invalid signature, should not be called.
    CHECK(false);
  }

  int32_t id{-1};
};

constexpr auto markAsExecuted = [](const auto &data,
                                   const PassResources &,
                                   void *) { data.executed = true; };

struct NoData
{

};

void test0() {
  FrameGraph fg;
  fg.addCallbackPass<NoData>(
      "Dummy",
      [](const FrameGraph::Builder &, auto &) {},
      [](const auto &, const PassResources &, void *) {});

  return;
}

void test1() {
  FrameGraph fg;

  struct TestPass {
    ResourceId foo;
    ResourceId bar;
    mutable bool executed{false};
  };
  auto &testPass = fg.addCallbackPass<TestPass>(
      "Test pass",
      [&fg](FrameGraph::Builder &builder, TestPass &data) {
        data.foo = builder.create<FrameGraphTexture>("foo", {128, 128});
        data.foo = builder.write(data.foo);
        REQUIRE(fg.isValid(data.foo));

        data.bar = builder.create<FrameGraphTexture>("bar", {256, 256});
        data.bar = builder.write(data.bar);
        REQUIRE(fg.isValid(data.bar));

        builder.setSideEffect();
      },
      [](const TestPass &data, PassResources &resources, void *) {
        CHECK(resources.get<FrameGraphTexture>(data.foo).id == 1);
        CHECK(resources.get<FrameGraphTexture>(data.bar).id == 2);

        data.executed = true;
      });

  fg.compile();
  //   std::ofstream{"basic_graph.dot"} << fg;

  fg.execute();
  return;
}

void test2() {
  static constexpr auto kBackbufferId = 777;

  FrameGraph fg;

  const auto backbuffer =
      fg.import("Backbuffer", {1280, 720}, FrameGraphTexture{kBackbufferId});
  REQUIRE(fg.isValid(backbuffer));

  struct TestPass {
    ResourceId backbuffer;
    mutable bool executed{false};
  };
  auto &testPass = fg.addCallbackPass<TestPass>(
      "Test pass",
      [&fg, backbuffer](FrameGraph::Builder &builder, TestPass &data) {
        const auto temp = backbuffer;
        data.backbuffer = builder.write(backbuffer);
        REQUIRE(fg.isValid(data.backbuffer));
        REQUIRE_FALSE(fg.isValid(temp));
      },
      [](const TestPass &data, PassResources &resources, void *) {
        CHECK(resources.get<FrameGraphTexture>(data.backbuffer).id ==
              kBackbufferId);
        data.executed = true;
      });

  fg.compile();
  //   std::ofstream{"imported_resource.dot"} << fg;

  fg.execute();
  REQUIRE(testPass.executed);
  return;
}

void test3() {
  FrameGraph fg;

  struct PassData {
    ResourceId foo;
    mutable bool executed{false};
  };
  auto &pass1 = fg.addCallbackPass<PassData>(
      "Pass1",
      [](FrameGraph::Builder &builder, PassData &data) {
        data.foo = builder.create<FrameGraphTexture>("foo", {});
        data.foo = builder.write(data.foo);
      },
      markAsExecuted);

  auto &pass2 = fg.addCallbackPass<PassData>(
      "Pass2",
      [&fg, &pass1](FrameGraph::Builder &builder, PassData &data) {
        data.foo = builder.write(builder.read(pass1.foo));
        REQUIRE_FALSE(fg.isValid(pass1.foo));
        REQUIRE(fg.isValid(data.foo));

        builder.setSideEffect();
      },
      markAsExecuted);

  fg.compile();
  //   std::ofstream{"renamed_resource.dot"} << fg;

  fg.execute();
  REQUIRE(pass1.executed);
  REQUIRE(pass2.executed);
  return;
}

void test4() {
  FrameGraph fg;

  struct TestPass {
    mutable bool executed{false};
  };
  auto &testPass = fg.addCallbackPass<TestPass>(
      "Test pass", [](const FrameGraph::Builder &, const auto &) {},
      markAsExecuted);

  fg.compile();
  //   std::ofstream{"culled_pass.dot"} << fg;

  fg.execute();
  REQUIRE_FALSE(testPass.executed);
  return;
}

void test5() {
  FrameGraph fg;
  auto backbufferId =
      fg.import("Backbuffer", {1280, 720}, FrameGraphTexture{117});

  const auto &desc = fg.getDescriptor<FrameGraphTexture>(backbufferId);

  struct DepthPass {
    ResourceId depth;
    mutable bool executed{false};
  };
  auto &depthPass = fg.addCallbackPass<DepthPass>(
      "Depth pass",
      [&desc](FrameGraph::Builder &builder, DepthPass &data) {
        data.depth = builder.create<FrameGraphTexture>("DepthBuffer", desc);
        data.depth = builder.write(data.depth);
      },
      markAsExecuted);

  struct GBufferPass {
    ResourceId depth;
    ResourceId position;
    ResourceId normal;
    ResourceId albedo;

    mutable bool executed{false};
  };
  auto &gbufferPass = fg.addCallbackPass<GBufferPass>(
      "GBuffer pass",
      [&desc, &depthPass](FrameGraph::Builder &builder, GBufferPass &data) {
        data.depth = builder.read(depthPass.depth);
        data.position =
            builder.create<FrameGraphTexture>("GBuffer/ Position", desc);
        data.position = builder.write(data.position);
        data.normal =
            builder.create<FrameGraphTexture>("GBuffer/ Normal", desc);
        data.normal = builder.write(data.normal);
        data.albedo =
            builder.create<FrameGraphTexture>("GBuffer/ Albedo", desc);
        data.albedo = builder.write(data.albedo);
      },
      markAsExecuted);

  struct LightingPass {
    ResourceId position;
    ResourceId normal;
    ResourceId albedo;
    ResourceId output;
    mutable bool executed{false};
  };
  auto &lightingPass = fg.addCallbackPass<LightingPass>(
      "Lighting pass",
      [&gbufferPass, backbufferId](FrameGraph::Builder &builder,
                                   LightingPass &data) {
        data.position = builder.read(gbufferPass.position);
        data.normal = builder.read(gbufferPass.normal);
        data.albedo = builder.read(gbufferPass.albedo);
        data.output = builder.write(backbufferId);
      },
      markAsExecuted);

  struct Dummy {
    mutable bool executed{false};
  };
  auto &dummyPass = fg.addCallbackPass<Dummy>(
      "Dummy pass", [](const FrameGraph::Builder &, auto &) {}, markAsExecuted);

  fg.compile();
  //   std::ofstream{"deferred_pipeline.dot"} << fg;

  fg.execute();
  REQUIRE(depthPass.executed);
  REQUIRE(gbufferPass.executed);
  REQUIRE(lightingPass.executed);
  REQUIRE_FALSE(dummyPass.executed);
  return;
}

int main() {
  test0();
  test1();
  test2();
  test3();
  test4();
  test5();

  return 0;
}