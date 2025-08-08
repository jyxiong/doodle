#pragma once

#include "pass.h"
#include "pass_node.h"
#include "resource.h"
#include "resource_node.h"

template <typename T>
concept Virtualizable = requires(T t) {
  requires std::conjunction_v<std::is_default_constructible<T>,
                              std::is_move_constructible<T>>;

  typename T::Desc;
  { t.create(typename T::Desc{}, (void *)nullptr) } -> std::same_as<void>;
  { t.destroy(typename T::Desc{}, (void *)nullptr) } -> std::same_as<void>;
};

class FrameGraph {
  friend class PassResources;

public:
  FrameGraph() = default;
  FrameGraph(const FrameGraph &) = delete;
  FrameGraph(FrameGraph &&) noexcept = delete;

  FrameGraph &operator=(const FrameGraph &) = delete;
  FrameGraph &operator=(FrameGraph &&) noexcept = delete;

  static constexpr auto kFlagsIgnored = ~0;

  class Builder final {
    friend class FrameGraph;

  public:
    Builder() = delete;
    Builder(const Builder &) = delete;
    Builder(Builder &&) noexcept = delete;

    Builder &operator=(const Builder &) = delete;
    Builder &operator=(Builder &&) noexcept = delete;

    /** Declares the creation of a resource. */
    template <Virtualizable T>
    NodeId create(const std::string_view name,
                                      const typename T::Desc &desc) {
      const auto nodeId = m_frameGraph._create<T>(ResourceEntry::Type::Transient,
                                              name, desc, T{});
      return m_passNode.m_creates.emplace_back(nodeId);
    }

    /** Declares read operation. */
    NodeId read(NodeId id);
    /**
     * Declares write operation.
     * @remark Writing to imported resource counts as side-effect.
     */
    NodeId write(NodeId id);

    /** Ensures that this pass is not culled during the compilation phase. */
    Builder &setSideEffect() {
      m_passNode.m_hasSideEffect = true;
      return *this;
    }

  private:
    Builder(FrameGraph &fg, PassNode &node)
        : m_frameGraph{fg}, m_passNode{node} {}

  private:
    FrameGraph &m_frameGraph;
    PassNode &m_passNode;
  };

  void reserve(uint32_t numPasses, uint32_t numResources);

  /**
   * @param setup Callback (lambda, may capture by reference), invoked
   * immediately, declare operations here.
   * @param exec Execution of this lambda is deferred until execute() phase
   * (must capture by value due to this).
   */
  template <typename Data, typename Setup, typename Execute>
  const Data &addCallbackPass(const std::string_view name, Setup &&setup,
                              Execute &&exec) {
    static_assert(std::is_invocable_v<Setup, Builder &, Data &>,
                  "Invalid setup callback");
    static_assert(
        std::is_invocable_v<Execute, const Data &, PassResources &, void *>,
        "Invalid exec callback");
    static_assert(sizeof(Execute) < 1024, "Execute captures too much");

    auto *pass = new FrameGraphPass<Data, Execute>(std::forward<Execute>(exec));
    auto &passNode = _createPassNode(
        name, std::unique_ptr<FrameGraphPass<Data, Execute>>(pass));
    Builder builder{*this, passNode};
    std::invoke(setup, builder, pass->data);
    return pass->data;
  }

  template <Virtualizable T>
  const typename T::Desc &getDescriptor(NodeId id) const {
    return _getResourceEntry(id).getDescriptor<T>();
  }

  /** Imports the given resource T into FrameGraph. */

  template <Virtualizable T>
  NodeId import(const std::string_view name, const typename T::Desc &desc,
                    T &&resource) {
    return _create<T>(ResourceEntry::Type::Imported, name, desc,
                      std::forward<T>(resource));
  }

  /** @return True if the given resource is valid for read/write operation. */
  bool isValid(NodeId id) const;

  /** Culls unreferenced resources and passes. */
  void compile();
  /** Invokes execution callbacks. */
  void execute(void *context = nullptr, void *allocator = nullptr);

private:
  PassNode &_createPassNode(const std::string_view name,
                            std::unique_ptr<FrameGraphPassConcept> &&);

  template <Virtualizable T>
  NodeId _create(const ResourceEntry::Type type,
                            const std::string_view name,
                            const typename T::Desc &desc, T &&resource) {
    const auto resourceId = static_cast<uint32_t>(m_resourceRegistry.size());
    m_resourceRegistry.emplace_back(
        ResourceEntry{type, resourceId, desc, std::forward<T>(resource)});
    return _createResourceNode(name, resourceId).getId();
  }

  ResourceNode &
  _createResourceNode(const std::string_view name, uint32_t resourceId,
                      uint32_t version = ResourceEntry::kInitialVersion);
  
                      /** Increments ResourceEntry version and produces a renamed handle. */
  NodeId _clone(NodeId id);

  const ResourceNode &_getResourceNode(NodeId id) const;
  const ResourceEntry &_getResourceEntry(NodeId id) const;
  const ResourceEntry &_getResourceEntry(const ResourceNode &) const;

  ResourceNode &_getResourceNode(NodeId id);
  ResourceEntry &_getResourceEntry(NodeId id);
  ResourceEntry &_getResourceEntry(const ResourceNode &node);

private:
  std::vector<PassNode> m_passNodes;
  std::vector<ResourceNode> m_resourceNodes;
  std::vector<ResourceEntry> m_resourceRegistry;
};

class PassResources {
  friend class FrameGraph;

public:
  PassResources() = delete;
  PassResources(const PassResources &) = delete;
  PassResources(PassResources &&) noexcept = delete;
  ~PassResources() = default;

  PassResources &operator=(const PassResources &) = delete;
  PassResources &operator=(PassResources &&) noexcept = delete;

  /**
   * @note Causes runtime-error with:
   * - Attempt to use obsolete handle (the one that has been renamed before)
   * - Incorrect resource type T
   */
  template <Virtualizable T> 
  T &get(NodeId id) {
    assert(m_passNode.reads(id) || m_passNode.creates(id) ||
           m_passNode.writes(id));
    return m_frameGraph._getResourceEntry(id).get<T>();
  }

  template <Virtualizable T>
  const typename T::Desc &getDescriptor(NodeId id) const {
    assert(m_passNode.reads(id) || m_passNode.creates(id) ||
           m_passNode.writes(id));
    return m_frameGraph.getDescriptor<T>(id);
  }

private:
  PassResources(FrameGraph &fg, const PassNode &node)
      : m_frameGraph{fg}, m_passNode{node} {}

private:
  FrameGraph &m_frameGraph;
  const PassNode &m_passNode;
};
