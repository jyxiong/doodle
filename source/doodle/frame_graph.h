#pragma once

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

#define _VIRTUALIZABLE_CONCEPT(T) Virtualizable T
#define _VIRTUALIZABLE_CONCEPT_IMPL(T) _VIRTUALIZABLE_CONCEPT(T)

class FrameGraph {
  friend class FrameGraphPassResources;

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

    template <_VIRTUALIZABLE_CONCEPT(T)>
    /** Declares the creation of a resource. */
    [[nodiscard]] ResourceId create(const std::string_view name,
                                            const typename T::Desc &);
    /** Declares read operation. */
    ResourceId read(ResourceId id,
                            uint32_t flags = kFlagsIgnored);
    /**
     * Declares write operation.
     * @remark Writing to imported resource counts as side-effect.
     */
    [[nodiscard]] ResourceId write(ResourceId id,
                                           uint32_t flags = kFlagsIgnored);

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

  struct NoData {};
  /**
   * @param setup Callback (lambda, may capture by reference), invoked
   * immediately, declare operations here.
   * @param exec Execution of this lambda is deferred until execute() phase
   * (must capture by value due to this).
   */
  template <typename Data = NoData, typename Setup, typename Execute>
  const Data &addCallbackPass(const std::string_view name, Setup &&setup,
                              Execute &&exec);

  template <_VIRTUALIZABLE_CONCEPT(T)>
  [[nodiscard]] const typename T::Desc &
  getDescriptor(ResourceId id) const;

  template <_VIRTUALIZABLE_CONCEPT(T)>
  /** Imports the given resource T into FrameGraph. */
  [[nodiscard]] ResourceId import(const std::string_view name,
                                          const typename T::Desc &, T &&);

  /** @return True if the given resource is valid for read/write operation. */
  [[nodiscard]] bool isValid(ResourceId id) const;

  /** Culls unreferenced resources and passes. */
  void compile();
  /** Invokes execution callbacks. */
  void execute(void *context = nullptr, void *allocator = nullptr);


private:
  [[nodiscard]] PassNode &
  _createPassNode(const std::string_view name,
                  std::unique_ptr<FrameGraphPassConcept> &&);

  template <_VIRTUALIZABLE_CONCEPT(T)>
  [[nodiscard]] ResourceId _create(const ResourceEntry::Type,
                                           const std::string_view name,
                                           const typename T::Desc &, T &&);

  [[nodiscard]] ResourceNode &
  _createResourceNode(const std::string_view name, uint32_t resourceId,
                      uint32_t version = ResourceEntry::kInitialVersion);
  /** Increments ResourceEntry version and produces a renamed handle. */
  [[nodiscard]] ResourceId _clone(ResourceId id);

  [[nodiscard]] const ResourceNode &
  _getResourceNode(ResourceId id) const;
  [[nodiscard]] const ResourceEntry &
  _getResourceEntry(ResourceId id) const;
  [[nodiscard]] const ResourceEntry &
  _getResourceEntry(const ResourceNode &) const;

  [[nodiscard]] decltype(auto) _getResourceNode(ResourceId id) {
    return const_cast<ResourceNode &>(
        const_cast<const FrameGraph *>(this)->_getResourceNode(id));
  }
  [[nodiscard]] decltype(auto) _getResourceEntry(ResourceId id) {
    return const_cast<ResourceEntry &>(
        const_cast<const FrameGraph *>(this)->_getResourceEntry(id));
  }
  [[nodiscard]] decltype(auto) _getResourceEntry(const ResourceNode &node) {
    return const_cast<ResourceEntry &>(
        const_cast<const FrameGraph *>(this)->_getResourceEntry(node));
  }

private:
  std::vector<PassNode> m_passNodes;
  std::vector<ResourceNode> m_resourceNodes;
  std::vector<ResourceEntry> m_resourceRegistry;
};

class FrameGraphPassResources {
  friend class FrameGraph;

public:
  FrameGraphPassResources() = delete;
  FrameGraphPassResources(const FrameGraphPassResources &) = delete;
  FrameGraphPassResources(FrameGraphPassResources &&) noexcept = delete;
  ~FrameGraphPassResources() = default;

  FrameGraphPassResources &operator=(const FrameGraphPassResources &) = delete;
  FrameGraphPassResources &
  operator=(FrameGraphPassResources &&) noexcept = delete;

  /**
   * @note Causes runtime-error with:
   * - Attempt to use obsolete handle (the one that has been renamed before)
   * - Incorrect resource type T
   */
  template <_VIRTUALIZABLE_CONCEPT(T)>
  [[nodiscard]] T &get(ResourceId id);
  template <_VIRTUALIZABLE_CONCEPT(T)>
  [[nodiscard]] const typename T::Desc &
  getDescriptor(ResourceId id) const;

private:
  FrameGraphPassResources(FrameGraph &fg, const PassNode &node)
      : m_frameGraph{fg}, m_passNode{node} {}

private:
  FrameGraph &m_frameGraph;
  const PassNode &m_passNode;
};

template <typename Data, typename Setup, typename Execute>
inline const Data &FrameGraph::addCallbackPass(const std::string_view name,
                                               Setup &&setup, Execute &&exec) {
  static_assert(std::is_invocable_v<Setup, Builder &, Data &>,
                "Invalid setup callback");
  static_assert(std::is_invocable_v<Execute, const Data &,
                                    FrameGraphPassResources &, void *>,
                "Invalid exec callback");
  static_assert(sizeof(Execute) < 1024, "Execute captures too much");

  auto *pass = new FrameGraphPass<Data, Execute>(std::forward<Execute>(exec));
  auto &passNode = _createPassNode(
      name, std::unique_ptr<FrameGraphPass<Data, Execute>>(pass));
  Builder builder{*this, passNode};
  std::invoke(setup, builder, pass->data);
  return pass->data;
}

template <_VIRTUALIZABLE_CONCEPT_IMPL(T)>
inline const typename T::Desc &
FrameGraph::getDescriptor(ResourceId id) const {
  return _getResourceEntry(id).getDescriptor<T>();
}

template <_VIRTUALIZABLE_CONCEPT_IMPL(T)>
inline ResourceId FrameGraph::import(const std::string_view name,
                                             const typename T::Desc &desc,
                                             T &&resource) {
  return _create<T>(ResourceEntry::Type::Imported, name, desc,
                    std::forward<T>(resource));
}

//
// (private):
//

template <_VIRTUALIZABLE_CONCEPT_IMPL(T)>
inline ResourceId
FrameGraph::_create(const ResourceEntry::Type type, const std::string_view name,
                    const typename T::Desc &desc, T &&resource) {
  const auto resourceId = static_cast<uint32_t>(m_resourceRegistry.size());
  m_resourceRegistry.emplace_back(
      ResourceEntry{type, resourceId, desc, std::forward<T>(resource)});
  return _createResourceNode(name, resourceId).getId();
}

//
// FrameGraph::Builder class:
//

template <_VIRTUALIZABLE_CONCEPT_IMPL(T)>
inline ResourceId
FrameGraph::Builder::create(const std::string_view name,
                            const typename T::Desc &desc) {
  const auto id =
      m_frameGraph._create<T>(ResourceEntry::Type::Transient, name, desc, T{});
  return m_passNode.m_creates.emplace_back(id);
}

//
// FrameGraphPassResources class:
//

template <_VIRTUALIZABLE_CONCEPT_IMPL(T)>
inline T &FrameGraphPassResources::get(ResourceId id) {
  assert(m_passNode.reads(id) || m_passNode.creates(id) ||
         m_passNode.writes(id));
  return m_frameGraph._getResourceEntry(id).get<T>();
}
template <_VIRTUALIZABLE_CONCEPT_IMPL(T)>
inline const typename T::Desc &
FrameGraphPassResources::getDescriptor(ResourceId id) const {
  assert(m_passNode.reads(id) || m_passNode.creates(id) ||
         m_passNode.writes(id));
  return m_frameGraph.getDescriptor<T>(id);
}
