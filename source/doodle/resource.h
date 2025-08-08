#pragma once

#include <cassert>
#include <memory>

#include "pass_node.h"

struct Concept {
  virtual ~Concept() = default;

  virtual void create(void *) = 0;
  virtual void destroy(void *) = 0;
};

template <typename T> struct Model final : Concept {
  Model(const typename T::Desc &desc, T &&obj)
      : descriptor{desc}, resource{std::move(obj)} {}

  void create(void *allocator) { resource.create(descriptor, allocator); }

  void destroy(void *allocator) override {
    resource.destroy(descriptor, allocator);
  }

  const typename T::Desc descriptor;
  T resource;
};

// Wrapper around a virtual resource.
class ResourceEntry final {
  friend class FrameGraph;

  enum class Type : uint8_t { Transient, Imported };

public:
  ResourceEntry() = delete;
  ResourceEntry(const ResourceEntry &) = delete;
  ResourceEntry(ResourceEntry &&) noexcept = default;

  ResourceEntry &operator=(const ResourceEntry &) = delete;
  ResourceEntry &operator=(ResourceEntry &&) noexcept = delete;

  static constexpr auto kInitialVersion{1u};

  uint32_t getId() const { return m_id; }
  uint32_t getVersion() const { return m_version; }
  bool isImported() const { return m_type == Type::Imported; }
  bool isTransient() const { return m_type == Type::Transient; }

  void create(void *allocator) {
    assert(isTransient());
    m_concept->create(allocator);
  }
  void destroy(void *allocator) {
    assert(isTransient());
    m_concept->destroy(allocator);
  }

  template <typename T> T &get() { return _getModel<T>()->resource; }
  template <typename T> const typename T::Desc &getDescriptor() const {
    return _getModel<T>()->descriptor;
  }

private:
  template <typename T>
  ResourceEntry(const Type type, uint32_t id, const typename T::Desc &desc,
                T &&obj)
      : m_type{type}, m_id{id}, m_version{kInitialVersion},
        m_concept{std::make_unique<Model<T>>(desc, std::forward<T>(obj))} {}

  // http://www.cplusplus.com/articles/oz18T05o/
  // https://www.modernescpp.com/index.php/c-core-guidelines-type-erasure-with-templates

  template <typename T> 
  Model<T> *_getModel() const {
    auto *model = dynamic_cast<Model<T> *>(m_concept.get());
    assert(model && "Invalid type");
    return model;
  }

private:
  const Type m_type;
  const uint32_t m_id;
  uint32_t m_version; // Incremented on each (unique) write declaration.
  std::unique_ptr<Concept> m_concept;

  PassNode *m_producer{nullptr};
  PassNode *m_last{nullptr};
};
