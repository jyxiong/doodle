#pragma once

#include <cassert>
#include <memory>

#include "pass_node.h"

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

  void create(void *allocator);
  void destroy(void *allocator);

  auto getId() const { return m_id; }
  auto getVersion() const { return m_version; }
  auto isImported() const { return m_type == Type::Imported; }
  auto isTransient() const { return m_type == Type::Transient; }

  template <typename T> T &get();
  template <typename T>
  const typename T::Desc &getDescriptor() const;

private:
  template <typename T>
  ResourceEntry(const Type, uint32_t id, const typename T::Desc &, T &&);

  // http://www.cplusplus.com/articles/oz18T05o/
  // https://www.modernescpp.com/index.php/c-core-guidelines-type-erasure-with-templates

  struct Concept {
    virtual ~Concept() = default;

    virtual void create(void *) = 0;
    virtual void destroy(void *) = 0;
  };
  template <typename T> struct Model final : Concept {
    Model(const typename T::Desc &, T &&);

    void create(void *allocator) override;
    void destroy(void *allocator) override;

    const typename T::Desc descriptor;
    T resource;
  };

  template <typename T> auto *_getModel() const;

private:
  const Type m_type;
  const uint32_t m_id;
  uint32_t m_version; // Incremented on each (unique) write declaration.
  std::unique_ptr<Concept> m_concept;

  PassNode *m_producer{nullptr};
  PassNode *m_last{nullptr};
};

//
// ResourceEntry class:
//

inline void ResourceEntry::create(void *allocator) {
  assert(isTransient());
  m_concept->create(allocator);
}
inline void ResourceEntry::destroy(void *allocator) {
  assert(isTransient());
  m_concept->destroy(allocator);
}

template <typename T> inline T &ResourceEntry::get() {
  return _getModel<T>()->resource;
}
template <typename T>
inline const typename T::Desc &ResourceEntry::getDescriptor() const {
  return _getModel<T>()->descriptor;
}

//
// (private):
//

template <typename T>
inline ResourceEntry::ResourceEntry(const Type type, uint32_t id,
                                    const typename T::Desc &desc, T &&obj)
    : m_type{type}, m_id{id}, m_version{kInitialVersion},
      m_concept{std::make_unique<Model<T>>(desc, std::forward<T>(obj))} {}

template <typename T> inline auto *ResourceEntry::_getModel() const {
  auto *model = dynamic_cast<Model<T> *>(m_concept.get());
  assert(model && "Invalid type");
  return model;
}

//
// ResourceEntry::Model class:
//

template <typename T>
inline ResourceEntry::Model<T>::Model(const typename T::Desc &desc, T &&obj)
    : descriptor{desc}, resource{std::move(obj)} {}

template <typename T>
inline void ResourceEntry::Model<T>::create(void *allocator) {
  resource.create(descriptor, allocator);
}
template <typename T>
inline void ResourceEntry::Model<T>::destroy(void *allocator) {
  resource.destroy(descriptor, allocator);
}

