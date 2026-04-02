#pragma once

#include <entt/entt.hpp>

namespace doodle {

class Scene;

class Entity {
public:
  Entity() = default;

  Entity(Scene *scene, entt::entity handle);

  Entity(const Entity &) = default;

  ~Entity() = default;

  bool operator==(const Entity &other) const;

  bool operator!=(const Entity &other) const;

  entt::registry &getRegistry();
  const entt::registry &getRegistry() const;

  // Add component
  template <typename T, typename... Args> T &addComponent(Args &&...args) {
    return getRegistry().emplace<T>(m_handle, std::forward<Args>(args)...);
  }

  // Get component
  template <typename T> T &getComponent() {
    return getRegistry().get<T>(m_handle);
  }

  template <typename T> const T &getComponent() const {
    return getRegistry().get<T>(m_handle);
  }

  // Try get component (returns nullptr if not present)
  template <typename T> T *tryGetComponent() {
    return getRegistry().try_get<T>(m_handle);
  }

  template <typename T> const T *tryGetComponent() const {
    return getRegistry().try_get<T>(m_handle);
  }

  // Has component
  template <typename T> bool hasComponent() const {
    return getRegistry().all_of<T>(m_handle);
  }

  // Has all components
  template <typename... T> bool hasAllComponents() const {
    return getRegistry().all_of<T...>(m_handle);
  }

  // Has any component
  template <typename... T> bool hasAnyComponent() const {
    return getRegistry().any_of<T...>(m_handle);
  }

  // Remove component
  template <typename T> void removeComponent() {
    getRegistry().remove<T>(m_handle);
  }

  // Replace component (remove and add)
  template <typename T, typename... Args> T &replaceComponent(Args &&...args) {
    getRegistry().replace<T>(m_handle, std::forward<Args>(args)...);
    return getRegistry().get<T>(m_handle);
  }

  // Get or add component
  template <typename T, typename... Args> T &getOrAddComponent(Args &&...args) {
    return getRegistry().get_or_emplace<T>(m_handle,
                                           std::forward<Args>(args)...);
  }

  // Validity check
  bool isValid() const;

  // Get underlying handle
  entt::entity getHandle() const;

  // Get scene
  Scene *getScene() const;

private:
  Scene *m_scene = nullptr;
  entt::entity m_handle = entt::null;
};

} // namespace doodle