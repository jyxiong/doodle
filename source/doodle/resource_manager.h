#pragma once

#include <vector>
#include <string_view>
#include <cstdint>
#include <utility>
#include "resource.h"

class ResourceManager {
public:
    ResourceManager() = default;
    ResourceManager(const ResourceManager&) = delete;
    ResourceManager(ResourceManager&&) noexcept = delete;
    ResourceManager& operator=(const ResourceManager&) = delete;
    ResourceManager& operator=(ResourceManager&&) noexcept = delete;

    // Add a new resource entry and return its resourceId
    template <typename T>
    uint32_t add(ResourceEntry::Type type, const std::string_view name, const typename T::Desc& desc, T&& resource) {
        uint32_t resourceId = static_cast<uint32_t>(m_registry.size());
        m_registry.emplace_back(ResourceEntry{type, resourceId, desc, std::forward<T>(resource)});
        return resourceId;
    }

    // Get a resource entry by resourceId (const)
    const ResourceEntry& get(uint32_t resourceId) const {
        return m_registry.at(resourceId);
    }

    // Get a resource entry by resourceId (non-const)
    ResourceEntry& get(uint32_t resourceId) {
        return m_registry.at(resourceId);
    }

    // Get the number of resources
    size_t size() const { return m_registry.size(); }

    // Reserve space for resources
    void reserve(size_t n) { m_registry.reserve(n); }

private:
    std::vector<ResourceEntry> m_registry;
};
