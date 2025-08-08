#pragma once

#include <vector>
#include <memory>

#include "graph_node.h"
#include "pass.h"

using ResourceId = uint32_t;

class PassNode final : public GraphNode {
  friend class FrameGraph;

public:
  PassNode(const PassNode &) = delete;
  PassNode(PassNode &&) noexcept = default;

  PassNode &operator=(const PassNode &) = delete;
  PassNode &operator=(PassNode &&) noexcept = delete;

  struct AccessDeclaration {
    ResourceId id;
    uint32_t flags;

    bool operator==(const AccessDeclaration &) const = default;
  };

  [[nodiscard]] bool creates(ResourceId id) const;
  [[nodiscard]] bool reads(ResourceId id) const;
  [[nodiscard]] bool writes(ResourceId id) const;

  [[nodiscard]] auto hasSideEffect() const { return m_hasSideEffect; }
  [[nodiscard]] auto canExecute() const {
    return getRefCount() > 0 || hasSideEffect();
  }

private:
  PassNode(const std::string_view name, uint32_t nodeId,
           std::unique_ptr<FrameGraphPassConcept> &&);

  ResourceId _read(ResourceId id, uint32_t flags);

  [[nodiscard]] ResourceId _write(ResourceId id,
                                          uint32_t flags);

private:
  std::unique_ptr<FrameGraphPassConcept> m_exec;

  std::vector<ResourceId> m_creates;
  std::vector<AccessDeclaration> m_reads;
  std::vector<AccessDeclaration> m_writes;

  bool m_hasSideEffect{false};
};

