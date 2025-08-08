#pragma once

#include <memory>
#include <vector>


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

  bool creates(ResourceId id) const;
  bool reads(ResourceId id) const;
  bool writes(ResourceId id) const;

  auto hasSideEffect() const { return m_hasSideEffect; }
  auto canExecute() const { return getRefCount() > 0 || hasSideEffect(); }

private:
  PassNode(const std::string_view name, uint32_t nodeId,
           std::unique_ptr<FrameGraphPassConcept> &&);

private:
  std::unique_ptr<FrameGraphPassConcept> m_exec;

  std::vector<ResourceId> m_creates;
  std::vector<ResourceId> m_reads;
  std::vector<ResourceId> m_writes;

  bool m_hasSideEffect{false};
};
