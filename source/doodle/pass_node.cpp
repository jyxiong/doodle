#include "pass_node.h"

#include <algorithm>
#include <cassert>

namespace {

bool hasId(const std::vector<ResourceId> &v, ResourceId id) {
  return std::ranges::find(v, id) != v.cend();
}

} // namespace

bool PassNode::creates(ResourceId id) const { return hasId(m_creates, id); }

bool PassNode::reads(ResourceId id) const { return hasId(m_reads, id); }

bool PassNode::writes(ResourceId id) const { return hasId(m_writes, id); }

PassNode::PassNode(const std::string_view name, uint32_t nodeId,
                   std::unique_ptr<FrameGraphPassConcept> &&exec)
    : GraphNode{name, nodeId}, m_exec{std::move(exec)} {
  m_creates.reserve(10);
  m_reads.reserve(10);
  m_writes.reserve(10);
}

