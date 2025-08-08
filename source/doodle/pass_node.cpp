#include "pass_node.h"

#include <algorithm>
#include <cassert>

namespace {

[[nodiscard]] bool hasId(const std::vector<ResourceId> &v,
                         ResourceId id) {
  return std::ranges::find(v, id) != v.cend();
}
[[nodiscard]] bool hasId(const std::vector<PassNode::AccessDeclaration> &v,
                         ResourceId id) {
  const auto match = [id](const auto &e) { return e.id == id; };
  return std::ranges::find_if(v, match) != v.cend();
}

[[nodiscard]] bool contains(const std::vector<PassNode::AccessDeclaration> &v,
                            PassNode::AccessDeclaration n) {
  return std::ranges::find(v, n) != v.cend();
}

} // namespace

bool PassNode::creates(ResourceId id) const {
  return hasId(m_creates, id);
}

bool PassNode::reads(ResourceId id) const { return hasId(m_reads, id); }

bool PassNode::writes(ResourceId id) const {
  return hasId(m_writes, id);
}

PassNode::PassNode(const std::string_view name, uint32_t nodeId,
                   std::unique_ptr<FrameGraphPassConcept> &&exec)
    : GraphNode{name, nodeId}, m_exec{std::move(exec)} {
  m_creates.reserve(10);
  m_reads.reserve(10);
  m_writes.reserve(10);
}

ResourceId PassNode::_read(ResourceId id, uint32_t flags) {
  assert(!creates(id) && !writes(id));
  return contains(m_reads, {id, flags})
             ? id
             : m_reads.emplace_back(AccessDeclaration{id, flags}).id;
}

ResourceId PassNode::_write(ResourceId id, uint32_t flags) {
  return contains(m_writes, {id, flags})
             ? id
             : m_writes.emplace_back(AccessDeclaration{id, flags}).id;
}
