#include "frame_graph.h"
#include <stack>

//
// FrameGraph class:
//

void FrameGraph::reserve(uint32_t numPasses, uint32_t numResources) {
  m_passNodes.reserve(numPasses);
  m_resourceNodes.reserve(numResources);
  m_resourceRegistry.reserve(numResources);
}

bool FrameGraph::isValid(NodeId id) const {
  const auto &node = _getResourceNode(id);
  return node.getVersion() == _getResourceEntry(node).getVersion();
}

void FrameGraph::compile() {
  for (auto &pass : m_passNodes) {
    pass.m_refCount = static_cast<int32_t>(pass.m_writes.size());
    for (const auto id : pass.m_reads) {
      auto &consumed = m_resourceNodes[id];
      consumed.m_refCount++;
    }
    for (const auto id : pass.m_writes) {
      auto &written = m_resourceNodes[id];
      written.m_producer = &pass;
    }
  }

  // -- Culling:

  std::stack<ResourceNode *> unreferencedResources;
  for (auto &node : m_resourceNodes) {
    if (node.m_refCount == 0)
      unreferencedResources.push(&node);
  }
  while (!unreferencedResources.empty()) {
    auto *unreferencedResource = unreferencedResources.top();
    unreferencedResources.pop();
    PassNode *producer{unreferencedResource->m_producer};
    if (producer == nullptr || producer->hasSideEffect())
      continue;

    assert(producer->m_refCount >= 1);
    if (--producer->m_refCount == 0) {
      for (const auto id : producer->m_reads) {
        auto &node = m_resourceNodes[id];
        if (--node.m_refCount == 0)
          unreferencedResources.push(&node);
      }
    }
  }

  // -- Calculate resources lifetime:

  for (auto &pass : m_passNodes) {
    if (pass.m_refCount == 0)
      continue;

    for (const auto id : pass.m_creates)
      _getResourceEntry(id).m_producer = &pass;
    for (const auto id : pass.m_writes)
      _getResourceEntry(id).m_last = &pass;
    for (const auto id : pass.m_reads)
      _getResourceEntry(id).m_last = &pass;
  }
}
void FrameGraph::execute(void *context, void *allocator) {
  for (const auto &pass : m_passNodes) {
    if (!pass.canExecute())
      continue;

    for (const auto id : pass.m_creates)
      _getResourceEntry(id).create(allocator);

    PassResources resources{*this, pass};
    std::invoke(*pass.m_exec, resources, context);

    for (auto &entry : m_resourceRegistry) {
      if (entry.m_last == &pass && entry.isTransient())
        entry.destroy(allocator);
    }
  }
}

//
// (private):
//

PassNode &
FrameGraph::_createPassNode(const std::string_view name,
                            std::unique_ptr<FrameGraphPassConcept> &&base) {
  const auto nodeId = static_cast<NodeId>(m_passNodes.size());
  return m_passNodes.emplace_back(PassNode{name, nodeId, std::move(base)});
}

ResourceNode &FrameGraph::_createResourceNode(const std::string_view name,
                                              uint32_t resourceId,
                                              uint32_t version) {
  const auto nodeId = static_cast<uint32_t>(m_resourceNodes.size());
  return m_resourceNodes.emplace_back(
      ResourceNode{name, nodeId, resourceId, version});
}

NodeId FrameGraph::_clone(NodeId id) {
  const auto &node = _getResourceNode(id);
  auto &entry = _getResourceEntry(node);
  entry.m_version++;

  const auto &clone = _createResourceNode(node.getName(), node.getResourceId(),
                                          entry.getVersion());
  return clone.getId();
}

const ResourceNode &FrameGraph::_getResourceNode(NodeId id) const {
  assert(id < m_resourceNodes.size());
  return m_resourceNodes[id];
}
const ResourceEntry &
FrameGraph::_getResourceEntry(NodeId id) const {
  return _getResourceEntry(_getResourceNode(id));
}
const ResourceEntry &
FrameGraph::_getResourceEntry(const ResourceNode &node) const {
  assert(node.m_resourceId < m_resourceRegistry.size());
  return m_resourceRegistry[node.m_resourceId];
}

ResourceNode &FrameGraph::_getResourceNode(NodeId id) {
  assert(id < m_resourceNodes.size());
  return m_resourceNodes[id];
}
ResourceEntry &
FrameGraph::_getResourceEntry(NodeId id) {
  return _getResourceEntry(_getResourceNode(id));
}
ResourceEntry &
FrameGraph::_getResourceEntry(const ResourceNode &node) {
  assert(node.m_resourceId < m_resourceRegistry.size());
  return m_resourceRegistry[node.m_resourceId];
}


//
// FrameGraph::Builder class:
//

NodeId FrameGraph::Builder::read(NodeId id) {
  assert(m_frameGraph.isValid(id));
  return m_passNode.m_reads.emplace_back(id);
}
NodeId FrameGraph::Builder::write(NodeId id) {
  assert(m_frameGraph.isValid(id));
  if (m_frameGraph._getResourceEntry(id).isImported())
    setSideEffect();

  if (m_passNode.creates(id)) {
    return m_passNode.m_writes.emplace_back(id);
  } else {
    // Writing to a texture produces a renamed handle.
    // This allows us to catch errors when resources are modified in
    // undefined order (when same resource is written by different passes).
    // Renaming resources enforces a specific execution order of the render
    // passes.
    m_passNode.m_reads.emplace_back(id);
    return m_passNode.m_writes.emplace_back(m_frameGraph._clone(id));
  }
}
