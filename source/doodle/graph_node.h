#pragma once

#include <string>

using NodeId = uint32_t;

class GraphNode {
  friend class FrameGraph;

public:
  GraphNode() = delete;
  GraphNode(const GraphNode &) = delete;
  GraphNode(GraphNode &&) noexcept = default;
  virtual ~GraphNode() = default;
  GraphNode &operator=(const GraphNode &) = delete;
  GraphNode &operator=(GraphNode &&) noexcept = delete;

  NodeId getId() const { return m_id; }
  std::string_view getName() const { return m_name; }
  int32_t getRefCount() const { return m_refCount; }

protected:
  GraphNode(const std::string_view name, NodeId id)
      : m_name{name}, m_id{id} {}

private:
  std::string m_name;
  const NodeId m_id;
  int32_t m_refCount{0};
};