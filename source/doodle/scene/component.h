#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace doodle {

/// Transform component (TRS representation)
struct Transform {
  glm::vec3 translation = glm::vec3(0.0f);
  glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // w, x, y, z
  glm::vec3 scale = glm::vec3(1.0f);
  
  glm::mat4 matrix() const {
    return glm::translate(glm::mat4(1.0f), translation) *
           glm::mat4_cast(rotation) *
           glm::scale(glm::mat4(1.0f), scale);
  }
};

struct Camera {
  glm::mat4 view = glm::mat4{ 1.0f };
  glm::mat4 inverseView = glm::mat4{ 1.0f };
  glm::mat4 projection = glm::mat4{ 1.0f };
};

struct Mesh {

};

struct Physics {

};
}