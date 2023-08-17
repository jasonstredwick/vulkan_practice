#pragma once


#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jms/vulkan/vulkan.hpp"
#include "jms/vulkan/memory.hpp"
#include "jms/vulkan/state.hpp"
#include "jms/vulkan/vertex_description.hpp"


struct Vertex {
    glm::vec4 position{};
    glm::vec4 color{};

    static std::vector<vk::VertexInputAttributeDescription> GetAttributeDesc(uint32_t binding) {
        return {
            vk::VertexInputAttributeDescription{
                .location=0,
                .binding=binding,
                .format=vk::Format::eR32G32B32A32Sfloat,
                .offset=offsetof(Vertex, position) // conditionally supported; no good alternative?  check hpp examples
            },
            vk::VertexInputAttributeDescription{
                .location=1,
                .binding=binding,
                .format=vk::Format::eR32G32B32A32Sfloat,
                .offset=offsetof(Vertex, color) // conditionally supported
            }
        };
    }

    static std::vector<vk::VertexInputBindingDescription> GetBindingDesc(uint32_t binding) {
        return {
            {
                .binding=binding,
                .stride=sizeof(Vertex),
                .inputRate=vk::VertexInputRate::eVertex
            }
        };
    }
};
static_assert(std::is_standard_layout_v<Vertex>); // requires for offsetof macro


std::vector<jms::vulkan::shader::Info> LoadShaders(const vk::raii::Device& device) {
    // need to investigate vector initialization with aggregate Info initialization with forbidden copy constructor.
    std::vector<jms::vulkan::shader::Info> out{};
    out.reserve(2);
    out.push_back({
        .shader_module=jms::vulkan::shader::Load(std::string{"shader.vert.spv"}, device),
        .stage=vk::ShaderStageFlagBits::eVertex,
        .entry_point_name=std::string{"main"}
    });
    out.push_back({
        .shader_module=jms::vulkan::shader::Load(std::string{"shader.frag.spv"}, device),
        .stage=vk::ShaderStageFlagBits::eFragment,
        .entry_point_name=std::string{"main"}
    });
    return out;
}


