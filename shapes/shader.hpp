#pragma once


#include <optional>
#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jms/vulkan/vulkan.hpp"
#include "jms/vulkan/graphics_pass.hpp"
#include "jms/vulkan/shader.hpp"


struct Vertex {
    glm::uint32_t model_index{};
    glm::vec3 position{};
};
static_assert(std::is_standard_layout_v<Vertex>); // requires for offsetof macro


struct VertexData {
    glm::vec4 color{};
};
static_assert(std::is_standard_layout_v<VertexData>); // requires for offsetof macro


struct ModelData {
    glm::mat4 transform{1.0f};
};
static_assert(std::is_standard_layout_v<ModelData>); // requires for offsetof macro


jms::vulkan::ShaderGroup CreateGroup() {
    return jms::vulkan::ShaderGroup{
        .vertex_attribute_desc={
           {
                .location=0,
                .binding=0,
                .format=vk::Format::eR32Uint,
                .offset=offsetof(Vertex, model_index)
            },
            {
                .location=1,
                .binding=0,
                .format=vk::Format::eR32G32B32Sfloat,
                .offset=offsetof(Vertex, position)
            }
        },

        .vertex_binding_desc={
            {
                .binding=0,
                .stride=sizeof(Vertex),
                .inputRate=vk::VertexInputRate::eVertex,
                .divisor=1
            }
        },

        .push_constant_ranges={},

        .set_layout_bindings={
            {
                {
                    .binding=0,
                    .descriptorType=vk::DescriptorType::eStorageBuffer,
                    .descriptorCount=1,
                    .stageFlags=vk::ShaderStageFlagBits::eVertex,
                    .pImmutableSamplers=nullptr
                },
                {
                    .binding=1,
                    .descriptorType=vk::DescriptorType::eStorageBuffer,
                    .descriptorCount=1,
                    .stageFlags=vk::ShaderStageFlagBits::eVertex,
                    .pImmutableSamplers=nullptr
                },
                {
                    .binding=2,
                    .descriptorType=vk::DescriptorType::eStorageBuffer,
                    .descriptorCount=1,
                    .stageFlags=vk::ShaderStageFlagBits::eVertex,
                    .pImmutableSamplers=nullptr
                }
            }
        },

        // need to investigate vector initialization with aggregate Info initialization with forbidden copy constructor.
        .shader_infos={
            {
                .flags=vk::ShaderCreateFlagBitsEXT::eLinkStage,
                .stage=vk::ShaderStageFlagBits::eVertex,
                .next_stage=vk::ShaderStageFlagBits::eFragment,
                .code_type=vk::ShaderCodeTypeEXT::eSpirv,
                .code=jms::vulkan::Load(std::string{"shader.vert.spv"}),
                .entry_point_name=std::string{"main"},
                .set_info_indices={0}
            },
            {
                .flags=vk::ShaderCreateFlagBitsEXT::eLinkStage,
                .stage=vk::ShaderStageFlagBits::eFragment,
                .code_type=vk::ShaderCodeTypeEXT::eSpirv,
                .code=jms::vulkan::Load(std::string{"shader.frag.spv"}),
                .entry_point_name=std::string{"main"}
            }
        }
    };
}


void BindShaders(vk::raii::CommandBuffer& command_buffer, jms::vulkan::GraphicsPass& graphics_pass) {
    graphics_pass.BindShaders(command_buffer, {0, 1});
}
