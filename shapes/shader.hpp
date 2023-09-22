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
    uint32_t model_index{};
    glm::vec3 position{};

    static std::vector<vk::VertexInputAttributeDescription2EXT> GetAttributeDesc(uint32_t binding) {
        return {
           {
                .location=0,
                .binding=binding,
                .format=vk::Format::eR32Uint,
                .offset=offsetof(Vertex, model_index)
            },
            {
                .location=1,
                .binding=binding,
                .format=vk::Format::eR32G32B32Sfloat,
                .offset=offsetof(Vertex, position)
            }
        };
    }

    static std::vector<vk::VertexInputBindingDescription2EXT> GetBindingDesc(uint32_t binding) {
        return {
            {
                .binding=binding,
                .stride=sizeof(Vertex),
                .inputRate=vk::VertexInputRate::eVertex,
                .divisor=1
            }
        };
    }
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


std::vector<jms::vulkan::shader::Info> LoadShaders(const vk::raii::Device& device) {
    // need to investigate vector initialization with aggregate Info initialization with forbidden copy constructor.
    std::vector<jms::vulkan::shader::Info> out{};
    out.reserve(2);
    out.push_back({
        .flags=vk::ShaderCreateFlagBitsEXT::eLinkStage,
        .stage=vk::ShaderStageFlagBits::eVertex,
        .next_stage=vk::ShaderStageFlagBits::eFragment,
        .code_type=vk::ShaderCodeTypeEXT::eSpirv,
        .code=jms::vulkan::shader::Load(std::string{"shader.vert.spv"}, device),
        .layout_bindings={
            vk::DescriptorSetLayoutBinding{
                .binding=0,
                .descriptorType=vk::DescriptorType::eStorageBuffer,
                .descriptorCount=1,
                .stageFlags=vk::ShaderStageFlagBits::eVertex,
                .pImmutableSamplers=nullptr
            },
            vk::DescriptorSetLayoutBinding{
                .binding=1,
                .descriptorType=vk::DescriptorType::eStorageBuffer,
                .descriptorCount=1,
                .stageFlags=vk::ShaderStageFlagBits::eVertex,
                .pImmutableSamplers=nullptr
            },
            vk::DescriptorSetLayoutBinding{
                .binding=2,
                .descriptorType=vk::DescriptorType::eStorageBuffer,
                .descriptorCount=1,
                .stageFlags=vk::ShaderStageFlagBits::eVertex,
                .pImmutableSamplers=nullptr
            }
        },
        .entry_point_name=std::string{"main"}
    });
    out.push_back({
        .flags=vk::ShaderCreateFlagBitsEXT::eLinkStage,
        .stage=vk::ShaderStageFlagBits::eFragment,
        .code_type=vk::ShaderCodeTypeEXT::eSpirv,
        .code=jms::vulkan::shader::Load(std::string{"shader.frag.spv"}, device),
        .entry_point_name=std::string{"main"}
    });
    return out;
}


jms::vulkan::shader::ShaderGroup CreateGroup(vk::raii::Device& device,
                                             const std::vector<jms::vulkan::shader::Info>& infos,
                                             vk::AllocationCallbacks* vk_allocation_callbacks = nullptr) {
    return {device, {infos[0], infos[1]}, vk_allocation_callbacks};
}


void BindShaders(vk::raii::CommandBuffer& command_buffer, jms::vulkan::shader::ShaderGroup& shader_group) {
    shader_group.Bind(command_buffer, {0, 1}, {vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eFragment});
}
