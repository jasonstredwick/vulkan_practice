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

    static std::vector<vk::VertexInputAttributeDescription> GetAttributeDesc(uint32_t binding) {
        return {
            vk::VertexInputAttributeDescription{
                .location=0,
                .binding=binding,
                .format=vk::Format::eR32Uint,
                .offset=offsetof(Vertex, model_index)
            },
            vk::VertexInputAttributeDescription{
                .location=1,
                .binding=binding,
                .format=vk::Format::eR32G32B32Sfloat,
                .offset=offsetof(Vertex, position)
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


struct VertexData {
    glm::vec4 color{};
};
static_assert(std::is_standard_layout_v<VertexData>); // requires for offsetof macro


struct ModelData {
    glm::mat4 transform{1.0f};
};
static_assert(std::is_standard_layout_v<ModelData>); // requires for offsetof macro


#if 0
class VertexPool {
    VertexPool() {
    }
};

// currently single thread implementation, chunking across multiple threads would require locks
class VertexBuffer {
    std::array<std::vector<vk::raii::DeviceMemory>, VK_MAX_MEMORY_TYPES> device_memory{{}};
    std::vector<Vertex> vertices{};
    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eVertexBuffer;
    vk::SharingMode sharing_mode = vk::SharingMode::eExclusive;

public:
    VertexBuffer(const vk::raii::PhysicalDevice& physical_device,
                 const vk::raii::Device& device,
                 const std::vector<Vertex>& vertices) {
        vk::raii::Buffer buffer = device.createBuffer({
            .pNext=nullptr,
            .flags={},
            .size=static_cast<vk::DeviceSize>(sizeof(Vertex) * vertices.size()),
            .usage=vk::BufferUsageFlags(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst),
            .sharingMode=vk::SharingMode::eExclusive
        });
        vk::MemoryRequirements mem_reqs = buffer.getMemoryRequirements();
        const vk::DeviceSize buffer_size_bytes = mem_reqs.size;
        const vk::DeviceSize buffer_align_bytes = mem_reqs.alignment;
        // bit_i == memory_type_index from PhysicalDevice::MemoryProperties::memoryTypeCount array
        const uint32_t memory_type_indices_supported = mem_reqs.memoryTypeBits;
        std::vector<uint32_t> optimal_indices = jms::vulkan::FindOptimalIndices(physical_device);
        std::vector<uint32_t> restricted_indices = jms::vulkan::RestrictMemoryTypes(physical_device,
                                                                                    optimal_indices,
                                                                                    memory_type_indices_supported,
                                                                                    {});
        if (optimal_indices.empty()) { throw std::runtime_error("Failed to create VertexBuffer ... no compatible memory type found."); }
        MapMemory(device, buffer, restricted_indices, buffer_size_bytes);
    }

    void MapMemory(const vk::raii::Device& device,
                   const vk::raii::Buffer& buffer,
                   const std::vector<uint32_t>& optimal_indices,
                   const vk::DeviceSize buffer_size_in_bytes) {
        uint32_t memory_type_index = optimal_indices[0];
        std::vector<vk::raii::DeviceMemory>& memory = device_memory[memory_type_index];
        if (memory.empty()) {
            vk::DeviceSize ALLOCATION_BLOCK_SIZE = std::pow(2, 20) * 256;
            vk::DeviceSize num_units = (buffer_size_in_bytes / ALLOCATION_BLOCK_SIZE) + (buffer_size_in_bytes % ALLOCATION_BLOCK_SIZE);
            vk::DeviceSize total_bytes = num_units * ALLOCATION_BLOCK_SIZE;
            memory.emplace_back(device, vk::MemoryAllocateInfo{
                .allocationSize=total_bytes,
                .memoryTypeIndex=memory_type_index
            });
        }
        vk::raii::DeviceMemory& device_memory = memory.back();
        device_memory.mapMemory(0, buffer_size_in_bytes);
    }
};
#endif


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


