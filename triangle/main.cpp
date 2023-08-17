#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <expected>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "jms/vulkan/vulkan.hpp"
#include "jms/vulkan/camera.hpp"
#include "jms/vulkan/commands.hpp"
#include "jms/vulkan/render_info.hpp"
#include "jms/vulkan/state.hpp"
#include "jms/vulkan/vertex_description.hpp"
#include "jms/wsi/glfw.hpp"
#include "jms/wsi/glfw.cpp"

#include "shader.hpp"


constexpr const size_t DIM_X = 768;
constexpr const size_t DIM_Y = 768;
constexpr const size_t WIN_WIDTH = 1024;
constexpr const size_t WIN_HEIGHT = 1024;


void DrawFrame(const jms::vulkan::State& vulkan_state, const vk::raii::Buffer& vertex_buffer, const vk::raii::Buffer& index_buffer, uint32_t num_indices);
jms::wsi::glfw::Window CreateEnvironment(jms::vulkan::State&, jms::wsi::glfw::Environment&);


int main(int argc, char** argv) {
    std::cout << "start" << std::endl;

    try {
        std::vector<Vertex> vertices = {
            {{0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
            {{0.5f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
            {{0.5f, 0.5f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}
        };
        std::vector<uint32_t> indices = {0, 2, 1};
        size_t vertex_buffer_size_in_bytes = sizeof(Vertex) * vertices.size();
        size_t indices_size_in_bytes = sizeof(uint32_t) * indices.size();

        jms::vulkan::State vulkan_state{};
        jms::wsi::glfw::Environment glfw_environment{};
        jms::wsi::glfw::Window window = CreateEnvironment(vulkan_state, glfw_environment);

        const vk::raii::PhysicalDevice& physical_device = vulkan_state.physical_devices.at(0);
        const vk::raii::Device& device = vulkan_state.devices.at(0);
        std::vector<uint32_t> optimal_indices = jms::vulkan::FindOptimalIndices(physical_device);

        /***
         * VERTICES
        */
        vk::raii::Buffer buffer_vs = device.createBuffer({
            .size=vertex_buffer_size_in_bytes,
            .usage=vk::BufferUsageFlags(vk::BufferUsageFlagBits::eTransferSrc),
            .sharingMode=vk::SharingMode::eExclusive
        });
        vk::MemoryRequirements mem_reqs_vs = buffer_vs.getMemoryRequirements();
        vk::MemoryPropertyFlags prohibited_flags_vs = vk::MemoryPropertyFlagBits::eDeviceLocal;
        std::vector<uint32_t> restricted_indices_vs = jms::vulkan::RestrictMemoryTypes(physical_device,
                                                                                       optimal_indices,
                                                                                       mem_reqs_vs.memoryTypeBits,
                                                                                       prohibited_flags_vs);
        if (restricted_indices_vs.empty()) { throw std::runtime_error("Failed to create vertex staging buffer ... no compatible memory type found."); }
        vk::raii::DeviceMemory device_memory_vs = device.allocateMemory({
            .allocationSize= mem_reqs_vs.size,
            .memoryTypeIndex=restricted_indices_vs[0]
        });
        buffer_vs.bindMemory(*device_memory_vs, 0);
        void* data_vs = device_memory_vs.mapMemory(0, vertex_buffer_size_in_bytes);
        std::memcpy(data_vs, vertices.data(), vertex_buffer_size_in_bytes);
        device_memory_vs.unmapMemory();

        vk::raii::Buffer buffer_v = device.createBuffer({
            .size=vertex_buffer_size_in_bytes,
            .usage=vk::BufferUsageFlags(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst),
            .sharingMode=vk::SharingMode::eExclusive
        });
        vk::MemoryRequirements mem_reqs_v = buffer_v.getMemoryRequirements();
        std::vector<uint32_t> restricted_indices_v = jms::vulkan::RestrictMemoryTypes(physical_device,
                                                                                      optimal_indices,
                                                                                      mem_reqs_v.memoryTypeBits,
                                                                                      {});
        if (restricted_indices_v.empty()) { throw std::runtime_error("Failed to create vertex buffer ... no compatible memory type found."); }
        vk::raii::DeviceMemory device_memory_v = device.allocateMemory({
            .allocationSize= mem_reqs_v.size,
            .memoryTypeIndex=restricted_indices_v[0]
        });
        buffer_v.bindMemory(*device_memory_v, 0);

        /***
         * INDICES
        */
        vk::raii::Buffer buffer_is = device.createBuffer({
            .size=indices_size_in_bytes,
            .usage=vk::BufferUsageFlags(vk::BufferUsageFlagBits::eTransferSrc),
            .sharingMode=vk::SharingMode::eExclusive
        });
        vk::MemoryRequirements mem_reqs_is = buffer_is.getMemoryRequirements();
        vk::MemoryPropertyFlags prohibited_flags_is = vk::MemoryPropertyFlagBits::eDeviceLocal;
        std::vector<uint32_t> restricted_indices_is = jms::vulkan::RestrictMemoryTypes(physical_device,
                                                                                       optimal_indices,
                                                                                       mem_reqs_is.memoryTypeBits,
                                                                                       prohibited_flags_is);
        if (restricted_indices_is.empty()) { throw std::runtime_error("Failed to create indices staging buffer ... no compatible memory type found."); }
        vk::raii::DeviceMemory device_memory_is = device.allocateMemory({
            .allocationSize= mem_reqs_is.size,
            .memoryTypeIndex=restricted_indices_is[0]
        });
        buffer_is.bindMemory(*device_memory_is, 0);
        void* data_is = device_memory_is.mapMemory(0, indices_size_in_bytes);
        std::memcpy(data_is, indices.data(), indices_size_in_bytes);
        device_memory_is.unmapMemory();

        vk::raii::Buffer buffer_i = device.createBuffer({
            .size=indices_size_in_bytes,
            .usage=vk::BufferUsageFlags(vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst),
            .sharingMode=vk::SharingMode::eExclusive
        });
        vk::MemoryRequirements mem_reqs_i = buffer_i.getMemoryRequirements();
        std::vector<uint32_t> restricted_indices_i = jms::vulkan::RestrictMemoryTypes(physical_device,
                                                                                      optimal_indices,
                                                                                      mem_reqs_i.memoryTypeBits,
                                                                                      {});
        if (restricted_indices_i.empty()) { throw std::runtime_error("Failed to create indices buffer ... no compatible memory type found."); }
        vk::raii::DeviceMemory device_memory_i = device.allocateMemory({
            .allocationSize= mem_reqs_i.size,
            .memoryTypeIndex=restricted_indices_i[0]
        });
        buffer_i.bindMemory(*device_memory_i, 0);

        /***
         * COPY DATA TO GPU
        */
        {
            std::vector<vk::raii::CommandBuffer> cbs = device.allocateCommandBuffers({
                .commandPool=*(vulkan_state.command_pools.at(0)),
                .level=vk::CommandBufferLevel::ePrimary,
                .commandBufferCount=1
            });
            std::vector<vk::CommandBuffer> vk_cbs{};
            std::ranges::transform(cbs, std::back_inserter(vk_cbs), [](auto& v) { return *v; });
            vk::raii::CommandBuffer& cb = cbs.at(0);
            cb.begin({.flags=vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cb.copyBuffer(*buffer_vs, *buffer_v, {{
                .srcOffset=0,
                .dstOffset=0,
                .size=vertex_buffer_size_in_bytes
            }});
            cb.copyBuffer(*buffer_is, *buffer_i, {{
                .srcOffset=0,
                .dstOffset=0,
                .size=indices_size_in_bytes
            }});
            cb.end();
            vulkan_state.graphics_queue.submit({{
                .commandBufferCount=static_cast<uint32_t>(vk_cbs.size()),
                .pCommandBuffers=vk_cbs.data()
            }});
            vulkan_state.graphics_queue.waitIdle();
        }

        std::cout << "---------------------\n";
        //std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();
        bool not_done = true;
        int ix = 500;
        while (not_done) {
            glfwPollEvents();
            int state = glfwGetKey(window.get(), GLFW_KEY_Q);
            if (state == GLFW_PRESS) {
                not_done = false;
            }
            //std::chrono::time_point<std::chrono::system_clock> t1 = std::chrono::system_clock::now();
            //if (std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() > 2) {
            //    glfwPollEvents();
            //}
            //if (std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() > 5) {
            //    not_done = false;
            //}
            //vertices[0].position.x = delta - 2.0 * delta * (static_cast<float>(ix) / 500.0f);
            //ix = (ix + 1) % 500;

            //const vk::raii::DeviceMemory& device_memory = vulkan_state.device_memory.at(0);
            //const vk::MemoryRequirements& buffers_mem_reqs = vulkan_state.buffers_mem_reqs.at(0);
            //void* data = device_memory.mapMemory(0, buffers_mem_reqs.size);
            //void* data = vulkan_state.mapped_buffers.at(0);
            //std::memcpy(data, vertices.data(), vertex_buffer_size_in_bytes);
            //device_memory.unmapMemory();

            DrawFrame(vulkan_state, buffer_v, buffer_i, indices.size() / 3 * 3);
            vulkan_state.devices.at(0).waitIdle();
        }
        //while (sys_window.ProcessEvents()) { ; }
        //std::cin >> a;
    } catch (std::exception const& exp) {
        std::cout << "Exception caught\n" << exp.what() << std::endl;
    }

    std::cout << fmt::format("End\n");
    return 0;
}


jms::wsi::glfw::Window CreateEnvironment(jms::vulkan::State& vulkan_state,
                                         jms::wsi::glfw::Environment& glfw_environment) {
    glfw_environment.EnableHIDPI();

    jms::wsi::glfw::Window window = jms::wsi::glfw::Window::DefaultCreate(WIN_WIDTH, WIN_HEIGHT);
    auto [width, height] = window.DimsPixel();
    std::cout << "Dims: (" << width << ", " << height << ")" << std::endl;
    std::vector<std::string> window_instance_extensions= jms::wsi::glfw::GetVulkanInstanceExtensions();

    std::vector<std::string> required_instance_extensions{};
    std::set<std::string> instance_extensions_set{window_instance_extensions.begin(),
                                                  window_instance_extensions.end()};
    for (auto& i : required_instance_extensions) { instance_extensions_set.insert(i); }
    std::vector<std::string> instance_extensions{instance_extensions_set.begin(), instance_extensions_set.end()};

    vulkan_state.InitInstance(jms::vulkan::InstanceConfig{
        .app_name=std::string{"tut4"},
        .engine_name=std::string{"tut4.e"},
        .layer_names={
            std::string{"VK_LAYER_KHRONOS_synchronization2"}
        },
        .extension_names=instance_extensions
    });

    // Create surface; must happen after instance creation, but before examining/creating devices.
    // will be moved from
    vulkan_state.surface = jms::wsi::glfw::CreateSurface(window, vulkan_state.instance);

    vk::raii::PhysicalDevice& physical_device = vulkan_state.physical_devices.at(0);
    vulkan_state.render_info = jms::vulkan::SurfaceInfo(vulkan_state.surface,
                                                        physical_device,
                                                        static_cast<uint32_t>(width),
                                                        static_cast<uint32_t>(height));
    uint32_t queue_family_index = 0;
    std::vector<float> queue_priority(2, 1.0f); // graphics + presentation
    vulkan_state.InitDevice(physical_device, jms::vulkan::DeviceConfig{
        .extension_names={std::string{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
        .queue_infos=std::move(std::vector<vk::DeviceQueueCreateInfo>{
            // graphics queue + presentation queue
            {
                .queueFamilyIndex=queue_family_index,
                .queueCount=static_cast<uint32_t>(queue_priority.size()),
                .pQueuePriorities=queue_priority.data()
            }
        })
    });
    vulkan_state.InitRenderPass(vulkan_state.devices.at(0), vulkan_state.render_info.format, vulkan_state.render_info.extent);
    vulkan_state.InitPipeline(vulkan_state.devices.at(0),
                              vulkan_state.render_passes.at(0),
                              vulkan_state.render_info.extent,
                              jms::vulkan::VertexDescription::Create<Vertex>(0),
                              std::vector<vk::DescriptorSetLayoutBinding>{/*
                                jms::vulkan::UniformBufferObject::Binding(0),
                                vk::DescriptorSetLayoutBinding{
                                    .binding=1,
                                    .descriptorType=vk::DescriptorType::eStorageImage,
                                    .descriptorCount=1,
                                    .stageFlags=vk::ShaderStageFlagBits::eFragment,
                                    .pImmutableSamplers=nullptr
                                }
                              */},
                              LoadShaders(vulkan_state.devices.at(0)));
    vulkan_state.InitQueues(vulkan_state.devices.at(0), queue_family_index);
    vulkan_state.InitSwapchain(vulkan_state.devices.at(0), vulkan_state.render_info, vulkan_state.surface, vulkan_state.render_passes.at(0));
    const vk::raii::Semaphore& image_available_semaphore = vulkan_state.semaphores.at(0);
    const vk::raii::Semaphore& render_finished_semaphore = vulkan_state.semaphores.at(0);
    const vk::raii::Fence& in_flight_fence = vulkan_state.fences.at(0);
    return window;
}


void DrawFrame(const jms::vulkan::State& vulkan_state, const vk::raii::Buffer& vertex_buffer, const vk::raii::Buffer& index_buffer, uint32_t num_indices) {
    const auto& image_available_semaphore = vulkan_state.semaphores.at(0);
    const auto& render_finished_semaphore = vulkan_state.semaphores.at(0);
    const auto& in_flight_fence = vulkan_state.fences.at(0);
    const auto& device = vulkan_state.devices.at(0);
    const auto& swapchain = vulkan_state.swapchain;
    const auto& swapchain_framebuffers = vulkan_state.swapchain_framebuffers;
    const auto& vs_command_buffers_0 = vulkan_state.command_buffers.at(0);
    const auto& command_buffer = vs_command_buffers_0.at(0);
    const auto& render_pass = vulkan_state.render_passes.at(0);
    const auto& swapchain_extent = vulkan_state.render_info.extent;
    const auto& pipeline = vulkan_state.pipelines.at(0);
    const auto& viewport = vulkan_state.viewports.front();
    const auto& scissor = vulkan_state.scissors.front();
    const auto& graphics_queue = vulkan_state.graphics_queue;
    const auto& present_queue = vulkan_state.present_queue;
    const auto& pipeline_layout = vulkan_state.pipeline_layouts.at(0);

    vk::Result result = device.waitForFences({*in_flight_fence}, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device.resetFences({*in_flight_fence});
    uint32_t image_index = 0;
    std::tie(result, image_index) = swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *image_available_semaphore);
    assert(result == vk::Result::eSuccess);
    assert(image_index < swapchain_framebuffers.size());
    vk::ClearValue clear_color{.color={std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}}};

    command_buffer.reset();
    command_buffer.begin({.pInheritanceInfo=nullptr});
    command_buffer.beginRenderPass({
        .renderPass=*render_pass,
        .framebuffer=*swapchain_framebuffers[image_index],
        .renderArea={
            .offset={0, 0},
            .extent=swapchain_extent
        },
        .clearValueCount=1,
        .pClearValues=&clear_color // count + values can be array
    }, vk::SubpassContents::eInline);
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
    command_buffer.setViewport(0, viewport);
    command_buffer.setScissor(0, scissor);
    command_buffer.bindVertexBuffers(0, {*vertex_buffer}, {0});
    command_buffer.bindIndexBuffer(*index_buffer, 0, vk::IndexType::eUint32);
    command_buffer.drawIndexed(num_indices, 1, 0, 0, 0);
    command_buffer.endRenderPass();
    command_buffer.end();

    std::vector<vk::Semaphore> wait_semaphores{*image_available_semaphore};
    std::vector<vk::Semaphore> signal_semaphores{*render_finished_semaphore};
    std::vector<vk::PipelineStageFlags> dst_stage_mask{vk::PipelineStageFlagBits::eColorAttachmentOutput};
    std::vector<vk::CommandBuffer> command_buffers{*command_buffer};
    graphics_queue.submit(std::array<vk::SubmitInfo, 1>{vk::SubmitInfo{
        .waitSemaphoreCount=static_cast<uint32_t>(wait_semaphores.size()),
        .pWaitSemaphores=wait_semaphores.data(),
        .pWaitDstStageMask=dst_stage_mask.data(),
        .commandBufferCount=static_cast<uint32_t>(command_buffers.size()),
        .pCommandBuffers=command_buffers.data(),
        .signalSemaphoreCount=static_cast<uint32_t>(signal_semaphores.size()),
        .pSignalSemaphores=signal_semaphores.data()
    }}, *in_flight_fence);
    std::vector<vk::SwapchainKHR> swapchains{*swapchain};
    result = present_queue.presentKHR({
        .waitSemaphoreCount=static_cast<uint32_t>(signal_semaphores.size()),
        .pWaitSemaphores=signal_semaphores.data(),
        .swapchainCount=static_cast<uint32_t>(swapchains.size()),
        .pSwapchains=swapchains.data(),
        .pImageIndices=&image_index,
        .pResults=nullptr
    });
}
