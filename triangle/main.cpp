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

#include "jms/utils/no_mutex.hpp"
#include "jms/vulkan/camera.hpp"
#include "jms/vulkan/commands.hpp"
#include "jms/vulkan/render_info.hpp"
#include "jms/vulkan/state.hpp"
#include "jms/vulkan/vertex_description.hpp"
#include "jms/vulkan/vulkan.hpp"
#include "jms/wsi/glfw.hpp"
#include "jms/wsi/glfw.cpp"

#include "shader.hpp"


constexpr const size_t WIN_WIDTH = 1024;
constexpr const size_t WIN_HEIGHT = 1024;


void DrawFrame(const jms::vulkan::State& vulkan_state, const vk::raii::Buffer& vertex_buffer, const vk::raii::Buffer& index_buffer, uint32_t num_indices);
jms::wsi::glfw::Window CreateEnvironment(jms::vulkan::State&, jms::wsi::glfw::Environment&);


int main(int argc, char** argv) {
    std::cout << std::format("Start\n");

    try {
        jms::vulkan::State vulkan_state{};
        jms::wsi::glfw::Environment glfw_environment{};
        jms::wsi::glfw::Window window = CreateEnvironment(vulkan_state, glfw_environment);

        vk::raii::PhysicalDevice& physical_device = vulkan_state.physical_devices.at(0);
        vk::raii::Device& device = vulkan_state.devices.at(0);
        vk::AllocationCallbacks vk_allocation_callbacks{};
        jms::vulkan::MemoryHelper memory_helper{physical_device, device, vk_allocation_callbacks};
        uint32_t memory_type_index = memory_helper.GetHostVisibleDeviceMemoryResourceCapableMemoryType();
        jms::vulkan::DeviceMemoryResource dmr = memory_helper.CreateDirectMemoryResource(memory_type_index);
        auto allocator = memory_helper.CreateHostVisibleDeviceMemoryResource<std::pmr::vector, jms::NoMutex>(dmr);

        std::pmr::vector<Vertex> vertices{&allocator};
        std::pmr::vector<uint32_t> indices{&allocator};

        vertices = {
            {{0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
            {{0.5f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}},
            {{0.5f, 0.5f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}}
        };
        indices = {0, 2, 1};

        // testing MemoryDevice re-allocation with std::pmr::vector
        vertices.push_back({{0.5f, 0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
        vertices.push_back({{0.0f, 0.5f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
        vertices.push_back({{0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}});
        indices.push_back(3);
        indices.push_back(5);
        indices.push_back(4);

        size_t vertex_buffer_size_in_bytes = sizeof(Vertex) * vertices.size();
        size_t indices_size_in_bytes = sizeof(uint32_t) * indices.size();
        vk::raii::Buffer vertex_buffer = allocator.AsBuffer(
            vertices.data(), vertex_buffer_size_in_bytes, vk::BufferUsageFlagBits::eVertexBuffer);
        vk::raii::Buffer index_buffer = allocator.AsBuffer(
            indices.data(), indices_size_in_bytes, vk::BufferUsageFlagBits::eIndexBuffer);

        std::cout << "---------------------\n";
        bool not_done = true;
        int ix = 500;
        while (not_done) {
            glfwPollEvents();
            int state = glfwGetKey(window.get(), GLFW_KEY_Q);
            if (state == GLFW_PRESS) { not_done = false; }
            DrawFrame(vulkan_state, vertex_buffer, index_buffer, indices.size() / 3 * 3);
            vulkan_state.devices.at(0).waitIdle();
        }
    } catch (std::exception const& exp) {
        std::cout << "Exception caught\n" << exp.what() << std::endl;
    }

    std::cout << std::format("End\n");
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
