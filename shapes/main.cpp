#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "jms/utils/no_mutex.hpp"
#include "jms/vulkan/glm.hpp"
#include "jms/vulkan/vulkan.hpp"
#include "jms/vulkan/camera.hpp"
#include "jms/vulkan/commands.hpp"
#include "jms/vulkan/memory.hpp"
#include "jms/vulkan/memory_resource.hpp"
#include "jms/vulkan/render_info.hpp"
#include "jms/vulkan/state.hpp"
#include "jms/vulkan/vertex_description.hpp"
#include "jms/wsi/glfw.hpp"
#include "jms/wsi/glfw.cpp"

#include "shader.hpp"


constexpr const size_t WIN_WIDTH = 1024;
constexpr const size_t WIN_HEIGHT = 1024;


struct DrawState {
    const jms::vulkan::State& vulkan_state;
    vk::Buffer vertex_buffer;
    vk::Buffer index_buffer;
    const uint32_t num_indices;
    vk::Viewport viewport;
    vk::Rect2D scissor;
};


void DrawFrame(const DrawState& draw_state);
jms::wsi::glfw::Window CreateEnvironment(jms::vulkan::State&,
                                         jms::wsi::glfw::Environment&,
                                         const jms::vulkan::VertexDescription& vertex_desc,
                                         const std::vector<vk::DescriptorSetLayoutBinding>& layout_bindings);


template <typename T> size_t NumBytes(const T& t) noexcept { return t.size() * sizeof(typename T::value_type); }


int main(int argc, char** argv) {
    std::cout << std::format("Start\n");

    try {
        jms::vulkan::State vulkan_state{};
        jms::wsi::glfw::Environment glfw_environment{};
        jms::wsi::glfw::Window window = CreateEnvironment(
            vulkan_state,
            glfw_environment,
            jms::vulkan::VertexDescription::Create<Vertex>(0),
            std::vector<vk::DescriptorSetLayoutBinding>{
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
            });


        vk::raii::PhysicalDevice& physical_device = vulkan_state.physical_devices.at(0);
        vk::raii::Device& device = vulkan_state.devices.at(0);
        jms::vulkan::MemoryHelper memory_helper{physical_device, device};
        auto dyn_memory_type_index = memory_helper.GetDeviceMemoryResourceMappedCapableMemoryType();
        if (dyn_memory_type_index == std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error{"Could not find a suitable, requested memory type."};
        }
        jms::vulkan::DeviceMemoryResource dyn_dmr = memory_helper.CreateDirectMemoryResource(dyn_memory_type_index);
        auto dyn_allocator = memory_helper.CreateDeviceMemoryResourceMapped<std::pmr::vector, jms::NoMutex>(dyn_dmr);
        auto obj_allocator = std::pmr::polymorphic_allocator{&dyn_allocator};

        std::pmr::vector<Vertex> vertices{{
            {.model_index=0, .position={0.0f, 0.0f, 0.0f}},
            {.model_index=0, .position={-5.0f, 0.0f, 0.0f}},
            {.model_index=0, .position={-5.0f, -5.0f, 0.0f}},

            {.model_index=1, .position={ 0.0f,  0.0f, 0.0f}},
            {.model_index=1, .position={13.0f,  0.0f, 0.0f}},
            {.model_index=1, .position={13.0f, 13.0f, 0.0f}},
            {.model_index=1, .position={ 0.0f, 13.0f, 0.0f}}
        }, &dyn_allocator};

        std::pmr::vector<uint32_t> indices{{0, 2, 1, 3, 5, 4, 3, 6, 5}, &dyn_allocator};

        std::pmr::vector<VertexData> vertex_data{{
            {.color={1.0f, 0.0f, 0.0f, 1.0f}},
            {.color={1.0f, 0.0f, 0.0f, 1.0f}},
            {.color={1.0f, 0.0f, 0.0f, 1.0f}},

            {.color={1.0f, 1.0f, 1.0f, 1.0f}},
            {.color={1.0f, 1.0f, 1.0f, 1.0f}},
            {.color={1.0f, 1.0f, 1.0f, 1.0f}},
            {.color={1.0f, 1.0f, 1.0f, 1.0f}}
        }, &dyn_allocator};

        std::pmr::vector<ModelData> model_transform_data{{
            ModelData{.transform=glm::mat4{1.0f}},
            ModelData{.transform=glm::mat4{1.0f}}
            //ModelData{.transform=glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.75, 0, 0, 1)},//glm::mat4{1.0f},
            //ModelData{.transform=glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.25, 0, 0, 1)}//glm::mat4{1.0f}
        }, &dyn_allocator};
        //model_transform_data = {
        //    ModelData{.transform=glm::mat4{1.0f}},
        //    ModelData{.transform=glm::mat4{1.0f}}
        //};

        /***
         * World
         *
         * Z Y
         * |/
         * .--X
         */
        glm::mat4 world{1.0f};
        //jms::vulkan::Camera camera = jms::vulkan::Camera(
        //    glm::radians(60.0f),
        //    static_cast<float>(WIN_WIDTH) / static_cast<float>(WIN_HEIGHT),
        //    0.1f);
        jms::vulkan::Camera camera = jms::vulkan::Camera{
            glm::lookAt(glm::vec3{-5.0f, -3.0f, -10.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, -1.0f, 0.0f}),
            glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f)//glm::mat4{1.0f}//glm::infinitePerspective(glm::radians(60.0f), static_cast<float>(WIN_WIDTH) / static_cast<float>(WIN_HEIGHT), 0.1f)
        };
        //jms::vulkan::UniqueMappedResource<jms::vulkan::Camera> camera{
        //    obj_allocator,
        //    glm::lookAt(glm::vec3{-5.0f, -3.0f, -10.0f}, glm::vec3{0.0f, 0.0f, 0.0f}, glm::vec3{0.0f, -1.0f, 0.0f}),
        //    glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.5f, 1.0f)//glm::mat4{1.0f}//glm::infinitePerspective(glm::radians(60.0f), static_cast<float>(WIN_WIDTH) / static_cast<float>(WIN_HEIGHT), 0.1f)
        //};
        //std::vector<glm::mat4> camera_data = {camera.projection * glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f), camera.model_view};
        glm::mat4 camera_view{1.0f};
        camera_view = glm::translate(camera_view, glm::vec3(0, 0, -5.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
        glm::mat4 mvp = projection * camera_view;
        std::pmr::vector<glm::mat4> camera_data{{mvp}, &dyn_allocator};
        //std::vector<glm::mat4> camera_data{mvp};

        vk::raii::Buffer vertex_buffer = dyn_allocator.AsBuffer(
            vertices.data(), NumBytes(vertices), vk::BufferUsageFlagBits::eVertexBuffer);
        vk::raii::Buffer index_buffer = dyn_allocator.AsBuffer(
            indices.data(), NumBytes(indices), vk::BufferUsageFlagBits::eIndexBuffer);
        vk::raii::Buffer vertex_data_buffer = dyn_allocator.AsBuffer(
            vertex_data.data(), NumBytes(vertex_data), vk::BufferUsageFlagBits::eStorageBuffer);
        vk::raii::Buffer model_data_buffer = dyn_allocator.AsBuffer(
            model_transform_data.data(), NumBytes(model_transform_data), vk::BufferUsageFlagBits::eStorageBuffer);
        vk::raii::Buffer camera_buffer = dyn_allocator.AsBuffer(
            camera_data.data(), NumBytes(camera_data), vk::BufferUsageFlagBits::eStorageBuffer);

        /***
         * Update draw state
         */

        std::vector<vk::DescriptorPoolSize> pool_sizes{
            {
                .type=vk::DescriptorType::eStorageBuffer,
                .descriptorCount=3
            }
        };
        vk::raii::DescriptorPool descriptor_pool = vulkan_state.devices.at(0).createDescriptorPool({
            .maxSets=1,
            .poolSizeCount=static_cast<uint32_t>(pool_sizes.size()),
            .pPoolSizes=pool_sizes.data()
        });
        std::vector<vk::DescriptorSetLayout> layouts{};
        std::ranges::transform(vulkan_state.descriptor_set_layouts, std::back_inserter(layouts), [](auto& dsl) {
            return *dsl;
        });
        vulkan_state.descriptor_sets = vulkan_state.devices.at(0).allocateDescriptorSets({
            .descriptorPool=*descriptor_pool,
            .descriptorSetCount=1,
            .pSetLayouts=layouts.data()
        });

        vk::DescriptorBufferInfo vertex_data_buffer_info{
            .buffer=*vertex_data_buffer,
            .offset=0,
            .range=NumBytes(vertex_data)
        };
        vk::DescriptorBufferInfo model_transform_storage_buffer_info{
            .buffer=*model_data_buffer,
            .offset=0,
            .range=NumBytes(model_transform_data)
        };
        vk::DescriptorBufferInfo camera_buffer_info{
            .buffer=*camera_buffer,
            .offset=0,
            .range=NumBytes(camera_data)
        };
        vulkan_state.devices.at(0).updateDescriptorSets({{
            .dstSet=*vulkan_state.descriptor_sets.at(0),
            .dstBinding=0,
            .dstArrayElement=0,
            .descriptorCount=1,
            .descriptorType=vk::DescriptorType::eStorageBuffer,
            .pImageInfo=nullptr,
            .pBufferInfo=&vertex_data_buffer_info,
            .pTexelBufferView=nullptr
        }, {
            .dstSet=*vulkan_state.descriptor_sets.at(0),
            .dstBinding=1,
            .dstArrayElement=0,
            .descriptorCount=1,
            .descriptorType=vk::DescriptorType::eStorageBuffer,
            .pImageInfo=nullptr,
            .pBufferInfo=&model_transform_storage_buffer_info,
            .pTexelBufferView=nullptr
        }, {
            .dstSet=*vulkan_state.descriptor_sets.at(0),
            .dstBinding=2,
            .dstArrayElement=0,
            .descriptorCount=1,
            .descriptorType=vk::DescriptorType::eStorageBuffer,
            .pImageInfo=nullptr,
            .pBufferInfo=&camera_buffer_info,
            .pTexelBufferView=nullptr
        }}, {});

        DrawState draw_state{
            .vulkan_state=vulkan_state,
            .vertex_buffer=*vertex_buffer,
            .index_buffer=*index_buffer,
            .num_indices=static_cast<uint32_t>(indices.size())
        };

        std::cout << std::format("---------------------\n");
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

            DrawFrame(draw_state);
            vulkan_state.devices.at(0).waitIdle();
        }
        //while (sys_window.ProcessEvents()) { ; }
        //std::cin >> a;
    } catch (std::exception const& exp) {
        std::cout << "Exception caught\n" << exp.what() << std::endl;
    }

    std::cout << std::format("End\n");
    return 0;
}


jms::wsi::glfw::Window CreateEnvironment(jms::vulkan::State& vulkan_state,
                                         jms::wsi::glfw::Environment& glfw_environment,
                                         const jms::vulkan::VertexDescription& vertex_desc,
                                         const std::vector<vk::DescriptorSetLayoutBinding>& layout_bindings) {
    glfw_environment.EnableHIDPI();

    jms::wsi::glfw::Window window = jms::wsi::glfw::Window::DefaultCreate(WIN_WIDTH, WIN_HEIGHT);
    auto [width, height] = window.DimsPixel();
    std::cout << std::format("Dims: ({}, {})\n", width, height);
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
                              std::vector<vk::DescriptorSetLayoutBinding>{
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
                              LoadShaders(vulkan_state.devices.at(0)));
    vulkan_state.InitQueues(vulkan_state.devices.at(0), queue_family_index);
    vulkan_state.InitSwapchain(vulkan_state.devices.at(0), vulkan_state.render_info, vulkan_state.surface, vulkan_state.render_passes.at(0));
    return window;
}


void DrawFrame(const DrawState& draw_state) {
    const auto& vulkan_state = draw_state.vulkan_state;
    const auto& image_available_semaphore = vulkan_state.semaphores.at(0);
    const auto& render_finished_semaphore = vulkan_state.semaphores.at(1);
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
    command_buffer.bindVertexBuffers(0, {draw_state.vertex_buffer}, {0});
    command_buffer.bindIndexBuffer(draw_state.index_buffer, 0, vk::IndexType::eUint32);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, {*vulkan_state.descriptor_sets[0]}, {});
    command_buffer.drawIndexed(draw_state.num_indices, 1, 0, 0, 0);
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
