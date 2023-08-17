#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <expected>
#include <iostream>
#include <memory>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "jms/vulkan/glm.hpp"
#include "jms/vulkan/vulkan.hpp"
#include "jms/vulkan/camera.hpp"
#include "jms/vulkan/commands.hpp"
#include "jms/vulkan/render_info.hpp"
#include "jms/vulkan/state.hpp"
#include "jms/vulkan/vertex_description.hpp"
#include "jms/wsi/glfw.hpp"
#include "jms/wsi/glfw.cpp"

#include "shader.hpp"
#include "scratch.hpp"


constexpr const size_t WIN_WIDTH = 1024;
constexpr const size_t WIN_HEIGHT = 1024;


struct DrawState {
    const jms::vulkan::State& vulkan_state;
    vk::Buffer vertex_buffer;
    vk::Buffer vertex_data_buffer;
    vk::Buffer index_buffer;
    const uint32_t num_indices;
    vk::Viewport viewport;
    vk::Rect2D scissor;
};


void DrawFrame(const DrawState& draw_state);
jms::wsi::glfw::Window CreateEnvironment(jms::vulkan::State&, jms::wsi::glfw::Environment&);


int main(int argc, char** argv) {
    std::cout << "start" << std::endl;

    try {
        jms::vulkan::State vulkan_state{};
        jms::wsi::glfw::Environment glfw_environment{};
        jms::wsi::glfw::Window window = CreateEnvironment(vulkan_state, glfw_environment);

        const vk::raii::PhysicalDevice& physical_device = vulkan_state.physical_devices.at(0);
        const vk::raii::Device& device = vulkan_state.devices.at(0);
        std::vector<uint32_t> optimal_indices = jms::vulkan::FindOptimalIndices(physical_device);

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
        std::vector<Vertex> vertices = {
            {.model_index=0, .position={0.0f, 0.0f, 0.0f}},
            {.model_index=0, .position={-5.0f, 0.0f, 0.0f}},
            {.model_index=0, .position={-5.0f, -5.0f, 0.0f}},

            {.model_index=1, .position={ 0.0f,  0.0f, 0.0f}},
            {.model_index=1, .position={13.0f,  0.0f, 0.0f}},
            {.model_index=1, .position={13.0f, 13.0f, 0.0f}},
            {.model_index=1, .position={ 0.0f, 13.0f, 0.0f}}
        };
        std::vector<VertexData> vertex_data = {
            {.color={1.0f, 0.0f, 0.0f, 1.0f}},
            {.color={1.0f, 0.0f, 0.0f, 1.0f}},
            {.color={1.0f, 0.0f, 0.0f, 1.0f}},

            {.color={1.0f, 1.0f, 1.0f, 1.0f}},
            {.color={1.0f, 1.0f, 1.0f, 1.0f}},
            {.color={1.0f, 1.0f, 1.0f, 1.0f}},
            {.color={1.0f, 1.0f, 1.0f, 1.0f}}
        };
        std::vector<ModelData> model_transform_data = {
            ModelData{.transform=glm::mat4{1.0f}},
            ModelData{.transform=glm::mat4{1.0f}}
            //ModelData{.transform=glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.75, 0, 0, 1)},//glm::mat4{1.0f},
            //ModelData{.transform=glm::mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -0.25, 0, 0, 1)}//glm::mat4{1.0f}
        };
        //std::vector<ModelData> model_transform_data = {
        //    ModelData{.transform=glm::mat4{1.0f}},
        //    ModelData{.transform=glm::mat4{1.0f}}
        //};
        std::vector<uint32_t> indices = {0, 2, 1, 3, 5, 4, 3, 6, 5};
        assert(indices.size() % 3 == 0);
        //std::vector<glm::mat4> camera_data = {camera.projection * glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f), camera.model_view};
        glm::mat4 camera_view{1.0f};
        camera_view = glm::translate(camera_view, glm::vec3(0, 0, -5.0f));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
        glm::mat4 mvp = projection * camera_view;
        std::vector<glm::mat4> camera_data{mvp};

        jms::vulkan::GPUStorageBuffer<Vertex> vertex_storage_buffer{
            device,
            physical_device,
            optimal_indices,
            vertices.size(),
            vk::BufferUsageFlags(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst)
        };
        jms::vulkan::GPUStorageBuffer<VertexData> vertex_data_storage_buffer{
            device,
            physical_device,
            optimal_indices,
            vertices.size(),
            vk::BufferUsageFlags(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst)
        };
        jms::vulkan::GPUStorageBuffer<uint32_t> indices_storage_buffer{
            device,
            physical_device,
            optimal_indices,
            indices.size(),
            vk::BufferUsageFlags(vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst)
        };
        jms::vulkan::GPUStorageBuffer<ModelData> model_transform_storage_buffer{
            device,
            physical_device,
            optimal_indices,
            model_transform_data.size(),
            vk::BufferUsageFlags(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst)
        };
        jms::vulkan::GPUStorageBuffer<glm::mat4> camera_buffer{
            device,
            physical_device,
            optimal_indices,
            1,
            vk::BufferUsageFlags(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst)
        };
        DrawState draw_state{
            .vulkan_state=vulkan_state,
            .vertex_buffer=*vertex_storage_buffer.buffer,
            .vertex_data_buffer=*vertex_data_storage_buffer.buffer,
            .index_buffer=*indices_storage_buffer.buffer,
            .num_indices=static_cast<uint32_t>(indices.size())
        };


        /***
         * COPY DATA TO GPU
         */
        {
            jms::vulkan::HostStorageBuffer<decltype(vertex_storage_buffer)::value_type_t> vertex_staging_buffer{
                device,
                physical_device,
                optimal_indices,
                vertex_storage_buffer.num_elements,
                vk::BufferUsageFlags(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferSrc),
                {}
            };
            jms::vulkan::HostStorageBuffer<decltype(vertex_data_storage_buffer)::value_type_t> vertex_data_staging_buffer{
                device,
                physical_device,
                optimal_indices,
                vertex_data_storage_buffer.num_elements,
                vk::BufferUsageFlags(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc),
                {}
            };
            jms::vulkan::HostStorageBuffer<decltype(indices_storage_buffer)::value_type_t> indices_staging_buffer{
                device,
                physical_device,
                optimal_indices,
                indices_storage_buffer.num_elements,
                vk::BufferUsageFlags(vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferSrc),
                {}
            };
            jms::vulkan::HostStorageBuffer<decltype(model_transform_storage_buffer)::value_type_t> model_transform_staging_buffer{
                device,
                physical_device,
                optimal_indices,
                model_transform_storage_buffer.num_elements,
                vk::BufferUsageFlags(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc),
                {}
            };
            jms::vulkan::HostStorageBuffer<decltype(camera_buffer)::value_type_t> camera_staging_buffer{
                device,
                physical_device,
                optimal_indices,
                camera_buffer.num_elements,
                vk::BufferUsageFlags(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc),
                {}
            };
            vertex_staging_buffer.Copy(vertices);
            vertex_data_staging_buffer.Copy(vertex_data);
            indices_staging_buffer.Copy(indices);
            model_transform_staging_buffer.Copy(model_transform_data);
            camera_staging_buffer.Copy(camera_data);
            std::vector<vk::raii::CommandBuffer> cbs = device.allocateCommandBuffers({
                .commandPool=*(vulkan_state.command_pools.at(0)),
                .level=vk::CommandBufferLevel::ePrimary,
                .commandBufferCount=1
            });
            std::vector<vk::CommandBuffer> vk_cbs{};
            std::ranges::transform(cbs, std::back_inserter(vk_cbs), [](auto& v) { return *v; });
            vk::raii::CommandBuffer& cb = cbs.at(0);
            cb.begin({.flags=vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            cb.copyBuffer(*vertex_staging_buffer.buffer, *vertex_storage_buffer.buffer, {{
                .srcOffset=0,
                .dstOffset=0,
                .size=vertex_storage_buffer.buffer_size
            }});
            cb.copyBuffer(*vertex_data_staging_buffer.buffer, *vertex_data_storage_buffer.buffer, {{
                .srcOffset=0,
                .dstOffset=0,
                .size=vertex_data_storage_buffer.buffer_size
            }});
            cb.copyBuffer(*indices_staging_buffer.buffer, *indices_storage_buffer.buffer, {{
                .srcOffset=0,
                .dstOffset=0,
                .size=indices_storage_buffer.buffer_size
            }});
            cb.copyBuffer(*model_transform_staging_buffer.buffer, *model_transform_storage_buffer.buffer, {{
                .srcOffset=0,
                .dstOffset=0,
                .size=model_transform_storage_buffer.buffer_size
            }});
            cb.copyBuffer(*camera_staging_buffer.buffer, *camera_buffer.buffer, {{
                .srcOffset=0,
                .dstOffset=0,
                .size=camera_staging_buffer.buffer_size
            }});
            cb.end();
            vulkan_state.graphics_queue.submit({{
                .commandBufferCount=static_cast<uint32_t>(vk_cbs.size()),
                .pCommandBuffers=vk_cbs.data()
            }});
            vulkan_state.graphics_queue.waitIdle();
        }

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
            .buffer=*vertex_data_storage_buffer.buffer,
            .offset=0,
            .range=vertex_data_storage_buffer.buffer_size
        };
        vk::DescriptorBufferInfo model_transform_storage_buffer_info{
            .buffer=*model_transform_storage_buffer.buffer,
            .offset=0,
            .range=model_transform_storage_buffer.buffer_size
        };
        vk::DescriptorBufferInfo camera_buffer_info{
            .buffer=*camera_buffer.buffer,
            .offset=0,
            .range=camera_buffer.buffer_size
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

            DrawFrame(draw_state);
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
            //std::string{"VK_LAYER_KHRONOS_synchronization2"}
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
