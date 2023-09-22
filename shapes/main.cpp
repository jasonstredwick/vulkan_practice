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
    jms::vulkan::State& vulkan_state;
    jms::vulkan::shader::ShaderGroup& shader_group;
    vk::Buffer vertex_buffer;
    vk::Buffer index_buffer;
    uint32_t num_indices;
    vk::Viewport viewport;
    vk::Rect2D scissor;
};


void DrawFrame(DrawState& draw_state);
jms::wsi::glfw::Window CreateEnvironment(jms::vulkan::State&, jms::wsi::glfw::Environment&);


template <typename T> size_t NumBytes(const T& t) noexcept { return t.size() * sizeof(typename T::value_type); }


int main(int argc, char** argv) {
    std::cout << std::format("Start\n");

    try {
        jms::vulkan::State vulkan_state{};
        jms::wsi::glfw::Environment glfw_environment{};
        jms::wsi::glfw::Window window = CreateEnvironment(vulkan_state, glfw_environment);
        vk::raii::PhysicalDevice& physical_device = vulkan_state.physical_devices.at(0);
        vk::raii::Device& device = vulkan_state.devices.at(0);

        std::vector<jms::vulkan::shader::Info> shader_info = LoadShaders(device);
        jms::vulkan::shader::ShaderGroup shader_group = CreateGroup(device, shader_info);

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
        std::ranges::transform(shader_group.layouts, std::back_inserter(layouts), [](auto& dsl) {
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
        vulkan_state.pipeline_layouts.push_back(device.createPipelineLayout({
            .setLayoutCount=static_cast<uint32_t>(layouts.size()),
            .pSetLayouts=layouts.data(),
            .pushConstantRangeCount=0,
            .pPushConstantRanges=nullptr
        }));

        DrawState draw_state{
            .vulkan_state=vulkan_state,
            .shader_group=shader_group,
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
                                         jms::wsi::glfw::Environment& glfw_environment) {
    glfw_environment.EnableHIDPI();

    jms::wsi::glfw::Window window = jms::wsi::glfw::Window::DefaultCreate(WIN_WIDTH, WIN_HEIGHT);
    auto [width, height] = window.DimsPixel();
    std::cout << std::format("Dims: ({}, {})\n", width, height);
    std::vector<std::string> window_instance_extensions= jms::wsi::glfw::GetVulkanInstanceExtensions();

    std::vector<std::string> required_instance_extensions{
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };
    std::set<std::string> instance_extensions_set{window_instance_extensions.begin(),
                                                  window_instance_extensions.end()};
    for (auto& i : required_instance_extensions) { instance_extensions_set.insert(i); }
    std::vector<std::string> instance_extensions{instance_extensions_set.begin(), instance_extensions_set.end()};

    std::vector<std::string> required_layers{
        std::string{"VK_LAYER_KHRONOS_synchronization2"},
        std::string{"VK_LAYER_KHRONOS_shader_object"}
    };
    vulkan_state.InitInstance(jms::vulkan::InstanceConfig{
        .app_name=std::string{"tut4"},
        .engine_name=std::string{"tut4.e"},
        .layer_names=required_layers,
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
    std::vector<std::string> required_device_extensions{
        VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
        VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
        VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
        VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
        VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME,
        VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
        VK_KHR_MAINTENANCE2_EXTENSION_NAME,
        VK_KHR_MULTIVIEW_EXTENSION_NAME,
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
    uint32_t queue_family_index = 0;
    std::vector<float> queue_priority(2, 1.0f); // graphics + presentation
    vulkan_state.InitDevice(physical_device, jms::vulkan::DeviceConfig{
        .layer_names={},
        .extension_names=required_device_extensions,
        .features={},
        .queue_infos=std::move(std::vector<vk::DeviceQueueCreateInfo>{
            // graphics queue + presentation queue
            {
                .queueFamilyIndex=queue_family_index,
                .queueCount=static_cast<uint32_t>(queue_priority.size()),
                .pQueuePriorities=queue_priority.data()
            }
        }),
        .pnext_features{
            vk::PhysicalDeviceShaderObjectFeaturesEXT{.shaderObject=true},
            vk::PhysicalDeviceDynamicRenderingFeatures{.dynamicRendering=true}
        }
    });
    vulkan_state.InitRenderPass(vulkan_state.devices.at(0), vulkan_state.render_info.format, vulkan_state.render_info.extent);
#if 0
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
#endif
    vulkan_state.InitQueues(vulkan_state.devices.at(0), queue_family_index);
    vulkan_state.InitSwapchain(vulkan_state.devices.at(0), vulkan_state.render_info, vulkan_state.surface, vulkan_state.render_passes.at(0));
    return window;
}


void DrawFrame(DrawState& draw_state) {
    auto& vulkan_state = draw_state.vulkan_state;
    auto& image_available_semaphore = vulkan_state.semaphores.at(0);
    auto& render_finished_semaphore = vulkan_state.semaphores.at(1);
    auto& in_flight_fence = vulkan_state.fences.at(0);
    auto& device = vulkan_state.devices.at(0);
    auto& swapchain = vulkan_state.swapchain;
    auto& swapchain_framebuffers = vulkan_state.swapchain_framebuffers;
    auto& vs_command_buffers_0 = vulkan_state.command_buffers.at(0);
    auto& command_buffer = vs_command_buffers_0.at(0);
    auto& render_pass = vulkan_state.render_passes.at(0);
    auto& target_extent = vulkan_state.render_info.extent;
    //auto& pipeline = vulkan_state.pipelines.at(0);
    auto& viewport = vulkan_state.viewports.front();
    auto& scissor = vulkan_state.scissors.front();
    auto& graphics_queue = vulkan_state.graphics_queue;
    auto& present_queue = vulkan_state.present_queue;
    auto& pipeline_layout = vulkan_state.pipeline_layouts.at(0);

    vk::Result result = device.waitForFences({*in_flight_fence}, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device.resetFences({*in_flight_fence});
    uint32_t image_index = 0;
    std::tie(result, image_index) = swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *image_available_semaphore);
    assert(result == vk::Result::eSuccess);
    assert(image_index < swapchain_framebuffers.size());

    jms::vulkan::VertexDescription2EXT vertex_desc = jms::vulkan::VertexDescription2EXT::Create<Vertex>(0);
    vk::ClearValue clear_value{.color={std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}}};

    std::vector<vk::RenderingAttachmentInfo> color_attachments{
        {
            .imageView=*vulkan_state.swapchain_image_views[image_index],
            .imageLayout=vk::ImageLayout::eColorAttachmentOptimal,
            .resolveMode={},
            .resolveImageView={},
            .resolveImageLayout={},
            .loadOp=vk::AttachmentLoadOp::eClear,
            .storeOp=vk::AttachmentStoreOp::eStore,
            .clearValue=clear_value
        }
    };

    vk::RenderingInfo rendering_info{
        .flags={},
        .renderArea={
            .offset={0, 0},
            .extent=target_extent
        },
        .layerCount=1,
        .viewMask=0,
        .colorAttachmentCount=static_cast<uint32_t>(color_attachments.size()),
        .pColorAttachments=color_attachments.data(),
        .pDepthAttachment=nullptr,
        .pStencilAttachment=nullptr
    };

    command_buffer.reset();
    command_buffer.begin({.pInheritanceInfo=nullptr});
    command_buffer.beginRendering(rendering_info);

    command_buffer.setViewportWithCountEXT({
        {
            .x=0.0f,
            .y=0.0f,
            .width=static_cast<float>(target_extent.width),
            .height=static_cast<float>(target_extent.height),
            .minDepth=0.0f,
            .maxDepth=1.0f
        }
    });
    command_buffer.setScissorWithCountEXT({{.offset={0, 0}, .extent=target_extent}});

    command_buffer.setPrimitiveTopologyEXT(vk::PrimitiveTopology::eTriangleList);
    //command_buffer.setPrimitiveRestartEnableEXT(false);

    // multisampling
    //command_buffer.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
    //command_buffer.setAlphaToCoverageEnableEXT(false);
    //command_buffer.setAlphaToOneEnableEXT(false);
    //command_buffer.setSampleMaskEXT(vk::SampleCountFlagBits::e1, {});
    //    ...others?  seems like it is missing some settings like minSampleShading

    // rasterization
    //command_buffer.setRasterizerDiscardEnableEXT(false);
    //command_buffer.setPolygonModeEXT(vk::PolygonMode::eFill);
    command_buffer.setCullModeEXT(vk::CullModeFlagBits::eBack);
    command_buffer.setFrontFaceEXT(vk::FrontFace::eCounterClockwise); //vk::FrontFace::eClockwise  ---- review
    //command_buffer.setLineWidth(1.0f);
    //command_buffer.setDepthClampEnableEXT(false);
    //command_buffer.setDepthBiasEnableEXT(false);
    //command_buffer.setDepthBias(0.0f, 0.0f, 0.0f);

    // DepthStencilState
    //command_buffer.setDepthTestEnableEXT(false);
    //command_buffer.setDepthBoundsTestEnableEXT(false); // VkPipelineDepthStencilStateCreateInfo::depthBoundsTestEnable
    //command_buffer.setDepthBounds(0.0f, 1.0f); // VkPipelineDepthStencilStateCreateInfo::minDepthBounds/maxDepthBounds
    //command_buffer.setDepthClipEnableEXT(true); // if not provided then VkPipelineRasterizationDepthClipStateCreateInfoEXT::depthClipEnable or if VkPipelineRasterizationDepthClipStateCreateInfoEXT is not provided then the inverse of setDepthClampEnableEXT
    //command_buffer.setDepthClipNegativeOneToOneEXT(false);
    //command_buffer.setDepthWriteEnableEXT(false);
    //command_buffer.setDepthCompareOpEXT(vk::CompareOp::eNever);
    //command_buffer.setStencilTestEnableEXT(false);

    // Stencil stuff
    //command_buffer.setStencilOpEXT({}, {}, {}, {}, {});
    //command_buffer.setStencilCompareMask({}, {});
    //command_buffer.setStencilWriteMask({}, {});
    //command_buffer.setStencilReference({}, {});

    //command_buffer.setFragmentShadingRateKHR({}, {});

    //command_buffer.setLogicOpEnableEXT(false);
    //command_buffer.setLogicOpEXT(vk::LogicOp::eCopy);
    /*
    command_buffer.setColorWriteMaskEXT(0, {
        {
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA
        }
    });
    */

    command_buffer.setVertexInputEXT(vertex_desc.binding_description, vertex_desc.attribute_description);

    command_buffer.bindVertexBuffers(0, {draw_state.vertex_buffer}, {0});
    command_buffer.bindIndexBuffer(draw_state.index_buffer, 0, vk::IndexType::eUint32);
    BindShaders(command_buffer, draw_state.shader_group);
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, {*vulkan_state.descriptor_sets[0]}, {});
    command_buffer.drawIndexed(draw_state.num_indices, 1, 0, 0, 0);
    command_buffer.endRendering();
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
