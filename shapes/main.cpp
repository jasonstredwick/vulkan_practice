#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "jms/utils/no_mutex.hpp"
#include "jms/vulkan/glm.hpp"
#include "jms/vulkan/vulkan.hpp"
#include "jms/vulkan/camera.hpp"
#include "jms/vulkan/info.hpp"
#include "jms/vulkan/memory.hpp"
#include "jms/vulkan/memory_resource.hpp"
#include "jms/vulkan/state.hpp"
#include "jms/vulkan/utils.hpp"
#include "jms/vulkan/variants.hpp"
#include "jms/wsi/glfw.hpp"
#include "jms/wsi/glfw.cpp"
#include "jms/wsi/surface.hpp"

#include "shader.hpp"


constexpr const size_t WINDOW_WIDTH{1024};
constexpr const size_t WINDOW_HEIGHT{1024};


struct AppState {
    size_t window_width{WINDOW_WIDTH};
    size_t window_height{WINDOW_HEIGHT};

    std::string app_name{"VULKAN_APP"};
    std::string engine_name{"VULKAN_APP_ENGINE"};

    std::vector<std::string> instance_extensions{
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };

    std::vector<std::string> instance_layers{
        std::string{"VK_LAYER_KHRONOS_synchronization2"},
        std::string{"VK_LAYER_KHRONOS_shader_object"}
    };

    std::vector<std::string> device_extensions{
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

    std::vector<std::string> device_layers{};
    vk::PhysicalDeviceFeatures device_features{};
    std::vector<jms::vulkan::DeviceCreateInfo2Variant> device_pnext_features{
        vk::PhysicalDeviceShaderObjectFeaturesEXT{.shaderObject=true},
        vk::PhysicalDeviceDynamicRenderingFeatures{.dynamicRendering=true}
    };

    std::vector<std::string> queue_family{"graphics", "presentation"};
    std::vector<float> queue_priority{1.0f, 1.0f};

    std::vector<vk::MemoryPropertyFlags> memory_types{
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    };

    jms::vulkan::GraphicsRenderingState default_graphics_rendering_state{
        .render_area{
            .extent{
                .width=static_cast<uint32_t>(WINDOW_WIDTH),
                .height=static_cast<uint32_t>(WINDOW_HEIGHT)
            }
        },

        .viewports={{
            .width=static_cast<float>(WINDOW_WIDTH),
            .height=static_cast<float>(WINDOW_HEIGHT)
        }},

        .scissors={{
            .extent{
                .width=static_cast<uint32_t>(WINDOW_WIDTH),
                .height=static_cast<uint32_t>(WINDOW_HEIGHT)
            }
        }}
    };

    jms::vulkan::ImageInfo render_target_info{
        .flags={},
        .image_type=vk::ImageType::e2D,
        .format=vk::Format::eR8G8B8A8Unorm,
        .extent={.width=WINDOW_WIDTH, .height=WINDOW_HEIGHT, .depth=1},
        .mip_levels=1,
        .array_layers=1,
        .samples=vk::SampleCountFlagBits::e1,
        .tiling=vk::ImageTiling::eOptimal,
        .usage=(vk::ImageUsageFlagBits::eColorAttachment |
                vk::ImageUsageFlagBits::eSampled |
                vk::ImageUsageFlagBits::eTransferSrc),
        .initial_layout=vk::ImageLayout::eUndefined
    };

    jms::vulkan::ImageViewInfo render_target_view_info{
        .flags={},
        .view_type=vk::ImageViewType::e2D,
        .format=vk::Format::eR8G8B8A8Unorm,
        .components={
            .r = vk::ComponentSwizzle::eIdentity,
            .g = vk::ComponentSwizzle::eIdentity,
            .b = vk::ComponentSwizzle::eIdentity,
            .a = vk::ComponentSwizzle::eIdentity
        },
        .subresource={
            .aspectMask=vk::ImageAspectFlagBits::eColor,
            .baseMipLevel=0,
            .levelCount=1,
            .baseArrayLayer=0,
            .layerCount=1
        }
    };
};


struct DrawState {
    jms::vulkan::State& vulkan_state;
    jms::vulkan::GraphicsPass& graphics_pass;
    vk::Buffer vertex_buffer;
    vk::Buffer index_buffer;
    uint32_t num_indices;
    std::vector<vk::DescriptorSet> descriptor_sets{};
    std::vector<vk::Image> targets{};
    std::vector<vk::ImageView> target_views{};
    std::vector<vk::Image> swapchain_images{};
};


void DrawFrame(DrawState& draw_state);
jms::vulkan::State CreateEnvironment(const AppState&, std::optional<jms::wsi::glfw::Window*> = std::nullopt);


template <typename T> size_t NumBytes(const T& t) noexcept { return t.size() * sizeof(typename T::value_type); }
template <typename T> size_t NumBytesCap(const T& t) noexcept { return t.capacity() * sizeof(typename T::value_type); }


int main(int argc, char** argv) {
    std::cout << std::format("Start\n");

    try {
        AppState app_state{};

        jms::wsi::glfw::Environment glfw_environment{};
        glfw_environment.EnableHIDPI();
        auto window = jms::wsi::glfw::Window::DefaultCreate(app_state.window_width, app_state.window_height);
        std::ranges::transform(jms::wsi::glfw::GetVulkanInstanceExtensions(),
                               std::back_inserter(app_state.instance_extensions),
                               [](auto& i) { return i; });

        jms::vulkan::State vulkan_state = CreateEnvironment(app_state, std::addressof(window));
        vk::raii::PhysicalDevice& physical_device = vulkan_state.physical_devices.at(0);
        vk::raii::Device& device = vulkan_state.devices.at(0);

        jms::vulkan::GraphicsPass graphics_pass{
            device, app_state.default_graphics_rendering_state, std::move(CreateGroup())};
        vk::raii::DescriptorPool descriptor_pool = graphics_pass.CreateDescriptorPool(device, 0, 1);
        std::vector<vk::DescriptorSet> descriptor_sets = graphics_pass.CreateDescriptorSets(
            device, descriptor_pool, {0});
        vulkan_state.semaphores.push_back(device.createSemaphore({}));
        vulkan_state.semaphores.push_back(device.createSemaphore({}));
        vulkan_state.semaphores.push_back(device.createSemaphore({}));
        vulkan_state.fences.push_back(device.createFence({.flags=vk::FenceCreateFlagBits::eSignaled}));



        using ImageResourceAllocator = jms::vulkan::ImageResourceAllocator<std::vector, jms::NoMutex>;
        using Image = jms::vulkan::Image<std::vector, jms::NoMutex>;

        auto image_allocator = vulkan_state.memory_helper.CreateImageAllocator(0);
        std::vector<Image> images{};
        images.reserve(3);
        std::ranges::generate_n(std::back_inserter(images), 3, [&a=image_allocator, &b=app_state.render_target_info]() {
            template <typename T> using Container_t = typename std::remove_cvref_t<decltype(a)>::Container_t<T>;
            using Mutex_t = typename std::remove_cvref_t<decltype(a)>::Mutex_t;
            return jms::vulkan::Image<Container_t, Mutex_t>{a, b};
        });

        std::vector<vk::raii::ImageView> image_views{};
        image_views.reserve(3);
        std::ranges::transform(images, std::back_inserter(image_views),
            [&info=app_state.render_target_view_info](const auto& image) { return image.CreateView(info); });

        auto dyn_allocator = vulkan_state.memory_helper.CreateDeviceMemoryResourceMapped(1);

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
        //auto obj_allocator = std::pmr::polymorphic_allocator{&dyn_allocator};
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
        graphics_pass.UpdateDescriptorSets(
            device, descriptor_sets.at(0), 0,
            {
                { 0, {.buffer=*vertex_data_buffer, .offset=0, .range=NumBytesCap(vertex_data)} },
                { 1, {.buffer=*model_data_buffer,  .offset=0, .range=NumBytesCap(model_transform_data)} },
                { 2, {.buffer=*camera_buffer,      .offset=0, .range=NumBytesCap(camera_data)} }
            },
            {},
            {}
        );


        /***
         * Update draw state
         */
        DrawState draw_state{
            .vulkan_state=vulkan_state,
            .graphics_pass=graphics_pass,
            .vertex_buffer=*vertex_buffer,
            .index_buffer=*index_buffer,
            .num_indices=static_cast<uint32_t>(indices.size()),
            .descriptor_sets=descriptor_sets,
            .swapchain_images=vulkan_state.swapchain.getImages()
        };
        std::ranges::transform(images, std::back_inserter(draw_state.targets), [](auto& i) { return i.AsVkImage(); });
        std::ranges::transform(image_views, std::back_inserter(draw_state.target_views), [](auto& i) { return *i; });

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


jms::vulkan::State CreateEnvironment(const AppState& app_state, std::optional<jms::wsi::glfw::Window*> window) {
    jms::vulkan::State vulkan_state{};

    vulkan_state.InitInstance(jms::vulkan::InstanceConfig{
        .app_name=app_state.app_name,
        .engine_name=app_state.engine_name,
        .layer_names=app_state.instance_layers,
        .extension_names=app_state.instance_extensions
    });

    vk::raii::PhysicalDevice& physical_device = vulkan_state.physical_devices.at(0);

    vulkan_state.InitDevice(physical_device, jms::vulkan::DeviceConfig{
        .layer_names=app_state.device_layers,
        .extension_names=app_state.device_extensions,
        .features=app_state.device_features,
        .queue_family_index=app_state.queue_family_index,
        .queue_priority=app_state.queue_priority,
        .pnext_features=app_state.device_pnext_features
    });

    vulkan_state.InitQueues(0);

    vk::raii::Device& device = vulkan_state.devices.at(0);

    vulkan_state.command_buffers.push_back(device.allocateCommandBuffers({
        .commandPool=*(vulkan_state.command_pools.at(0)),
        .level=vk::CommandBufferLevel::ePrimary,
        .commandBufferCount=2
    }));

    vulkan_state.memory_helper = {physical_device, device, app_state.memory_types};

    if (window.has_value()) {
        vulkan_state.surface = jms::wsi::glfw::CreateSurface(*window.value(), vulkan_state.instance);
        auto surface_render_info = jms::wsi::FromSurface(vulkan_state.surface,
                                                         physical_device,
                                                         static_cast<uint32_t>(app_state.window_width),
                                                         static_cast<uint32_t>(app_state.window_height));
        vulkan_state.InitSwapchain(device, vulkan_state.surface, surface_render_info);
    }

    return vulkan_state;
}


class RenderTargets {
    vk::raii::SwapchainKHR& swapchain;
    std::vector<vk::raii::ImageView>& targets;
    std::vector<vk::raii::ImageView>& swapchain_targets;
    std::vector<vk::raii::ImageView>::iterator target;
    std::vector<vk::raii::ImageView>::iterator sc_target;
    std::mutex mutex{};

public:
    RenderTargets(vk::raii::SwapchainKHR& swapchain_in,
                  std::vector<vk::raii::ImageView>& targets_in,
                  std::vector<vk::raii::ImageView>& swapchain_targets)
    : swapchain{swapchain_in},
      targets{targets_in},
      swapchain_targets{swapchain_targets},
      target{targets.end()},
      sc_target{targets.end()}
    {
        if (targets.size() < 2) { std::runtime_error{"RenderTargets require minimum of two targets."}; }
    }
    vk::raii::ImageView& NextTarget() {
        std::lock_guard lock{mutex};
        if (target != targets.end()) {
            target = std::ranges::next(target);
            if (target == targets.end()) { target = targets.begin(); }
        } else {
            target = targets.begin();
        }
        if (target == sc_target) {
            target = std::ranges::next(target);
            if (target == targets.end()) { target = targets.begin(); }
        }
        return *target;
    }
};

void DrawFrame(DrawState& draw_state) {
    auto& vulkan_state = draw_state.vulkan_state;
    auto& device = vulkan_state.devices.at(0);
    auto& image_available_semaphore = vulkan_state.semaphores.at(0);
    auto& render_finished_semaphore = vulkan_state.semaphores.at(1);
    auto& present_finished_semaphore = vulkan_state.semaphores.at(2);
    auto& in_flight_fence = vulkan_state.fences.at(0);
    auto& graphics_queue = vulkan_state.graphics_queue;
    auto& present_queue = vulkan_state.present_queue;
    auto& vs_command_buffers_0 = vulkan_state.command_buffers.at(0);
    auto& command_buffer_0 = vs_command_buffers_0.at(0);
    auto& command_buffer_1 = vs_command_buffers_0.at(0);
    auto& swapchain = vulkan_state.swapchain;

    vk::Result result = device.waitForFences({*in_flight_fence}, VK_TRUE, std::numeric_limits<uint64_t>::max());
    device.resetFences({*in_flight_fence});

    uint32_t swapchain_image_index = 0;
    std::tie(result, swapchain_image_index) = swapchain.acquireNextImage(std::numeric_limits<uint64_t>::max(),
                                                                         *image_available_semaphore);
    assert(result == vk::Result::eSuccess);
    assert(swapchain_image_index < draw_state.swapchain_images.size());

    vk::ImageView target_view = draw_state.target_views.at(0);
    vk::Image target_image = draw_state.targets.at(0);
    vk::Image swapchain_image = draw_state.swapchain_images.at(swapchain_image_index);

    command_buffer_0.reset();
    command_buffer_0.begin({.pInheritanceInfo=nullptr});
    draw_state.graphics_pass.ToCommands(
        command_buffer_0,
        {target_view},
        {},
        draw_state.descriptor_sets,
        {},
        [a=0, b=std::vector<vk::Buffer>{draw_state.vertex_buffer}, c=std::vector<vk::DeviceSize>{0}](auto& cb) {
            cb.bindVertexBuffers(a, b, c);
        },
        [&a=draw_state.index_buffer, b=0, c=vk::IndexType::eUint32](auto& cb) { cb.bindIndexBuffer(a, b, c); },
        [&a=draw_state.graphics_pass, &F=BindShaders](auto& cb) { F(cb, a); },
        [&a=draw_state.num_indices, b=1, c=0, d=0, e=0](auto& cb) { cb.drawIndexed(a, b, c, d, e); }
    );
    command_buffer_0.end();

    command_buffer_1.reset();
    command_buffer_1.begin({.pInheritanceInfo=nullptr});
    command_buffer_1.copyImage(target_image, vk::ImageLayout::eColorAttachmentOptimal,
                               swapchain_image, vk::ImageLayout::eColorAttachmentOptimal, {});
    command_buffer_1.end();

    std::vector<vk::Semaphore> wait_semaphores{*image_available_semaphore};
    std::vector<vk::Semaphore> render_semaphores{*render_finished_semaphore};
    std::vector<vk::Semaphore> present_semaphores{*present_finished_semaphore};
    std::vector<vk::PipelineStageFlags> dst_stage_mask{vk::PipelineStageFlagBits::eColorAttachmentOutput};
    std::vector<vk::CommandBuffer> command_buffers_0{*command_buffer_0};
    std::vector<vk::CommandBuffer> command_buffers_1{*command_buffer_1};

    graphics_queue.submit(std::array<vk::SubmitInfo, 1>{vk::SubmitInfo{
        .waitSemaphoreCount=static_cast<uint32_t>(wait_semaphores.size()),
        .pWaitSemaphores=jms::vulkan::VectorAsPtr(wait_semaphores),
        .pWaitDstStageMask=dst_stage_mask.data(),
        .commandBufferCount=static_cast<uint32_t>(command_buffers_0.size()),
        .pCommandBuffers=jms::vulkan::VectorAsPtr(command_buffers_0),
        .signalSemaphoreCount=static_cast<uint32_t>(render_semaphores.size()),
        .pSignalSemaphores=jms::vulkan::VectorAsPtr(render_semaphores)
    }}, VK_NULL_HANDLE);

    graphics_queue.submit(std::array<vk::SubmitInfo, 1>{vk::SubmitInfo{
        .waitSemaphoreCount=static_cast<uint32_t>(render_semaphores.size()),
        .pWaitSemaphores=jms::vulkan::VectorAsPtr(render_semaphores),
        .pWaitDstStageMask=jms::vulkan::VectorAsPtr(dst_stage_mask),
        .commandBufferCount=static_cast<uint32_t>(command_buffers_0.size()),
        .pCommandBuffers=jms::vulkan::VectorAsPtr(command_buffers_0),
        .signalSemaphoreCount=static_cast<uint32_t>(present_semaphores.size()),
        .pSignalSemaphores=jms::vulkan::VectorAsPtr(present_semaphores)
    }}, *in_flight_fence);

    std::vector<vk::SwapchainKHR> swapchains{*swapchain};
    present_queue.presentKHR({
        .waitSemaphoreCount=static_cast<uint32_t>(present_semaphores.size()),
        .pWaitSemaphores=jms::vulkan::VectorAsPtr(present_semaphores),
        .swapchainCount=static_cast<uint32_t>(swapchains.size()),
        .pSwapchains=jms::vulkan::VectorAsPtr(swapchains),
        .pImageIndices=&swapchain_image_index,
        .pResults=nullptr
    });
}
