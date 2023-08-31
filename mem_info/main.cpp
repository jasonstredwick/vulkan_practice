#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <expected>
#include <iostream>
#include <memory>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "jms/vulkan/vulkan.hpp"


struct LayerInfo {
    vk::LayerProperties layer_props;
    std::vector<vk::ExtensionProperties> ext_props;
};


void Print(const vk::LayerProperties& props) {
    std::cout << std::format("{}: Version: {} API Version: {}.{}.{} Description: {}\n",
        static_cast<const char*>(props.layerName),
        props.implementationVersion,
        props.specVersion >> 22,
        (props.specVersion >> 12) & 0x03FF,
        props.specVersion & 0xFFF,
        static_cast<const char*>(props.description));
}


void Print(const vk::ExtensionProperties& props) {
    std::cout << std::format("{}  Version: {}\n",
                             static_cast<const char*>(props.extensionName),
                             props.specVersion);
}


void Run() {
    try {
        vk::raii::Context context{};

        uint32_t api_version = context.enumerateInstanceVersion();
        std::cout << std::format("Version- {}.{}.{}\n", std::to_string(VK_VERSION_MAJOR(api_version)),
                                                        std::to_string(VK_VERSION_MINOR(api_version)),
                                                        std::to_string(VK_VERSION_PATCH(api_version)));
        std::cout << std::endl;

        std::vector<LayerInfo> layers_info{};
        std::ranges::transform(context.enumerateInstanceLayerProperties(), std::back_inserter(layers_info),
            [&context](const vk::LayerProperties& lp) {
                std::string layer_name = std::string{static_cast<const char*>(lp.layerName)};
                return LayerInfo{.layer_props=lp, .ext_props=context.enumerateInstanceExtensionProperties(layer_name)};
        });
        std::ranges::sort(layers_info, [](const LayerInfo& lhs, const LayerInfo& rhs) {
            return std::string{static_cast<const char*>(lhs.layer_props.layerName)} <
                   std::string{static_cast<const char*>(rhs.layer_props.layerName)};
        });

        std::cout << "Layers-\n";
        for (LayerInfo& info: layers_info) {
            Print(info.layer_props);
            for (auto& ext_props : info.ext_props) {
                std::cout << std::format("    Ext: {}  Version: {}\n",
                                         static_cast<const char*>(ext_props.extensionName),
                                         ext_props.specVersion);
            }
        }
        std::cout << std::endl;

        std::cout << "Extensions-\n";
        std::vector<vk::ExtensionProperties> extensions_props = context.enumerateInstanceExtensionProperties();
        std::ranges::sort(extensions_props, [](const auto& lhs, const auto& rhs) {
            return std::string{static_cast<const char*>(lhs.extensionName)} <
                   std::string{static_cast<const char*>(rhs.extensionName)};
        });
        for (auto& ext_props : extensions_props) { Print(ext_props); }
        std::cout << std::endl;

        vk::ApplicationInfo app_info{.engineVersion=1, .apiVersion=api_version};
        std::vector<const char*> use_layers{
            "VK_LAYER_KHRONOS_synchronization2",
            "VK_LAYER_KHRONOS_shader_object"
        };
        std::vector<std::string> use_exts{};
        for (auto& ext_props : extensions_props) {
            std::string name{static_cast<const char*>(ext_props.extensionName)};
            if (name.starts_with(std::string{"VK_KHR_"})) { use_exts.push_back(name); }
        }
        std::vector<const char*> vk_use_exts{}; // "VK_KHR_surface", "VK_KHR_win32_surface"
        std::ranges::transform(use_exts, std::back_inserter(vk_use_exts), [](auto& s) { return s.c_str(); });
        vk::InstanceCreateInfo instance_create_info{
            .pApplicationInfo=&app_info,
            .enabledLayerCount=static_cast<uint32_t>(use_layers.size()),
            .ppEnabledLayerNames=use_layers.data(),
            .enabledExtensionCount=static_cast<uint32_t>(vk_use_exts.size()),
            .ppEnabledExtensionNames=vk_use_exts.data()
        };
        vk::raii::Instance instance = context.createInstance(instance_create_info);

        std::vector<vk::raii::PhysicalDevice> physical_devices = instance.enumeratePhysicalDevices();
        for (auto index : std::views::iota(static_cast<size_t>(0), physical_devices.size())) {
            vk::raii::PhysicalDevice& physical_device = physical_devices.at(index);
            auto dev_props = physical_device.getProperties2();
            std::cout << std::format("PhysicalDevice {} [{}]-\n", index, static_cast<const char*>(dev_props.properties.deviceName));
#if 0
            std::vector<vk::LayerProperties> layer_props = physical_device.enumerateDeviceLayerProperties();
            std::vector<vk::ExtensionProperties> extensions_props = physical_device.enumerateDeviceExtensionProperties();
            std::cout << "Layers-\n";
            for (auto& props : layer_props) { Print(props); }
            std::cout << "\n";
            std::cout << "Extensions-\n";
            for (auto& props : extensions_props) { Print(props); }
            std::cout << "\n";
#endif
            auto mem_props2 = physical_device.getMemoryProperties2<vk::PhysicalDeviceMemoryProperties2,
                                                                   vk::PhysicalDeviceMemoryBudgetPropertiesEXT>();
            const auto& mem_props = mem_props2.get<vk::PhysicalDeviceMemoryProperties2>().memoryProperties;
            std::span<const vk::MemoryHeap> vk_heaps{mem_props.memoryHeaps.begin(), mem_props.memoryHeapCount};
            std::span<const vk::MemoryType> vk_types{mem_props.memoryTypes.begin(), mem_props.memoryTypeCount};
            std::cout << "Types-\n";
            for (auto& i : vk_types) {
                std::cout << std::format("Heap: {}, Type: {}\n", i.heapIndex, vk::to_string(i.propertyFlags));
            }
            std::cout << "\n";

            //vkGetDeviceBufferMemoryRequirements
            //VkDeviceImageMemoryRequirements
#if 0
            std::vector<Memory::Heap> heaps{};
            heaps.reserve(mem_props.memoryHeapCount);
            std::ranges::transform(vk_heaps, std::back_inserter(heaps), [](auto& heap_info) {
                return Memory::Heap{
                    .size=heap_info.size,
                    .flags=heap_info.flags
                };
            });
            return {.heaps=std::move(heaps)};
#endif
        }
    } catch (std::exception const& exp) {
        std::cout << "Exception caught\n" << exp.what() << std::endl;
    }
}


/***
 * Patterns-
 * 1. Heap - no types -> Ignore
 * 2. DEVICE_LOCAL + DEVICE_HOST_VISIBLE < 1GB  ->  Staging/
*/

int main(int argc, char** argv) {
    std::cout << std::format("Start\n");
    Run();
    std::cout << std::format("End\n");
    return 0;
}
