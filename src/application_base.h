#pragma once

#include <cstdint>
#include <memory>

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <webgpu/webgpu.hpp>

#pragma warning(push, 3)
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#pragma warning(pop)

class ApplicationBase {
public:
    ApplicationBase(const char* title);
    ApplicationBase(const ApplicationBase&) = delete;
    ApplicationBase(ApplicationBase&&);
    virtual ~ApplicationBase();

    ApplicationBase& operator=(const ApplicationBase&) = delete;
    ApplicationBase& operator=(ApplicationBase&&) = delete;

    void run();

protected:
    virtual void on_frame(wgpu::CommandEncoder&, wgpu::TextureView&, wgpu::TextureView&);
    virtual void on_resize();

    wgpu::Device& device();
    const wgpu::Device& device() const;

    uint32_t surface_width() const;
    uint32_t surface_height() const;
    wgpu::TextureFormat surface_format() const;
    wgpu::TextureFormat depth_stencil_format() const;

private:
    void configure_surface();
    void configure_depth_stencil();
    void inspect_adapter(wgpu::Adapter&) const;
    void inspect_surface(wgpu::Adapter&, wgpu::Surface&) const;

    GLFWwindow* m_window;
    ImGuiContext* m_imgui_context;
    wgpu::Instance m_instance;
    wgpu::Surface m_surface;
    std::unique_ptr<wgpu::ErrorCallback> m_error_callback;
    wgpu::Device m_device;
    wgpu::TextureFormat m_surface_format;
    wgpu::TextureFormat m_depth_stencil_format;
    wgpu::Texture m_depth_stencil_texture;
    uint32_t m_window_width;
    uint32_t m_window_height;
};