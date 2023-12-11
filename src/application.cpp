#include <application.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <numbers>
#include <utility>

Application::Application()
    : ApplicationBase { "Exercise 8" }
    , m_slice_shader_module { nullptr }
    , m_slice_bind_group_layout { nullptr }
    , m_slice_pipeline_layout { nullptr }
    , m_slice_render_pipeline { nullptr }
    , m_iso_contours_shader_module { nullptr }
    , m_iso_contours_bind_group_layout { nullptr }
    , m_iso_contours_pipeline_layout { nullptr }
    , m_iso_contours_render_pipeline { nullptr }
    , m_slice_texture { nullptr }
    , m_slice_samples {}
    , m_slice_texture_changed { false }
    , m_iso_contours_buffer { nullptr }
    , m_iso_contour_lines {}
    , m_volume { std::nullopt }
    , m_dataset { Dataset::Baby }
    , m_plane { SlicePlane::Axial }
    , m_plane_offset { 0.0f }
    , m_plane_rotation { 0.0f }
    , m_iso_value { 0.0f }
{
    this->init_slice_render_pipeline();
    this->init_iso_contours_render_pipeline();
    this->init_slice_texture();
    this->init_iso_contours_buffer();
}

Application::Application(Application&& app)
    : ApplicationBase { std::move(app) }
    , m_slice_shader_module { std::exchange(app.m_slice_shader_module, nullptr) }
    , m_slice_bind_group_layout { std::exchange(app.m_slice_bind_group_layout, nullptr) }
    , m_slice_pipeline_layout { std::exchange(app.m_slice_pipeline_layout, nullptr) }
    , m_slice_render_pipeline { std::exchange(app.m_slice_render_pipeline, nullptr) }
    , m_iso_contours_shader_module { std::exchange(app.m_iso_contours_shader_module, nullptr) }
    , m_iso_contours_bind_group_layout { std::exchange(app.m_iso_contours_bind_group_layout, nullptr) }
    , m_iso_contours_pipeline_layout { std::exchange(app.m_iso_contours_pipeline_layout, nullptr) }
    , m_iso_contours_render_pipeline { std::exchange(app.m_iso_contours_render_pipeline, nullptr) }
    , m_slice_texture { std::exchange(app.m_slice_texture, nullptr) }
    , m_slice_samples { std::exchange(app.m_slice_samples, {}) }
    , m_slice_texture_changed { std::exchange(app.m_slice_texture_changed, false) }
    , m_iso_contours_buffer { std::exchange(app.m_iso_contours_buffer, nullptr) }
    , m_iso_contour_lines { std::exchange(app.m_iso_contour_lines, {}) }
    , m_volume { std::exchange(app.m_volume, std::nullopt) }
    , m_dataset { std::exchange(app.m_dataset, Dataset::Baby) }
    , m_plane { std::exchange(app.m_plane, SlicePlane::Axial) }
    , m_plane_offset { std::exchange(app.m_plane_offset, 0.0f) }
    , m_plane_rotation { std::exchange(app.m_plane_rotation, 0.0f) }
{
}

Application::~Application()
{
    if (this->m_iso_contours_buffer) {
        this->m_iso_contours_buffer.release();
    }
    if (this->m_slice_texture) {
        this->m_slice_texture.release();
    }

    if (this->m_iso_contours_render_pipeline) {
        this->m_iso_contours_render_pipeline.release();
    }
    if (this->m_iso_contours_pipeline_layout) {
        this->m_iso_contours_pipeline_layout.release();
    }
    if (this->m_iso_contours_bind_group_layout) {
        this->m_iso_contours_bind_group_layout.release();
    }
    if (this->m_iso_contours_shader_module) {
        this->m_iso_contours_shader_module.release();
    }

    if (this->m_slice_render_pipeline) {
        this->m_slice_render_pipeline.release();
    }
    if (this->m_slice_pipeline_layout) {
        this->m_slice_pipeline_layout.release();
    }
    if (this->m_slice_bind_group_layout) {
        this->m_slice_bind_group_layout.release();
    }
    if (this->m_slice_shader_module) {
        this->m_slice_shader_module.release();
    }
}

void Application::on_frame(wgpu::CommandEncoder& encoder, wgpu::TextureView& frame)
{
    ImGui::Begin("Config");
    ImGui::Text("Dataset");
    ImGui::SameLine();
    int dataset = static_cast<int>(this->m_dataset);
    bool dataset_changed = ImGui::RadioButton("Baby", &dataset, 0);
    ImGui::SameLine();
    dataset_changed |= ImGui::RadioButton("CT-Head", &dataset, 1);
    ImGui::SameLine();
    dataset_changed |= ImGui::RadioButton("Fuel", &dataset, 2);
    dataset_changed |= !this->m_volume.has_value();
    if (dataset_changed) {
        this->m_dataset = static_cast<Dataset>(dataset);
        switch (this->m_dataset) {
        case Dataset::Baby:
            this->m_volume = PVMVolume { "resources/Baby.pvm" };
            break;
        case Dataset::CT_Head:
            this->m_volume = PVMVolume { "resources/CT-Head.pvm" };
            break;
        case Dataset::Fuel:
            this->m_volume = PVMVolume { "resources/Fuel.pvm" };
            break;
        default:
            std::cerr << "Invalid dataset!" << std::endl;
        }
        auto extents { this->m_volume->extents() };
        glm::vec3 extents_float { extents.x, extents.y, extents.z };

        this->m_plane = SlicePlane::Axial;
        this->m_plane_offset = 0.0f;
        this->m_plane_rotation = 0.0f;
    }
    bool plane_changed = ImGui::SliderFloat("Plane offset (%)", &this->m_plane_offset, 0.0f, 100.0f);
    plane_changed |= ImGui::SliderFloat("Plane rotation", &this->m_plane_rotation, 0.0f, 360.0f);

    if (ImGui::Button("Axial")) {
        plane_changed = true;
        this->m_plane = SlicePlane::Axial;
        this->m_plane_offset = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Sagittal")) {
        plane_changed = true;
        this->m_plane = SlicePlane::Sagittal;
        this->m_plane_offset = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("Coronal")) {
        plane_changed = true;
        this->m_plane = SlicePlane::Coronal;
        this->m_plane_offset = 0.0f;
    }

    ImGui::End();

    bool update_slice_texture = this->m_slice_texture_changed || dataset_changed || plane_changed;
    this->m_slice_texture_changed = false;
    if (update_slice_texture) {
        this->update_slice_samples_and_texture();
    }

    wgpu::TextureView slice_texture_view = this->m_slice_texture.createView();

    std::array<wgpu::BindGroupEntry, 1> bind_group_entries { wgpu::Default };
    bind_group_entries[0].binding = 0;
    bind_group_entries[0].textureView = slice_texture_view;

    wgpu::BindGroupDescriptor bind_group_desc { wgpu::Default };
    bind_group_desc.layout = this->m_slice_bind_group_layout;
    bind_group_desc.entryCount = bind_group_entries.size();
    bind_group_desc.entries = bind_group_entries.data();
    wgpu::BindGroup bind_group = this->device().createBindGroup(bind_group_desc);

    auto color_attachments = std::array { wgpu::RenderPassColorAttachment { wgpu::Default } };
    color_attachments[0].view = frame;
    color_attachments[0].loadOp = wgpu::LoadOp::Clear;
    color_attachments[0].storeOp = wgpu::StoreOp::Store;
    color_attachments[0].clearValue = wgpu::Color { 0.45f, 0.55f, 0.60f, 1.0f };

    wgpu::RenderPassDescriptor pass_desc { wgpu::Default };
    pass_desc.colorAttachmentCount = color_attachments.size();
    pass_desc.colorAttachments = color_attachments.data();
    auto pass_encoder = encoder.beginRenderPass(pass_desc);
    if (!pass_encoder) {
        std::cerr << "Could not create render pass!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    pass_encoder.setPipeline(this->m_slice_render_pipeline);
    pass_encoder.setBindGroup(0, bind_group, 0, nullptr);
    pass_encoder.draw(6, 1, 0, 0);

    wgpu::BindGroup contours_bind_group { nullptr };
    if (!this->m_iso_contour_lines.empty()) {
        std::array<wgpu::BindGroupEntry, 1> contours_bind_group_entries { wgpu::Default };
        contours_bind_group_entries[0].binding = 0;
        contours_bind_group_entries[0].buffer = this->m_iso_contours_buffer;
        contours_bind_group_entries[0].size = WGPU_WHOLE_SIZE;

        wgpu::BindGroupDescriptor contours_bind_group_desc { wgpu::Default };
        contours_bind_group_desc.layout = this->m_iso_contours_bind_group_layout;
        contours_bind_group_desc.entryCount = contours_bind_group_entries.size();
        contours_bind_group_desc.entries = contours_bind_group_entries.data();
        contours_bind_group = this->device().createBindGroup(contours_bind_group_desc);

        pass_encoder.setPipeline(this->m_iso_contours_render_pipeline);
        pass_encoder.setBindGroup(0, contours_bind_group, 0, nullptr);
        pass_encoder.draw(6, static_cast<uint32_t>(this->m_iso_contour_lines.size()), 0, 0);
    }

    pass_encoder.end();
    pass_encoder.release();
    bind_group.release();

    if (contours_bind_group) {
        contours_bind_group.release();
    }
}

void Application::on_resize()
{
    ApplicationBase::on_resize();
    this->init_slice_texture();
}

void Application::init_slice_render_pipeline()
{
    wgpu::ShaderModuleWGSLDescriptor wgsl_module_desc { wgpu::Default };
    wgsl_module_desc.code = R"(
        @group(0)
        @binding(0)
        var slice_texture: texture_2d<f32>;

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>
        }

        @vertex
        fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
            var VERTEX_BUFFER = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(-1.0, 1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(-1.0, 1.0),
            );
            var UV_BUFFER = array<vec2<f32>, 6>(
                vec2<f32>(0.0, 0.0),
                vec2<f32>(1.0, 0.0),
                vec2<f32>(0.0, 1.0),
                vec2<f32>(1.0, 0.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(0.0, 1.0),
            );

            let pos = vec4(VERTEX_BUFFER[in_vertex_index], 0.0, 1.0);
            let uv = UV_BUFFER[in_vertex_index];
            return VertexOutput(pos, uv);
        }

        @fragment
        fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let dimensions = vec2<f32>(textureDimensions(slice_texture));
            let texel = vec2<u32>(dimensions * uv);
            return textureLoad(slice_texture, texel, 0);
        }
    )";
    wgpu::ShaderModuleDescriptor module_desc { wgpu::Default };
    module_desc.nextInChain = reinterpret_cast<wgpu::ChainedStruct*>(&wgsl_module_desc);
    this->m_slice_shader_module = this->device().createShaderModule(module_desc);
    if (!this->m_slice_shader_module) {
        std::cerr << "Failed to create the shader module" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::array<wgpu::BindGroupLayoutEntry, 1> binding_layout_entries { wgpu::Default };
    binding_layout_entries[0].binding = 0;
    binding_layout_entries[0].visibility = wgpu::ShaderStage::Fragment;
    binding_layout_entries[0].texture.sampleType = wgpu::TextureSampleType::Float;
    binding_layout_entries[0].texture.viewDimension = wgpu::TextureViewDimension::_2D;

    wgpu::BindGroupLayoutDescriptor group_layout_desc { wgpu::Default };
    group_layout_desc.entryCount = binding_layout_entries.size();
    group_layout_desc.entries = binding_layout_entries.data();
    this->m_slice_bind_group_layout = this->device().createBindGroupLayout(group_layout_desc);
    if (!this->m_slice_bind_group_layout) {
        std::cerr << "Failed to create the bind group layout" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    wgpu::PipelineLayoutDescriptor layout_desc { wgpu::Default };
    layout_desc.bindGroupLayoutCount = 1;
    layout_desc.bindGroupLayouts = (WGPUBindGroupLayout*)&this->m_slice_bind_group_layout;
    this->m_slice_pipeline_layout = this->device().createPipelineLayout(layout_desc);
    if (!this->m_slice_pipeline_layout) {
        std::cerr << "Failed to create the pipeline layout" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    wgpu::RenderPipelineDescriptor pipeline_desc { wgpu::Default };
    pipeline_desc.layout = this->m_slice_pipeline_layout;
    pipeline_desc.vertex.module = this->m_slice_shader_module;
    pipeline_desc.vertex.entryPoint = "vs_main";

    auto fragment_targets = std::array { wgpu::ColorTargetState { wgpu::Default } };
    fragment_targets[0].format = this->surface_format();
    fragment_targets[0].writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fragment_state { wgpu::Default };
    fragment_state.module = this->m_slice_shader_module;
    fragment_state.entryPoint = "fs_main";
    fragment_state.targetCount = fragment_targets.size();
    fragment_state.targets = fragment_targets.data();
    fragment_state.constantCount = 0;
    fragment_state.constants = nullptr;
    pipeline_desc.fragment = &fragment_state;
    pipeline_desc.depthStencil = nullptr;
    pipeline_desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipeline_desc.multisample.count = 1;
    pipeline_desc.multisample.mask = 0xFFFFFFFF;
    this->m_slice_render_pipeline = this->device().createRenderPipeline(pipeline_desc);
    if (!this->m_slice_render_pipeline) {
        std::cerr << "Failed to create the render pipeline" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Application::init_iso_contours_render_pipeline()
{
    wgpu::ShaderModuleWGSLDescriptor wgsl_module_desc { wgpu::Default };
    wgsl_module_desc.code = R"(
        struct IsoContourLine {
            start: vec2<f32>,
            end: vec2<f32>
        }

        @group(0)
        @binding(0)
        var<storage, read> contour_lines: array<IsoContourLine>;

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) normal: vec2<f32>,
        }

        fn get_line_alpha(normal: vec2<f32>) -> f32 {
            let feather: f32 = 0.5;
            let one_minus_feather: f32 = 1.0 - feather;

            let distance = length(normal);
            if distance <= one_minus_feather {
                return 1.0;
            } else if distance <= 1.0 {
                let t = (distance - feather) / one_minus_feather;
                return mix(1.0, 0.0, t);
            }

            return 0.0;
        }

        @vertex
        fn vs_main(
            @builtin(vertex_index) in_vertex_index: u32,
            @builtin(instance_index) in_instance_idx: u32
        ) -> VertexOutput {
            let line = contour_lines[in_instance_idx];

            let start_x = mix(-1.0, 1.0, line.start.x);
            let start_y = mix(-1.0, 1.0, line.start.y);
            let end_x = mix(-1.0, 1.0, line.end.x);
            let end_y = mix(-1.0, 1.0, line.end.y);

            let line_start = vec2(start_x, start_y);
            let line_end = vec2(end_x, end_y);

            let line_vector = normalize(line_end - line_start);
            let line_unit_cos = line_vector.x;
            let line_unit_sin = line_vector.y;

            var INDEX_BUFFER = array<u32, 6>(0u, 1u, 2u, 1u, 3u, 2u);
            var VERTEX_NORMALS_BUFFER = array<vec2<f32>, 4>(
                vec2<f32>(0.0, -1.0),
                vec2<f32>(0.0, 1.0),
                vec2<f32>(0.0, -1.0),
                vec2<f32>(0.0, 1.0),
            );

            let rotation_matrix = mat2x2<f32>(
                line_unit_cos,
                line_unit_sin,    // column 1: [cos theta, sin theta]
                -line_unit_sin,
                line_unit_cos,   // column 2: [-sin theta, cos theta]
            );
            let index = INDEX_BUFFER[in_vertex_index];
            let vertex_normal = rotation_matrix * VERTEX_NORMALS_BUFFER[index];
            let vertex_pos = select(line_start, line_end, vec2<bool>(index <= 1u));

            let delta = vec4<f32>(vertex_normal * 0.002, 0.0, 0.0);
            let pos = vec4<f32>(vertex_pos, 0.0, 1.0);
            let offset_position = (pos + delta);

            return VertexOutput(offset_position, vertex_normal);
        }

        @fragment
        fn fs_main(@location(0) normal: vec2<f32>) -> @location(0) vec4<f32> {
            let alpha = get_line_alpha(normal);
            let green = vec4(0.0, 1.0, 0.0, 1.0);
            return green * alpha;
        }
    )";
    wgpu::ShaderModuleDescriptor module_desc { wgpu::Default };
    module_desc.nextInChain = reinterpret_cast<wgpu::ChainedStruct*>(&wgsl_module_desc);
    this->m_iso_contours_shader_module = this->device().createShaderModule(module_desc);
    if (!this->m_iso_contours_shader_module) {
        std::cerr << "Failed to create the shader module" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::array<wgpu::BindGroupLayoutEntry, 1> binding_layout_entries { wgpu::Default };
    binding_layout_entries[0].binding = 0;
    binding_layout_entries[0].visibility = wgpu::ShaderStage::Vertex;
    binding_layout_entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutDescriptor group_layout_desc { wgpu::Default };
    group_layout_desc.entryCount = binding_layout_entries.size();
    group_layout_desc.entries = binding_layout_entries.data();
    this->m_iso_contours_bind_group_layout = this->device().createBindGroupLayout(group_layout_desc);
    if (!this->m_iso_contours_bind_group_layout) {
        std::cerr << "Failed to create the bind group layout" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    wgpu::PipelineLayoutDescriptor layout_desc { wgpu::Default };
    layout_desc.bindGroupLayoutCount = 1;
    layout_desc.bindGroupLayouts = (WGPUBindGroupLayout*)&this->m_iso_contours_bind_group_layout;
    this->m_iso_contours_pipeline_layout = this->device().createPipelineLayout(layout_desc);
    if (!this->m_iso_contours_pipeline_layout) {
        std::cerr << "Failed to create the pipeline layout" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    wgpu::RenderPipelineDescriptor pipeline_desc { wgpu::Default };
    pipeline_desc.layout = this->m_iso_contours_pipeline_layout;
    pipeline_desc.vertex.module = this->m_iso_contours_shader_module;
    pipeline_desc.vertex.entryPoint = "vs_main";

    wgpu::BlendState fragment_blend_state = { wgpu::Default };
    fragment_blend_state.alpha.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    fragment_blend_state.alpha.operation = wgpu::BlendOperation::Add;
    fragment_blend_state.alpha.srcFactor = wgpu::BlendFactor::One;
    fragment_blend_state.color.dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha;
    fragment_blend_state.color.operation = wgpu::BlendOperation::Add;
    fragment_blend_state.color.srcFactor = wgpu::BlendFactor::One;

    auto fragment_targets = std::array { wgpu::ColorTargetState { wgpu::Default } };
    fragment_targets[0].format = this->surface_format();
    fragment_targets[0].blend = &fragment_blend_state;
    fragment_targets[0].writeMask = wgpu::ColorWriteMask::All;

    wgpu::FragmentState fragment_state { wgpu::Default };
    fragment_state.module = this->m_iso_contours_shader_module;
    fragment_state.entryPoint = "fs_main";
    fragment_state.targetCount = fragment_targets.size();
    fragment_state.targets = fragment_targets.data();
    fragment_state.constantCount = 0;
    fragment_state.constants = nullptr;
    pipeline_desc.fragment = &fragment_state;
    pipeline_desc.depthStencil = nullptr;
    pipeline_desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipeline_desc.multisample.count = 1;
    pipeline_desc.multisample.mask = 0xFFFFFFFF;
    this->m_iso_contours_render_pipeline = this->device().createRenderPipeline(pipeline_desc);
    if (!this->m_iso_contours_render_pipeline) {
        std::cerr << "Failed to create the render pipeline" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Application::init_slice_texture()
{
    if (this->m_slice_texture) {
        this->m_slice_texture.release();
        this->m_slice_texture = { nullptr };
    }

    wgpu::Device& device = this->device();
    uint32_t width { this->surface_width() };
    uint32_t height { this->surface_height() };

    wgpu::TextureDescriptor desc { wgpu::Default };
    desc.label = "slice texture";
    desc.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::TextureBinding;
    desc.dimension = wgpu::TextureDimension::_2D;
    desc.size = wgpu::Extent3D { width, height, 1 };
    desc.format = wgpu::TextureFormat::RGBA8Unorm;
    desc.mipLevelCount = 1;
    desc.sampleCount = 1;
    desc.viewFormatCount = 0;
    desc.viewFormats = nullptr;
    this->m_slice_texture = device.createTexture(desc);
    this->m_slice_texture_changed = true;
    if (!this->m_slice_texture) {
        std::cerr << "Could not create slice texture!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Application::init_iso_contours_buffer()
{
    if (this->m_iso_contours_buffer) {
        this->m_iso_contours_buffer.release();
        this->m_iso_contours_buffer = { nullptr };
    }

    wgpu::Device& device = this->device();

    size_t buffer_size { this->m_iso_contour_lines.size() * sizeof(IsoContourLine) };
    if (buffer_size == 0) {
        buffer_size = sizeof(IsoContourLine);
    }

    wgpu::BufferDescriptor desc { wgpu::Default };
    desc.label = "isocontours buffer";
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage;
    desc.mappedAtCreation = false;
    desc.size = static_cast<uint64_t>(buffer_size);
    this->m_iso_contours_buffer = device.createBuffer(desc);
    if (!this->m_iso_contours_buffer) {
        std::cerr << "Could not create contours buffer!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Application::update_slice_samples_and_texture()
{
    // Compute the plane.
    glm::vec4 normal { 0.0f };
    glm::vec4 top_left { 0.0f };
    glm::vec4 bottom_left { 0.0f };
    glm::vec4 bottom_right { 0.0f };

    auto extents { this->m_volume->extents() };
    glm::vec3 extents_f { static_cast<float>(extents.x), static_cast<float>(extents.y), static_cast<float>(extents.z) };
    glm::vec3 plane_max { extents_f - glm::vec3 { 1.0f } };

    switch (this->m_plane) {
    case SlicePlane::Axial: {
        normal = glm::vec4 { 0.0f, 0.0f, 1.0f, 0.0f };
        glm::vec4 offset { normal * plane_max.z * (this->m_plane_offset / 100.0f) };
        top_left = glm::vec4 { 0.0f, plane_max.y, 0.0f, 1.0f } + offset;
        bottom_left = glm::vec4 { 0.0f, 0.0f, 0.0f, 1.0f } + offset;
        bottom_right = glm::vec4 { plane_max.x, 0.0f, 0.0f, 1.0f } + offset;
        break;
    }
    case SlicePlane::Sagittal: {
        normal = glm::vec4 { 1.0f, 0.0f, 0.0f, 0.0f };
        glm::vec4 offset { normal * plane_max.x * (this->m_plane_offset / 100.0f) };
        top_left = glm::vec4 { 0.0f, 0.0f, plane_max.z, 1.0f } + offset;
        bottom_left = glm::vec4 { 0.0f, 0.0f, 0.0f, 1.0f } + offset;
        bottom_right = glm::vec4 { 0.0f, plane_max.y, 0.0f, 1.0f } + offset;
        break;
    }
    case SlicePlane::Coronal: {
        normal = glm::vec4 { 0.0f, 1.0f, 0.0f, 0.0f };
        glm::vec4 offset { normal * plane_max.y * (this->m_plane_offset / 100.0f) };
        top_left = glm::vec4 { 0.0f, 0.0f, plane_max.z, 1.0f } + offset;
        bottom_left = glm::vec4 { 0.0f, 0.0f, 0.0f, 1.0f } + offset;
        bottom_right = glm::vec4 { plane_max.x, 0.0f, 0.0f, 1.0f } + offset;
        break;
    }
    }

    auto plane_rotation = glm::radians(this->m_plane_rotation);
    glm::quat rotation { glm::quat(normal * plane_rotation) };
    glm::vec4 center { (top_left + bottom_right) / 2.0f };
    top_left = center + rotation * (top_left - center);
    bottom_left = center + rotation * (bottom_left - center);
    bottom_right = center + rotation * (bottom_right - center);

    Plane plane {
        .top_left = top_left,
        .bottom_left = bottom_left,
        .bottom_right = bottom_right,
    };

    // Allocate memory for the sample buffer.
    uint32_t width { this->surface_width() };
    uint32_t height { this->surface_height() };
    size_t size { static_cast<size_t>(width) * static_cast<size_t>(height) };
    this->m_slice_samples.resize(size, -1.0);

    // Sample the slice.
    this->sample_slice(*this->m_volume, this->m_slice_samples, plane, width, height);

    // Allocate memory for the color buffer.
    std::vector<Color> color_buffer { size, Color {} };

    // Fill the color buffer.
    this->color_slice(this->m_slice_samples, color_buffer);

    // Copy the color buffer to the gpu texture.
    wgpu::Device& device { this->device() };
    wgpu::Queue queue { device.getQueue() };

    wgpu::ImageCopyTexture destination { wgpu::Default };
    destination.texture = this->m_slice_texture;
    destination.mipLevel = 0;
    destination.origin = wgpu::Origin3D { 0, 0, 0 };
    destination.aspect = wgpu::TextureAspect::All;
    wgpu::TextureDataLayout data_layout { wgpu::Default };
    data_layout.offset = 0;
    data_layout.bytesPerRow = width * 4;
    data_layout.rowsPerImage = height;
    wgpu::Extent3D write_size { width, height, 1 };
    queue.writeTexture(destination, &color_buffer.data()->r, 4 * size, data_layout, write_size);
}

void Application::update_iso_contours()
{
    // Compute the contour lines.
    uint32_t width { this->surface_width() };
    uint32_t height { this->surface_height() };
    glm::vec2 iso_range { this->m_volume->component_range(0) };
    this->m_iso_contour_lines = this->compute_iso_contours(this->m_slice_samples, width, height, this->m_iso_value, iso_range);
    this->init_iso_contours_buffer();

    // Upload the lines to the buffer.
    wgpu::Device& device { this->device() };
    wgpu::Queue queue { device.getQueue() };
    queue.writeBuffer(this->m_iso_contours_buffer, 0, this->m_iso_contour_lines.data(), this->m_iso_contour_lines.size() * sizeof(IsoContourLine));
}

float Application::interpolate_trilinear(VoxelCell cell, float t_x, float t_y, float t_z) const
{
    float bottom_front = (cell.bottom_front_left * (1.0f - t_x)) + (cell.bottom_front_right * t_x);
    float bottom_back = (cell.bottom_back_left * (1.0f - t_x)) + (cell.bottom_back_right * t_x);
    float top_front = (cell.top_front_left * (1.0f - t_x)) + (cell.top_front_right * t_x);
    float top_back = (cell.top_back_left * (1.0f - t_x)) + (cell.top_back_right * t_x);

    float bottom = (bottom_front * (1.0f - t_y)) + (bottom_back * t_y);
    float top = (top_front * (1.0f - t_y)) + (top_back * t_y);

    return (bottom * (1.0f - t_z)) + (top * t_z);
}

float Application::sample_at_position(const PVMVolume& volume, glm::vec3 position) const
{
    auto extents = volume.extents();
    glm::vec3 extents_float { glm::vec3 { extents.x, extents.y, extents.z } - glm::vec3 { 1.0f } };

    glm::vec3 min_position = glm::floor(position);
    if (glm::any(glm::greaterThan(glm::vec3 { 0 }, min_position))) {
        return -1.0;
    }

    glm::vec3 max_position = glm::ceil(position);
    if (glm::any(glm::greaterThan(max_position, extents_float))) {
        return -1.0;
    }

    glm::vec3 ts = glm::fract(position);
    float t_x = ts.x;
    float t_y = ts.y;
    float t_z = ts.z;

    std::size_t x = static_cast<std::size_t>(min_position.x);
    std::size_t y = static_cast<std::size_t>(min_position.y);
    std::size_t z = static_cast<std::size_t>(min_position.z);

    std::size_t x_max = static_cast<std::size_t>(max_position.x);
    std::size_t y_max = static_cast<std::size_t>(max_position.y);
    std::size_t z_max = static_cast<std::size_t>(max_position.z);

    float bottom_front_left = volume.voxel_normalized(x, y, z);
    float bottom_front_right = volume.voxel_normalized(x_max, y, z);
    float bottom_back_left = volume.voxel_normalized(x, y_max, z);
    float bottom_back_right = volume.voxel_normalized(x_max, y_max, z);
    float top_front_left = volume.voxel_normalized(x, y, z_max);
    float top_front_right = volume.voxel_normalized(x_max, y, z_max);
    float top_back_left = volume.voxel_normalized(x, y_max, z_max);
    float top_back_right = volume.voxel_normalized(x_max, y_max, z_max);
    VoxelCell cell {
        .bottom_front_left = bottom_front_left,
        .bottom_front_right = bottom_front_right,
        .bottom_back_left = bottom_back_left,
        .bottom_back_right = bottom_back_right,
        .top_front_left = top_front_left,
        .top_front_right = top_front_right,
        .top_back_left = top_back_left,
        .top_back_right = top_back_right,
    };
    return this->interpolate_trilinear(cell, t_x, t_y, t_z);
}

void Application::sample_slice(const PVMVolume& volume, std::span<float> plane_buffer, Plane plane,
    uint32_t buffer_width, uint32_t buffer_height) const
{
    glm::vec3 top_vector { (plane.top_left - plane.bottom_left) / static_cast<float>(buffer_height) };
    glm::vec3 right_vector { (plane.bottom_right - plane.bottom_left) / static_cast<float>(buffer_width) };

    for (uint32_t y { 0 }; y < buffer_height; ++y) {
        for (uint32_t x { 0 }; x < buffer_width; ++x) {
            glm::vec3 top_offset = static_cast<float>(y) * top_vector;
            glm::vec3 right_offset = static_cast<float>(x) * right_vector;
            glm::vec3 pos { plane.bottom_left + right_offset + top_offset };

            float value { this->sample_at_position(volume, pos) };

            std::size_t index { x + (y * buffer_width) };
            plane_buffer[index] = value;
        }
    }
}

Color Application::sample_transfer_function(float t) const
{
    uint8_t value = static_cast<uint8_t>(t * 255.0);
    return Color {
        .r = value,
        .g = value,
        .b = value,
        .a = 255
    };
}

void Application::color_slice(std::span<const float> samples, std::span<Color> color_buffer) const
{
    for (std::size_t i { 0 }; i < samples.size(); i++) {
        float sample { samples[i] };

        if (sample < 0.0) {
            color_buffer[i] = Color { .r = 255, .g = 0, .b = 0, .a = 255 };
        } else {
            Color color = this->sample_transfer_function(sample);
            color_buffer[i] = color;
        }
    }
}

void Application::compute_marching_squares_cell(std::vector<IsoContourLine>& lines, Cell2D cell, float iso_value,
    glm::vec2 cell_start, glm::vec2 cell_size) const
{
    // Implement method.
}

std::vector<IsoContourLine> Application::compute_iso_contours(std::span<const float> samples, uint32_t width,
    uint32_t height, float iso_value, glm::vec2 iso_range) const
{
    // Replace with implementation.
    return { IsoContourLine { .start_x = 0.0, .start_y = 0.0, .end_x = 1.0, .end_y = 1.0 } };
}
