#include <application.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numbers>
#include <utility>

Application::Application()
    : ApplicationBase { "Exercise 10" }
    , m_slice_shader_module { nullptr }
    , m_slice_bind_group_layout { nullptr }
    , m_slice_pipeline_layout { nullptr }
    , m_slice_render_pipeline { nullptr }
    , m_iso_contours_shader_module { nullptr }
    , m_iso_contours_bind_group_layout { nullptr }
    , m_iso_contours_pipeline_layout { nullptr }
    , m_iso_contours_render_pipeline { nullptr }
    , m_iso_surface_shader_module { nullptr }
    , m_iso_surface_bind_group_layout { nullptr }
    , m_iso_surface_pipeline_layout { nullptr }
    , m_iso_surface_render_pipeline { nullptr }
    , m_slice_texture { nullptr }
    , m_slice_samples {}
    , m_slice_texture_changed { false }
    , m_iso_contours_buffer { nullptr }
    , m_iso_contour_lines {}
    , m_uniforms_buffer { nullptr }
    , m_iso_surface_vertex_buffer { nullptr }
    , m_iso_surface_index_buffer { nullptr }
    , m_vertex_count { 0 }
    , m_index_count { 0 }
    , m_volume { std::nullopt }
    , m_dataset { Dataset::Baby }
    , m_projection_mat { glm::identity<glm::mat4>() }
    , m_view_mat { glm::identity<glm::mat4>() }
    , m_model_mat { glm::identity<glm::mat4>() }
    , m_normal_mat { glm::identity<glm::mat3>() }
    , m_view_pos { 0.0f, 0.0f, 0.0f }
    , m_plane { SlicePlane::Axial }
    , m_plane_offset { 0.0f }
    , m_plane_rotation { 0.0f }
    , m_iso_value { 0.0f }
    , m_camera_distance { 2.0f }
    , m_camera_theta { std::numbers::pi_v<float> / 2.0f }
    , m_camera_phi { 0.0f }
{
    this->init_slice_render_pipeline();
    this->init_iso_contours_render_pipeline();
    this->init_iso_surface_render_pipeline();
    this->init_slice_texture();
    this->init_iso_contours_buffer();
    this->init_uniform_buffer();
    this->init_iso_surface_buffer();
    this->init_projection_matrix();
    this->init_view_matrix();
    this->init_model_matrix();
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
    , m_iso_surface_shader_module { std::exchange(app.m_iso_surface_shader_module, nullptr) }
    , m_iso_surface_bind_group_layout { std::exchange(app.m_iso_surface_bind_group_layout, nullptr) }
    , m_iso_surface_pipeline_layout { std::exchange(app.m_iso_surface_pipeline_layout, nullptr) }
    , m_iso_surface_render_pipeline { std::exchange(app.m_iso_surface_render_pipeline, nullptr) }
    , m_slice_texture { std::exchange(app.m_slice_texture, nullptr) }
    , m_slice_samples { std::exchange(app.m_slice_samples, {}) }
    , m_slice_texture_changed { std::exchange(app.m_slice_texture_changed, false) }
    , m_iso_contours_buffer { std::exchange(app.m_iso_contours_buffer, nullptr) }
    , m_iso_contour_lines { std::exchange(app.m_iso_contour_lines, {}) }
    , m_uniforms_buffer { std::exchange(app.m_uniforms_buffer, nullptr) }
    , m_iso_surface_vertex_buffer { std::exchange(app.m_iso_surface_vertex_buffer, nullptr) }
    , m_iso_surface_index_buffer { std::exchange(app.m_iso_surface_index_buffer, nullptr) }
    , m_vertex_count { std::exchange(app.m_vertex_count, 0) }
    , m_index_count { std::exchange(app.m_index_count, 0) }
    , m_volume { std::exchange(app.m_volume, std::nullopt) }
    , m_dataset { std::exchange(app.m_dataset, Dataset::Baby) }
    , m_projection_mat { std::exchange(app.m_projection_mat, glm::identity<glm::mat4>()) }
    , m_view_mat { std::exchange(app.m_view_mat, glm::identity<glm::mat4>()) }
    , m_model_mat { std::exchange(app.m_model_mat, glm::identity<glm::mat4>()) }
    , m_normal_mat { std::exchange(app.m_normal_mat, glm::identity<glm::mat3>()) }
    , m_view_pos { std::exchange(app.m_view_pos, { 0.0f, 0.0f, 0.0f }) }
    , m_plane { std::exchange(app.m_plane, SlicePlane::Axial) }
    , m_plane_offset { std::exchange(app.m_plane_offset, 0.0f) }
    , m_plane_rotation { std::exchange(app.m_plane_rotation, 0.0f) }
    , m_iso_value { std::exchange(app.m_iso_value, 0.0f) }
    , m_camera_distance { std::exchange(app.m_camera_distance, 2.0f) }
    , m_camera_theta { std::exchange(app.m_camera_theta, std::numbers::pi_v<float> / 2.0f) }
    , m_camera_phi { std::exchange(app.m_camera_phi, 0.0f) }
{
}

Application::~Application()
{
    if (this->m_iso_surface_index_buffer) {
        this->m_iso_surface_index_buffer.release();
    }

    if (this->m_iso_surface_vertex_buffer) {
        this->m_iso_surface_vertex_buffer.release();
    }

    if (this->m_uniforms_buffer) {
        this->m_uniforms_buffer.release();
    }

    if (this->m_iso_contours_buffer) {
        this->m_iso_contours_buffer.release();
    }
    if (this->m_slice_texture) {
        this->m_slice_texture.release();
    }

    if (this->m_iso_surface_render_pipeline) {
        this->m_iso_surface_render_pipeline.release();
    }
    if (this->m_iso_surface_pipeline_layout) {
        this->m_iso_surface_pipeline_layout.release();
    }
    if (this->m_iso_surface_bind_group_layout) {
        this->m_iso_surface_bind_group_layout.release();
    }
    if (this->m_iso_surface_shader_module) {
        this->m_iso_surface_shader_module.release();
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

void Application::on_frame(wgpu::CommandEncoder& encoder, wgpu::TextureView& frame, wgpu::TextureView& depth_stencil)
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
        this->init_model_matrix();
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

    auto iso_value_range = this->m_volume->component_range(0);
    bool iso_value_changed = ImGui::SliderFloat("Iso-value", &this->m_iso_value, iso_value_range.x, iso_value_range.y);

    bool camera_changed = ImGui::SliderFloat("Camera distance (%)", &this->m_camera_distance, 0.0f, 5.0f);
    camera_changed |= ImGui::SliderAngle("Camera inclination", &this->m_camera_theta, 0.0f, 180.0f);
    camera_changed |= ImGui::SliderAngle("Camera azimuth", &this->m_camera_phi, 0.0f, 360.0f);
    if (dataset_changed || camera_changed) {
        this->init_view_matrix();
    }

    ImGui::End();

    bool update_slice_texture = this->m_slice_texture_changed || dataset_changed || plane_changed;
    this->m_slice_texture_changed = false;
    if (update_slice_texture) {
        this->update_slice_samples_and_texture();
    }

    if (update_slice_texture || iso_value_changed) {
        this->update_iso_contours();
    }

    if (dataset_changed || iso_value_changed) {
        this->update_iso_surface();
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

    wgpu::RenderPassDepthStencilAttachment depth_stencil_attachment { wgpu::Default };
    depth_stencil_attachment.view = depth_stencil;
    depth_stencil_attachment.depthClearValue = 1.0f;
    depth_stencil_attachment.depthLoadOp = wgpu::LoadOp::Clear;
    depth_stencil_attachment.depthStoreOp = wgpu::StoreOp::Store;
    depth_stencil_attachment.depthReadOnly = false;
    depth_stencil_attachment.stencilClearValue = 0;
    depth_stencil_attachment.stencilLoadOp = wgpu::LoadOp::Clear;
    depth_stencil_attachment.stencilStoreOp = wgpu::StoreOp::Store;
    depth_stencil_attachment.stencilReadOnly = true;

    wgpu::RenderPassDescriptor pass_desc { wgpu::Default };
    pass_desc.colorAttachmentCount = color_attachments.size();
    pass_desc.colorAttachments = color_attachments.data();
    pass_desc.depthStencilAttachment = &depth_stencil_attachment;
    auto pass_encoder = encoder.beginRenderPass(pass_desc);
    if (!pass_encoder) {
        std::cerr << "Could not create render pass!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    float width { static_cast<float>(this->surface_width()) / 2.0f };
    float height { static_cast<float>(this->surface_height()) };

    pass_encoder.setViewport(0.0f, 0.0f, width, height, 0.0f, 1.0f);
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

    wgpu::BindGroup iso_surface_bind_group { nullptr };
    if (this->m_vertex_count > 0) {
        this->init_uniform_buffer();

        std::array<wgpu::BindGroupEntry, 1> surface_bind_group_entries { wgpu::Default };
        surface_bind_group_entries[0].binding = 0;
        surface_bind_group_entries[0].buffer = this->m_uniforms_buffer;
        surface_bind_group_entries[0].size = WGPU_WHOLE_SIZE;

        wgpu::BindGroupDescriptor surface_bind_group_desc { wgpu::Default };
        surface_bind_group_desc.layout = this->m_iso_surface_bind_group_layout;
        surface_bind_group_desc.entryCount = surface_bind_group_entries.size();
        surface_bind_group_desc.entries = surface_bind_group_entries.data();
        iso_surface_bind_group = this->device().createBindGroup(surface_bind_group_desc);

        std::uint64_t vertex_buffer_size = this->m_vertex_count * sizeof(float) * 3 * 2;
        std::uint64_t index_buffer_size = this->m_index_count * sizeof(std::uint32_t);

        pass_encoder.setViewport(width, 0.0f, width, height, 0.0f, 1.0f);
        pass_encoder.setPipeline(this->m_iso_surface_render_pipeline);
        pass_encoder.setBindGroup(0, iso_surface_bind_group, 0, nullptr);
        pass_encoder.setVertexBuffer(0, this->m_iso_surface_vertex_buffer, 0, vertex_buffer_size);
        pass_encoder.setIndexBuffer(this->m_iso_surface_index_buffer, wgpu::IndexFormat::Uint32, 0, index_buffer_size);
        pass_encoder.drawIndexed(this->m_index_count, 1, 0, 0, 0);
    }

    pass_encoder.end();
    pass_encoder.release();
    bind_group.release();

    if (contours_bind_group) {
        contours_bind_group.release();
    }

    if (iso_surface_bind_group) {
        iso_surface_bind_group.release();
    }
}

void Application::on_resize()
{
    ApplicationBase::on_resize();
    this->init_slice_texture();
    this->init_projection_matrix();
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

    wgpu::DepthStencilState depth_stencil_state { wgpu::Default };
    depth_stencil_state.format = this->depth_stencil_format();
    depth_stencil_state.depthWriteEnabled = false;
    depth_stencil_state.depthCompare = wgpu::CompareFunction::Less;
    depth_stencil_state.stencilReadMask = 0;
    depth_stencil_state.stencilWriteMask = 0;

    pipeline_desc.depthStencil = &depth_stencil_state;
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

    wgpu::DepthStencilState depth_stencil_state { wgpu::Default };
    depth_stencil_state.format = this->depth_stencil_format();
    depth_stencil_state.depthWriteEnabled = false;
    depth_stencil_state.depthCompare = wgpu::CompareFunction::Less;
    depth_stencil_state.stencilReadMask = 0;
    depth_stencil_state.stencilWriteMask = 0;

    pipeline_desc.depthStencil = &depth_stencil_state;
    pipeline_desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipeline_desc.multisample.count = 1;
    pipeline_desc.multisample.mask = 0xFFFFFFFF;
    this->m_iso_contours_render_pipeline = this->device().createRenderPipeline(pipeline_desc);
    if (!this->m_iso_contours_render_pipeline) {
        std::cerr << "Failed to create the render pipeline" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Application::init_iso_surface_render_pipeline()
{
    wgpu::ShaderModuleWGSLDescriptor wgsl_module_desc { wgpu::Default };
    wgsl_module_desc.code = R"(
        struct State {
            mvp_matrix: mat4x4<f32>,
            model_matrix: mat4x4<f32>,
            normal_matrix: mat3x3<f32>,
            view_pos: vec3<f32>,
        }

        @group(0)
        @binding(0)
        var<uniform> state: State;

        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) normal: vec3<f32>,
        }

        struct VertexOutput {
            @builtin(position) pos: vec4<f32>,
            @location(0) position: vec3<f32>,
            @location(1) normal: vec3<f32>,
        }

        @vertex
        fn vs_main(in: VertexInput) -> VertexOutput {
            var out: VertexOutput;

            out.pos = state.mvp_matrix * vec4(in.position, 1.0);
            out.position = (state.model_matrix * vec4(in.position, 1.0)).xyz;
            out.normal = state.normal_matrix * in.normal;

            return out;
        }

        const LIGHT_POS = vec3<f32>(100.0, 100.0, 100.0);
        const AMBIENT_COLOR = vec3<f32>(0.1, 0.1, 0.1);
        const LIGHT_COLOR = vec3<f32>(0.7, 0.7, 0.7);
        const SHININESS = 16.0;

        @fragment
        fn fs_main(
            @location(0) position: vec3<f32>,
            @location(1) normal: vec3<f32>,
        ) -> @location(0) vec4<f32> {
            let norm = normalize(normal);
            let light_dir = normalize(LIGHT_POS - position);

            var lambertian = max(dot(light_dir, norm), 0.0);
            var specular = 0.0;

            if lambertian > 0.0 {
                let view_dir = normalize(state.view_pos - position);
                
                // Blinn Phong
                let half_dir = normalize(light_dir + view_dir);
                let spec_angle = max(dot(half_dir, norm), 0.0);
                specular = pow(spec_angle, SHININESS);
            }

            let color = AMBIENT_COLOR + (LIGHT_COLOR * lambertian) + (LIGHT_COLOR * specular);
            return vec4(color, 1.0);
        }
    )";
    wgpu::ShaderModuleDescriptor module_desc { wgpu::Default };
    module_desc.nextInChain = reinterpret_cast<wgpu::ChainedStruct*>(&wgsl_module_desc);
    this->m_iso_surface_shader_module = this->device().createShaderModule(module_desc);
    if (!this->m_iso_surface_shader_module) {
        std::cerr << "Failed to create the shader module" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::array<wgpu::BindGroupLayoutEntry, 1> binding_layout_entries { wgpu::Default };
    binding_layout_entries[0].binding = 0;
    binding_layout_entries[0].visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    binding_layout_entries[0].buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutDescriptor group_layout_desc { wgpu::Default };
    group_layout_desc.entryCount = binding_layout_entries.size();
    group_layout_desc.entries = binding_layout_entries.data();
    this->m_iso_surface_bind_group_layout = this->device().createBindGroupLayout(group_layout_desc);
    if (!this->m_iso_surface_bind_group_layout) {
        std::cerr << "Failed to create the bind group layout" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    wgpu::PipelineLayoutDescriptor layout_desc { wgpu::Default };
    layout_desc.bindGroupLayoutCount = 1;
    layout_desc.bindGroupLayouts = (WGPUBindGroupLayout*)&this->m_iso_surface_bind_group_layout;
    this->m_iso_surface_pipeline_layout = this->device().createPipelineLayout(layout_desc);
    if (!this->m_iso_surface_pipeline_layout) {
        std::cerr << "Failed to create the pipeline layout" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    wgpu::RenderPipelineDescriptor pipeline_desc { wgpu::Default };
    pipeline_desc.layout = this->m_iso_surface_pipeline_layout;

    std::array<wgpu::VertexAttribute, 2> vertex_attributes { { wgpu::Default } };
    vertex_attributes[0].shaderLocation = 0;
    vertex_attributes[0].offset = 0;
    vertex_attributes[0].format = wgpu::VertexFormat::Float32x3;
    vertex_attributes[1].shaderLocation = 1;
    vertex_attributes[1].offset = 3 * sizeof(float);
    vertex_attributes[1].format = wgpu::VertexFormat::Float32x3;

    std::array<wgpu::VertexBufferLayout, 1> vertex_buffers { { wgpu::Default } };
    vertex_buffers[0].attributes = vertex_attributes.data();
    vertex_buffers[0].attributeCount = vertex_attributes.size();
    vertex_buffers[0].stepMode = wgpu::VertexStepMode::Vertex;
    vertex_buffers[0].arrayStride = 6 * sizeof(float);

    pipeline_desc.vertex.module = this->m_iso_surface_shader_module;
    pipeline_desc.vertex.entryPoint = "vs_main";
    pipeline_desc.vertex.buffers = vertex_buffers.data();
    pipeline_desc.vertex.bufferCount = vertex_buffers.size();

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
    fragment_state.module = this->m_iso_surface_shader_module;
    fragment_state.entryPoint = "fs_main";
    fragment_state.targetCount = fragment_targets.size();
    fragment_state.targets = fragment_targets.data();
    fragment_state.constantCount = 0;
    fragment_state.constants = nullptr;
    pipeline_desc.fragment = &fragment_state;

    wgpu::DepthStencilState depth_stencil_state { wgpu::Default };
    depth_stencil_state.format = this->depth_stencil_format();
    depth_stencil_state.depthWriteEnabled = true;
    depth_stencil_state.depthCompare = wgpu::CompareFunction::Less;
    depth_stencil_state.stencilReadMask = 0;
    depth_stencil_state.stencilWriteMask = 0;

    pipeline_desc.depthStencil = &depth_stencil_state;
    pipeline_desc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipeline_desc.primitive.stripIndexFormat = wgpu::IndexFormat::Undefined;
    pipeline_desc.primitive.frontFace = wgpu::FrontFace::CCW;
    pipeline_desc.primitive.cullMode = wgpu::CullMode::None;
    pipeline_desc.multisample.count = 1;
    pipeline_desc.multisample.mask = 0xFFFFFFFF;
    this->m_iso_surface_render_pipeline = this->device().createRenderPipeline(pipeline_desc);
    if (!this->m_iso_surface_render_pipeline) {
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
    uint32_t width { this->surface_width() / 2 };
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

void Application::init_uniform_buffer()
{
    struct alignas(16) UniformState {
        glm::mat4 mvp_mat;
        glm::mat4 model_mat;
        glm::mat4x3 normal_mat;
        glm::vec3 view_pos;
    };

    wgpu::Device& device = this->device();
    wgpu::Queue queue { device.getQueue() };

    if (!this->m_uniforms_buffer) {
        wgpu::BufferDescriptor desc { wgpu::Default };
        desc.label = "surface uniform buffer";
        desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Uniform;
        desc.mappedAtCreation = false;
        desc.size = static_cast<uint64_t>(sizeof(UniformState));
        this->m_uniforms_buffer = device.createBuffer(desc);
        if (!this->m_uniforms_buffer) {
            std::cerr << "Could not create surface uniform buffer!" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    UniformState state {
        this->m_projection_mat * this->m_view_mat * this->m_model_mat,
        this->m_model_mat,
        this->m_normal_mat,
        this->m_view_pos,
    };
    queue.writeBuffer(this->m_uniforms_buffer, 0, &state, sizeof(UniformState));
}

void Application::init_iso_surface_buffer()
{
    if (this->m_iso_surface_vertex_buffer) {
        this->m_iso_surface_vertex_buffer.release();
        this->m_iso_surface_vertex_buffer = { nullptr };
    }

    if (this->m_iso_surface_index_buffer) {
        this->m_iso_surface_index_buffer.release();
        this->m_iso_surface_index_buffer = { nullptr };
    }

    wgpu::Device& device = this->device();

    size_t vertex_buffer_size { this->m_vertex_count * 2 * 3 * sizeof(float) };
    size_t index_buffer_size { this->m_index_count * sizeof(std::uint32_t) };
    if (this->m_index_count == 0) {
        vertex_buffer_size = sizeof(float) * 6;
        index_buffer_size = sizeof(std::uint32_t);
    }

    wgpu::BufferDescriptor desc { wgpu::Default };
    desc.label = "surface vertex buffer";
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Vertex;
    desc.mappedAtCreation = false;
    desc.size = static_cast<uint64_t>(vertex_buffer_size);
    this->m_iso_surface_vertex_buffer = device.createBuffer(desc);
    if (!this->m_iso_surface_vertex_buffer) {
        std::cerr << "Could not create surface vertex buffer!" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    desc.label = "surface index buffer";
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Index;
    desc.mappedAtCreation = false;
    desc.size = static_cast<uint64_t>(index_buffer_size);
    this->m_iso_surface_index_buffer = device.createBuffer(desc);
    if (!this->m_iso_surface_index_buffer) {
        std::cerr << "Could not create surface index buffer!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void Application::init_projection_matrix()
{
    constexpr float fov { 70.0f * std::numbers::pi_v<float> / 180.0 };
    constexpr float near { 0.001f };
    constexpr float far { 10000.0f };

    float width { static_cast<float>(this->surface_width()) / 2.0f };
    float height { static_cast<float>(this->surface_height()) };
    float ratio { width / height };

    this->m_projection_mat = glm::perspective(fov, ratio, near, far);
}

void Application::init_view_matrix()
{
    if (!this->m_volume.has_value()) {
        this->m_view_pos = glm::vec3 { 0.0f };
        this->m_view_mat = glm::identity<glm::mat4>();
        return;
    }

    float distance { this->m_camera_distance };
    float theta { this->m_camera_theta };
    float phi { this->m_camera_phi };

    auto extents { this->m_volume->extents() };
    glm::vec3 extents_float { extents.x, extents.y, extents.z };

    auto scale { this->m_volume->scale() };
    glm::vec3 volume_size { extents_float * scale };
    glm::vec3 radii { volume_size * distance };

    float x { radii.x * glm::sin(theta) * glm::cos(phi) };
    float y { radii.y * glm::sin(theta) * glm::sin(phi) };
    float z { radii.z * glm::cos(theta) };

    this->m_view_pos = glm::vec3 { x, y, z };
    this->m_view_mat = glm::lookAt(this->m_view_pos, glm::vec3 { 0.0f }, glm::vec3 { 0.0f, 0.0f, 1.0f });
}

void Application::init_model_matrix()
{
    if (!this->m_volume.has_value()) {
        this->m_model_mat = glm::identity<glm::mat4>();
        this->m_normal_mat = glm::identity<glm::mat3>();
        return;
    }

    auto extents { this->m_volume->extents() };
    glm::vec3 extents_float { extents.x, extents.y, extents.z };

    auto scale { this->m_volume->scale() };
    glm::vec3 volume_size { extents_float * scale };
    glm::vec3 center { volume_size / 2.0f };

    this->m_model_mat = glm::identity<glm::mat4>();
    this->m_model_mat = glm::translate(this->m_model_mat, -center);
    this->m_normal_mat = glm::mat3 { glm::transpose(glm::inverse(this->m_model_mat)) };
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
    uint32_t width { this->surface_width() / 2 };
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
    uint32_t width { this->surface_width() / 2 };
    uint32_t height { this->surface_height() };
    glm::vec2 iso_range { this->m_volume->component_range(0) };
    this->m_iso_contour_lines = this->compute_iso_contours(this->m_slice_samples, width, height, this->m_iso_value, iso_range);
    this->init_iso_contours_buffer();

    // Upload the lines to the buffer.
    wgpu::Device& device { this->device() };
    wgpu::Queue queue { device.getQueue() };
    queue.writeBuffer(this->m_iso_contours_buffer, 0, this->m_iso_contour_lines.data(), this->m_iso_contour_lines.size() * sizeof(IsoContourLine));
}

void Application::update_iso_surface()
{
    if (!this->m_volume) {
        return;
    }

    // Compute the surface.
    glm::vec2 iso_range { this->m_volume->component_range(0) };
    std::vector<IsoSurfaceTriangle> triangles { this->compute_iso_surface(*this->m_volume, this->m_iso_value, iso_range) };

    // Extract the vertex and index buffers.
    std::size_t vertex_count { 0 };
    std::vector<std::array<float, 3>> vertex_buffer {};
    std::vector<std::uint32_t> index_buffer {};
    std::map<std::array<float, 3>, std::uint32_t> index_map {};

    auto init_vertex = [&](const glm::vec3& vertex) {
        std::array<float, 3> vertex_ { vertex.x, vertex.y, vertex.z };
        auto pos { index_map.find(vertex_) };
        if (pos == index_map.end()) {
            std::uint32_t index = vertex_count++;
            vertex_buffer.push_back({ vertex.x, vertex.y, vertex.z });
            vertex_buffer.push_back({ 0.0f, 0.0f, 0.0f });
            index_map.insert({ vertex_, index });
            index_buffer.push_back(index);
            return index;
        } else {
            index_buffer.push_back(pos->second);
            return pos->second;
        }
    };

    auto get_normal = [&](std::size_t index) -> glm::vec3 {
        auto normal { vertex_buffer[index * 2 + 1] };
        return { normal[0], normal[1], normal[2] };
    };

    auto set_normal = [&](std::size_t index, glm::vec3 value) {
        auto& normal { vertex_buffer[index * 2 + 1] };
        normal[0] = value.x;
        normal[1] = value.x;
        normal[2] = value.x;
    };

    // Compute the smooth normals for each vertex.
    for (const auto& triangle : triangles) {
        std::uint32_t p0_idx = init_vertex(triangle.p0);
        std::uint32_t p1_idx = init_vertex(triangle.p1);
        std::uint32_t p2_idx = init_vertex(triangle.p2);

        glm::vec3 p0_normal = get_normal(p0_idx);
        glm::vec3 p1_normal = get_normal(p1_idx);
        glm::vec3 p2_normal = get_normal(p2_idx);

        glm::vec3 normal = glm::cross(triangle.p1 - triangle.p0, triangle.p2 - triangle.p0);
        p0_normal += normal;
        p1_normal += normal;
        p2_normal += normal;

        set_normal(p0_idx, p0_normal);
        set_normal(p1_idx, p1_normal);
        set_normal(p2_idx, p2_normal);
    }

    for (std::size_t i { 0 }; i < vertex_count; i++) {
        glm::vec3 normal = glm::normalize(get_normal(i));
        set_normal(i, normal);
    }

    // Resize the buffers.
    this->m_vertex_count = vertex_count;
    this->m_index_count = index_buffer.size();
    this->init_iso_surface_buffer();

    // Upload the triangles to the buffer.
    wgpu::Device& device { this->device() };
    wgpu::Queue queue { device.getQueue() };
    queue.writeBuffer(this->m_iso_surface_vertex_buffer, 0, vertex_buffer.data(), vertex_buffer.size() * sizeof(float) * 3);
    queue.writeBuffer(this->m_iso_surface_index_buffer, 0, index_buffer.data(), index_buffer.size() * sizeof(std::uint32_t));
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
    int case_num { 0 };
    case_num |= 0b0001 * (cell.bottom_left > iso_value);
    case_num |= 0b0010 * (cell.bottom_right > iso_value);
    case_num |= 0b0100 * (cell.top_right > iso_value);
    case_num |= 0b1000 * (cell.top_left > iso_value);

    if (case_num > 15) {
        std::cerr << "Invalid case " << case_num << std::endl;
    }

    std::array<glm::vec2, 4> ms_vertices {
        cell_start,
        cell_start + glm::vec2 { cell_size.x, 0.0 },
        cell_start + glm::vec2 { cell_size.x, cell_size.y },
        cell_start + glm::vec2 { 0.0, cell_size.y },
    };

    constexpr std::array<std::array<int, 2>, 4> ms_edges {
        std::array<int, 2> { 0, 1 },
        std::array<int, 2> { 1, 2 },
        std::array<int, 2> { 2, 3 },
        std::array<int, 2> { 3, 0 },
    };

    constexpr std::array<std::array<int, 4>, 16> ms_cases {
        std::array<int, 4> { -1, -1, -1, -1 },
        std::array<int, 4> { 0, 3, -1, -1 },
        std::array<int, 4> { 0, 1, -1, -1 },
        std::array<int, 4> { 1, 3, -1, -1 },
        std::array<int, 4> { 1, 2, -1, -1 },
        std::array<int, 4> { 0, 1, 2, 3 },
        std::array<int, 4> { 0, 2, -1, -1 },
        std::array<int, 4> { 2, 3, -1, -1 },
        std::array<int, 4> { 2, 3, -1, -1 },
        std::array<int, 4> { 0, 2, -1, -1 },
        std::array<int, 4> { 0, 3, 1, 2 },
        std::array<int, 4> { 1, 2, -1, -1 },
        std::array<int, 4> { 1, 3, -1, -1 },
        std::array<int, 4> { 0, 1, -1, -1 },
        std::array<int, 4> { 0, 3, -1, -1 },
        std::array<int, 4> { -1, -1, -1, -1 },
    };

    std::array<float, 4> values { cell.bottom_left, cell.bottom_right, cell.top_right, cell.top_left };
    for (int i { 0 }; i < 4; i += 2) {
        int start_edge { ms_cases[case_num][i] };
        int end_edge { ms_cases[case_num][i + 1] };
        if (start_edge == -1) {
            break;
        }

        int start_idx_0 { ms_edges[start_edge][0] };
        int start_idx_1 { ms_edges[start_edge][1] };
        int end_idx_0 { ms_edges[end_edge][0] };
        int end_idx_1 { ms_edges[end_edge][1] };

        float start_sample_1 { values[start_idx_0] };
        float start_sample_2 { values[start_idx_1] };
        float end_sample_1 { values[end_idx_0] };
        float end_sample_2 { values[end_idx_1] };

        float start_t { (start_sample_1 - iso_value) / (start_sample_1 - start_sample_2) };
        float end_t { (end_sample_1 - iso_value) / (end_sample_1 - end_sample_2) };

        glm::vec2 start_position_0 { ms_vertices[start_idx_0] };
        glm::vec2 start_position_1 { ms_vertices[start_idx_1] };
        glm::vec2 end_position_0 { ms_vertices[end_idx_0] };
        glm::vec2 end_position_1 { ms_vertices[end_idx_1] };

        glm::vec2 line_start { start_position_0 + ((1.0f - start_t) * (start_position_1 - start_position_0)) };
        glm::vec2 line_end { end_position_0 + ((1.0f - end_t) * (end_position_1 - end_position_0)) };

        lines.push_back(IsoContourLine { line_start, line_end });
    }
}

std::vector<IsoContourLine> Application::compute_iso_contours(std::span<const float> samples, uint32_t width,
    uint32_t height, float iso_value, glm::vec2 iso_range) const
{
    std::vector<IsoContourLine> lines {};
    float normalized_iso_value = (iso_value - iso_range.x) / (iso_range.y - iso_range.x);
    glm::vec2 cell_size {
        1.0 / static_cast<float>(width - 1),
        1.0 / static_cast<float>(height - 1),
    };

    for (uint32_t y { 0 }; y < height - 1; y++) {
        for (uint32_t x { 0 }; x < width - 1; x++) {
            std::size_t bottom_left_idx { x + (y * width) };
            std::size_t bottom_right_idx { (x + 1) + (y * width) };
            std::size_t top_left_idx { x + ((y + 1) * width) };
            std::size_t top_right_idx { (x + 1) + ((y + 1) * width) };

            float bottom_left = samples[bottom_left_idx];
            float bottom_right = samples[bottom_right_idx];
            float top_left = samples[top_left_idx];
            float top_right = samples[top_right_idx];
            Cell2D cell {
                .top_left = top_left,
                .top_right = top_right,
                .bottom_left = bottom_left,
                .bottom_right = bottom_right
            };

            glm::vec2 cell_start {
                static_cast<float>(x) / static_cast<float>(width),
                static_cast<float>(y) / static_cast<float>(height),
            };
            this->compute_marching_squares_cell(lines, cell, normalized_iso_value, cell_start, cell_size);
        }
    }

    return lines;
}

void Application::compute_marching_cubes_cell(std::vector<IsoSurfaceTriangle>& triangles, VoxelCell cell,
    float iso_value, glm::vec3 cell_start, glm::vec3 cell_size) const
{
    // Implement the function.
}

std::vector<IsoSurfaceTriangle> Application::compute_iso_surface(const PVMVolume& volume, float iso_value,
    glm::vec2 iso_range) const
{
    // Replace with implementation.
    std::vector<IsoSurfaceTriangle> triangles {};
    return triangles;
}
