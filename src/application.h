#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <vector>

#include <application_base.h>
#include <extrema_graph.h>
#include <pvm_volume.h>

/**
 * A color with 8-bit rgba values.
 * Each channel is in the range [0, 255].
 */
struct Color {
    uint8_t r; // Red
    uint8_t g; // Green
    uint8_t b; // Blue
    uint8_t a; // Alpha
};

/**
 * Grid values of a voxel cell.
 * Each value is in the range [0.0, 1.0].
 *
 * Left/Right corresponds to x/x+1
 * Front/Back corresponds to y/y+1
 * Bottom/Top corresponds to z/z+1
 */
struct VoxelCell {
    float bottom_front_left;
    float bottom_front_right;
    float bottom_back_left;
    float bottom_back_right;
    float top_front_left;
    float top_front_right;
    float top_back_left;
    float top_back_right;
};

/**
 * A two-dimensional cell.
 */
struct Cell2D {
    float top_left;
    float top_right;
    float bottom_left;
    float bottom_right;
};

/**
 * Extents of the slice plane.
 */
struct Plane {
    glm::vec3 top_left; // Top left coordinate
    glm::vec3 bottom_left; // Bottom left coordinate
    glm::vec3 bottom_right; // Bottom right coordinate
};

/**
 * Available datasets.
 */
enum class Dataset {
    Baby,
    CT_Head,
    Fuel
};

/**
 * Possible plane orientations.
 */
enum class SlicePlane {
    Axial,
    Sagittal,
    Coronal,
};

/**
 * A line given in texture coordinates.
 *
 * The position (0, 0) corresponds to the bottom left
 * of the screen, while (1, 1) corresponds to the top
 * right.
 */
struct alignas(8) IsoContourLine {
    glm::vec2 start;
    glm::vec2 end;
};

/**
 * A triangle in local coordinates.
 *
 * The position (0, 0, 0) corresponds to the position of the
 * first grid point. The maximum position of
 * volume_extents * volume_cell_size corresponds to the position
 * of the last grid cell. The individual points spanning the
 * triangle must be defined in counter-clockwise order, for
 * example top left, bottom left, bottom right.
 */
struct IsoSurfaceTriangle {
    glm::vec3 p0;
    glm::vec3 p1;
    glm::vec3 p2;
};

class Application final : public ApplicationBase {
public:
    Application();
    Application(const Application&) = delete;
    Application(Application&&);
    ~Application();

    Application& operator=(const Application&) = delete;
    Application& operator=(Application&&) = delete;

protected:
    void on_frame(wgpu::CommandEncoder&, wgpu::TextureView&, wgpu::TextureView&) override;
    void on_resize() override;

private:
    void init_slice_render_pipeline();
    void init_iso_contours_render_pipeline();
    void init_extrema_graph_render_pipeline();
    void init_iso_surface_render_pipeline();

    void init_slice_texture();
    void init_iso_contours_buffer();
    void init_uniform_buffer();
    void init_iso_surface_buffer();

    void init_extrema_graph_2d();

    void init_projection_matrix();
    void init_view_matrix();
    void init_model_matrix();

    void update_slice_samples_and_texture();
    void update_iso_contours();
    void update_iso_surface();

    /**
     * Computes the value inside the cell by applying a trilinear interpolation
     * of the grid values of the voxel cell.
     *
     * @param cell grid values around the cell
     * @param t_x cell-local x coordinate, in range [0.0, 1.0].
     * @param t_y cell-local y coordinate, in range [0.0, 1.0].
     * @param t_z cell-local z coordinate, in range [0.0, 1.0].
     * @return interpolated value
     */
    float interpolate_trilinear(VoxelCell cell, float t_x, float t_y, float t_z) const;

    /**
     * Samples the volume at the given position.
     *
     * This returns the normalized value at the given grid points. If the
     * position lies inside a voxel cell, the resulting value is interpolated
     * through trilinear interpolation. If the position lies outside of the
     * volume, it returns `-1.0`.
     *
     * @param volume volume to sample
     * @param position grid point position
     * @return interpolated normalized sample
     */
    float sample_at_position(const PVMVolume& volume, glm::vec3 position) const;

    /**
     * Samples a slice from the provided volume.
     *
     * The samples are written into the buffer of size buffer_width * buffer_height.
     * The index 0 of the color buffer corresponds to the sample at the bottom left
     * position of the provided plane, while the last index corresponds to the upper
     * right position of the plane. The sample buffer is given in row-major order,
     * i.e., consecutive elements of a row are contiguous in memory.
     *
     * @param volume volume to slice
     * @param plane_buffer buffer where the slice is written to
     * @param plane slicing plane
     * @param buffer_width width of the color buffer
     * @param buffer_height height of the color buffer
     */
    void sample_slice(const PVMVolume& volume, std::span<float> plane_buffer, Plane plane,
        uint32_t buffer_width, uint32_t buffer_height) const;

    /**
     * Samples the transfer function at a given position.
     * The transfer function is a continuous grayscale color map, where position
     * t=0.0 corresponds to black, wheras position t=1.0 is white.
     *
     * @param t sample position, in range [0.0, 1.0].
     * @return sampled color
     */
    Color sample_transfer_function(float t) const;

    /**
     * Assigns a color to each sample.
     *
     * If a sample is `< 0.0`, then it assigns the color red.
     *
     * @param samples buffer of samples
     * @param color_buffer colors of the samples
     */
    void color_slice(std::span<const float> samples, std::span<Color> color_buffer) const;

    /**
     * Applies the marching squares algorithm on a single cell.
     *
     * The cell start position is given in texture coordinates, with (0, 0)
     * being the bottom left of the screen and (1, 1) being the top right.
     *
     * @param lines lines buffer
     * @param cell cell to process
     * @param iso_value normalized iso-value in the range [0.0, 1.0]
     * @param cell_start position of the bottom left sample of the cell
     * @param cell_size size of the cell
     */
    void compute_marching_squares_cell(std::vector<IsoContourLine>& lines, Cell2D cell,
        float iso_value, glm::vec2 cell_start, glm::vec2 cell_size,
        glm::vec<2, std::uint32_t> cell_idx,
        std::vector<std::tuple<uint32_t, uint32_t>>& cell_stack) const;

    /**
     * Computes the Isocontour of the sampled slice.
     *
     * The iso-value is given in the range [iso_range.x, iso_range.y].
     *
     * @param samples sampled slice
     * @param width width of the slice
     * @param height height of the slice
     * @param iso_value ios-value to compute the isocontour of
     * @param iso_range minimum/maximum value of the iso-value
     * @return std::vector<IsoContourLine>
     */
    std::vector<IsoContourLine> compute_iso_contours(std::span<const float> samples, uint32_t width,
        uint32_t height, float iso_value, glm::vec2 iso_range) const;

    /**
     * Applies the marching cubes algorithm on a single cell.
     *
     * The cell and triangle positions are given in the local space of the volume,
     * with position (0, 0, 0) corresponding to the position of the first grid value,
     * and position (x, y, z) * cell_size corresponding to the position of the grid
     * value (x, y, z).
     *
     * @param triangles triangles buffer
     * @param cell cell to process
     * @param iso_value normalized iso-value in the range [0.0, 1.0]
     * @param cell_start start position of the cell
     * @param cell_size size of the cell in local coordinate space
     */
    void compute_marching_cubes_cell(std::vector<IsoSurfaceTriangle>& triangles, VoxelCell cell,
        float iso_value, glm::vec3 cell_start, glm::vec3 cell_size) const;

    /**
     * Computes the isosurface of the volume.
     *
     * The iso-value is given in the range [iso_range.x, iso_range.y].
     *
     * @param volume volume to sample from
     * @param iso_value ios-value to compute the isosurface of
     * @param iso_range minimum/maximum value of the iso-value
     * @return triangles of the isosurface
     */
    std::vector<IsoSurfaceTriangle> compute_iso_surface(const PVMVolume& volume, float iso_value,
        glm::vec2 iso_range) const;

    /**
     * Computes the extrema graph of the two-dimensional slice.
     *
     * @param samples slice samples
     * @param width width of the slice
     * @param height height of the slice
     * @return extrema graph of the slice
     */
    ExtremaGraph<2> compute_extrema_graph(std::span<const float> samples, uint32_t width,
        uint32_t height) const;

    wgpu::ShaderModule m_slice_shader_module;
    wgpu::BindGroupLayout m_slice_bind_group_layout;
    wgpu::PipelineLayout m_slice_pipeline_layout;
    wgpu::RenderPipeline m_slice_render_pipeline;

    wgpu::ShaderModule m_iso_contours_shader_module;
    wgpu::BindGroupLayout m_iso_contours_bind_group_layout;
    wgpu::PipelineLayout m_iso_contours_pipeline_layout;
    wgpu::RenderPipeline m_iso_contours_render_pipeline;

    wgpu::ShaderModule m_extrema_graph_shader_module;
    wgpu::BindGroupLayout m_extrema_graph_bind_group_layout;
    wgpu::PipelineLayout m_extrema_graph_pipeline_layout;
    wgpu::RenderPipeline m_extrema_graph_render_pipeline;

    wgpu::ShaderModule m_iso_surface_shader_module;
    wgpu::BindGroupLayout m_iso_surface_bind_group_layout;
    wgpu::PipelineLayout m_iso_surface_pipeline_layout;
    wgpu::RenderPipeline m_iso_surface_render_pipeline;

    wgpu::Texture m_slice_texture;
    std::vector<float> m_slice_samples;
    bool m_slice_texture_changed;

    wgpu::Buffer m_iso_contours_buffer;
    std::vector<IsoContourLine> m_iso_contour_lines;

    wgpu::Buffer m_extrema_graph_buffer;
    ExtremaGraph<2> m_extrema_graph_2d;
    uint32_t m_extrema_graph_2d_edges;

    wgpu::Buffer m_uniforms_buffer;
    wgpu::Buffer m_iso_surface_vertex_buffer;
    wgpu::Buffer m_iso_surface_index_buffer;
    std::size_t m_vertex_count;
    std::size_t m_index_count;

    std::optional<PVMVolume> m_volume;
    Dataset m_dataset;

    glm::mat4 m_projection_mat;
    glm::mat4 m_view_mat;
    glm::mat4 m_model_mat;
    glm::mat3 m_normal_mat;
    glm::vec3 m_view_pos;

    SlicePlane m_plane;
    float m_plane_offset;
    float m_plane_rotation;
    float m_iso_value;
    float m_camera_distance;
    float m_camera_theta;
    float m_camera_phi;
};