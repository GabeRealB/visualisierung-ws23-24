#pragma once

#ifdef _MSC_VER
#pragma warning(push, 3)
#endif
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_LEFT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

/**
 * An extremum in a scalar field.
 *
 * @tparam N number of dimensions.
 */
template <size_t N>
struct Extrema {
    /**
     * Grid position of the extremum.
     */
    glm::vec<N, std::size_t> pos;
    /**
     * Sample value at the extremum.
     */
    float value;
};

/**
 * A node of the extrema graph
 *
 * The node consists of the extremum and the connected neighbors.
 * The marker index may be used during the construction of the
 * extrema graph.
 *
 * @tparam N number of dimensions
 */
template <size_t N>
struct ExtremaNode {
    Extrema<N> extremum;
    size_t marker_idx;
    std::vector<size_t> neighbor_idxs;
};

template <size_t N>
class ExtremaGraph {
public:
    ExtremaGraph() = default;
    ExtremaGraph(const ExtremaGraph&) = default;
    ExtremaGraph(ExtremaGraph&&) = default;

    /**
     * Constructs a new graph from a list of extrema.
     *
     * @param extrema list of extrema in a field
     */
    explicit ExtremaGraph(std::vector<Extrema<N>>& extrema);

    ExtremaGraph& operator=(const ExtremaGraph&) = default;
    ExtremaGraph& operator=(ExtremaGraph&&) = default;

    const std::vector<ExtremaNode<N>>& nodes() const;

    /**
     * Queries a list of grid points that can be used to seed the iso surface generation.
     *
     * The method receives the iso value of the surface and a callable that returns
     * whether a grid point at the specified position participates in the generation
     * of the iso surface. The callable `iso_value_in_cell` can be used like any other
     * function, with the syntax `iso_value_in_cell(grid_point, iso_value)`. You can
     * use the already provided method `calculate_arc`, to calculate a line between
     * two extrema which may contain the grid point that participates in the generation
     * of the iso surface.
     *
     * @param iso_value iso surface to query for
     * @param iso_value_in_cell callable
     * @return list of seed grid points
     */
    std::vector<glm::vec<N, size_t>> query_starting_points(
        float iso_value,
        std::predicate<glm::vec<N, size_t>, float> auto& iso_value_in_cell) const;

private:
    /**
     * Returns an arc of grid points from the start node to the end node.
     *
     * @param start first node
     * @param end second node
     * @return list of grid points between start and end
     */
    std::vector<glm::vec<N, size_t>> calculate_arc(const ExtremaNode<N>& start, const ExtremaNode<N>& end) const;

    std::vector<ExtremaNode<N>> m_nodes;
};

template <size_t N>
const std::vector<ExtremaNode<N>>& ExtremaGraph<N>::nodes() const
{
    return this->m_nodes;
}

template <size_t N>
std::vector<glm::vec<N, size_t>> ExtremaGraph<N>::calculate_arc(const ExtremaNode<N>& start, const ExtremaNode<N>& end) const
{
    std::vector<glm::vec<N, size_t>> cells {};
    return cells;
}

template <>
inline std::vector<glm::vec<2, size_t>> ExtremaGraph<2>::calculate_arc(const ExtremaNode<2>& start, const ExtremaNode<2>& end) const
{
    using svec2 = glm::vec<2, std::make_signed_t<size_t>>;

    svec2 signed_start_pos = start.extremum.pos;
    svec2 signed_end_pos = end.extremum.pos;
    svec2 delta = glm::abs(signed_end_pos - signed_start_pos);
    delta.y *= -1;

    svec2 x = glm::lessThan(signed_start_pos, signed_end_pos);
    svec2 s = (svec2(2) * x) - svec2(1);

    std::make_signed_t<size_t> error = delta.x + delta.y;

    std::vector<glm::vec<2, size_t>> cells {};
    glm::vec<2, size_t> current = start.extremum.pos;
    while (true) {
        cells.push_back(current);
        if (current == end.extremum.pos)
            break;
        std::make_signed_t<size_t> e2 = 2 * error;
        if (e2 >= delta.y) {
            if (current.x == end.extremum.pos.x)
                break;
            error = error + delta.y;
            current.x += s.x;
        }
        if (e2 <= delta.x) {
            if (current.y == end.extremum.pos.y)
                break;
            error = error + delta.x;
            current.y += s.y;
        }
    }

    return cells;
}

template <>
inline std::vector<glm::vec<3, size_t>> ExtremaGraph<3>::calculate_arc(const ExtremaNode<3>& start, const ExtremaNode<3>& end) const
{
    using svec3 = glm::vec<3, std::make_signed_t<size_t>>;

    svec3 signed_start_pos = start.extremum.pos;
    svec3 signed_end_pos = end.extremum.pos;
    svec3 delta = glm::abs(signed_end_pos - signed_start_pos);

    svec3 x = glm::lessThan(signed_start_pos, signed_end_pos);
    svec3 s = (svec3(2) * x) - svec3(1);

    size_t delta_max = std::max({ delta.x, delta.y, delta.z });
    size_t i = delta_max;

    std::vector<glm::vec<3, size_t>> cells {};
    svec3 c_start = start.extremum.pos;
    svec3 c_end { std::make_signed_t<size_t>(delta_max / 2) };
    for (;;) {
        cells.emplace_back(c_start);
        if (i-- == 0)
            break;
        c_end -= delta;
        svec3 mask = glm::lessThan(c_end, svec3(0));
        c_end += mask * svec3(std::make_signed_t<size_t>(delta_max));
        c_start += mask * s;
    }

    return cells;
}

template <size_t N>
ExtremaGraph<N>::ExtremaGraph(std::vector<Extrema<N>>& extrema)
    : m_nodes {}
{
}

template <size_t N>
std::vector<glm::vec<N, size_t>> ExtremaGraph<N>::query_starting_points(
    float iso_value,
    std::predicate<glm::vec<N, size_t>, float> auto& iso_value_in_cell) const
{
    std::vector<glm::vec<N, size_t>> starting_points {};
    return starting_points;
}