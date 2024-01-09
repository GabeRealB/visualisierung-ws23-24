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
    // assign markers to extrema
    for (size_t i = 0; i < extrema.size(); ++i) {
        this->m_nodes.push_back(ExtremaNode<N> {
            .extremum = extrema[i],
            .marker_idx = i,
            .neighbor_idxs = {},
        });
    }

    auto lookup_group_leader = [&](size_t node_idx) {
        const ExtremaNode<N>* current = &this->m_nodes[node_idx];
        size_t current_idx = node_idx;
        while (current->marker_idx != current_idx) {
            current_idx = current->marker_idx;
            current = &this->m_nodes[current_idx];
        }

        return current_idx;
    };

    // connect nodes
    for (size_t node_idx = 0; node_idx < this->m_nodes.size(); ++node_idx) {
        auto& node = this->m_nodes[node_idx];
        size_t current_group_leader_idx = lookup_group_leader(node.marker_idx);

        bool found = false;
        size_t minimal_idx = -1;
        size_t found_other_group_leader_idx = -1;
        float minimal_distance = std::numeric_limits<float>::max();
        for (size_t other_idx = 0; other_idx < this->m_nodes.size(); ++other_idx) {
            auto& other = this->m_nodes[other_idx];
            size_t other_group_leader_idx = lookup_group_leader(other.marker_idx);

            if (current_group_leader_idx == other_group_leader_idx)
                continue;

            glm::vec<N, float> node_pos = node.extremum.pos;
            glm::vec<N, float> other_node_pos = other.extremum.pos;
            const float dist = glm::distance(node_pos, other_node_pos);
            if (dist < minimal_distance) {
                minimal_idx = other_idx;
                minimal_distance = dist;
                found_other_group_leader_idx = other_group_leader_idx;
                found = true;
            }
        }

        if (found) {
            // we point the bigger group leader to the smaller one
            auto& group_leader = this->m_nodes[current_group_leader_idx];
            auto& other_group_leader = this->m_nodes[found_other_group_leader_idx];
            if (group_leader.marker_idx < other_group_leader.marker_idx) {
                other_group_leader.marker_idx = group_leader.marker_idx;
            } else {
                group_leader.marker_idx = other_group_leader.marker_idx;
            }

            // connect closest nodes uni-directionally
            node.neighbor_idxs.push_back(minimal_idx);
        }
    }
}

template <size_t N>
std::vector<glm::vec<N, size_t>> ExtremaGraph<N>::query_starting_points(
    float iso_value,
    std::predicate<glm::vec<N, size_t>, float> auto& iso_value_in_cell) const
{
    std::vector<glm::vec<N, size_t>> starting_points {};
    for (auto& node : this->m_nodes) {
        for (auto& neighbor_idx : node.neighbor_idxs) {
            auto& neighbor = this->m_nodes[neighbor_idx];
            float min = std::min(node.extremum.value, neighbor.extremum.value);
            float max = std::max(node.extremum.value, neighbor.extremum.value);

            if (min <= iso_value && iso_value <= max) {
                const auto arc = calculate_arc(node, neighbor);
                for (auto& cell : arc) {
                    if (iso_value_in_cell(cell, iso_value)) {
                        starting_points.push_back(cell);
                        break;
                    }
                }
            }
        }
    }
    return starting_points;
}