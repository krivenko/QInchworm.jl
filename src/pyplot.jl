using .PyPlot

function _make_arc(x1, x2; kwargs...)
    width = x2 - x1
    height = width
    arc = matplotlib.patches.Arc((x2 - width/2, 0.0), width, height, theta2=180.0; kwargs...)
    return arc
end

function plot_topology!(ax, vertices, t::diagrammatics.Topology, k::Int = 0)
    connected, disconnected = diagrammatics.split_k_connected(t.pairs, k)
    diagrammatics.traverse_crossing_graph_dfs!(connected, disconnected)

    for p in connected
      arc = _make_arc(vertices[p.first], vertices[p.second]; edgecolor="black")
      ax.add_patch(arc)
    end

    for p in disconnected
      arc = _make_arc(vertices[p.first], vertices[p.second]; edgecolor="red")
      ax.add_patch(arc)
    end
end
