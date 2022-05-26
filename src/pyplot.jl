using .PyPlot

function _make_arc(x1, x2)
    width = x2 - x1
    height = width
    arc = matplotlib.patches.Arc((x2 - width/2, 0.0), width, height, theta2=180.0)
    return arc
end

function plot_topology!(ax, t::QInchworm.diagrammatics.Topology)
    vertices = range(0.0, 1.0, 2*t.order)
    ax.plot(vertices, zero(vertices), "k.")
    for p in t.pairs
        arc = _make_arc(vertices[p.first], vertices[p.second])
        ax.add_patch(arc)
    end
end
