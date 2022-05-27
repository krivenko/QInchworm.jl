using .PyPlot

function _make_arc(x1, x2; kwargs...)
    width = x2 - x1
    height = width
    arc = matplotlib.patches.Arc((x2 - width/2, 0.0), width, height, theta2=180.0; kwargs...)
    return arc
end

function plot_topology!(ax, t::QInchworm.diagrammatics.Topology, k::Int = 0)
    vertices = range(0.0, 1.0, 2*t.order)
    ax.plot(vertices, zero(vertices), "k.")
    for p in t.pairs
      if p.first <= k || p.second <= k
        style = Dict(:edgecolor => "red")
      else
        style = Dict(:edgecolor => "black")
      end
      arc = _make_arc(vertices[p.first], vertices[p.second]; style...)
      ax.add_patch(arc)
    end
end
