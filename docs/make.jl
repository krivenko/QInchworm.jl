# QInchworm.jl
#
# Copyright (C) 2021-2023 I. Krivenko, H. U. R. Strand and J. Kleinhenz
#
# QInchworm.jl is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# QInchworm.jl is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Igor Krivenko, Hugo U. R. Strand, Joseph Kleinhenz

using QInchworm
using Documenter

makedocs(;
    modules=[QInchworm],
    authors="Igor Krivenko <igor.s.krivenko@gmail.com>, " *
            "Hugo U. R. Strand <hugo.strand@gmail.com>, " *
            "Joseph Kleinhenz <kleinhenz.joseph@gmail.com>",
    repo=Documenter.Remotes.GitHub("krivenko", "QInchworm.jl"),
    sitename="QInchworm.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    warnonly=true,
    pages=[
        "Home" => "index.md",
        "Example" => "example.md",
        "Public API" => "api.md",
        "Internals" => String[
            "modules/expansion.md",
            "modules/inchworm.md",
            "modules/ppgf.md",
            "modules/sector_block_matrix.md",
            "modules/spline_gf.md",
            "modules/utility.md",
            "modules/mpi.md",
            "modules/diagrammatics.md",
            "modules/topology_eval.md",
            "modules/qmc_integrate.md",
            "modules/randomization.md"
        ],
        "About" => "about.md"
    ]
)
