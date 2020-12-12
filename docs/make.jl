using QInchworm
using Documenter

makedocs(;
    modules=[QInchworm],
    authors="Igor Krivenko <igor.s.krivenko@gmail.com>, Hugo U. R. Strand <hugo.strand@gmail.com>",
    repo="https://github.com/krivenko/QInchworm.jl/blob/{commit}{path}#L{line}",
    sitename="QInchworm.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
