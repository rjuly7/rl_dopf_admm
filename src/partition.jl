using Pkg
Pkg.activate(".")
using Ipopt 
using PowerModels
using JuMP
using BSON
using Suppressor
using KaHyPar
using SparseArrays

# Partition a system into n areas using KaHyPar partition algorithm

# # Arguments:
# - data::Dict{String, <:Any} : dictionary contains case in PowerModel format
# - n::Int : number of areas
# - configuration::Symbol=:edge_cut : partition meteric (:edge_cut or :connectivity)
# - print_info::Bool=false : print partition algorithm information
# """
function partition_system!(data::Dict, n::Int64; configuration::Symbol=:edge_cut, print_info::Bool=false)
    nbus = length(data["bus"])
    nbranch = length(data["branch"])
    bus_index = [x.second["index"] for x in data["bus"]]
    branch_index = [x.second["index"] for x in data["branch"]]

    sort!(bus_index)
    sort!(branch_index)

    W = zeros(nbus,nbranch)

    for (i,branch) in data["branch"]
        f_bus = findfirst(x->x==branch["f_bus"], bus_index)
        t_bus = findfirst(x->x==branch["t_bus"], bus_index)
        indx = branch["index"]
        W[f_bus,indx] = 1
        W[t_bus,indx] = 1
    end
    W = SparseArrays.sparse(W)
    h = KaHyPar.HyperGraph(W)

    info = @capture_out begin
        partitions = KaHyPar.partition(h, n, configuration=configuration)
    end
    partitions = Dict([bus_index[i]=>partitions[i]+1 for i in 1:nbus])

    for (i,bus) in data["bus"]
        bus["area"] = partitions[bus["index"]]
    end

    if print_info
        println(info)
    end
end

casename = "pglib_opf_case588_sdet"
n_areas = 8

case_path = "data/$casename"*".m";

data = parse_file(case_path);

partition_system!(data, n_areas)

new_path = "data/$casename"*"_$n_areas.m";

export_matpower(new_path, data)

