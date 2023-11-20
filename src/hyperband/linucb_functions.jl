
#n i.i.d samples from some distribution defined over the hyperparameter configuration space 
function get_hyperparameter_configuration(data_area,pq_bounds,vt_bounds)
    alpha_configs = Dict(area_id => deepcopy(data_area[area_id]["alpha"]) for area_id in keys(data_area)) 
    alpha_vector = []
    for area_id in keys(data_area)
        data = data_area[area_id]
        for neighbor in keys(data["alpha"])
            for opf_var in keys(data["alpha"][neighbor])
                for i in keys(data["alpha"][neighbor][opf_var])
                    if opf_var == "pt" || opf_var == "pf" || opf_var == "qt" || opf_var == "qf"
                        aa = rand()*(pq_bounds[2]-pq_bounds[1])+pq_bounds[1] 
                        alpha_configs[area_id][neighbor][opf_var][i] = aa 
                        push!(alpha_vector, aa)
                    else
                        aa =  rand()*(vt_bounds[2]-vt_bounds[1])+vt_bounds[1] 
                        alpha_configs[area_id][neighbor][opf_var][i] = aa
                        push!(alpha_vector, aa)
                    end
                end
            end
        end
    end
    return alpha_configs, alpha_vector 
end
