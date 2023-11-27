hat_thetas = []
for i in eachindex(trace_params["V"])
    V = trace_params["V"][i]
    y = trace_params["y"][i]
    push!(hat_thetas, inv(V)*y)
end

rr = 100
inv_V = inv(trace_params["V"][rr])
hat_theta = hat_thetas[rr]

val_list = []
for a_idx=1:nv 
    a_vec = zeros(nv,1)
    a_vec[a_idx] = 1         
    push!(val_list, dot(a_vec,hat_theta) + sqrt(beta)*sqrt(transpose(a_vec)*inv_V*a_vec)[1])
end