
eta = 3
R = 5
total = get_run_count(eta,R)

arrrs = []
for rst=10:total
    push!(arrrs, eta*(1-total/rst*(1-1/eta)))
end

rst = 19
arr = eta*(1-total/rst*(1-1/eta))
k = log(1/eta,arr)

#Use start_R = 19,  k = 2 