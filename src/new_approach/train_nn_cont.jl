#Next, generate dataset of 1000 
run_num = 2
n_iters = 150
n_runs = 4000
X,y = gen_dataset(n_runs,n_iters,data,alpha_config,primal_map,dual_map,model_primal,model_dual)
bson("$path/data/new_approach/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))
n_epochs = 50 
batchsize = Int(n_runs/2) 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, model_primal, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
model_dual, loss_tr, val_tr, test_loss = train_dual(n_epochs, casename, batchsize, model_dual, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))

#Next, generate dataset of 1000 
run_num = 3
n_iters = 200
n_runs = 6000
X,y = gen_dataset(n_runs,n_iters,data,alpha_config,primal_map,dual_map,model_primal,model_dual)
bson("$path/data/new_approach/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))
n_epochs = 50 
batchsize = Int(n_runs/2) 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, model_primal, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
model_dual, loss_tr, val_tr, test_loss = train_dual(n_epochs, casename, batchsize, model_dual, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))

#Next, generate dataset of 1000 
run_num = 4
n_iters = 250
n_runs = 8000
X,y = gen_dataset(n_runs,n_iters,data,alpha_config,primal_map,dual_map,model_primal,model_dual)
bson("$path/data/new_approach/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))
n_epochs = 50 
batchsize = Int(n_runs/2) 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, model_primal, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
model_dual, loss_tr, val_tr, test_loss = train_dual(n_epochs, casename, batchsize, model_dual, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))

#Next, generate dataset of 1000 
run_num = 5
n_iters = 300
n_runs = 10000
X,y = gen_dataset(n_runs,n_iters,data,alpha_config,primal_map,dual_map,model_primal,model_dual)
bson("$path/data/new_approach/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))
n_epochs = 50 
batchsize = Int(n_runs/2) 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, model_primal, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
model_dual, loss_tr, val_tr, test_loss = train_dual(n_epochs, casename, batchsize, model_dual, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))

#Next, generate dataset of 1000 
run_num = 5
n_iters = 400
n_runs = 15000
X,y = gen_dataset(n_runs,n_iters,data,alpha_config,primal_map,dual_map,model_primal,model_dual)
bson("$path/data/new_approach/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))
n_epochs = 50 
batchsize = Int(n_runs/2) 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, model_primal, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
model_dual, loss_tr, val_tr, test_loss = train_dual(n_epochs, casename, batchsize, model_dual, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))


#Next, generate dataset of 1000 
run_num = 5
n_iters = 500
n_runs = 10000
X,y = gen_dataset(n_runs,n_iters,data,alpha_config,primal_map,dual_map,model_primal,model_dual)
bson("$path/data/new_approach/$casename"*"_dataset_$run_num.bson", Dict("X" => X, "y" => y))
n_epochs = 50 
batchsize = Int(n_runs/2) 
#Train primal NN for some epochs 
model_primal, loss_tr, val_tr, test_loss = train_primal(n_epochs, casename, batchsize, model_primal, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_primal.bson", Dict(:nn => model_primal, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
model_dual, loss_tr, val_tr, test_loss = train_dual(n_epochs, casename, batchsize, model_dual, adp)
bson("$path/data/new_approach/$casename"*"_nn$nn_num"*"_dual.bson", Dict(:nn => model_dual, :test_loss => test_loss, :loss_tr => loss_tr, :val_tr => val_tr))
