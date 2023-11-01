"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""

import numpy as np
import torch
import pickle

from problem import get_problem

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.surrogate_model import GaussianProcess
from mobo.transformation import StandardTransform

from model import ParetoSetModel

# -----------------------------------------------------------------------------
# list of 15 test problems, which are defined in problem.py
# ins_list = ['f1','f2','f3','f4','f5','f6',
#             'vlmop1','vlmop2', 'vlmop3', 'dtlz2',
#             're21', 're23', 're33','re36','re37']
ins_list = ['mdtlz1_4_1', 'mdtlz1_4_2', 'mdtlz1_4_3', 'mdtlz1_4_4',
            'mdtlz2_4_1', 'mdtlz2_4_2', 'mdtlz2_4_3', 'mdtlz2_4_4',
            'mdtlz3_4_1', 'mdtlz3_4_2', 'mdtlz3_4_3', 'mdtlz3_4_4']

# number of independent runs
n_run = 2 #20
# number of initialized solutions
n_init = 10
# number of iterations, and batch size per iteration
n_iter = 20 - n_init
n_sample = 1

# PSL 
# number of learning steps
n_steps = 1000 
# number of sampled preferences per step
n_pref_update = 10 
# coefficient of LCB
coef_lcb = 0.1
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000 
# number of optional local search
n_local = 1
# device
device = 'cpu'
# -----------------------------------------------------------------------------

hv_list = {}

for test_ins in ins_list:
    print(test_ins)
    
    # get problem info
    hv_all_value = np.zeros([n_run, n_iter])
    problem = get_problem(test_ins)
    n_dim = problem.n_dim
    n_obj = problem.n_obj

    ref_point = problem.nadir_point
    ref_point = [1.1*x for x in ref_point]
    
    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        
        # TODO: better I/O between torch and np
        # currently, the Pareto Set Model is on torch, and the Gaussian Process Model is on np 
        
        # initialize n_init solutions 
        x_init = lhs(n_dim, n_init)
        y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
        
        X = x_init
        Y = y_init.cpu().numpy()
    
        z = torch.zeros(n_obj).to(device)
        
        # n_iter batch selections 
        for i_iter in range(n_iter):
            
            # intitialize the model and optimizer 
            psmodel = ParetoSetModel(n_dim, n_obj)
            psmodel.to(device)
                
            # optimizer
            optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
          
            # solution normalization
            transformation = StandardTransform([0,1])
            transformation.fit(X, Y)
            X_norm, Y_norm = transformation.do(X, Y) 
            
            # train GP surrogate model 
            surrogate_model = GaussianProcess(n_dim, n_obj, nu = 5)
            surrogate_model.fit(X_norm,Y_norm)
            
            z =  torch.min(torch.cat((z.reshape(1,n_obj),torch.from_numpy(Y_norm).to(device) - 0.1)), axis = 0).values.data
            
            # nondominated X, Y 
            nds = NonDominatedSorting()
            idx_nds = nds.do(Y_norm)
            
            X_nds = X_norm[idx_nds[0]]
            Y_nds = Y_norm[idx_nds[0]]
            
            # t_step Pareto Set Learning with Gaussian Process
            for t_step in range(n_steps):
                psmodel.train()
                
                # sample n_pref_update preferences
                alpha = np.ones(n_obj)
                pref = np.random.dirichlet(alpha,n_pref_update)
                pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
                
                # get the current coressponding solutions
                x = psmodel(pref_vec)
                x_np = x.detach().cpu().numpy()
                
                # obtain the value/grad of mean/std for each obj
                mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
                mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)
                
                std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
                std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).to(device)
                
                # calculate the value/grad of tch decomposition with LCB
                value = mean - coef_lcb * std
                value_grad = mean_grad - coef_lcb * std_grad
               
                tch_idx = torch.argmax((1 / pref_vec) * (value - z), axis = 1)
                tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
                tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update,1) * \
                           value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis = 1)

                tch_grad = tch_grad / torch.norm(tch_grad, dim = 1)[:, None]
                
                # gradient-based pareto set model update 
                optimizer.zero_grad()
                psmodel(pref_vec).backward(tch_grad)
                optimizer.step()  
                
            # solutions selection on the learned Pareto set
            psmodel.eval()
            
            # sample n_candidate preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha, n_candidate)
            pref = torch.tensor(pref).to(device).float() + 0.0001
    
            # generate correponding solutions, get the predicted mean/std
            X_candidate = psmodel(pref).to(torch.float64)
            X_candidate_np = X_candidate.detach().cpu().numpy()
            Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
            
            Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            
            # optional TCH-based local Exploitation 
            if n_local > 0:
                X_candidate_tch = X_candidate_np
                z_candidate = z.cpu().numpy()
                pref_np = pref.cpu().numpy()
                for j in range(n_local):
                    candidate_mean =  surrogate_model.evaluate(X_candidate_tch)['F']
                    candidate_mean_grad =  surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']
                    
                    candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
                    candidate_std_grad = surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']
                    
                    candidate_value = candidate_mean - coef_lcb * candidate_std
                    candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad
                    
                    candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis = 1)
                    candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)]
                    
                    candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)].reshape(n_candidate,1) * candidate_grad[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)] 
                    candidate_tch_grad +=  0.01 * np.sum(candidate_grad, axis = 1) 
                    
                    X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
                    X_candidate_tch[X_candidate_tch <= 0]  = 0
                    X_candidate_tch[X_candidate_tch >= 1]  = 1  
                    
                X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])
                
                Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
                Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
                
                Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            
            # greedy batch selection 
            best_subset_list = []
            Y_p = Y_nds 
            for b in range(n_sample):
                hv = HV(ref_point=np.max(np.vstack([Y_p, Y_candidate]), axis = 0))
                best_hv_value = 0
                best_subset = None
                
                for k in range(len(Y_candidate)):
                    Y_subset = Y_candidate[k]
                    Y_comb = np.vstack([Y_p, Y_subset])
                    hv_value_subset = hv(Y_comb)
                    if hv_value_subset > best_hv_value:
                        best_hv_value = hv_value_subset
                        best_subset = [k]
                        
                Y_p = np.vstack([Y_p, Y_candidate[best_subset]])
                best_subset_list.append(best_subset)  
                
            best_subset_list = np.array(best_subset_list).T[0]
            
            # evaluate the selected n_sample solutions
            X_candidate = torch.tensor(X_candidate_np).to(device)
            X_new = X_candidate[best_subset_list]
            Y_new = problem.evaluate(X_new)
            
            # update the set of evaluated solutions (X,Y)
            X = np.vstack([X, X_new.detach().cpu().numpy()])
            Y = np.vstack([Y, Y_new.detach().cpu().numpy()])
            
            # check the current HV for evaluated solutions
            hv = HV(ref_point=np.array(ref_point))
            hv_value = hv(Y)
            hv_all_value[run_iter, i_iter] = hv_value
            # in some certain iterations we generate IGD results or RMSE results
            
            
            print("hv", "{:.2e}".format(np.mean(hv_value)))
            print("***")
        
        # store the final performance
        hv_list[test_ins] = hv_all_value
        
        print("************************************************************")

        with open('hv_psl_mobo.pickle', 'wb') as output_file:
            pickle.dump([hv_list], output_file)

