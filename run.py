"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""

import numpy as np
import torch
import time
import pickle

from problem import get_problem
from utils import igd, rmse

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

# time slot to store rmse results
rmse_list = [25, 50, 75, 99]

# number of independent runs
n_run = 10 #20
# number of initialized solutions
n_init = 20
# number of iterations, and batch size per iteration
n_iter = 100
n_sample = 1

# PSL 
# number of learning steps
n_steps = 500
# number of sampled preferences per step
n_pref_update = 10 
# coefficient of LCB
coef_lcb = 0.1
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000 
# number of optional local search
n_local = 1
# device
device = 'cuda'
# benchmark or hyper
if_hyper = False
# -----------------------------------------------------------------------------

hv_list = {}

problem_id = [0, 4, 8]
problem_range = [4, 4, 4]

for range_id, test_id in enumerate(problem_id):
    print("Start with {}.".format(ins_list[test_id]))
    
    # get problem info
    # We only use hv records for debugging
    hv_all_value = np.zeros([n_run, n_iter])
    # Append all the info into the info list
    problem_list = []
    n_dim_list = []
    n_obj_list = []
    for temp_id in range(problem_range[range_id]):
        problem = get_problem(ins_list[test_id+temp_id])
        n_dim = problem.n_dim
        n_obj = problem.n_obj

        problem_list.append(problem)
        n_dim_list.append(n_dim)
        n_obj_list.append(n_obj)

    # get the temp storage vector
    pareto_records_list = []
    igd_records_list = []
    rmse_records_list = []
    time_list = []

    # not sure whether ref_point make sense if we use no hv
    ref_point = problem.nadir_point
    ref_point = [1.1*x for x in ref_point]
    
    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        # record the starting time
        time_s = time.time()
        
        # TODO: better I/O between torch and np
        # currently, the Pareto Set Model is on torch, and the Gaussian Process Model is on np 

        # We split the single-task evaluation into multi-task settings
        X_list = []
        Y_list = []
        Z_list = []
        for id_id, problem in enumerate(problem_list):
            # initialize n_init solutions
            x_init = lhs(n_dim_list[id_id], n_init)
            y_init = problem.evaluate(torch.from_numpy(x_init).to(device))

            X = x_init
            Y = y_init.cpu().numpy()
            z = torch.zeros(n_obj_list[id_id]).to(device)

            # Store the init results to the list
            X_list.append(X)
            Y_list.append(Y)
            Z_list.append(z)

        # We supply the IGD_records, RMSE_records, and Pareto_records
        # To enable the computation, we prepare the font_list and weight_list
        pareto_records = [torch.zeros(n_iter, 1) for i in range(problem_range[range_id])]
        igd_records = [torch.zeros(n_iter, 1) for i in range(problem_range[range_id])]
        rmse_records = []
        front_list = []
        weight_list = []

        # prepare the groud truth PF and weights for evaluation (con't)
        for task_id in range(problem_range[range_id]):
            if not if_hyper:
                weight_item, front_item = problem_list[task_id].ref_and_obj()
                front_list.append(front_item)
                weight_list.append(weight_item)

        # print("DEBUG init")
        psmodel_list = []
        optimizer_list = []
        train_input_list = []
        train_output_list = []
        # n_iter batch selections 
        for i_iter in range(n_init, n_iter):
            print("Start of iteration {}.".format(i_iter))
            for task_id in range(problem_range[range_id]):

                # intitialize the model and optimizer
                psmodel = ParetoSetModel(n_dim_list[task_id], n_obj_list[task_id])
                psmodel.to(device)
                # psmodel_list.append(psmodel)

                # optimizer
                optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-3)
                # optimizer_list.append(optimizer)

                # fetch the X and Y
                X = X_list[task_id]
                Y = Y_list[task_id]

                # solution normalization
                transformation = StandardTransform([0, 1])
                transformation.fit(X, Y)
                X_norm, Y_norm = transformation.do(X, Y)
            
                # train GP surrogate model
                surrogate_model = GaussianProcess(n_dim_list[task_id], n_obj_list[task_id], nu=5)
                surrogate_model.fit(X_norm, Y_norm)
            
                z = torch.min(torch.cat((Z_list[task_id].reshape(1, n_obj_list[task_id]),
                                         torch.from_numpy(Y_norm).to(device) - 0.1)), axis=0).values.data
                Z_list[task_id] = z
            
                # nondominated X, Y
                nds = NonDominatedSorting()
                idx_nds = nds.do(Y_norm)

                X_nds = X_norm[idx_nds[0]]
                Y_nds = Y_norm[idx_nds[0]]
            
                # t_step Pareto Set Learning with Gaussian Process
                for t_step in range(n_steps):
                    psmodel.train()

                    # sample n_pref_update preferences
                    alpha = np.ones(n_obj_list[task_id])
                    pref = np.random.dirichlet(alpha, n_pref_update)
                    pref_vec = torch.tensor(pref).to(device).float() + 0.0001

                    # get the current coressponding solutions
                    x = psmodel(pref_vec)
                    x_np = x.detach().cpu().numpy()

                    # obtain the value/grad of mean/std for each obj
                    mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
                    mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)

                    std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
                    std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).\
                        to(device)

                    # calculate the value/grad of tch decomposition with LCB
                    value = mean - coef_lcb * std
                    value_grad = mean_grad - coef_lcb * std_grad

                    tch_idx = torch.argmax((1 / pref_vec) * (value - z), axis=1)
                    tch_idx_mat = [torch.arange(len(tch_idx)), tch_idx]
                    tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update, 1) * \
                        value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis=1)

                    tch_grad = tch_grad / torch.norm(tch_grad, dim=1)[:, None]

                    # gradient-based pareto set model update
                    optimizer.zero_grad()
                    psmodel(pref_vec).backward(tch_grad)
                    optimizer.step()

                # solutions selection on the learned Pareto set
                psmodel.eval()
            
                # sample n_candidate preferences
                alpha = np.ones(n_obj_list[task_id])
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
                        candidate_mean = surrogate_model.evaluate(X_candidate_tch)['F']
                        candidate_mean_grad = \
                            surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']

                        candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
                        candidate_std_grad = \
                            surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']

                        candidate_value = candidate_mean - coef_lcb * candidate_std
                        candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad

                        candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis=1)
                        candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)), list(candidate_tch_idx)]

                        candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)), list(candidate_tch_idx)].reshape(n_candidate, 1) * \
                        candidate_grad[np.arange(len(candidate_tch_idx)), list(candidate_tch_idx)]
                        candidate_tch_grad += 0.01 * np.sum(candidate_grad, axis=1)

                        X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
                        X_candidate_tch[X_candidate_tch <= 0] = 0
                        X_candidate_tch[X_candidate_tch >= 1] = 1

                    X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])

                    Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
                    Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']

                    Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            
                # greedy batch selection
                best_subset_list = []
                Y_p = Y_nds
                for b in range(n_sample):
                    hv = HV(ref_point=np.max(np.vstack([Y_p, Y_candidate]), axis=0))
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
                Y_new = problem_list[task_id].evaluate(X_new)
            
                # update the set of evaluated solutions (X,Y)
                X = np.vstack([X, X_new.detach().cpu().numpy()])
                Y = np.vstack([Y, Y_new.detach().cpu().numpy()])

                # update the X set and Y set to the whole set
                X_list[task_id] = X
                Y_list[task_id] = Y

                # update the stats vector for supervision
                # update pareto set size
                pareto_records[task_id][i_iter, :] = X_nds.shape[0]
                # update igd value
                igd_records[task_id][i_iter] = igd(front_list[task_id], torch.from_numpy(Y))
                # print("DEBUG")
                # update rmse value
                if i_iter in rmse_list:
                    # w --> x
                    predict_x = psmodel(weight_list[task_id].to(device)).to(torch.float64).to(device)
                    # predict_x = predict_x.detach().cpu().numpy()

                    # x --> y
                    current_result = problem_list[task_id].evaluate(predict_x)
                    current_rmse, _ = rmse(front_list[task_id].to(device), current_result)
                    rmse_records.append(current_rmse)
            
            # check the current HV for evaluated solutions
            # hv = HV(ref_point=np.array(ref_point))
            # hv_value = hv(Y)
            # hv_all_value[run_iter, i_iter] = hv_value
            # in some certain iterations we generate IGD results or RMSE results
        
        # store the final performance
        # hv_list[test_ins] = hv_all_value

        # record the ending time
        time_t = time.time()
        time_list.append(time_t - time_s)
        print("*********The end of run id {}***********".format(run_iter))
        # At the end of each run
        pareto_records_list.append(pareto_records)
        igd_records_list.append(igd_records)
        rmse_records_list.append(rmse_records)

        # with open('hv_psl_mobo.pickle', 'wb') as output_file:
        #     pickle.dump([hv_list], output_file)

    my_dict = dict()
    my_dict['pareto'] = pareto_records_list
    my_dict['igd'] = igd_records_list
    my_dict['rmse'] = rmse_records_list
    my_dict['time'] = time_list
    my_dict['dim'] = n_dim_list[0]
    my_dict['obj'] = n_obj_list[0]

    torch.save(my_dict, "./server/{}_obj{}_dim{}_{}.pth".
               format(problem_list[0].current_name,
                      my_dict['obj'],
                      my_dict['dim'],
                      "PSL"))
