import itertools
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from math import *
import scipy
import sys

from utils import *
from lag_test_utils import *

# structural stability -- rewrite the prev code for the objects
def G_mat_hwa(spc_list, dep_order):
    '''
    spc: species obbject list of length N=R
    dep_order: array([R])
    '''
    R = len(dep_order)
    N = R
    G = np.zeros([N, R])
    for i_n, spc in enumerate(spc_list):
        for i_t in range(R):
            Rs = np.array([1*(res+1 in dep_order[i_t:]) for res in range(R)])
            spc.lag_left=0
            spc.GetEating(Rs)
            G[i_n, i_t] = spc.GetGrowthRate()            
    return G

def NicheGrowthState(spcs, dep_order, t_order):
    '''
    spcs: list of species (object; SeqUt_alt or CoUt_alt) 
    dep_order: np array; order of resources (1 to R) depleted
    t_order: np array; time spans of each temporal niche. 
    '''
    # RE: resource_eating for [spc, niche], 
    # in an np array with boolean list of elements
    N, R = len(spcs), len(dep_order)
    RE = np.zeros([N, R], dtype=object)
    for i_n in range(N):
        for i_t in range(R):
            Rs = np.zeros(R)
            for res in dep_order[i_t:]:
                Rs[res-1] = 1
            spcs[i_n].lag_left = 0
            spcs[i_n].GetEating(Rs)
            RE[i_n, i_t] = spcs[i_n].eating
    # here we define a niche growth state matrix S, whose elements can be 0, 1, or 2
    # 0 means no growth; 1 means growth during full niche; 2 means part of of niche is taken by lag
    # and define a tau_mod - the lag time in each niche for those 2 elements in S. 
    # initialize by all niche - all growth
    S = np.ones([N, R])
    tau_mod = np.zeros([N, R])
    for i_n, spc in enumerate(spcs):
        in_lagphase = 0
        tau_new = 0
        for niche in np.arange(0, R):
            # if next time niche is a different resource, 
            # we assume the species would renew the current lag by the lagtime of
            # switching from the previous (actively consuming) nutrient to the current nutrient
            # regardless of whether it was already in the middle of a lag phase or not. 
            if(niche==0 or not np.array_equal(RE[i_n, niche], RE[i_n, niche-1])):
                # update the lag
                if(niche==0):
                    tau_new = spc.tau_init
                else:
                    Rs = np.array([int(i+1 in dep_order[niche:]) for i in range(R)]) # if res is in leftover resources then 1; else 0
                    tau_new = spc.tau_f(Rs, spc.rho, spc.tau0)
                # check if this lag outlasts the t-niche
                if(tau_new < t_order[niche]):
                    S[i_n, niche] = 2
                    tau_mod[i_n, niche] = tau_new
                    in_lagphase = 0
                else:
                    S[i_n, niche] = 0
                    tau_new -= max(0, t_order[niche])
                    in_lagphase = 1
            # if the next time niche is a same resource, 
            # when the species has not yet finished a lag, it would continue being in that lag
            # with no need to renew the lagtime value. 
            elif(in_lagphase == 1):
                if(tau_new < t_order[niche]):
                    S[i_n, niche] = 2
                    tau_mod[i_n, niche] = tau_new
                    in_lagphase = 0
                else:
                    S[i_n, niche] = 0
                    tau_new -= max(0, t_order[niche])
            # else: the spc was growing and the next niche it's eating the same stuff. S=1 and tau_mod=0. 
    return S, tau_mod

def TsolveIter(spc_list, dep_order, logD, T=24): # N = R must be ensured
    # species-niche growth state S
    N = len(spc_list)
    R = N
    S = np.ones([N, R]) * 2 # initialize S at 2. 0 is all lag in this niche and 1 is all growth
    # before the first iter
    converged = 0
    t_iter_compare = np.zeros(R)
    G = G_mat_hwa(spc_list, dep_order)
    # for the first iteration of S and tau_mod, go with all t_order=max
    S, tau_mod = NicheGrowthState(spc_list, dep_order, np.ones(R)*T)
    # to keep time short do 10 iters at most
    for count in range(10): 
        # rhs: [Gt = logD]'s rhs, vec of len R
        rhs = logD + np.diag( (G*(S>0)) @ np.transpose(tau_mod) )
        if(np.linalg.matrix_rank(G*(S>0))>=N):
            t_iter = np.linalg.inv(G*(S>0))@rhs
            # update S and tau_mod based on this set of new t_iter
            S, tau_mod = NicheGrowthState(spc_list, dep_order, t_iter)
            if ((t_iter_compare==t_iter).all() and np.sum(t_iter)<=24):
                converged = 1
                break
            t_iter_compare = t_iter
        else:
            converged = 0
            break
    return converged, t_iter_compare, S, tau_mod

def FMatLag(spcs, dep_order, t_order, S, tau_mod):
    N, R = len(spcs), len(dep_order)
    F_mat = np.zeros([N, R])
    for i_n in range(N):
        coeff = 1
        for i_t in range(R):
            Rs = np.zeros(R)
            for res in dep_order[i_t:]:
                Rs[res-1] = 1
            spcs[i_n].GetEating(Rs)
            g = spcs[i_n].GetGrowthRate()
            vec_dep = spcs[i_n].GetDep()
            if(S[i_n, i_t] == 1):
                dep = coeff * (exp(g*t_order[i_t]) - 1)
                coeff *= exp(g*t_order[i_t])
            elif S[i_n, i_t] == 2:
                dep = coeff * (exp(g*(t_order[i_t]-tau_mod[i_n, i_t])) - 1)
                coeff *= exp(g*(t_order[i_t]-tau_mod[i_n, i_t]))
            else:
                dep=0
            F_mat[i_n, :] += dep * vec_dep
    return np.transpose(F_mat)



if __name__=="__main__":
    ind = int(sys.argv[1])
    stabilities = []
    parameters_list = [[100, 0.1, lambda x: np.random.uniform(0.2, 0.4)], 
                       [10000, 0.1, lambda x: np.random.uniform(0.2, 0.4)], 
                       [1000, 0.075, lambda x: np.random.uniform(0.2, 0.4)], 
                       [1000, 0.2, lambda x: np.random.uniform(0.2, 0.4)], 
                       [1000, 0.1, lambda x: np.random.uniform(0, 0)], 
                       [1000, 0.1, lambda x: np.random.uniform(0.05, 0.1)], 
                       [1000, 0.1, lambda x: np.random.uniform(0.4, 0.8)]]
    n_Nseq = 5
    Nseq = ind % n_Nseq
    Ncout = 4 - Nseq
    parameters_ind = ind // n_Nseq
    D, gsigma, tau_0_dist = parameters_list[parameters_ind]
    tau_init_dist = lambda x: np.random.uniform(2, 3)
    N_community = 10000
    gmean, gC, R, T_dilute = 0.5, 1.0, 4, 24
    n_rho = 40
    rholist = np.linspace(1e-4, 0.4, n_rho)
    rho = rholist[14]

    for i in range(N_community):
        g_seq = generate_g(Nseq, R, gmean, gsigma)
        g_cout = generate_g(Ncout, R, gmean, gsigma)
        permutations = list(itertools.permutations(list(range(1, R+1))))
        pref_list = np.array(random.choices(permutations, k=Nseq))
        col = np.argmax(g_seq, axis=1) # find column in g where it's the largest
        for row_i, row in enumerate(pref_list):
            index = np.where(row==col[row_i]+1)[0][0] # and switch this resource with the first-consumed one
            row[0], row[index] = row[index], row[0]
        species_list = []
        for i in range(Nseq):
            species_list.append(SeqUt_alt(rho=rho, g_enz=g_seq[i], gC=gC, pref_list=pref_list[i], biomass=0.01, id=i))
        for i in np.arange(Nseq, Nseq+Ncout):
            species_list.append(CoUt_alt(rho=rho, g_enz=g_cout[i-Nseq], gC=gC, biomass=0.01, id=i))
        for species in species_list:
            tau0 = tau_0_dist(0)
            tau_init = tau_init_dist(0)
            if(species.cat=="Seq"):
                species.SetLag(tau0, TaufSeq, tau_init)
                species.b = 0.01
            else:
                species.SetLag(tau0, TaufCout, tau_init)
                species.b = 0.01
            species.RezeroLag()
        dep_orders = np.array(permutations)
        N_dep_order = len(dep_orders)
        for j in range(N_dep_order):
            dep_order = dep_orders[j]
            G = G_mat_hwa(species_list, dep_order)
            if(np.linalg.matrix_rank(G)>=Nseq+Ncout):
                converged, t_iter, S, tau_mod = TsolveIter(species_list, dep_order, logD=np.log(D), T=24)
                if(converged == 1 and sum(t_iter>0) >= len(t_iter)):
                    F = FMatLag(species_list, dep_order, t_iter, S, tau_mod)
                    if(abs(np.linalg.det(F))>1e-10):
                        stabilities.append( (log(np.abs(np.linalg.det(F))/(exp(log(D))-1)**R)/log(10))/(R-1) )
    pickle.dump(stabilities, open(f"stabilities_Nseq={Nseq}_parameters={parameters_ind}.pkl", "wb"))