import itertools
import numpy as np
import random
import pickle
from math import *
from scipy.stats import pearsonr


# generate g
def generate_g(N, R, mu=0.5, sigma=0.1):
    g = np.random.normal(mu, sigma, (N, R))
    cutoff = mu-0.01
    g = np.clip(g, a_min=mu-cutoff, a_max=mu+cutoff)
    return abs(g)

# generate x for the co-utilizers
def generate_x(N, R):
    # x = np.random.rand(N, R)
    mu, sigma = 1, 0.2
    x = np.random.normal(mu, sigma, (N, R))
    cutoff = mu - 4 * sigma
    x = np.clip(x, a_min=cutoff, a_max=None)
    norms = x.sum(axis=1)
    return x / norms[:, np.newaxis]

# randomly generate pref orders (not used for smart bugs)
def make_preference_list(N, R):
    permutations = list(itertools.permutations(list(range(1, R+1))))
    preference_list = random.choices(permutations, k=N)
    return np.array(preference_list)

# generate preference orders for smart bugs with given g matrix
def smart_preference_list(g):
    return np.argsort(-g, axis=1) + 1

# a hand-waving pref order complementarity
def comp(pref, N):
    unique_counts = [len(np.unique(pref[:, col])) for col in range(pref.shape[1])]
    return sum(unique_counts)/N**2

# separate complementarity on each resource
def comp_sep(pref, N):
    unique_counts = [len(np.unique(pref[:, col]))/N for col in range(pref.shape[1])]
    return unique_counts

# check allowed depletion orders for a certain set of bugs
def allowed_orders(pref_list):
    R = len(pref_list[0])
    permutations = list(itertools.permutations(list(range(1, R+1))))
    allowed_orders_list = []
    for i in permutations:
        mark=0
        temp_list = [j for j in pref_list]
        while(mark<R):
            res_pool = [j[0] for j in temp_list]
            if(i[mark] not in res_pool):
                mark=-1
                break
            else:
                temp_list = [[k for k in j if k!= i[mark]] for j in temp_list]
                mark+=1
        if(mark != -1):
            allowed_orders_list.append(i)
    return allowed_orders_list

# for a given depletion order, generate G matrix
# here we only consider 1 season setup -- where nutrients just deplete one by one. 
def G_mat(g, pref, dep_order, N, R):
    G = np.zeros([N, R])
    for i_n in range(N):
        for i_t in range(R):
            for i in range(R):
                top_resource = pref[i_n][i]
                if(top_resource in dep_order[i_t:]):
                    break
            G[i_n, i_t] = g[i_n, top_resource-1]
    return G
# G_mat for co-utilizers
def G_mat_co(g, x, dep_order, N, R):
    G = np.zeros([N, R])
    for i_n in range(N):
        for i_t in range(R):
            present_res = dep_order[i_t:]-1
            if(sum(x[i_n, present_res])) == 0:
                G[i_n, i_t] = 0
            else: G[i_n, i_t] = g[i_n, present_res]@x[i_n, present_res] / sum(x[i_n, present_res])
    return G


# find the corresponding t values.
def t_niches(g, pref, dep_order, logD, N, R):
    G = G_mat(g, pref, dep_order, N, R)
    return np.linalg.inv(G)@np.ones(R)*logD

# make the resource-to-species conversion matrix based on t's. 
def F_mat(g, pref, dep_order, logD, N, R):
    F_mat = np.zeros([R, N])
    G = G_mat(g, pref, dep_order, N, R)
    t = np.linalg.inv(G)@np.ones(R)*logD
    for i_n in range(N):
        coeff = 1
        start = 0
        for i in range(R):
            if(start < R and pref[i_n][i] in dep_order[start:]):
                end = dep_order.index(pref[i_n][i]) + 1
                delta_t = sum(t[start:end])
                g_temp = g[i_n, pref[i_n][i]-1]
                F_mat[pref[i_n][i]-1, i_n] = coeff * (exp(g_temp*delta_t) - 1)
                coeff *= exp(g_temp*delta_t)
                start = end
            else:
                continue
    return F_mat

# F_mat for co utilizers
def F_mat_co(g, x, dep_order, logD, N, R):
    F_mat = np.zeros([R, N])
    G = G_mat_co(g, x, dep_order, N, R)
    t = np.linalg.inv(G)@np.ones(R)*logD
    for i_n in range(N):
        coeff = 1
        for i_t in range(R):
            present_res = dep_order[i_t:]-1
            for r in present_res:
                # here need to consider if g@x==0
                if( g[i_n, present_res]@x[i_n, present_res] == 0 ):
                    F_mat[r][i_n] += 0
                else:
                    F_mat[r][i_n] += coeff * (( x[i_n, r]*g[i_n, r] / (g[i_n, present_res]@x[i_n, present_res]) ) * (exp(G[i_n, i_t]*t[i_t]) - 1) )
            coeff *= exp(G[i_n, i_t]*t[i_t])
    return F_mat

# alternative F if we know t niches -- this is for R>N
def F_mat_alt(g, pref, dep_order, t, N, R):
    # print(g, pref, dep_order, t)
    F_mat = np.zeros([R, N])
    for i_n in range(N):
        coeff = 1
        start = 0
        for i in range(R):
            if(start < R and pref[i_n][i] in dep_order[start:]):
                end = dep_order.index(pref[i_n][i]) + 1
                delta_t = sum(t[start:end])
                g_temp = g[i_n, pref[i_n][i]-1]
                F_mat[pref[i_n][i]-1, i_n] = coeff * (exp(g_temp*delta_t) - 1)
                coeff *= exp(g_temp*delta_t)
                start = end
            else:
                continue
    return F_mat

# binning for plots. 
# input: one or many sets of (x, y). like:
# x = [[first set of var_x], [another set of var_x], ...]
# y = [[first set of var_y], ...]
# output: for each section in x, what is the mean and err of y. 
def binning(x, y, n_bins):
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    bin_indices = np.digitize(x, bins)
    bin_means = np.array([[np.mean(y[j][bin_indices[j] == i]) for i in range(1, n_bins + 1)] for j in range(y.shape[0])])
    bin_err = np.array([[np.std(y[j][bin_indices[j] == i])/sqrt(len(y[j][bin_indices[j] == i])) for i in range(1, n_bins + 1)] for j in range(y.shape[0])])
    return bins, bin_means, bin_err

# binning but only for histogram.
# input: one or many sets of x.
# output: for each section of x, what is the frequency. 
def bin_hist(x, n_bins):
    x = np.atleast_2d(x)
    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    histlist = []
    for i in range(x.shape[0]):
        hist, _ = np.histogram(x[i, :], bins=bins)
        histlist.append(hist)
    return bins[:-1], np.array(histlist)


# dynamical stability mod for the diauxers
def b_to_b(g, dep_order, G, t, F, env, i, j):
    effect = int(i==j)
    R = env["R"]
    # how B changes T
    term1 = np.zeros(R)
    for k in range(1, R+1):
        ind = dep_order.index(k)
        B_list = np.exp(G[:, :ind+1]@t[:ind+1]) # every bug's growth by Rk depletion
        g_list = G[:, ind] # every bug's growth rate by Rk depletion
        term1[k-1] = 1 / ( B_list[G[:, ind]==g[:, k-1]] @ g_list[G[:, ind]==g[:, k-1]] ) # only consider those bugs eating Rk
    term1 = term1 * (-F[:, i])
    # how T changes another B
    term2 = np.zeros(R)
    for k in range(1, R):
        term2[dep_order[k-1]-1] = G[j, k-1] - G[j, k]
    term2[dep_order[-1]-1] = G[j, R-1]
    effect += term1@term2
    # print(i, j, term1, term2)
    return effect

def Pert_mat(g, dep_order, G, t, F, env):
    N = env["N"]
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i, j] = b_to_b(g, dep_order, G, t, F, env, i, j)
    return P

# dynamical stability, for the co-utilizers
def b_to_b_co(g, x, dep_order, G, t, F, env, i, j):
    effect = int(i==j)
    R = env["R"]
    # how B changes T
    term1 = np.zeros(R)
    for k in range(1, R+1):
        ind = dep_order.index(k)
        B_list = np.exp(G[:, :ind+1]@t[:ind+1]) # every bug's growth by Rk depletion
        g_list = G[:, ind] # every bug's growth rate by Rk depletion
        available_resources = np.array(dep_order[ind:])-1
        coeffs = x[:, k-1]*g[:, k-1] / np.sum(g[:, available_resources]*x[:, available_resources], axis=1)
        term1[k-1] = 1 / ( (coeffs*B_list) @ g_list ) # everyone is eating Rk
    term1 = term1 * (-F[:, i])
    # how T changes another B
    term2 = np.zeros(R)
    for k in range(1, R):
        term2[dep_order[k-1]-1] = G[j, k-1] - G[j, k]
    term2[dep_order[-1]-1] = G[j, R-1]
    effect += term1@term2
    # print(i, j, term1, term2)
    return effect
def Pert_mat_co(g, x, dep_order, G, t, F, env):
    N = env["N"]
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i, j] = b_to_b_co(g, x, dep_order, G, t, F, env, i, j)
    return P

# for plotting the cloud diagram
def ticking(xlo, xhi):
    xticks= []
    xticklabels=[]
    x1, x2 = floor(xlo), ceil(xhi)
    for i in range(x1, x2):
        xticks.extend([log10(j*(10**i)) for j in range(1, 10)])
        xticklabels.append(str(10**i))
        xticklabels.extend(["" for j in range(8)])
    xticks.append(x2)
    xticklabels.append(str(10**x2))
    xticklabels = [xticklabels[idx] for idx, i in enumerate(xticks) if xlo<=i<=xhi]
    xticks = [i for i in xticks if xlo<=i<=xhi]
    return xticks, xticklabels


###################################################################################################
# everything related to terry hwa's growth theory
def g_convert(g_tilde, gC):
    return 1/(1/g_tilde+1/gC)
def g_comb(g_vec, gC): # the combined growth rate when multiple resources are present
    return ( ( g_vec @ ((1-g_vec/gC)**-1) )**-1 + 1/gC )**-1
def G_mat_co_hwa(g, gC, dep_order, N, R):
    G = np.zeros([N, R])
    for i_n in range(N):
        for i_t in range(R):
            present_res = dep_order[i_t:]-1
            G[i_n, i_t] = g_comb(g[i_n, present_res], gC)
    return G
def F_mat_co_hwa(g, gC, G, dep_order, logD, N, R):
    g_tilde = 1/(1/g-1/gC) # scales with the carbon flux
    F_mat = np.zeros([R, N])
    t = np.linalg.inv(G)@np.ones(R)*logD
    for i_n in range(N):
        coeff = 1
        for i_t in range(R):
            present_res = dep_order[i_t:]-1
            for r in present_res:
                F_mat[r][i_n] += coeff * (( g_tilde[i_n, r] / np.sum(g_tilde[i_n, present_res]) ) * (exp(G[i_n, i_t]*t[i_t]) - 1) )
            coeff *= exp(G[i_n, i_t]*t[i_t])
    return F_mat
def b_to_b_co_hwa(g, gC, dep_order, G, t, F, env, i, j):
    g_tilde = 1/(1/g-1/gC)
    effect = int(i==j)
    R = env["R"]
    # how B changes T
    term1 = np.zeros(R)
    for k in range(1, R+1):
        ind = dep_order.index(k)
        B_list = np.exp(G[:, :ind+1]@t[:ind+1]) # every bug's growth by Rk depletion
        g_list = G[:, ind] # every bug's growth rate by Rk depletion
        available_resources = np.array(dep_order[ind:])-1
        coeffs = g_tilde[:, k-1] / np.sum(g_tilde[:, available_resources], axis=1)
        term1[k-1] = 1 / ( (coeffs*B_list) @ g_list ) # everyone is eating Rk
    term1 = term1 * (-F[:, i])
    # how T changes another B
    term2 = np.zeros(R)
    for k in range(1, R):
        term2[dep_order[k-1]-1] = G[j, k-1] - G[j, k]
    term2[dep_order[-1]-1] = G[j, R-1]
    effect += term1@term2
    # print(i, j, term1, term2)
    return effect
def Pert_mat_co_hwa(g, gC, dep_order, G, t, F, env):
    N = env["N"]
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i, j] = b_to_b_co_hwa(g, gC, dep_order, G, t, F, env, i, j)
    return P

##################################################################################################################
# lag related functions
# what resource is the species eating
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