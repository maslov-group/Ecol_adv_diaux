import sys

from utils import *
from lag_test_utils import *
from scipy.optimize import root_scalar



Nmax = 5000 
N_trials = 100
gmean, gsigma, gC, R, D, T_dilute = 0.5, 0.1, 1.0, 4, 1000, 24

def PureCommunity(Npair, rho):
    Nseq, Ncout = Npair
    outputs = []
    for idx, trial in enumerate(range(N_trials)):
        for i in range(Nmax):
            # generate comm
            flag = 0
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

# here in supp fig make all lags to be zero
            tau_init_dist = lambda x: np.random.uniform(2, 3) *0
            tau_0_dist = lambda x: np.random.uniform(0.2, 0.4)*2 *0
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
            # filter for coexistence for 4 species 
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
                            flag=1
                            break
            if(flag): # if coexist in linalg, test it in simulation
                bs = np.array([1,1,1,1])/D
                for jdx, spc in enumerate(species_list):
                    spc.b = bs[jdx]
                    spc.RezeroLag()
                Rs = np.sum(F, axis=1)/D
                C = EcoSystem(species_list)
                b_list, id_list = [], []
                for i in range(100):
                    C.OneCycle(Rs, T_dilute)
                    b_list.append(C.last_cycle['bs'][-1])
                    id_list.append(C.last_cycle['ids'])
                    C.MoveToNext(D)
                if(len(C.last_cycle["ids"])>=R):
                    break
        outputs.append(C.last_cycle)
    # print(outputs)
    return outputs

def PureSingle(Npair, rho):
    Nseq, Ncout = Npair
    outputs = []
    for idx, trial in enumerate(range(N_trials)):
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

        tau_init_dist = lambda x: np.random.uniform(2, 3) 
        tau_0_dist = lambda x: np.random.uniform(0.2, 0.4)*2 
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
        bs = np.array([1,1,1,1])/D/R
        for jdx, spc in enumerate(species_list):
            spc.b = bs[jdx]
            spc.RezeroLag()
        Rs = np.array([1, 1, 1, 1])/1.0/R
        C = EcoSystem(species_list)
        b_list, id_list = [], []
        for i in range(100):
            C.OneCycle(Rs, T_dilute)
            b_list.append(C.last_cycle['bs'][-1])
            id_list.append(C.last_cycle['ids'])
            C.MoveToNext(D)
        outputs.append(C.last_cycle)
    # print(outputs)
    return outputs

if __name__=="__main__":
    ind = int(sys.argv[1])
    n_rho = 40
    rho_ind = ind % n_rho
    Npair_ind = ind // n_rho
    
    rholist = np.linspace(1e-4, 0.4, n_rho)
    rho = rholist[rho_ind]
    comm_list = [(0, 4), (4, 0)]
    single_list = [(0, 1), (1, 0)]
    Npair = comm_list[Npair_ind]
    outputs = PureCommunity(Npair, rho)
    pickle.dump(outputs, open(f"../data/seq_vs_co_lag_new/fitness_rho={rho}_Npair={Npair}.pkl", "wb"))
    Npair = single_list[Npair_ind]
    outputs = PureSingle(Npair, rho)
    pickle.dump(outputs, open(f"../data/seq_vs_co_lag_new/fitness_rho={rho}_Npair={Npair}.pkl", "wb"))