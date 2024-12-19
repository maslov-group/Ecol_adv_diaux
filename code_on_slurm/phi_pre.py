from lag_test_utils import *
import sys

T_dilute = 24
STEADY_CRIT = 1e-6
MAX_CYCLE = 5e3
MIN_RHO = 1e-200
gmean, gsigma, gC, R, D = 0.5, 0.1, 1.0, 4, 1000
tau_init_dist = lambda x: np.random.uniform(2, 3)
tau_0_dist = lambda x: np.random.uniform(0.2, 0.4)

def GeneratePool(N, rho_dist, pool_type):
    if(pool_type=="seq"):
        Nseq, Ncout = N, 0
    elif pool_type=="cout":
        Nseq, Ncout = 0, N
    elif pool_type=="mix":
        Nseq, Ncout = int(N/2), int(N/2)
    g_seq = generate_g(Nseq, R, gmean, gsigma)
    permutations = list(itertools.permutations(list(range(1, R+1))))
    pref_list = np.array(random.choices(permutations, k=Nseq))
    col = np.argmax(g_seq, axis=1) # find column in g where it's the largest
    for row_i, row in enumerate(pref_list):
        index = np.where(row==col[row_i]+1)[0][0] # and switch this resource with the first-consumed one
        row[0], row[index] = row[index], row[0]
    g_cout = generate_g(Ncout, R, gmean, gsigma)
    species_list = []
    for i in range(Nseq):
        species_list.append(SeqUt_alt(rho=rho_dist(i), g_enz=g_seq[i], gC=gC, pref_list=pref_list[i], biomass=0.01, id=i))
    for i in np.arange(Nseq, Nseq+Ncout):
        species_list.append(CoUt_alt(rho=rho_dist(i), g_enz=g_cout[i-Nseq], gC=gC, biomass=0.01, id=i))
    # assign lags
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
    random.shuffle(species_list)
    return species_list

def AssemblePool(species_list, Rs):
    C = EcoSystem([])
    history = []
    b_list, id_list = [], []
    for idx, species in enumerate(species_list):
        if(C.CheckInvade(species, D)):
            C.Invade(species)
            steady = False
            count = 0
            while count < MAX_CYCLE and steady == False:
                count += 1
                C.OneCycle(Rs, T_dilute)
                b_list.append(C.last_cycle['bs'][-1])
                id_list.append(C.last_cycle['ids'])
                if(len(b_list)>1 and len(b_list[-1])==len(b_list[-2])):
                    b_diff = np.abs((b_list[-1]-b_list[-2])/b_list[-1])
                    if(np.max(b_diff)<STEADY_CRIT):
                        steady = True
                C.MoveToNext(D)
            history.append([idx, C.last_cycle])
    return history

if __name__=="__main__":
    ind = int(sys.argv[1])
    type_list = ["seq", "cout", "mix"]
    pool_per_run = 50
    # runs = 100*3
    rho_dist = lambda x: 10**np.random.uniform(-4, -1)*4
    N = 1000
    pool_type = type_list[ind % 3]

    outputs = []
    for i in range(pool_per_run):
        spc_list = GeneratePool(N, rho_dist, pool_type)
        history = AssemblePool(spc_list, np.ones(R))
        outputs.append({"invasions": [frame[0] for frame in history], 
                        "rhos":[spc_list[frame[0]].rho for frame in history], 
                        "survivors":[frame[1]["ids"] for frame in history]})
    pickle.dump(outputs, open(f"../data/phi_pre/type={pool_type}_{ind//3}.pkl", "wb"))