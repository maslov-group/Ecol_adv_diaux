from lag_test_utils import *
import sys

T_dilute = 24
STEADY_CRIT = 1e-6
MAX_CYCLE = 5e3
MIN_RHO = 1e-200
gmean, gsigma, gC, R, D = 0.5, 0.1, 1.0, 4, 1000
tau_init_dist = lambda x: np.random.uniform(2, 3)
tau_0_dist = lambda x: np.random.uniform(0.2, 0.4)

def GeneratePool(N, rho):
    # generate all species
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
        species_list.append(SeqUt_alt(rho=rho, g_enz=g_seq[i], gC=gC, pref_list=pref_list[i], biomass=0.01, id=i))
    for i in np.arange(Nseq, Nseq+Ncout):
        species_list.append(CoUt_alt(rho=rho, g_enz=g_cout[i-Nseq], gC=gC, biomass=0.01, id=i))
    # find the 8 best species to be ranked at the top of invasion list
    # in order to accelerate the simulation
    for spc in species_list:
        spc.GetEating(np.ones(4))
    def get_top_n(lst, f, g, y, N=4):
        # Filter the list by condition g(x) = y
        filtered_lst = [x for x in lst if g(x) == y]
        # Sort the filtered list by f(x) in descending order
        sorted_lst = sorted(filtered_lst, key=f, reverse=True)
        return sorted_lst[:N]
    new_species_list = []
    # best diaux for 4 res
    for i in range(R): 
        new_species_list.extend(get_top_n(species_list[:Nseq], f=lambda x: x.GetGrowthRate(), 
                                          g = lambda x: x.pref[0], y=i+1, N=1))
    # 4 best cout on 1st niche
    new_species_list.extend(get_top_n(species_list[Nseq:], f=lambda x: x.GetGrowthRate(),
                                        g = lambda x: 1, y=1, N=4))
    random.shuffle(species_list)
    random.shuffle(new_species_list)
    species_list = new_species_list + [spc for spc in species_list if spc not in new_species_list]
    # assign lags to all the species
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

def RunTest(rho, n_community, pool_size):
    output = {}
    params = {"gmean":gmean, "gsigma":gsigma, "gC":gC, "R":R, "R":D}
    data = []
    for ic in range(n_community):  
        data.append([])
        # make the random supply of species -- uniformly sample on symplex
        cuts = np.sort(np.random.rand(R-1))
        cuts = np.insert(cuts, 0, 0)
        cuts = np.insert(cuts, R, 1)
        Rs = (cuts[1:]-cuts[:-1])*R
        pool0 = GeneratePool(pool_size, rho)
        history = AssemblePool(pool0, Rs)
        data[-1].append(history[-1])
    output["params"]=params
    output["data"]=data
    return output

if __name__=="__main__":
    note = "tau_init=2,3; tau_0=0.2,0.4"
    ind = int(sys.argv[1])
    n_rho = 40
    rho_ind = ind % n_rho
    pool_size_ind = ind // n_rho
    rho = np.linspace(1e-4, 0.4, n_rho)[rho_ind]
    pool_size = [2, 5, 10, 20, 50, 100, 200, 500, 1000][pool_size_ind]
    n_community = [2000, 2000, 1000, 500, 200, 200, 200, 200, 200][pool_size_ind]
    output = RunTest(rho, n_community, pool_size)
    output["note"]=note
    pickle.dump(output, open(f"../data/seq_vs_co_lag_new/rho={rho}_size={pool_size}.pkl", "wb"))
