import itertools
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
from math import *
import scipy
from tqdm import tqdm
import copy

from utils import *
from scipy.optimize import root_scalar

DIEOUT_BOUND = 1e-8
INVADE_BOLUS = 1e-8

# lag with coutilizers: all the same span for each lag -- except for the initial lag, 
# which is more by a diff tau0*log(nR)
class CoUt_alt: # coutilizer
    def __init__(self, rho, g_enz, gC, biomass, id):
        '''
        growth_Rate_list_single: float array[1, N_res]
        gC: float
        biomass: float
        '''
        self.cat = "Cout"
        self.id = id
        self.rho = rho
        self.alive = True
        self.nr = len(g_enz)
        self.g = g_enz # g here is already converted from gaussian and by Terry's physiology
        self.gC = gC
        self.b = biomass
        self.eating = np.array([False for i in range(self.nr)]) # default - not eating anything
        self.lag_left = 0.0
    def Dilute(self, D):
        '''
        D: float, dilution factor
        '''
        self.b /= D
        if(self.b<DIEOUT_BOUND):
            self.alive = False
    def GetEating(self, Rs):
        '''
        Rs: float array of all resources
        '''
        if(self.lag_left > 0 and len([Ri for Ri in Rs if Ri<=0])==0): # initial lag
            self.eating = (Rs>0)*False
        else:
            self.eating = (Rs>0)
    def GetGrowthRate(self): # growth rate of the sepcies with non-zero resources in R_left
        g_vec = self.g[self.eating]
        n = len(self.g)
        n_eat = np.sum(self.eating)
        if(True not in self.eating): # including the case of initial lag
            return 0
        else:
            if (self.lag_left>0):
                # return ( ( np.sum(g_vec) * (self.rho*(n_eat+1) + (1-self.rho)*n)/(n_eat+1) )**-1 + 1/self.gC )**-1
                return 0
            else:
                return ( ( np.sum(g_vec) * (self.rho*n_eat + (1-self.rho)*n)/n_eat )**-1 + 1/self.gC )**-1
    def GetDep(self):
        '''
        In all cases assume yield Y=1
        Get the fraction of each resource in the biomass gained by this co-utilizer
        '''
        dep = np.zeros(self.nr)
        dep[self.eating] = self.g[self.eating]/np.sum(self.g[self.eating])
        return dep
    def SetLag(self, tau0, tau_f, tau_init):
        '''
        tau0: coefficient of lag. 
        tau_f: function that takes tau0 , rho and Rs as input and returns the lag time. 
        tau_init: function that takes tau0, rho and Rs as input and returns the initial lag time. 
        '''
        self.tau0 = tau0
        self.tau_f = tau_f
        self.tau_init = tau_init
    def GetLag(self, R_dep, Rs): # update lag state after 1 resource gets depleted
        '''
        R_dep: int, the last depleted resource
        Rs: float array of all resources. At this point, R_dep should be 0 in Rs. 
        '''
        if(np.sum(Rs)==0): # if no more resources present, straight up rezero everything
            self.lag_left = 0.0
            return 0
        if(self.lag_left==0):
            self.lag_left += self.tau_f(Rs, self.rho, self.tau0)
        else:
            self.lag_left += self.tau_f(Rs, self.rho, self.tau0)
    ## apply the initial lag in the simplest way
    def RezeroLag(self):
        # tau_init = self.tau_init(np.ones(self.nr), self.rho, self.tau0)
        self.lag_left = self.tau_init
    
class SeqUt_alt: # sequential utilizer, this time with lags
    def __init__(self, rho, g_enz, gC, pref_list, biomass, id):
        '''
        growth_Rate_list_single: float array[N_res], generated from gaussian
        pref_list: int array[N_res]
        biomass: float
        '''
        self.cat = "Seq"
        self.id = id
        self.gC = gC
        self.rho = rho
        self.alive = True
        self.nr = len(g_enz)
        self.g = g_enz # g here is already converted from gaussian and by Terry's physiology
        self.pref = pref_list
        self.b = biomass
        self.eating = np.array([False for i in range(self.nr)]) # default - not eating anything
        self.lag_from = -1
        self.lag_to = -1
        self.lag_left = 0.0
    def Dilute(self, D):
        '''
        D: float
        '''
        self.b /= D
        if(self.b<DIEOUT_BOUND):
            self.alive = False
    def GetEating(self, Rs):
        '''
        Rs: float array of all resources
        '''
        self.eating = np.array([False for i in range(self.nr)])
        if(self.lag_left==0):
            for r in self.pref:
                if(Rs[r-1]>0):
                    self.eating[r-1] = True
                    break
    def GetGrowthRate(self):
        if(self.lag_left>0):
            return 0
        else:
            n = len(self.g)
            gtilde = self.g * (self.rho + (1-self.rho)*n)
            return (gtilde/(1+gtilde/self.gC)) @ self.eating
    def GetDep(self):
        '''
        In all cases assume yield Y=1
        '''
        return self.eating.astype(float)
    def SetLag(self, tau0, tau_f, tau_init):
        '''
        taus: np.array(R, R)
        '''
        self.tau0 = tau0
        self.tau_f = tau_f
        self.tau_init = tau_init
    def GetLag(self, R_dep, Rs): # update lag state after 1 resource gets depleted
        '''
        R_dep: int, the last depleted resource
        Rs: float array of all resources. At this point, R_dep should be 0 in Rs. 
        '''
        if(np.sum(Rs)==0): # if no more resources present, straight up rezero everything
            self.lag_from = -1
            self.lag_to = -1
            self.lag_left = 0.0
            return 0
        if(self.lag_left==0):
            if(self.eating[R_dep-1]>0):
                self.lag_from = R_dep
                for r in self.pref:
                    if(Rs[r-1]>0):
                        self.lag_to = r
                self.lag_left += self.tau_f(Rs, self.rho, self.tau0)
        else:
            if(R_dep==self.lag_to):
                for r in self.pref:
                    if(Rs[r-1]>0):
                        self.lag_to = r
                self.lag_left += self.tau_f(Rs, self.rho, self.tau0)
    ## apply the initial lag in the simplest way
    def RezeroLag(self):
        self.lag_from = self.pref[-1]
        self.lag_to = self.pref[0]
        # tau_init = self.tau_init(np.ones(self.nr), self.rho, self.tau0)
        self.lag_left = self.tau_init

class EcoSystem: 
    def __init__(self, species=[]):
        '''
        Rs_init: float array [N_res]
        species: list of species; species are SeqUt, CoUt etc examples
        '''
        self.res = np.array([])
        self.species = species
        for species in self.species:
            species.alive = True
        self.last_cycle = {'ids':[species.id for species in self.species], 'ts':[], 'cs':[], 'bs':[]}
    def OneCycle(self, R0, T_dilute):
        '''
        R0: float array [N_res] added in this cycle
        T_dilute: float, cutoff of dilution time
        At the beginning of each cycle, res are at the scale of 1 and species are at the scale of 1/D
        '''
        self.species = [species for species in self.species if species.alive]
        ts, cs, bs = [], [], []
        t_switch = 0
        self.res = copy.deepcopy(R0)
        nr = len(R0)
        state_flag = -1 # -1: nothing happens; -2: finished a lag; >=0: a resource is depleted. 
        ts.append(t_switch)
        cs.append(copy.deepcopy(self.res))
        bs.append(np.array([species.b for species in self.species]))
        while t_switch < T_dilute:
            # print(t_switch)
            t_step = T_dilute - t_switch
            for species in self.species:
                species.GetEating(self.res)
                if (species.lag_left>0):
                    t_i = species.lag_left
                    t_step = min(t_step, t_i)
                    state_flag = -2
            for r_id, r in enumerate(self.res):
                if r>0:
                    def remain(t):
                        return r - sum([species.b * (exp(species.GetGrowthRate()*t)-1) * species.GetDep()[r_id] for species in self.species])
                    if remain(t_step)<0:
                        t_i = root_scalar(remain, bracket = [0, t_step], method='brenth').root
                        t_step = t_i
                        state_flag = r_id
            t_switch = t_switch + t_step
            # update the system according to the t_step
            
            if (state_flag==-2): # if it's a lag
                # first update res before species, because we need b at the previous timepoint for res change
                for r_id, r in enumerate(self.res):
                    self.res[r_id] = r - sum([species.b * (exp(species.GetGrowthRate()*t_step)-1) * species.GetDep()[r_id] for species in self.species])
                # update the species abundance and their lag_left
                for species in self.species:
                    if(species.lag_left>0):
                        # if(species.cat=="Cout"):
                        #     species.b = species.b * exp(species.GetGrowthRate()*t_step)
                        species.lag_left = species.lag_left - t_step
                    else:
                        species.b = species.b * exp(species.GetGrowthRate()*t_step)

            elif(state_flag>-1): # if it's a depletion of resource
                for r_id, r in enumerate(self.res):
                    self.res[r_id] = r - sum([species.b * (exp(species.GetGrowthRate()*t_step)-1) * species.GetDep()[r_id] for species in self.species])
                    self.res[state_flag] = 0
                # update species abundance; update lag_left
                for species in self.species:
                    if(species.lag_left>0):
                        species.lag_left = species.lag_left - t_step
                        # if(species.cat=="Cout"):
                        #     species.b = species.b * exp(species.GetGrowthRate()*t_step)
                    else:
                        species.b = species.b * exp(species.GetGrowthRate()*t_step)
                    species.GetLag(state_flag+1, self.res)
                    # print("B:", species.b)
                # print("R:", self.res)
            else:
                for r_id, r in enumerate(self.res):
                    self.res[r_id] = r - sum([species.b * (exp(species.GetGrowthRate()*t_step)-1) * species.GetDep()[r_id] for species in self.species])
                # update species abundance and lag_left
                for species in self.species:
                    species.b = species.b * exp(species.GetGrowthRate()*t_step)
                    if(species.lag_left>0):
                        species.lag_left = species.lag_left - t_step
            state_flag = -1
            ts.append(t_switch)
            cs.append(copy.deepcopy(self.res))
            bs.append(np.array([species.b for species in self.species]))
        self.last_cycle = {'ids':[species.id for species in self.species], 'ts':ts, 'cs': cs, 'bs': bs}
    def MoveToNext(self, D):
        '''
        D: float, dilution rate
        This does not include adding new resources
        '''
        self.res /= D
        for species in self.species:
            species.Dilute(D)
            species.RezeroLag()
    def CheckInvade(self, invader, D):
        '''
        invader: a species
        D: float, dilution rate
        '''
        # TBD
        if(len(self.species) == 0):
            return True
        if(invader.id in [species.id for species in self.species]):
            return False
        ts, cs= self.last_cycle["ts"], self.last_cycle["cs"],
        growth = 0
        invader.RezeroLag()
        for idx, t_pt in enumerate(ts[:-1]):
            if(np.sum(cs[idx])>0):
                delta_t = ts[idx+1] - ts[idx]
                if(invader.lag_left==0):
                    invader.GetEating(cs[idx])
                    growth += invader.GetGrowthRate()*delta_t
                    # update the lag state
                    for r in range(len(self.res)):
                        if(cs[idx][r]!=0 and cs[idx+1][r]==0):
                            invader.GetLag(r+1, cs[idx+1])
                else:
                    if(invader.lag_left >= delta_t):
                        invader.lag_left -= delta_t
                    else:
                        invader.lag_left = 0
                        delta_t -= invader.lag_left
                        invader.GetEating(cs[idx])
                        growth += invader.GetGrowthRate()*delta_t
                        # update the lag state
                        for r in range(len(self.res)):
                            if(cs[idx][r]!=0 and cs[idx+1][r]==0):
                                invader.GetLag(r+1, cs[idx+1])
        return growth>log(D)
    def Invade(self, invader):
        '''
        invader: a species
        '''
        invader.alive = True
        invader.b = INVADE_BOLUS
        self.species.append(invader)

# make some function about plotting the biomass across cycles
def vis_biomass(id_list, blist):
    '''
    id_list: list of list of int; each element is the ID of species that are present at the end of a cycle
    blist: list of np.array of float; each element is the abundance of species at the end of a cycle
    '''
    all_keys = set(sum(id_list, []))
    all_info_dict = {key:[] for key in all_keys}
    for cycle, ids in enumerate(id_list):
        for key in all_keys:
            all_info_dict[key].append(0)
        for idx, id in enumerate(ids):
            all_info_dict[id][-1] = blist[cycle][idx]
    for key in all_info_dict:
        plt.plot(range(len(blist)), all_info_dict[key])
    plt.xlabel("Dilution cycles")
    plt.ylabel("Species abundance")


def TaufSeq(Rs, rho, tau0):
    nr = len(Rs)
    return tau0*log((1-(nr-1)*rho/nr)/(rho/nr))
def TaufCout(Rs, rho, tau0):
    nr = len(Rs)
    nr_pres = np.sum(Rs>0)
    if(nr_pres==nr):
        # initial lag; replaced with tau_init
        return tau0*log((1/nr)/(rho/nr))
    else:
        return tau0*log((1-(nr-nr_pres)*rho/nr)/(1-(nr-nr_pres-1)*rho/nr) * (nr_pres+1)/nr_pres)
    


