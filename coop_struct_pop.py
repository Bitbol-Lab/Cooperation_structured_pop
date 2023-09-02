import numpy as np
from numba import njit
import numba as nb
import random

#=======================================================================
@njit
def count_nb_ele_nonzero(arr): 
    compt = 0
    for i in range(len(arr)):
        if arr[i] != 0:
            compt += 1
    return compt
    
@njit
def put_to0_negative_ele(arr):
    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = 0
    return arr

@njit
def initialization10(Nm0, Nw0, D): #We start from one fully mutant deme in a wild-type structure; the mutant deme corresponds to the element of index 0 in the arrays Nm and Nw
    Nm    = np.zeros(D)
    Nm[0] = Nm0
    Nw    = Nw0*np.ones(D)
    Nw[0] = 0
    #print('Initial state  ', 'Nm0: ', Nm, ' ', 'Nw0: ', Nw)
    return Nm, Nw

@njit
def initialization_lattice(Nm0, Nw0, D): #We start from one fully mutant deme in a wild-type structure; the mutant deme corresponds to the element of index 0 in the arrays Nm and Nw
    Nm    = np.zeros(D)
    Nm[4] = Nm0
    Nw    = Nw0*np.ones(D)
    Nw[4] = 0
    #print('Initial state  ', 'Nm0: ', Nm, ' ', 'Nw0: ', Nw)
    return Nm, Nw

@njit
def initialization01(Nm0, Nw0, D): #We start from one fully mutant deme in a wild-type structure; the mutant deme corresponds to the element of index D-1 in the arrays Nm and Nw
    Nm    = np.zeros(D)
    Nm[D-1] = Nm0
    Nw    = Nw0*np.ones(D)
    Nw[D-1] = 0
    #print('Initial state (t = 0): ','Nm0: ', Nm, ' ', 'Nw0: ', Nw)
    return Nm, Nw

@njit
def mig_rate_star(mO, mI, D): #We create the migration rate matrix for the star; see Fig. 1 to understand its construction
    arrmig_rate = np.zeros((D,D))
    arrmig_rate[0]   = mO*np.ones(D)
    arrmig_rate[:,0] = mI*np.ones(D)
    np.fill_diagonal(arrmig_rate, 0)
    return arrmig_rate

@njit
def mig_rate_clique(m, D): #We create the migration rate matrix for the clique; see Fig. 1 to understand its construction
    arrmig_rate = m*np.ones((D,D))
    np.fill_diagonal(arrmig_rate, 0)
    return arrmig_rate

@njit
def mig_rate_cycle(mA, mC, D): #code for D = 5, we create the migration rate matrix for the cycle; see Fig. 1 to understand its construction
    arrmig_rate = np.zeros((D,D))
    arrmig_rate[0] = np.array([0, mC, 0, 0, mA])
    arrmig_rate[1] = np.array([mA, 0, mC, 0, 0])
    arrmig_rate[2] = np.array([0, mA, 0, mC, 0])
    arrmig_rate[3] = np.array([0, 0, mA, 0, mC])
    arrmig_rate[4] = np.array([mC, 0, 0, mA, 0])
    return arrmig_rate

@njit
def mig_rate_lattice3x3(mN, mS, mE, mW, D): #code for D = 9, we create the migration rate matrix for the lattice; see Fig. 1 to understand its construction
    arrmig_rate = np.zeros((D,D))
    arrmig_rate[0] = np.array([0, mE, mW, mS, 0, 0, mN, 0, 0])
    arrmig_rate[1] = np.array([mW, 0, mE, 0, mS, 0, 0, mN, 0])
    arrmig_rate[2] = np.array([mE, mW, 0, 0, 0, mS, 0, 0, mN])
    arrmig_rate[3] = np.array([mN, 0, 0, 0, mE, mW, mS, 0, 0])
    arrmig_rate[4] = np.array([0, mN, 0, mW, 0, mE, 0, mS, 0])
    arrmig_rate[5] = np.array([0, 0, mN, mE, mW, 0, 0, 0, mS])
    arrmig_rate[6] = np.array([mS, 0, 0, mN, 0, 0, 0, mE, mW])
    arrmig_rate[7] = np.array([0, mS, 0, 0, mN, 0, mW, 0, mE])
    arrmig_rate[8] = np.array([0, 0, mS, 0, 0, mN, mE, mW, 0])
    return arrmig_rate
    
@njit
def fitnessM(b, c, w, Nm, Nw, K, D): #We create a np array containing the mutant fitnesses for each deme 
    fm = np.zeros(D)
    for i in range(D): 
        fm[i] = 1 - w + w * ((b-c)*(Nm[i]-1)/(Nm[i]+Nw[i]-1) - c*Nw[i]/(Nm[i]+Nw[i]-1))
    return fm

@njit
def fitnessW(b, c, w, Nm, Nw, K, D): #We create a np array containing the wild-type fitnesses for each deme 
    fw = np.zeros(D)
    for i in range(D):
        fw[i] = 1 - w + w * b*Nm[i]/(Nm[i]+Nw[i]-1)
    return fw
    
@njit
def compute_death_reactionrate(deathrate, D):
    return deathrate*np.ones(D)
    
@njit
def compute_division_reactionrate(b, c, w, Nm, Nw, f, K, D):
    k = np.zeros(D)
    for i in range(D):
        k[i] = f[i]*(1-(Nw[i]+Nm[i])/K) 
    return k
    
@njit
def compute_total_rate(b, c, w, Nm, Nw, gm, gw, fm, fw, mig_rate, kw_p, km_p, kw_n, km_n, K, D):
    #kw_p: division reaction rate for the wild-types
    #km_p: division reaction rate for the mutants
    #kw_n: death reaction rate for the wild-types
    #km_n: death reaction rate for the mutants
    
    ktot = 0
    for i in range(D):
        ktot += (kw_p[i] + kw_n[i])*Nw[i] + (km_p[i] + km_n[i])*Nm[i]
    for i in range(D):
        for j in range(D):
            ktot += mig_rate[i][j]*(Nw[i] + Nm[i])
    return ktot

@njit
def build_changetower(D): #We create an array change_tower in which we store all modifications which can happen at one Gillespie step 
    nb_reac = 4*D + 2*D**2 
    ind_arr = np.array([D, 2*D, 3*D, 4*D, 4*D + D**2, 4*D+2*D**2], dtype=np.int64)
    change_tower = np.empty((nb_reac,2), dtype=np.int64)
    change_tower[0,:] = [0,+1] 
    
    for i in range(1,ind_arr[0]):
        change_tower[i,:] = [0,+1] #means here that we have 1 more wild-type in the deme i as we are looking at wild-type division reactions 
    
    for i in range(ind_arr[0], ind_arr[1]):
        change_tower[i,:] = [+1,0] #means here that we have 1 more mutant in the deme i as we are looking at mutant division reactions
        
    for i in range(ind_arr[1], ind_arr[2]):
        change_tower[i,:] = [0,-1] #means here that we have lost one wild-type in the deme i as we are looking at wild-type death reactions 
        
    for i in range(ind_arr[2], ind_arr[3]):
        change_tower[i,:] = [-1,0] #means here that we have lost one mutant in the deme i as we are looking at mutant death reactions 
        
    ind_dictw = []
    ind_dictm = []
    
    for i in range(D):
        for j in range(D):
            ind_dictw.append(np.array([i,j])) #index i corresponds to starting deme and index j to ending deme, ind_dictw stores from and to which deme the migration of a wild-type takes place     
            ind_dictm.append(np.array([i,j]))
            
    for i in range(ind_arr[3], ind_arr[4]):
        change_tower[i,:] = ind_dictw[i-ind_arr[3]]

    for i in range(ind_arr[4], ind_arr[5]):
        change_tower[i,:] = ind_dictm[i-ind_arr[4]]

    return change_tower
    
@njit
def build_samplingtower(b, c, w, Nm, Nw, gm, gw, fm, fw, mig_rate, kw_p, km_p, kw_n, km_n, K, D): #We create an array reac_tower in which we store all the possible reactions and from which we will randomly choose an event to happen in the simulation 
    nb_reac = 4*D + 2*D**2 #total number of possible reactions in Gillespie algorithm
    ind_arr = np.array([D, 2*D, 3*D, 4*D, 4*D + D**2, 4*D+2*D**2], dtype=np.int64) #Reactions are grouped by package, first the mutants divisions in the D demes, second the wild-types divisions in the D demes, third the mutants deaths in the D demes, fourth the wild-tpes deaths in the D demes, fifth the mutants migrations between the linked demes, and last the wild-types migrations between the linked demes    
    reac_tower = np.zeros(nb_reac, dtype=np.float64)  
    
    ktot = compute_total_rate(b, c, w, Nm, Nw, gm, gw, fm, fw, mig_rate, kw_p, km_p, kw_n, km_n, K, D)

    reac_tower[0] = kw_p[0]*Nw[0]/ktot

    for i in range(1,ind_arr[0]):
        reac_tower[i] = reac_tower[i-1] + kw_p[i]*Nw[i]/ktot
    
    for i in range(ind_arr[0], ind_arr[1]):
        reac_tower[i] = reac_tower[i-1] + km_p[i-ind_arr[0]]*Nm[i-ind_arr[0]]/ktot
        
    for i in range(ind_arr[1], ind_arr[2]):
        reac_tower[i] = reac_tower[i-1] + kw_n[i-ind_arr[1]]*Nw[i-ind_arr[1]]/ktot
        
    for i in range(ind_arr[2], ind_arr[3]):
        reac_tower[i] = reac_tower[i-1] + km_n[i-ind_arr[2]]*Nm[i-ind_arr[2]]/ktot
        
    mij_sumw  = []
    mij_summ  = []
    
    for i in range(D):
        for j in range(D):
            mij_sumw.append(mig_rate[i][j]*Nw[i])
            mij_summ.append(mig_rate[i][j]*Nm[i])
            
    for i in range(ind_arr[3], ind_arr[4]):
        reac_tower[i] = reac_tower[i-1] + mij_sumw[i-ind_arr[3]]/ktot

    for i in range(ind_arr[4], ind_arr[5]):
        reac_tower[i] = reac_tower[i-1] + mij_summ[i-ind_arr[4]]/ktot

    return reac_tower
    
    
@njit
def finalpopulation(b, c, w, Nm, Nw, gm, gw, arrmig_rate, K, D): #We apply here the Gillespie algorithm until one type of individuals (mutant or wild-type) has fixed in the population; this function returns the final populations Nm and Nw after running the Gillespie algorithm
    
    nb_loop  = 0
    time = 0
    
    ind_arr  = np.array([D, 2*D, 3*D, 4*D, 4*D+D**2, 4*D+2*D**2])
    change_tower = build_changetower(D)
    
    while (count_nb_ele_nonzero(Nm) > 0) and (count_nb_ele_nonzero(Nw) > 0):
        
        nb_loop += 1

        fm   = fitnessM(b, c, w, Nm, Nw, K, D)
        fw   = fitnessW(b, c, w, Nm, Nw, K, D)
        
        #kw_p: division reaction rate for the wild-types
        kw_p = compute_division_reactionrate(b, c, w, Nm, Nw, fw, K, D)
        kw_p = put_to0_negative_ele(kw_p) #done to prevent the demes to be overflowed, likelihood to have a division event in a deme is put to 0 when the deme has already reached its maximal capacity

        #km_p: division reaction rate for the mutants
        km_p = compute_division_reactionrate(b, c, w, Nm, Nw, fm, K, D)
        km_p = put_to0_negative_ele(km_p)
        
        #kw_n: death reaction rate for the wild-types
        kw_n = compute_death_reactionrate(gw, D)
        
        #km_n: death reaction rate for the mutants
        km_n = compute_death_reactionrate(gm, D)

        ##ktot: total reaction rate for all individuals
        ktot = compute_total_rate(b, c, w, Nm, Nw, gm, gw, fm, fw, arrmig_rate,  kw_p, km_p, kw_n, km_n, K, D)
        
        reac_tower =  build_samplingtower(b, c, w, Nm, Nw, gm, gw, fm, fw, arrmig_rate, kw_p, km_p, kw_n, km_n, K, D)
        #Recall the meanings of reac_tower et change_tower:
        #reac_tower: tower that will store all the possible reactions and from which we will randomly choose an event to happen in the simulation
        #change_tower: store the resulting change in the population for each corresponding reaction

        r    = np.random.uniform(0,1) 
        dt   = (1/ktot)*np.log(1/r)
        time += dt 
        
        r_   = np.random.uniform(0,1)
        ir   = 0
        while reac_tower[ir] <= r_:
            ir += 1  
            
        if ir < ind_arr[3]:
            addNm = change_tower[ir,0]
            addNw = change_tower[ir,1]
        
            if ir < D:
                Nm[ir] += addNm
                Nw[ir] += addNw
            
            if ir < 2*D and ir >= D:
                Nm[ir-D] += addNm
                Nw[ir-D] += addNw
            
            if ir < 3*D and ir >= 2*D:
                Nm[ir-2*D] += addNm
                Nw[ir-2*D] += addNw
        
            if ir < 4*D and ir >= 3*D:
                Nm[ir-3*D] += addNm
                Nw[ir-3*D] += addNw

        if ir < ind_arr[4] and ir >= ind_arr[3]:
            deme_i = change_tower[ir,0]
            deme_j = change_tower[ir,1]
            Nw[deme_i] -= 1
            Nw[deme_j] += 1
            
        if ir < ind_arr[5] and ir >= ind_arr[4]:
            deme_i = change_tower[ir,0]
            deme_j = change_tower[ir,1]
            Nm[deme_i] -= 1
            Nm[deme_j] += 1
            
    return Nm, Nw

@njit
def main(b, c, w, gm, gw, arrmig_rate, K, D, N_iter):

    nb_fix   = 0

    for i in range(N_iter):
        #Nm0, Nw0   = initialization01(18, 18, D) #initialization star starting from 1 mutant leaf deme
        Nm0, Nw0  = initialization10(18, 18, D) #initialization star starting from mutant center deme
        #Nm0, Nw0  = initialization_lattice(18, 18, D)
        Nm_f, Nw_f = finalpopulation(b, c, w, Nm0, Nw0, gm, gw, arrmig_rate, K, D) 
        
        if count_nb_ele_nonzero(Nm_f) > 0:
            nb_fix += 1
            
    return nb_fix/N_iter 


N_iter = 10
b      = 2
c      = 1
D      = 5
K      = 20
gw     = 0.1
gm     = 0.1
w_list = [0.01] #[1e-40, 0.01, 0.02, 0.03, 0.04, 0.05] #[1e-40, 0.01, 0.02, 0.03, 0.04, 0.05] #[1e-40, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
mI_list= [1*1e-5]
mO_list= [4*1e-5]
#mA_list= [4*1e-5]
#mC_list= [1*1e-5]
#m_list = [4*1e-5]
#mE_list = [1*1e-5]
#mN_list = [1*1e-5]
#mW_list = [1*1e-5]
#mS_list = [1*1e-5]


for mO, mI in zip(mO_list, mI_list): #for the star
#for mA, mC in zip(mA_list, mC_list): #for the cycle
#for m in m_list: #for the clique
#for mN, mS, mE, mW in zip(mN_list, mS_list, mE_list, mW_list): #for the lattice

    arrmig_rate = mig_rate_star(mO, mI, D)
    #arrmig_rate = mig_rate_cycle(mA, mC, D)
    #arrmig_rate = mig_rate_clique(m, D)
    #arrmig_rate = mig_rate_lattice3x3(mN, mS, mE, mW, D)
    prob_fix    = []
    
    for w in w_list:
        print(w)
        prob = main(b, c, w, gm, gw, arrmig_rate, K, D, N_iter) 
        prob_fix.append(prob)
    
    print(prob_fix)
    

