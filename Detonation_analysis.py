"""
    Algorithms for computations in this project are based on resources of California Institute of Technology,
    Numerical Solution Methods for Shock and Detonation Jump Conditions, S.
    Browne, J. Ziegler, and J. E. Shepherd, GALCIT Report FM2006.006 - R3,
    http://shepherd.caltech.edu/EDL/PublicResources/sdt/
"""
from sdtoolbox.postshock import CJspeed, PostShock_fr, PostShock_eq
from sdtoolbox.znd import zndsolve
from sdtoolbox.cv import cvsolve
from sdtoolbox.utilities import CJspeed_plot, znd_plot, znd_fileout
import cantera as ct
import numpy as np

def gavrikov(delta,theta,Tvn,T0):
  
    # Constants
    a = -0.007843787493
    b = 0.1777662961
    c = 0.02371845901
    d = 1.477047968
    e = 0.1545112957
    f = 0.01547021569
    g = -1.446582357
    h = 8.730494354
    i = 4.599907939
    j = 7.443410379
    k = 0.4058325462
    m = 1.453392165
    # Define nondimensional parameters
    X = theta
    Y = Tvn/T0
    z = Y*(a*Y-b) + X*(c*X-d + (e-f*Y)*Y) + g*np.log(Y) + h*np.log(X) + Y*(i/X - k*Y/X**m) - j
    lam = delta*np.power(10,z)
    return lam

def ng(delta,chi):
    
    # Constants
    A0 = 30.465860763763;
    a1 = 89.55438805808153;
    a2 = -130.792822369483;
    a3 = 42.02450507117405;
    b1 = -0.02929128383850;
    b2 = 1.0263250730647101E-5;
    b3 = -1.031921244571857E-9;
    lam = delta*(A0 + ((a3/chi + a2/chi)/chi + a1)/chi + ((b3*chi + b2*chi)*chi + b1)*chi);
    return lam

P1 = 100000; T1 = 300
q = 'H2:2 O2:1 N2:3.76'
mech = 'Mevel2017.cti'
fname = 'wyniki_koncowe'

# Find CJ speed and related data, make CJ diagnostic plots
cj_speed,R2,plot_data = CJspeed(P1,T1,q,mech,fullOutput=True)

# Set up gas object
gas1 = ct.Solution(mech)
gas1.TPX = T1,P1,q

# Find equilibrium post shock state for given speed
gas = PostShock_eq(cj_speed, P1, T1, q, mech)
u_cj = cj_speed*gas1.density/gas.density

# Find frozen post shock state for given speed
gas = PostShock_fr(cj_speed, P1, T1, q, mech)

# Solve ZND ODEs, make ZND plots
out = zndsolve(gas,gas1,cj_speed,t_end=1e-3,advanced_output=True)

# Find CV parameters including effective activation energy
gas.TPX = T1,P1,q
gas = PostShock_fr(cj_speed, P1, T1, q, mech)
Ts = gas.T; Ps = gas.P
Ta = Ts*1.02
gas.TPX = Ta,Ps,q
CVout1 = cvsolve(gas)
Tb = Ts*0.98
gas.TPX = Tb,Ps,q
CVout2 = cvsolve(gas)
# Approximate effective activation energy for CV explosion
taua = CVout1['ind_time']
taub = CVout2['ind_time']
if taua==0 and taub==0:
    theta_effective_CV = 0
else:
    theta_effective_CV = 1/Ts*((np.log(taua)-np.log(taub))/((1/Ta)-(1/Tb)))


limit_species = 'H2'
i_limit = gas.species_index(limit_species)
gas.TPX = Ts,Ps,q
X_initial = gas.X[i_limit]
gas.equilibrate('UV')
X_final = gas.X[i_limit]
T_final = gas.T
X_gav = 0.5*(X_initial - X_final) + X_final
T_west = 0.5*(T_final - Ts) + Ts


for i,X in enumerate(CVout1['speciesX'][i_limit,:]):
    if X > X_gav:
        t_gav = CVout1['time'][i]
        
x_gav = t_gav*out['U'][0]        
for i,T in enumerate(CVout1['T']):
    if T < T_west:
        t_west = CVout1['time'][i]
x_west = t_west*out['U'][0]

max_thermicity_width_ZND = u_cj/out['max_thermicity_ZND']
chi_ng = theta_effective_CV*out['ind_len_ZND']/max_thermicity_width_ZND
cell_gav = gavrikov(x_gav,theta_effective_CV, Ts, T1)
cell_ng = ng(out['ind_len_ZND'], chi_ng)

print('ZND model computation results: ')
print('Reaction zone computation end time = {:8.3e} s'.format(out['tfinal']))
print('Reaction zone computation end distance = {:8.3e} m'.format(out['xfinal']))
print(' ')
print('T (K), initial = {:1.5g}, final = {:1.5g}, max = {:1.5g} '.format(out['T'][0],out['T'][-1],max(out['T'])))
print('P (Pa), initial = {:1.5g}, final = {:1.5g}, max = {:1.5g} '.format(out['P'][0],out['P'][-1],max(out['P'])))
print('M, initial = {:1.5g}, final = {:1.5g}, max = {:1.5g}'.format(out['M'][0],out['M'][-1],max(out['M'])))
print('u (m/s), initial = {:1.5g}, final = {:1.5g}, u_cj = {:1.5g}'.format(out['U'][0],out['U'][-1],u_cj))
print(' ')
print('Reaction zone width (u_cj/sigmadot_max) = {:8.3e} m'.format(max_thermicity_width_ZND))
print(' ')
print('Detonation cell zone parameters: ')
print('Characteristical dimension = {:8.3e} m'.format(cell_gav))


#Computation of reaction zone thermodynamical parameters:
file_name = 'results.txt'

# Find CJ speed and related data, make CJ diagnostic plots
cj_speed,R2,plot_data = CJspeed(P1,T1,q,mech,fullOutput=True)

# Set up gas object
gas1 = ct.Solution(mech)
gas1.TPX = T1,P1,q

# Find post shock state for given speed
gas = PostShock_fr(cj_speed, P1, T1, q, mech)

# Solve ZND ODEs, make ZND plots
znd_out = zndsolve(gas,gas1,cj_speed,t_end=1e-3,advanced_output=True)
znd_fileout(file_name,znd_out)

