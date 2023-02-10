# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:48:55 2021

@author: wyue
"""

import numpy as np
import matplotlib.pyplot as plt
import helpers as hp
import matplotlib.pyplot as plt
from scipy import stats
import nlopt

#number of components
m = 3
## miu is shifted for Na times
Na = 100
## Each shift, Nb components are produced
Nb = 10000
## N samples are generated
N = Na*Nb

smallvalue = 1e-2 # Lower bound is 0, to prevent dividing by zero, set lower bond to a small value
largevalue= 20

X = np.array([55.30, 22.86, 101.60])
sigmaX = np.array([0.080, 0.042, 0.105])

e1L = 3
e2L = 3
e3L = 3
e1R = 3
e2R = 3
e3R = 3
SYMMETRY = False

grad_r = np.zeros(m)

if SYMMETRY:
    epsilon = np.array([e1L, e2L, e3L])
    grad_epsilon = np.zeros(m)
else:
    epsilon = np.array([e1L, e2L, e3L, e1R, e2R, e3R])
    grad_epsilon = np.zeros(2*m)

## miu is shifted for M times
M = 500

A = np.array([0.98, 0.52, 1.23])
B = np.array([2.136, 1.8, 2.568])

E = np.array([0.032, 0.02136, 0.05336])
F = np.array([0.0004, 0.0002, 0.0006]) 

#Scrap cost of a product
Sp = np.sum(A)/10

## Symmetrical case
G = np.array([0.98, 0.52, 1.23])
H = np.array([2.136, 1.8, 2.568])
V = np.array([0.001, 0.001, 0.001])

## Control cost (asymmetrical case)
# GL = np.array([0.98, 0.52, 1.23])*0.15
# HL = np.array([2.136, 1.8, 2.568])*0.06
# VL = np.array([0.001, 0.001, 0.001])*0.27
# GR = np.array([0.98, 0.52, 1.23])*0.15
# HR = np.array([2.136, 1.8, 2.568])*0.06
# VR = np.array([0.001, 0.001, 0.001])*0.27

GL = np.array([0.98, 0.52, 1.23])
HL = np.array([2.136, 1.8, 2.568])
VL = np.array([0.001, 0.001, 0.001])
GR = np.array([0.98, 0.52, 1.23])
HR = np.array([2.136, 1.8, 2.568])
VR = np.array([0.001, 0.001, 0.001])

#Nominal value of Y
miuY = 0.122
##Upper specification limit
USY = miuY + 0.035
LSY = miuY - 0.035

D1 = hp.dy_dx1_clutch(X[0],X[1],X[2])
D2 = hp.dy_dx2_clutch(X[0],X[1],X[2])
D3 = hp.dy_dx3_clutch(X[0],X[1],X[2])
D = np.array([D1,D2,D3])

#Concatenate r and epsilon into a numpy array
r = np.array([5.0,5.0,5.0])
x = np.concatenate((r,epsilon),axis=0)


x_vec = np.zeros((3,M,N))

def obj(x,grad,para):
    #retrieve r as the optimization variable x. (k will not be optimized, so just use const)
    A = para[0]
    B = para[1]
    E = para[2]
    F = para[3]
    G = para[4]
    H = para[5]
    V = para[6]
    D = para[7]
    r = x[0:m]
    num_m = int(x.size/2)
    epsilon = x[num_m:]
    sigmaX = hp.sigma(E,F,r)
    sigmaX_loaf = hp.sigma_loaf(E,F,r,epsilon)
    
    sigmaY_Taylor = hp.sigmaY(sigmaX_loaf,D)
    #Compute Unit Cost
    Cprocess = hp.Cprocess(A,B,r)
    Ccontrol = hp.Ccontrol(G,H,V,epsilon)
    U = hp.U(Cprocess,Ccontrol,USY,miuY,sigmaY_Taylor,Sp)
    
    for i in range(0,m):  # Change this for loop to vectorization         
        ## Gradient of r
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dsigmai_loaf_dri_v = hp.dsigmai_loaf_dri(F[i],r[i],epsilon[i])
        dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_loaf_dri_v)        
        grad_r[i] = hp.dU_dri(USY,miuY,sigmaY_Taylor,Cprocess,Ccontrol,dsigmaY_dri_v,dCprocessi_dri_v,Sp)
        ## Gradient of epsilon
        dCi_depsiloni_v = hp.dCcontrol_depsiloni(H[i],V[i],epsilon[i],SYMMETRY)
        dsigmai_loaf_depsiloni_v = hp.dsigmai_loaf_depsiloni(epsilon[i], sigmaX[i])
        dsigmaY_depsiloni_v = hp.dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v)
        grad_epsilon[i] = hp.dU_depsiloni(USY, miuY,sigmaY_Taylor,Cprocess,
                                                   Ccontrol,dsigmaY_depsiloni_v,dCi_depsiloni_v,Sp)
        
    grad_combine = np.concatenate((grad_r,grad_epsilon),axis=0)
    
    if grad.size > 0:
        grad[:] = grad_combine #Make sure to assign value using [:]
    #print(U)
    return U

def obj_as(x,grad,para):
    #retrieve r as the optimization variable x. (k will not be optimized, so just use const)
    A = para[0]
    B = para[1]
    E = para[2]
    F = para[3]
    GL = para[4]
    HL = para[5]
    VL = para[6]
    GR = para[7]
    HR = para[8]
    VR = para[9]
    D = para[10]
    r = x[0:m]
    num_m = m
    epsilon = x[num_m:]
    sigmaX = hp.sigma(E,F,r)
    sigmaX_loaf = hp.sigma_loaf_as(E,F,r,epsilon,m)
    miuX = hp.miu_x_as(X, epsilon, sigmaX, m)
    sigmaY_Taylor = hp.sigmaY(sigmaX_loaf,D)
    #Compute Unit Cost
    Cprocess = hp.Cprocess(A,B,r)
    Ccontrol = hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsilon,m)
    ##beta is product pass rate
    beta = hp.productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    U = hp.U_as(Cprocess,Ccontrol,Sp,beta)
    
    for i in range(0,m):  # Change this for loop to vectorization         
        epsilonL_i = epsilon[i]
        epsilonR_i = epsilon[i+m]
        
        ## Gradient of r
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dsigmai_dr = hp.dsigmai_dri(F[i],r[i])
        dmiuy_dri_v = hp.dmiuY_dri(i,miuX,epsilonL_i,epsilonR_i,dsigmai_dr)
        dsigmai_loaf_dri_v = hp.dsigmai_loaf_dri_as(F[i],r[i],epsilonL_i,epsilonR_i)
        dsigmay_dri_v = hp.dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_loaf_dri_v)
        dbeta_dri_v = hp.dbeta_dri_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_dri_v, dsigmay_dri_v)
        grad_r[i] = hp.dU_dri_as(Cprocess,Ccontrol,Sp,dCprocessi_dri_v, beta, dbeta_dri_v)
        
        ## Gradient of epsilon left
        dCcontrol_depsiloni_L_v = hp.dCcontrol_depsiloni(HL[i],VL[i],epsilonL_i,SYMMETRY)
        dmiuy_depsiloni_L_v = hp.dmiuY_depsiloni(i,miuX,sigmaX[i],left=True)
        dsigmai_loaf_depsiloni_v = hp.dsigmai_loaf_depsiloni_as(epsilonL_i,epsilonR_i,sigmaX[i])
        dsigmaY_depsiloni_v = hp.dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v)
        dbeta_depsiloni_L_v = hp.dbeta_depsiloni_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_depsiloni_L_v, dsigmaY_depsiloni_v)
        grad_epsilon[i] = hp.dU_depsiloni_as(Cprocess,Ccontrol,Sp,dCcontrol_depsiloni_L_v, beta, dbeta_depsiloni_L_v)
        
        ## Gradient of epsilon right
        dCcontrol_depsiloni_R_v = hp.dCcontrol_depsiloni(HR[i],VR[i],epsilonR_i,SYMMETRY)
        dmiuy_depsiloni_R_v = hp.dmiuY_depsiloni(i,miuX,sigmaX[i],left=False)
        dsigmai_loaf_depsiloni_v = hp.dsigmai_loaf_depsiloni_as(epsilonL_i,epsilonR_i,sigmaX[i])
        dsigmaY_depsiloni_v = hp.dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v)
        dbeta_depsiloni_R_v = hp.dbeta_depsiloni_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_depsiloni_R_v, dsigmaY_depsiloni_v)
        grad_epsilon[i+m] = hp.dU_depsiloni_as(Cprocess,Ccontrol,Sp,dCcontrol_depsiloni_R_v, beta, dbeta_depsiloni_R_v)
    grad_combine = np.concatenate((grad_r,grad_epsilon),axis=0)
    
    if grad.size > 0:
        grad[:] = grad_combine #Make sure to assign value using [:]
    #print(U)
    return U

def optimize(prnt,para):
    #Unit cost of initial values
    #sigmaX = hp.sigma(E,F,r)    
    #sigmaY_Taylor = hp.sigmaY(sigmaX,D)
    #cost = hp.C(A,B,r)
    result = {}    
    
    opt = nlopt.opt(nlopt.LD_MMA, 2*m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA
    #ubK = 10.0
    opt.set_lower_bounds([smallvalue,smallvalue,smallvalue,smallvalue,smallvalue,smallvalue])
    #opt.set_upper_bounds([largevalue,largevalue,largevalue,5,5,5])
    opt.set_min_objective(lambda x,grad: obj(x,grad,para))

        
    opt.set_xtol_rel(1e-4)
    x0 = np.concatenate((r,epsilon),axis = 0)
    x = opt.optimize(x0)
    #U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
    minf = opt.last_optimum_value()
    result['U'] = minf
    result['r'] = x[0:m]
    result['epsilon'] = x[m:]        
    if prnt==True:              
        print("optimum at ", x)
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())    
    return result

def optimize_as(prnt,para):
    result = {}    
    
    opt = nlopt.opt(nlopt.LD_MMA, 3*m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA
    #ubK = 10.0
    opt.set_lower_bounds([smallvalue]*9)
    #opt.set_upper_bounds([largevalue,largevalue,largevalue,5,5,5,5,5,5])
    opt.set_min_objective(lambda x,grad: obj_as(x,grad,para))

        
    opt.set_xtol_rel(1e-4)
    x0 = np.concatenate((r,epsilon),axis = 0)
    x = opt.optimize(x0)
    #U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
    minf = opt.last_optimum_value()
    result['U'] = minf
    result['r'] = x[0:m]
    result['epsilon_L'] = x[m:2*m]
    result['epsilon_R'] = x[2*m:]
    if prnt==True:              
        print("optimum at ", x)
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())    
    return result
    
    
    
def casestudy_U():
    para = np.array([A,B,E,F,G,H,V,D])
    prnt=True
    result = optimize(prnt,para)
    U_equation = result['U']
    r_opt = result['r']
    epsilon_opt = result['epsilon']
    #sigma_opt = hp.sigma(E,F,r_opt)
    # M = hp.estimateM(X,E,F,r_opt,epsilon_opt,USY,miuY,Na,Nb)
    M,U_simulation,Y = hp.U_simulation(r_opt,epsilon_opt,para,X,USY,miuY,Sp,Na,Nb,m,SYMMETRY)
    hp.compare_SigmaY(Y,r_opt,epsilon_opt,D,E,F,SYMMETRY, m)

    print('U Equation: ', U_equation)
    print('U Simulation: ', U_simulation)
    print("# Satisfactory product",M)
    print('error: ', (U_equation-U_simulation)/U_simulation*100 ,'%')
    # satisfactionrate = hp.satisfactionrate_component_product(miu,E,F,r_opt,k,NSample,USY,miuY,scenario)
    # print('beta: ', satisfactionrate['beta'])
    # print('sigmaY: ', hp.sigmaY(sigma_opt,D,scenario,k_opt))
    # print('opt cost:', hp.C(A,B,r_opt))
    # print('N: ',N)
    # print('M: ', M)
    # print('Gama', satisfactionrate['gammas'])
    hp.plot(r_opt, epsilon_opt, E,F,m,Na,Nb,X,SYMMETRY)
    return (M,U_simulation,Y)

def casestudy_U_as():
    para = np.array([A,B,E,F,GL,HL,VL,GR,HR,VR,D])
    prnt=True
    result = optimize_as(prnt,para)
    U_equation = result['U']
    r_opt = result['r']
    epsilon_opt_L = result['epsilon_L']
    epsilon_opt_R = result['epsilon_R']
    epsilon_opt = np.concatenate((epsilon_opt_L,epsilon_opt_R),axis=0)
    #sigma_opt = hp.sigma(E,F,r_opt)
    # M = hp.estimateM(X,E,F,r_opt,epsilon_opt,USY,miuY,Na,Nb)
    M,U_simulation,Y = hp.U_simulation(r_opt,epsilon_opt,para,X,USY,miuY,Sp,Na,Nb,m,SYMMETRY)
    hp.compare_SigmaY(Y,r_opt,epsilon_opt,D,E,F,SYMMETRY,m)

    print('U Equation: ', U_equation)
    print('U Simulation: ', U_simulation)
    print("# Satisfactory product",M)
    print('error: ', (U_equation-U_simulation)/U_simulation*100 ,'%')
    # satisfactionrate = hp.satisfactionrate_component_product(miu,E,F,r_opt,k,NSample,USY,miuY,scenario)
    # print('beta: ', satisfactionrate['beta'])
    # print('sigmaY: ', hp.sigmaY(sigma_opt,D,scenario,k_opt))
    # print('opt cost:', hp.C(A,B,r_opt))
    # print('N: ',N)
    # print('M: ', M)
    # print('Gama', satisfactionrate['gammas'])
    hp.plot(r_opt, epsilon_opt, E,F,m,Na,Nb,X,SYMMETRY)
    return (M,U_simulation,Y)

if SYMMETRY:
    M,U,Y = casestudy_U()
else:
    M,U,Y = casestudy_U_as()
"""
## For each component
for p in range(m):
    ## Mean deviate for M times (uniform distribution)
    for i in range(M):
        miu_low = X[p] - epsilon[p] * sigmaX[p]
        miu_high = X[p] + epsilon[p] * sigmaX[p]
        miu = np.random.uniform(miu_low,miu_high,1)
        x_vec[p][i][:] = np.random.normal(miu,sigmaX[p],N)

x_vec = x_vec.reshape(3,-1)
y = hp.assembly(x_vec)

#hp.plot_hist(y,30,False)
#hp.normaltest(Y)
#hp.probability_plot(Y)
_, sigmaY_simulation = hp.stat_parameters_simulation(y)

## Analytical values
sigmaX_loaf = hp.sigma_loaf(sigmaX,epsilon)
sigmaY_Taylor = hp.stat_parameters_analysis(X,sigmaX_loaf)

error = (sigmaY_simulation - sigmaY_Taylor)/ sigmaY_simulation
print("error: ",error)
"""