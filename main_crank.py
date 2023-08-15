# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 20:26:51 2021

@author: wyue
"""

import numpy as np
import matplotlib.pyplot as plt
import helpers as hp
import nlopt

CLUTCH = False
RECIPROCAL = False
#number of components
m = 5
## miu is shifted for Na times
Na = 500
## Each shift, Nb components are produced
Nb = 2000
## N samples are generated
N = Na*Nb

smallvalue = 0 #1e-2 # Lower bound is 0, to prevent dividing by zero, set lower bond to a small value
largevalue= 100

## miu is shifted for M times
M = 500

A = np.array([2.4, 2.6, 6.15, 4.9,7.0 ])
B = np.array([1.91, 1.8, 3.2, 2.14, 3.8])

E = np.array([0.11, 0.23, 0.12, 0.15, 0.1]) 
F = np.array([0.0041, 0.0060 ,0.0071, 0.0128, 0.0073])

## Control cost
if RECIPROCAL:
    GL = np.array([0.38, 0.52, 0.73, 0.98, 0.52]) * 0.2
    HL = np.array([2.136, 1.8, 2.568, 2.136, 1.8]) * 0.2
    VL = np.array([0.001, 0.001, 0.001, 0.001, 0.001])* 0.2
    GR = np.array([0.98, 0.52, 1.23, 0.98, 0.52])* 0.2
    HR = np.array([2.136, 1.8, 2.568, 2.136, 1.8])* 0.2
    VR = np.array([0.001, 0.001, 0.001, 0.001, 0.001])* 0.2
else:
    GL = np.array([0.076, 0.104, 0.146, 0.136, 0.172]) * 1.5
    HL = np.array([0.10272, 0.36, 0.5136 , 0.4272 , 0.6]) * 0.5
    VL = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    GR = np.array([0.076, 0.104, 0.146, 0.136, 0.172]) * 1.5
    HR = np.array([0.10272, 0.36, 0.5136 , 0.4272 , 0.6]) * 0.5
    VR = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
#Scrap cost of a product
Sp = np.sum(A)/10.0

X = np.array([500, 300, 375, 750, 125])
D1 = hp.dy_dx1_crank(X[0],X[1],X[2],X[3],X[4])
D2 = hp.dy_dx2_crank(X[0],X[1],X[2],X[3],X[4])
D3 = hp.dy_dx3_crank(X[0],X[1],X[2],X[3],X[4])
D4 = hp.dy_dx4_crank(X[0],X[1],X[2],X[3],X[4])
D5 = hp.dy_dx5_crank(X[0],X[1],X[2],X[3],X[4])

D = np.array([D1,D2,D3,D4,D5])

#Nominal value of Y
Y0 = 992.9096
##Upper specification limit
USY = Y0 + 4
LSY = Y0 - 4

def obj(x,grad,para):
    #retrieve r as the optimization variable x. (k will not be optimized, so just use const)
    # A = para[0]
    # B = para[1]
    # E = para[2]
    # F = para[3]
    # GL = para[4]
    # HL = para[5]
    # VL = para[6]
    # GR = para[7]
    # HR = para[8]
    # VR = para[9]
    # D = para[10]

    A = para["A"]
    B = para["B"]
    E = para["E"]
    F = para["F"]
    GL = para["GL"]
    HL = para["HL"]
    VL = para["VL"]
    GR = para["GR"]
    HR = para["HR"]
    VR = para["VR"]
    D = para["D"]
    grad_r = para["grad_r"]
    grad_epsilon = para["grad_epsilon"]


    r = x[0:m]
    num_m = m
    epsilon = x[num_m:]
    sigmaX = hp.sigma(E,F,r)
    sigmaX_loaf = hp.sigma_loaf_as(E,F,r,epsilon,m)
    miuX = hp.miu_x_as(X, epsilon, sigmaX, m)
    sigmaY_Taylor = hp.sigmaY(sigmaX_loaf,D)
    #Compute Unit Cost
    Cprocess = hp.Cprocess(A,B,r)
    Ccontrol = hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsilon,m,RECIPROCAL)
    ##beta is product pass rate
    miuY = Y0
    beta = hp.productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    U = hp.U_as(Cprocess,Ccontrol,Sp,beta)
    
    for i in range(0,m):  # Change this for loop to vectorization         
        epsilonL_i = epsilon[i]
        epsilonR_i = epsilon[i+m]
        
        ## Gradient of r
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dsigmai_dr = hp.dsigmai_dri(F[i],r[i])
        dmiuy_dri_v = hp.dmiuY_dri(i,miuX,epsilonL_i,epsilonR_i,dsigmai_dr)
        dsigmai_loaf_dri_v = hp.dsigmai_loaf_dri_as(F[i],r[i],epsilonL_i,epsilonR_i)
        dsigmay_dri_v = hp.dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_loaf_dri_v)
        dbeta_dri_v = hp.dbeta_dri_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_dri_v, dsigmay_dri_v)
        grad_r[i] = hp.dU_dri_as(Cprocess,Ccontrol,Sp,dCprocessi_dri_v, beta, dbeta_dri_v)
        
        ## Gradient of epsilon left
        dCcontrol_depsiloni_L_v = hp.dCcontrol_depsiloni(HL[i],VL[i],epsilonL_i,RECIPROCAL)
        dmiuy_depsiloni_L_v = hp.dmiuY_depsiloni(i,miuX,sigmaX[i],left=True)
        dsigmai_loaf_depsiloni_v = hp.dsigmai_loaf_depsiloni_as(epsilonL_i,epsilonR_i,sigmaX[i])
        dsigmaY_depsiloni_v = hp.dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v)
        dbeta_depsiloni_L_v = hp.dbeta_depsiloni_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_depsiloni_L_v, dsigmaY_depsiloni_v)
        grad_epsilon[i] = hp.dU_depsiloni_as(Cprocess,Ccontrol,Sp,dCcontrol_depsiloni_L_v, beta, dbeta_depsiloni_L_v)
        
        ## Gradient of epsilon right
        dCcontrol_depsiloni_R_v = hp.dCcontrol_depsiloni(HR[i],VR[i],epsilonR_i,RECIPROCAL)
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

def obj_single_side(x,grad,para):
    A = para["A"]
    B = para["B"]
    E = para["E"]
    F = para["F"]
    G = para["G"]
    H = para["H"]
    V = para["V"]

    # D = para["D"]
    grad_r = para["grad_r"]
    grad_epsilon = para["grad_epsilon"]

    r = x[0:m]
    num_m = m
    epsilon = x[num_m:]
    sigmaX = hp.sigma(E,F,r)
    sigmaX_loaf = hp.sigma_loaf_as(E,F,r,epsilon,m, double_side=False)
    miuX = hp.miu_x_as(X, epsilon, sigmaX, m, double_side=False)

    D = hp.get_D(miuX)

    sigmaY_Taylor = hp.sigmaY(sigmaX_loaf,D)
    #Compute Unit Cost
    Cprocess = hp.Cprocess(A,B,r)
    Ccontrol = hp.Ccontrol_as_single_side(G,H,V,epsilon,RECIPROCAL)
    ##beta is product pass rate
    miuY = hp.assem_fun(miuX)
    beta = hp.productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    U = hp.U_as(Cprocess,Ccontrol,Sp,beta)
    
    for i in range(0,m):  # Change this for loop to vectorization      
        ## This is single-sided. Set the shifting to the left as 0   
        epsilonL_i = 0
        epsilonR_i = epsilon[i]
        
        ## Gradient of r
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dsigmai_dr = hp.dsigmai_dri(F[i],r[i])
        dmiuy_dri_v = hp.dmiuY_dri(i,miuX,epsilonL_i,epsilonR_i,dsigmai_dr)
        dsigmai_loaf_dri_v = hp.dsigmai_loaf_dri_as(F[i],r[i],epsilonL_i,epsilonR_i)
        dsigmay_dri_v = hp.dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_loaf_dri_v)
        dbeta_dri_v = hp.dbeta_dri_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_dri_v, dsigmay_dri_v)
        grad_r[i] = hp.dU_dri_as(Cprocess,Ccontrol,Sp,dCprocessi_dri_v, beta, dbeta_dri_v)
        
        ## Gradient of epsilon
        dCcontrol_depsiloni_R_v = hp.dCcontrol_depsiloni(H[i],V[i],epsilonR_i,RECIPROCAL)
        dmiuy_depsiloni_R_v = hp.dmiuY_depsiloni(i,miuX,sigmaX[i],left=False)
        dsigmai_loaf_depsiloni_v = hp.dsigmai_loaf_depsiloni_as(epsilonL_i,epsilonR_i,sigmaX[i])
        dsigmaY_depsiloni_v = hp.dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v)
        dbeta_depsiloni_R_v = hp.dbeta_depsiloni_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_depsiloni_R_v, dsigmaY_depsiloni_v)
        grad_epsilon[i] = hp.dU_depsiloni_as(Cprocess,Ccontrol,Sp,dCcontrol_depsiloni_R_v, beta, dbeta_depsiloni_R_v)
        
    grad_combine = np.concatenate((grad_r,grad_epsilon),axis=0)
    
    if grad.size > 0:
        grad[:] = grad_combine #Make sure to assign value using [:]
    #print(U)
    return U

def obj_jcp(x,grad,para):
    A = para["A"]
    B = para["B"]
    E = para["E"]
    F = para["F"]
    GL = para["GL"]
    HL = para["HL"]
    VL = para["VL"]
    GR = para["GR"]
    HR = para["HR"]
    VR = para["VR"]
    D = para["D"]
    control_cost = para["control_cost"]
    grad_r = para["grad_r"]
    # grad_epsilon = para["grad_epsilon"]


    r = x[0:m]
    # fix epsilon at 0. In the JCP method, the model does not consider epsilon.
    epsilon_zero = np.array([0] * (2 * m))
    sigmaX = hp.sigma(E,F,r)
    sigmaX_loaf = hp.sigma_loaf_as(E,F,r,epsilon_zero,m)
    miuX = hp.miu_x_as(X, epsilon_zero, sigmaX, m)
    sigmaY_Taylor = hp.sigmaY(sigmaX_loaf,D)
    #Compute Unit Cost
    Cprocess = hp.Cprocess(A,B,r)

    Ccontrol = np.zeros(m)
    ## For a fair comparison, the control cost need to be added
    if control_cost:
        epsilon_fix = para['epsilon_fix']
        epsilon = np.array([epsilon_fix] * (2 * m))
        Ccontrol = hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsilon,m,RECIPROCAL)
    ##beta is product pass rate
    miuY = Y0
    beta = hp.productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    U = hp.U_as(Cprocess,Ccontrol,Sp,beta)
    
    for i in range(0,m):  # Change this for loop to vectorization         
        epsilonL_i = 0
        epsilonR_i = 0
        
        ## Gradient of r
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dsigmai_dr = hp.dsigmai_dri(F[i],r[i])
        dmiuy_dri_v = hp.dmiuY_dri(i,miuX,epsilonL_i,epsilonR_i,dsigmai_dr)
        dsigmai_loaf_dri_v = hp.dsigmai_loaf_dri_as(F[i],r[i],epsilonL_i,epsilonR_i)
        dsigmay_dri_v = hp.dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_loaf_dri_v)
        dbeta_dri_v = hp.dbeta_dri_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_dri_v, dsigmay_dri_v)
        grad_r[i] = hp.dU_dri_as(Cprocess,Ccontrol,Sp,dCprocessi_dri_v, beta, dbeta_dri_v)
    
    if grad.size > 0:
        grad[:] = grad_r #Make sure to assign value using [:]
    #print(U)
    return U

def obj_jcp_single_side(x,grad,para):
    A = para["A"]
    B = para["B"]
    E = para["E"]
    F = para["F"]
    G = para["G"]
    H = para["H"]
    V = para["V"]

    D = para["D"]
    control_cost = para["control_cost"]
    grad_r = para["grad_r"]
    # grad_epsilon = para["grad_epsilon"]


    r = x[0:m]
    # fix epsilon at 0. In the JCP method, the model does not consider epsilon.
    epsilon_zero = np.array([0] * (m))
    sigmaX = hp.sigma(E,F,r)
    sigmaX_loaf = hp.sigma_loaf_as(E,F,r,epsilon_zero,m, double_side=False)
    miuX = hp.miu_x_as(X, epsilon_zero, sigmaX, m,double_side=False)
    sigmaY_Taylor = hp.sigmaY(sigmaX_loaf,D)
    #Compute Unit Cost
    Cprocess = hp.Cprocess(A,B,r)

    Ccontrol = np.zeros(m)
    ## For a fair comparison, the control cost need to be added
    if control_cost:
        epsilon_fix = para['epsilon_fix']
        epsilon = np.array([epsilon_fix] * (m))
        Ccontrol = hp.Ccontrol_as_single_side(G,H,V,epsilon,RECIPROCAL)
    ##beta is product pass rate
    miuY = Y0
    beta = hp.productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    U = hp.U_as(Cprocess,Ccontrol,Sp,beta)
    
    for i in range(0,m):  # Change this for loop to vectorization         
        epsilonL_i = 0
        epsilonR_i = 0
        
        ## Gradient of r
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
        dsigmai_dr = hp.dsigmai_dri(F[i],r[i])
        dmiuy_dri_v = hp.dmiuY_dri(i,miuX,epsilonL_i,epsilonR_i,dsigmai_dr)
        dsigmai_loaf_dri_v = hp.dsigmai_loaf_dri_as(F[i],r[i],epsilonL_i,epsilonR_i)
        dsigmay_dri_v = hp.dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_loaf_dri_v)
        dbeta_dri_v = hp.dbeta_dri_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_dri_v, dsigmay_dri_v)
        grad_r[i] = hp.dU_dri_as(Cprocess,Ccontrol,Sp,dCprocessi_dri_v, beta, dbeta_dri_v)
    
    if grad.size > 0:
        grad[:] = grad_r #Make sure to assign value using [:]
    #print(U)
    return U

def optimize(prnt,para, r0, epsilon0, upper_bounds=False):
    result = {}    
    
    opt = nlopt.opt(nlopt.LD_MMA, 3*m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA
    #ubK = 10.0
    ## the lower bound of r cannot be too small, otherwise, sigmaY is too small, this cause
    ## computational issues when computing dbeta_dr
    low_bnd = [2.0]*5+[smallvalue]*10
    opt.set_lower_bounds(low_bnd)
    if upper_bounds:
        max_sigma = para["max_epsilon"]
        ub = [largevalue]*m + [max_sigma] * (2 * m)
        opt.set_upper_bounds(ub)
    opt.set_min_objective(lambda x,grad: obj(x,grad,para))

        
    opt.set_xtol_rel(1e-4)
    x0 = np.concatenate((r0,epsilon0),axis = 0)
    x = opt.optimize(x0)
    #U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
    minf = opt.last_optimum_value()
    rst = opt.last_optimize_result()
    if rst < 0:
        print("NLOPT failed!!!!")
    result['U'] = minf
    result['r'] = x[0:m]
    result['epsilon_L'] = x[m:2*m]
    result['epsilon_R'] = x[2*m:]
    if prnt==True:              
        print("optimum at ", x)
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())    
    return result

def optimize_single_side(prnt,para, r0, epsilon0, upper_bounds=False):
    result = {}    
    
    opt = nlopt.opt(nlopt.LD_MMA, 2*m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA
    #ubK = 10.0
    ## the lower bound of r cannot be too small, otherwise, sigmaY is too small, this cause
    ## computational issues when computing dbeta_dr
    low_bnd = [2.0] * 5 + [smallvalue] * 5
    opt.set_lower_bounds(low_bnd)
    if upper_bounds:
        max_sigma = para["max_epsilon"]
        ub = [largevalue]*m + [max_sigma] * ( m)
        opt.set_upper_bounds(ub)
    opt.set_min_objective(lambda x,grad: obj_single_side(x,grad,para))

        
    opt.set_xtol_rel(1e-4)
    x0 = np.concatenate((r0,epsilon0),axis = 0)
    x = opt.optimize(x0)
    #U_init = hp.U_scrap(cost,USY,miuY,sigmaY_Taylor,k,Sp,Sc)
    minf = opt.last_optimum_value()

    rst = opt.last_optimize_result()
    if rst < 0:
        print("NLOPT failed!!!!")

    result['U'] = minf
    result['r'] = x[0:m]
    result['epsilon'] = x[m:]
    if prnt==True:              
        print("optimum at ", x)
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())    
    return result
    
def optimize_jcp(para, r0, prnt=False):
    result = {}    
    
    opt = nlopt.opt(nlopt.LD_MMA, m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA

    ## the lower bound of r cannot be too small, otherwise, sigmaY is too small, this cause
    ## computational issues when computing dbeta_dr
    low_bnd = [2.0] * m
    opt.set_lower_bounds(low_bnd)
    #opt.set_upper_bounds([largevalue,largevalue,largevalue])
    opt.set_min_objective(lambda x,grad: obj_jcp(x,grad,para))

        
    opt.set_xtol_rel(1e-4)
    x0 = r0[:]
    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    result['U'] = minf
    result['r'] = x[0:m]

    if prnt==True:              
        print("optimum at ", x)
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())    
    return result

def optimize_jcp_single_side(para, r0, prnt=False):
    result = {}    
    
    opt = nlopt.opt(nlopt.LD_MMA, m) # MMA (Method of Moving Asymptotes) and CCSA LD_MMA

    ## the lower bound of r cannot be too small, otherwise, sigmaY is too small, this cause
    ## computational issues when computing dbeta_dr
    low_bnd = [2.0] * m
    opt.set_lower_bounds(low_bnd)
    #opt.set_upper_bounds([largevalue,largevalue,largevalue])
    opt.set_min_objective(lambda x,grad: obj_jcp_single_side(x,grad,para))

        
    opt.set_xtol_rel(1e-4)
    x0 = r0[:]
    x = opt.optimize(x0)
    minf = opt.last_optimum_value()
    result['U'] = minf
    result['r'] = x[0:m]

    if prnt==True:              
        print("optimum at ", x)
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())    
    return result

def casestudy_U():
    #para = np.array([A,B,E,F,GL,HL,VL,GR,HR,VR,D])
    para = {"A":A,"B":B,"E":E,"F":F,"GL":GL,"HL":HL,"VL":VL,"GR":GR,"HR":HR,"VR":VR,"D":D}
    para["control_cost"] = True
    para["grad_r"] = np.zeros(m)
    para["grad_epsilon"] = np.zeros(2*m)
    prnt=True
    epsilon0 = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    r0 = np.ones(5) * 10.0
    result = optimize(prnt,para, r0, epsilon0)
    U_equation = result['U']
    r_opt = result['r']
    epsilon_opt_L = result['epsilon_L']
    epsilon_opt_R = result['epsilon_R']
    epsilon_opt = np.concatenate((epsilon_opt_L,epsilon_opt_R),axis=0)
    
    miuY = Y0
    beta_equation = hp.beta_equation(E,F,r_opt,epsilon_opt,m,LSY,USY, miuY,D,double_side=True)

    # M_equation = hp.M_equation(X,E,F,r_opt,epsilon_opt,USY,miuY,Na,Nb)

    M_simulation,U_simulation,Y = hp.U_simulation(r_opt,epsilon_opt,para,X,USY,miuY,Sp,Na,Nb,m,CLUTCH,RECIPROCAL)
    hp.compare_SigmaY(Y,r_opt,epsilon_opt,D,E,F,m)

    print('U Equation: ', U_equation)
    print('U Simulation: ', U_simulation)
    print('error of U: ', (U_equation-U_simulation)/U_simulation*100 ,'%')

    sigma_opt = hp.sigma(E,F,r_opt)
    print(f"Optimal sigma: {sigma_opt}")
    # sigma_Y_equation = hp.sigmaY(sigma_opt, D)

    print("# Satisfactory product (simulation)", M_simulation)
    print(f"Pass rate beta (equation): {beta_equation}")
    print(f"Error: of pass rate: {(beta_equation - M_simulation/(Na*Nb))/(M_simulation/(Na*Nb)) * 100} %")



    print('Process cost:', hp.Cprocess(A,B,r_opt))
    print('Maintenance cost:', hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsilon_opt,m,RECIPROCAL))

    hp.plot_test(r_opt, epsilon_opt, E,F,m,Na,Nb,X,CLUTCH)
    return (M,U_simulation,Y,result)

def casestudy_U_single_side():
    #para = np.array([A,B,E,F,GL,HL,VL,GR,HR,VR,D])
    para = {"A":A,"B":B,"E":E,"F":F,"G":GL,"H":HL,"V":VL}
    para["control_cost"] = True
    para["grad_r"] = np.zeros(m)
    para["grad_epsilon"] = np.zeros(m)
    prnt=True
    epsilon0 = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
    r0 = np.ones(5) * 10.0
    result = optimize_single_side(prnt,para, r0, epsilon0)
    U_equation = result['U']
    r_opt = result['r']

    epsilon_opt =  result['epsilon']
    sigma_opt = hp.sigma(E,F,r_opt)

    miuX = hp.miu_x_as(X, epsilon_opt, sigma_opt, m, double_side=False)
    D = hp.get_D(miuX)

    miuY = hp.assem_fun(miuX)
    beta_equation = hp.beta_equation(E,F,r_opt,epsilon_opt,m,LSY,USY, miuY,D,double_side=False)

    # M_equation = hp.M_equation(X,E,F,r_opt,epsilon_opt,USY,miuY,Na,Nb)

    M_simulation,U_simulation,Y = hp.U_simulation_single_side(r_opt,epsilon_opt,para,X,USY,Y0,Sp,Na,Nb,m,CLUTCH,RECIPROCAL)



    hp.compare_SigmaY(Y,r_opt,epsilon_opt,D,E,F,m,double_side=False)

    print('U Equation: ', U_equation)
    print('U Simulation: ', U_simulation)
    print('error of U: ', (U_equation-U_simulation)/U_simulation*100 ,'%')

    sigma_opt = hp.sigma(E,F,r_opt)
    print(f"Optimal sigma: {sigma_opt}")
    # sigma_Y_equation = hp.sigmaY(sigma_opt, D)

    print("# Satisfactory product (simulation)", M_simulation)
    print(f"Pass rate beta (equation): {beta_equation}")
    print(f"Error: of pass rate: {(beta_equation - M_simulation/(Na*Nb))/(M_simulation/(Na*Nb)) * 100} %")



    print('Process cost:', hp.Cprocess(A,B,r_opt))
    print('Maintenance cost:', hp.Ccontrol_as_single_side(para['G'],para['H'],para['V'], epsilon_opt,RECIPROCAL))

    hp.plot_test(r_opt, epsilon_opt, E,F,m,Na,Nb,X,CLUTCH,double_side=False)
    return (M,U_simulation,Y,result)

def casestudy_U_jcp(max_epsilon,control_cost=True):
    """
    The old method proposed in JCLP
    Equation: fix ε at 0 and optimize r only. 
    Simulation: let the mean shift with the given epsilon.  
    """
    # para = np.array([A,B,E,F,GL,HL,VL,GR,HR,VR,D])
    para = {"A":A,"B":B,"E":E,"F":F,"GL":GL,"HL":HL,"VL":VL,"GR":GR,"HR":HR,"VR":VR,"D":D}
    para["control_cost"] = control_cost
    para["epsilon_fix"] = max_epsilon
    para["grad_r"] = np.zeros(m)
    r0 = np.ones(5) * 10.0
    result = optimize_jcp(para,r0)
    U_eq = result['U']
    r_opt = result['r']
    sigma_opt = hp.sigma(E,F,r_opt)
    sigmaY_eq = hp.sigmaY(sigma_opt, D)


    epsilon_fixed = np.array([max_epsilon] * (2 * m))
    miuY = Y0
    _, U_simu, Y = hp.U_simulation(r_opt,epsilon_fixed,para,X,USY,miuY,Sp,Na,Nb,m,CLUTCH,RECIPROCAL)
    sigmaY_simu = np.std(Y)

    return U_eq, U_simu, sigmaY_eq, sigmaY_simu

def casestudy_U_jcp_single_side(max_epsilon,control_cost=True):
    """
    The old method proposed in JCLP
    Equation: fix ε at 0 and optimize r only. 
    Simulation: let the mean shift with the given epsilon.  
    Single sided epsilon
    """
    # para = np.array([A,B,E,F,GL,HL,VL,GR,HR,VR,D])
    para = {"A":A,"B":B,"E":E,"F":F,"G":GL,"H":HL,"V":VL,"D":D}
    para["control_cost"] = control_cost
    para["epsilon_fix"] = max_epsilon
    para["grad_r"] = np.zeros(m)
    r0 = np.ones(5) * 10.0
    result = optimize_jcp_single_side(para,r0)
    U_eq = result['U']
    r_opt = result['r']
    sigma_opt = hp.sigma(E,F,r_opt)
    sigmaY_eq = hp.sigmaY(sigma_opt, D)


    epsilon_fixed = np.array([max_epsilon] * (m))
    miuY = Y0
    _, U_simu, Y = hp.U_simulation_single_side(r_opt,epsilon_fixed,para,X,USY,miuY,Sp,Na,Nb,m,CLUTCH,RECIPROCAL)
    sigmaY_simu = np.std(Y)

    return U_eq, U_simu, sigmaY_eq, sigmaY_simu

def casestudy_U_this(max_epsilon, control_cost=True):
    """
    The method proposed in this paper
    Optimize r and epsilon, with the upper bound of epsilon being max_epsilon
    """
    para = {"A":A,"B":B,"E":E,"F":F,"GL":GL,"HL":HL,"VL":VL,"GR":GR,"HR":HR,"VR":VR,"D":D}
    para["control_cost"] = control_cost
    para["max_epsilon"] = max_epsilon
    para["grad_r"] = np.zeros(m)
    para["grad_epsilon"] = np.zeros(2*m)
    epsilon0 = np.array([max_epsilon/2]*(2*m))
    r0 = np.ones(5) * 10.0
    prnt=True
    result = optimize(prnt,para,r0, epsilon0, upper_bounds=True)
    U_equation = result['U']
    r_opt = result['r']
    epsilon_opt_L = result['epsilon_L']
    epsilon_opt_R = result['epsilon_R']
    epsilon_opt = np.concatenate((epsilon_opt_L,epsilon_opt_R),axis=0)
    
    miuY = Y0
    beta_equation = hp.beta_equation(E,F,r_opt,epsilon_opt,m,LSY,USY, miuY,D)

    # M_equation = hp.M_equation(X,E,F,r_opt,epsilon_opt,USY,miuY,Na,Nb)

    M_simulation,U_simulation,Y = hp.U_simulation(r_opt,epsilon_opt,para,X,USY,miuY,Sp,Na,Nb,m,CLUTCH,RECIPROCAL)
    # sigmaY_simu = np.std(Y)
    sigmaY_eq, sigmaY_simu = hp.compare_SigmaY(Y,r_opt,epsilon_opt,D,E,F,m)

    print('U Equation: ', U_equation)
    print('U Simulation: ', U_simulation)
    print('error of U: ', (U_equation-U_simulation)/U_simulation*100 ,'%')

    sigma_opt = hp.sigma(E,F,r_opt)
    print(f"Optimal sigma: {sigma_opt}")
    # sigma_Y_equation = hp.sigmaY(sigma_opt, D)

    print("# Satisfactory product (simulation)", M_simulation)
    print(f"Pass rate beta (equation): {beta_equation}")
    print(f"Error: of pass rate: {(beta_equation - M_simulation/(Na*Nb))/(M_simulation/(Na*Nb)) * 100} %")



    print('Process cost:', hp.Cprocess(A,B,r_opt))
    print('Maintenance cost:', hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsilon_opt,m,RECIPROCAL))


    return U_equation, U_simulation, sigmaY_eq, sigmaY_simu

def casestudy_U_this_single_side(max_epsilon, control_cost=True):
    """
    The method proposed in this paper
    Optimize r and epsilon, with the upper bound of epsilon being max_epsilon
    """
    para = {"A":A,"B":B,"E":E,"F":F,"G":GL,"H":HL,"V":VL}
    para["control_cost"] = control_cost
    para["max_epsilon"] = max_epsilon
    para["grad_r"] = np.zeros(m)
    para["grad_epsilon"] = np.zeros(m)
    epsilon0 = np.array([max_epsilon/2]*(m))
    r0 = np.ones(5) * 10.0
    prnt=True
    result = optimize_single_side(prnt,para,r0, epsilon0, upper_bounds=True)
    U_equation = result['U']
    r_opt = result['r']

    epsilon_opt =  result['epsilon']
    
    miuY = Y0
    beta_equation = hp.beta_equation(E,F,r_opt,epsilon_opt,m,LSY,USY, miuY,D,double_side=False)

    # M_equation = hp.M_equation(X,E,F,r_opt,epsilon_opt,USY,miuY,Na,Nb)

    M_simulation,U_simulation,Y = hp.U_simulation_single_side(r_opt,epsilon_opt,para,X,USY,miuY,Sp,Na,Nb,m,CLUTCH,RECIPROCAL)
    # sigmaY_simu = np.std(Y)
    sigmaY_eq, sigmaY_simu = hp.compare_SigmaY(Y,r_opt,epsilon_opt,D,E,F,m,double_side=False)

    print('U Equation: ', U_equation)
    print('U Simulation: ', U_simulation)
    print('error of U: ', (U_equation-U_simulation)/U_simulation*100 ,'%')

    sigma_opt = hp.sigma(E,F,r_opt)
    print(f"Optimal sigma: {sigma_opt}")
    # sigma_Y_equation = hp.sigmaY(sigma_opt, D)

    print("# Satisfactory product (simulation)", M_simulation)
    print(f"Pass rate beta (equation): {beta_equation}")
    print(f"Error: of pass rate: {(beta_equation - M_simulation/(Na*Nb))/(M_simulation/(Na*Nb)) * 100} %")



    print('Process cost:', hp.Cprocess(A,B,r_opt))
    print('Maintenance cost:', hp.Ccontrol_as_single_side(GL,HL,VL,epsilon_opt,RECIPROCAL))


    return U_equation, U_simulation, sigmaY_eq, sigmaY_simu


# def plot_compare_error(x, y1, y2, x_label, y1_label, y2_label, label1, label2, \
#     color1='green', color2='blue', marker1='+', marker2='x', fname="comparison.tif"):
#     fig, ax1 = plt.subplots()

#     ax2 = ax1.twinx()
#     lns1 = ax1.plot(x, y1, color=color1,linestyle='solid', marker=marker1, label=label1)
#     lns2 = ax2.plot(x, y2, color=color2,linestyle='solid', marker=marker2, label=label2)

#     ax1.set_xlabel(x_label)
#     ax1.set_ylabel(y1_label, color=color1)
#     ax2.set_ylabel(y2_label, color=color2)
#     lns = lns1 + lns2
#     labs = [l.get_label() for l in lns]
#     ax1.legend(lns, labs,loc="center left")
#     ax1.grid(True)
#     fig.savefig(fname,dpi=300)
#     plt.show()

def plot_compare_error(x, y1, y2, x_label, y1_label, y2_label, label1, label2, \
    color1='green', color2='blue', marker1='+', marker2='x', fname="comparison.tif"):
    fig, ax1 = plt.subplots()


    lns1 = ax1.plot(x, y1, color=color1,linestyle='solid', marker=marker1, label=label1)
    lns2 = ax1.plot(x, y2, color=color2,linestyle='solid', marker=marker2, label=label2)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color='black')
    
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs) #,loc="center left"
    ax1.grid(True)
    plt.yscale("log")
    fig.savefig(fname,dpi=300)
    plt.show()

def plot_compare_U(x, U1, U2, x_label, y_label,label1, label2,\
    color1='red', color2='cyan', marker1='o', marker2='s',fname="compare_U.tif"):
    fig, ax1 = plt.subplots()

    ax1.plot(x, U1, color=color1, linestyle='solid', marker=marker1,label=label1)
    ax1.plot(x, U2, color=color2, linestyle='solid', marker=marker2,label=label2)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.legend()
    ax1.grid(True)
    fig.savefig(fname,dpi=300)
    plt.show()

def comparison(min_epsilon=0, max_epsilon=4, count=10,r=None):
    """
    Compare the proposed method to the method proposed in JCP paper.
    """
    epsilons = np.linspace(min_epsilon, max_epsilon, count, endpoint=True)
    rst_U_eq_jcp = np.zeros(count)
    rst_U_simu_jcp = np.zeros(count)
    rst_sigmaY_eq_jcp = np.zeros(count)
    rst_sigmaY_simu_jcp = np.zeros(count)
    rst_U_eq = np.zeros(count)
    rst_U_simu = np.zeros(count)
    rst_sigmaY_eq = np.zeros(count)
    rst_sigmaY_simu = np.zeros(count)

    ## Compare sigmaY
    for i, epsilon in enumerate(epsilons):
        ## JCP
        sigma_opt = hp.sigma(E,F,r)
        sigmaY_eq_jcp = hp.sigmaY(sigma_opt, D)
        epsilon_fixed = np.array([epsilon] * (2 * m))
        miuY = Y0
        _, Y, _ = hp.estimateM(X,E,F,r,epsilon_fixed,USY,miuY,Na,Nb,m,clutch=False)
        sigmaY_simu_jcp = np.std(Y)
        rst_sigmaY_eq_jcp[i] = sigmaY_eq_jcp
        rst_sigmaY_simu_jcp[i] = sigmaY_simu_jcp

        ## This paper
        sigmaY_eq, sigmaY_simu = hp.compare_SigmaY(Y,r,epsilon_fixed,D,E,F,m)
        rst_sigmaY_eq[i] = sigmaY_eq
        rst_sigmaY_simu[i] = sigmaY_simu

    error_sigY_eq_jcp = (rst_sigmaY_eq_jcp - rst_sigmaY_simu_jcp) / rst_sigmaY_simu_jcp * 100
    error_sigY_eq_jcp = np.abs(error_sigY_eq_jcp)
    error_sigY_eq = (rst_sigmaY_eq - rst_sigmaY_simu) / rst_sigmaY_simu * 100
    error_sigY_eq = np.abs(error_sigY_eq)
    print(f"Error of U by JCP: {error_sigY_eq_jcp}%")
    print(f"Error of U by this research: {error_sigY_eq}%")
    
    plot_compare_error(epsilons, error_sigY_eq, error_sigY_eq_jcp, r'$\it{\tau}$', \
        y1_label=r'Relative error (%) of $\it{\sigma}_Y$', \
        y2_label=r'Relative error (%) of $\it{\sigma}_Y$ (Wang et al. 2021)',\
        label1 = "This paper",\
        label2 = "Wang et al. 2021",\
        marker1='^', marker2='p',\
        fname="compare_error_sigY.tiff")


    for i, epsilon in enumerate(epsilons):
        ## get the U, and sigmaY from both equation and simulation of the JCP method
        U_eq_jcp,U_simu_jcp, _, _ = \
            casestudy_U_jcp(epsilon, control_cost=True)
        rst_U_eq_jcp[i] = U_eq_jcp
        rst_U_simu_jcp[i] = U_simu_jcp
        # rst_sigmaY_eq_jcp[i] = sigmaY_eq_jcp
        # rst_sigmaY_simu_jcp[i] = sigmaY_simu_jcp
        U_eq, U_simu, _, _ = \
            casestudy_U_this(epsilon, control_cost=True)
        rst_U_eq[i] = U_eq
        rst_U_simu[i] = U_simu
        # rst_sigmaY_eq[i] = sigmaY_eq
        # rst_sigmaY_simu[i] = sigmaY_simu

    # error_sigY_eq_jcp = (rst_sigmaY_eq_jcp - rst_sigmaY_simu_jcp) / rst_sigmaY_simu_jcp * 100
    # error_sigY_eq_jcp = np.abs(error_sigY_eq_jcp)
    # error_sigY_eq = (rst_sigmaY_eq - rst_sigmaY_simu) / rst_sigmaY_simu * 100
    # error_sigY_eq = np.abs(error_sigY_eq)
    # print(f"Error of U by JCP: {error_sigY_eq_jcp}%")
    # print(f"Error of U by this research: {error_sigY_eq}%")
    
    # plot_compare_error(epsilons, error_sigY_eq, error_sigY_eq_jcp, r'$\it{\tau}$', \
    #     y1_label=r'Relative error (%) of $\it{\sigma}_Y$', \
    #     y2_label=r'Relative error (%) of $\it{\sigma}_Y$ (Wang et al. 2021)',\
    #     label1 = "This paper",\
    #     label2 = "Wang et al. 2021",\
    #     marker1='^', marker2='p',\
    #     fname="compare_error_sigY.tiff")

    error_U_eq_jcp = (rst_U_eq_jcp - rst_U_simu_jcp) / rst_U_simu_jcp * 100
    error_U_eq_jcp = np.abs(error_U_eq_jcp)
    error_U_eq = (rst_U_eq - rst_U_simu) / rst_U_simu * 100
    error_U_eq = np.abs(error_U_eq)
    print(f"Error of U by JCP: {error_U_eq_jcp}%")
    print(f"Error of U by this research: {error_U_eq}%")
    
    plot_compare_error(epsilons, error_U_eq, error_U_eq_jcp, r'$\it{\tau_{max}}$', \
        y1_label=r'Relative error (%) of $U$', \
        y2_label=r'Relative error (%) of $U$ (Wang et al. 2021)',\
        label1 = "This paper",\
        label2 = "Wang et al. 2021",\
        marker1='+', marker2='x',\
        fname="compare_error_U.tiff")
    
    plot_compare_U(epsilons, rst_U_simu, rst_U_simu_jcp, r'$\it{\tau_{max}}$',\
         y_label="$U$",label1="This paper",label2="Wang et al. 2021")

def comparison_single_side(min_epsilon=0, max_epsilon=4, count=10):
    """
    Compare the proposed method to the method proposed in JCP paper.
    Consider single sided epsilon
    """
    epsilons = np.linspace(min_epsilon, max_epsilon, count, endpoint=True)
    rst_U_eq_jcp = np.zeros(count)
    rst_U_simu_jcp = np.zeros(count)
    rst_sigmaY_eq_jcp = np.zeros(count)
    rst_sigmaY_simu_jcp = np.zeros(count)
    rst_U_eq = np.zeros(count)
    rst_U_simu = np.zeros(count)
    rst_sigmaY_eq = np.zeros(count)
    rst_sigmaY_simu = np.zeros(count)
    for i, epsilon in enumerate(epsilons):
        ## get the U, and sigmaY from both equation and simulation of the JCP method
        U_eq_jcp,U_simu_jcp, sigmaY_eq_jcp, sigmaY_simu_jcp = \
            casestudy_U_jcp_single_side(epsilon, control_cost=True)
        rst_U_eq_jcp[i] = U_eq_jcp
        rst_U_simu_jcp[i] = U_simu_jcp
        rst_sigmaY_eq_jcp[i] = sigmaY_eq_jcp
        rst_sigmaY_simu_jcp[i] = sigmaY_simu_jcp
        U_eq, U_simu, sigmaY_eq, sigmaY_simu = \
            casestudy_U_this_single_side(epsilon, control_cost=True)
        rst_U_eq[i] = U_eq
        rst_U_simu[i] = U_simu
        rst_sigmaY_eq[i] = sigmaY_eq
        rst_sigmaY_simu[i] = sigmaY_simu

    error_sigY_eq_jcp = (rst_sigmaY_eq_jcp - rst_sigmaY_simu_jcp) / rst_sigmaY_simu_jcp * 100
    error_sigY_eq_jcp = np.abs(error_sigY_eq_jcp)
    error_sigY_eq = (rst_sigmaY_eq - rst_sigmaY_simu) / rst_sigmaY_simu * 100
    error_sigY_eq = np.abs(error_sigY_eq)
    print(f"Error of U by JCP: {error_sigY_eq_jcp}%")
    print(f"Error of U by this research: {error_sigY_eq}%")
    
    # plot_compare_error(epsilons, error_sigY_eq, error_sigY_eq_jcp, r'$\it{\tau_{max}}$', \
    #     y1_label=r'Relative error (%) of $\it{\sigma}_Y$ (this paper)', \
    #     y2_label=r'Relative error (%) of $\it{\sigma}_Y$ (Wang et al. 2021)',\
    #     label1 = "This paper",\
    #     label2 = "Wang et al. 2021",\
    #     marker1='^', marker2='p',\
    #     fname="compare_error_sigY.tiff")

    # error_U_eq_jcp = (rst_U_eq_jcp - rst_U_simu_jcp) / rst_U_simu_jcp * 100
    # error_U_eq_jcp = np.abs(error_U_eq_jcp)
    # error_U_eq = (rst_U_eq - rst_U_simu) / rst_U_simu * 100
    # error_U_eq = np.abs(error_U_eq)
    # print(f"Error of U by JCP: {error_U_eq_jcp}%")
    # print(f"Error of U by this research: {error_U_eq}%")
    # plot_compare_error(epsilons, error_U_eq, error_U_eq_jcp, x_label=r'$\it{\tau_{max}}$', \
    #     y1_label=r'Relative error (%) of $U$ (this paper)', \
    #     y2_label=r'Relative error (%) of $U$ (Wang et al. 2021)',\
    #     label1 = "This paper",\
    #     label2 = "Wang et al. 2021",\
    #     marker1='+', marker2='x',\
    #     fname="compare_error_U.tiff")
    
    plot_compare_error(epsilons, error_sigY_eq, error_sigY_eq_jcp, r'$\it{\tau_{max}}$', \
        y1_label=r'Relative error (%) of $\it{\sigma}_Y$', \
        y2_label=r'Relative error (%) of $\it{\sigma}_Y$ (Wang et al. 2021)',\
        label1 = "This paper",\
        label2 = "Wang et al. 2021",\
        marker1='^', marker2='p',\
        fname="compare_error_sigY.tiff")

    error_U_eq_jcp = (rst_U_eq_jcp - rst_U_simu_jcp) / rst_U_simu_jcp * 100
    error_U_eq_jcp = np.abs(error_U_eq_jcp)
    error_U_eq = (rst_U_eq - rst_U_simu) / rst_U_simu * 100
    error_U_eq = np.abs(error_U_eq)
    print(f"Error of U by JCP: {error_U_eq_jcp}%")
    print(f"Error of U by this research: {error_U_eq}%")
    plot_compare_error(epsilons, error_U_eq, error_U_eq_jcp, x_label=r'$\it{\tau_{max}}$', \
        y1_label=r'Relative error (%) of $U$ (this paper)', \
        y2_label=r'Relative error (%) of $U$ (Wang et al. 2021)',\
        label1 = "This paper",\
        label2 = "Wang et al. 2021",\
        marker1='+', marker2='x',\
        fname="compare_error_U.tiff")

    plot_compare_U(epsilons, rst_U_simu, rst_U_simu_jcp, r'$\it{\tau_{max}}$',\
         y_label="$U$",label1="This paper",label2="Wang et al. 2021")

if __name__ == "__main__":
    M,U,Y,result = casestudy_U()
    print("################## Experiment 1 ###################")
    comparison(r=result['r'])
    # M,U,Y,result = casestudy_U_single_side()
    # print("################## Experiment 1 ###################")
    # comparison_single_side(max_epsilon=4)