# -*- coding: utf-8 -*-
"""
Created on Sun May  9 12:21:27 2021

@author: wyue
"""
import numpy as np
from scipy.special import erf
import math
from scipy.special import factorial
from scipy.stats import norm 
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

#n is the order of polimonial function to approximate the error function and its derivative
n = 120
THETA = 1/erf(4/erf(2))


#Cost-Rate function. The ith component is the ith component in the returned array
def Cprocess(A,B,r):
    return np.add(A,np.divide(B,r))

#Cost-epsilon function. The ith component is the ith component in the returned array
def Ccontrol(G,H,V,epsilon):
    return np.add(G,np.divide(H,np.add(epsilon,V)))

def Ccontrol_as(GL,HL,VL,GR,HR,VR,epsilon,m,reciprocal=True):
    if reciprocal:
        epsilon_L_vec = epsilon[0:m]
        epsilon_R_vec = epsilon[m:]
        return 0.5*np.add(GL,np.divide(HL,np.add(epsilon_L_vec,VL))) + 0.5*np.add(GR,np.divide(HR,np.add(epsilon_R_vec,VR)))
    else:
        epsilon_L_vec = epsilon[0:m]
        epsilon_R_vec = epsilon[m:]
        v1 = GL + HL * np.exp(-epsilon_L_vec * VL)
        v2 = GR + HR * np.exp(-epsilon_R_vec * VR)
        return v1 + v2

def Ccontrol_as_single_side(G,H,V,epsilon,reciprocal=True):
    if reciprocal:
        pass
    else:
        v = G + H * np.exp(-epsilon * V)
        return v

def assem_fun(x):
    """
    The design function
    y = f(x)
    """
    x1, x2, x3, x4, x5 = x
    y = x2 + np.sqrt(x4*x4-0.5*x3*x3-np.power(x1-x5-np.sqrt(2)/2*x3,2))
    return y
def assembly(x,clutch=True):
    if clutch:
        x1=np.copy(x[0])
        x2=np.copy(x[1])
        x3=np.copy(x[2])
        ## Shuffle for random assembly
        np.random.shuffle(x1)
        np.random.shuffle(x2)
        np.random.shuffle(x3)
        dividV = np.divide((x1+x2),(x3-x2))
        dividV = dividV[np.logical_and(dividV>=-1,dividV<=1)]
        y=np.arccos(dividV)
        #Y=np.arccos(np.divide((X1+X2),(X3-X2))) 
    else:
        x1=np.copy(x[0])
        x2=np.copy(x[1])
        x3=np.copy(x[2])
        x4=np.copy(x[3])
        x5=np.copy(x[4])
        ## Shuffle for random assembly
        np.random.shuffle(x1)
        np.random.shuffle(x2)
        np.random.shuffle(x3)
        np.random.shuffle(x4)
        np.random.shuffle(x5)
        y = x2 + np.sqrt(x4*x4-0.5*x3*x3-np.power(x1-x5-np.sqrt(2)/2*x3,2))
    return y



# Design function of clutch
def f_clutch(X):
    X1=X[0]
    X2=X[1]
    X3=X[2]
    dividV = np.divide((X1+X2),(X3-X2))
    dividV = dividV[np.logical_and(dividV>=-1,dividV<=1)]
    Y=np.arccos(dividV)
    #Y=np.arccos(np.divide((X1+X2),(X3-X2))) 
    return Y
    
# Design function of crank
def f_crank(x):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    x5=x[4]

    y = x2 + np.sqrt(x4*x4-0.5*x3*x3-np.power(x1-x5-np.sqrt(2)/2*x3,2))
    return y


def plot_hist(vec,num_bins=10,log=False):
    fig, ax = plt.subplots(1, ncols=1)
    ax.hist(vec, num_bins, log=log)
    ax.set_title('Y')
    
    fig.tight_layout()
    plt.show()
    #fig.savefig(fname='hist',dpi=300)
    

# =============================================================================
# def normaltest(vec,alpha = 1e-3):
#     k2, p = stats.normaltest(vec)
#     print("p = {:g}".format(p))
#     
#     if p < alpha:  # null hypothesis: x comes from a normal distribution
#         print("The null hypothesis can be rejected")
#     else:
#         print("The null hypothesis cannot be rejected")
# 
# def probability_plot(x):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     res = stats.probplot(x, plot=plt)
#     ax.set_title("Probability plot")
# =============================================================================
    
def stat_parameters_simulation(y):
    print("******Simulation******")
    miu_y = np.mean(y)
    #print("Mean: ", miu_y)
    std_y = np.std(y)
    #print("Sigma: ", std_y)
    return (miu_y,std_y)
    
    
    
#The derivative of X1 to the clutch design function: dy/dx1. (Also noted as D1)
def dy_dx1_clutch(x1,x2,x3):
    return -1.0/(math.sqrt(1-math.pow((x1+x2)/(x3-x2),2))*(x3-x2))
    
#The derivative of X2 to the clutch design function: dy/dx2 (Also noted as D2)
def dy_dx2_clutch(x1,x2,x3):
    return -(x1+x3)/(math.sqrt(1.0-math.pow((x1+x2)/(x3-x2),2))*math.pow(x3-x2,2))

#The derivative of X3 to the clutch design function: dy/dx3 (Also noted as D3)
def dy_dx3_clutch(x1,x2,x3):
    return (x1+x2)/(math.sqrt(1.0-math.pow((x1+x2)/(x3-x2),2))*math.pow(x3-x2,2))

#The derivative of X1 to the crank design function: dy/dx1. (Also noted as D1)
def dy_dx1_crank(x1,x2,x3,x4,x5):
    v1 = x1-x5-np.sqrt(2)/2*x3
    v2 = np.power(x4*x4-0.5*x3*x3-v1*v1,-0.5)
    return -v1*v2

#The derivative of X2 to the crank design function: dy/dx2. (Also noted as D2)
def dy_dx2_crank(x1,x2,x3,x4,x5):
    return 1

#The derivative of X3 to the crank design function: dy/dx3. (Also noted as D3)
def dy_dx3_crank(x1,x2,x3,x4,x5):
    v1 = x1-x5-np.sqrt(2)/2*x3
    v2 = 0.5*np.power(x4*x4-0.5*x3*x3-v1*v1,-0.5)
    v3 = -x3 + np.sqrt(2)*v1
    return v2*v3

#The derivative of X4 to the crank design function: dy/dx4. (Also noted as D4)
def dy_dx4_crank(x1,x2,x3,x4,x5):
    v1 = x1-x5-np.sqrt(2)/2*x3
    return x4*np.power(x4*x4-0.5*x3*x3-v1*v1,-0.5)

#The derivative of X5 to the crank design function: dy/dx5. (Also noted as D5)
def dy_dx5_crank(x1,x2,x3,x4,x5):
    v1 = x1-x5-np.sqrt(2)/2*x3
    v2 = np.power(x4*x4-0.5*x3*x3-v1*v1,-0.5)
    return v1*v2

def sigmaY(sigmaX,D):
    VY = np.sum(np.multiply(np.power(D,2),np.power(sigmaX,2)))
    V = np.sqrt(VY)
    return V


# def stat_parameters_analysis(X,sigmaX):
#     print("******Analysis******")
#     D1 = dy_dx1_clutch(X[0],X[1],X[2])
#     D2 = dy_dx2_clutch(X[0],X[1],X[2])
#     D3 = dy_dx3_clutch(X[0],X[1],X[2])
    
#     D = np.array([D1,D2,D3])
    
#     sigmaY_Taylor = sigmaY(sigmaX,D)  
#     print("sigma",sigmaY_Taylor)
#     return sigmaY_Taylor

# =============================================================================
# #Cost-Rate function. The ith component is the ith component in the returned array
# def sigma(E,F,r):
#     return np.add(E,np.multiply(F,np.power(r,2)))
# =============================================================================

def sigma_loaf(E,F,r, epsilon):
    sigma = np.add(E,np.multiply(F,np.power(r,2)))
    variance = np.power(sigma,2)
    tem = 1.0 + np.power(epsilon,2)/3 
    sigma_new = np.multiply(tem,variance)
    return np.sqrt(sigma_new)

def sigma_loaf_as(E,F,r,epsilon,m,double_side=True):
    if double_side:
        sigma = np.add(E,np.multiply(F,np.power(r,2)))
        variance = np.power(sigma,2)
        epsilon_L_vec = epsilon[0:m]
        epsilon_R_vec = epsilon[m:]
        tem = 1 + np.power(epsilon_L_vec+epsilon_R_vec,2)/12
        sigma_new = np.multiply(tem,variance)
    else:
        sigma = np.add(E,np.multiply(F,np.power(r,2)))
        variance = np.power(sigma,2)
        tem = 1 + np.power(epsilon,2)/12
        sigma_new = np.multiply(tem,variance)
    return np.sqrt(sigma_new)

def sigma(E,F,r):
    return np.add(E,np.multiply(F,np.power(r,2)))

#Objective function: Unit cost of a product
def U(Cprocess,Ccontrol,USY,miuY,sigmaY,Sp):
 #The parameters are arrays.   
    USp = USY - miuY
    sqrt2 = np.sqrt(2)
    erfV = erf(USp/sqrt2/sigmaY) 
    term1 = (np.sum(Cprocess)+np.sum(Ccontrol))/erfV
    term2 = (1.0/erfV-1.0) * Sp
    
    return term1 + term2

#Objective function of asymetrical case: Unit cost of a product
def U_as(Cprocess,Ccontrol,Sp,beta):
    return (np.sum(Cprocess) + np.sum(Ccontrol) + Sp*(1-beta))/beta


#dci_dri
def dCiprocess_dri(Bi,ri):
    return -Bi*math.pow(ri,-2)

#dci_depsiloni
def dCcontrol_depsiloni(Hi,Vi,epsiloni, reciprocal=True):
    if reciprocal:
        return -0.5*Hi*math.pow(epsiloni+Vi,-2)
    else:
        return -Hi*Vi*np.exp(-epsiloni*Vi)

#dsigma/dr
def dsigmai_dri(Fi,ri):
    return 2*Fi*ri

#dsigma_loaf/dr
def dsigmai_loaf_dri(Fi,ri,epsilon):
    dsigmai_dri_v = dsigmai_dri(Fi,ri)
    return dsigmai_dri_v*np.sqrt(1+epsilon*epsilon/3)

def dsigmai_loaf_dri_as(Fi,ri,epsilon_i_L,epsilon_i_R):
    dsigmai_dri_v = dsigmai_dri(Fi,ri)
    return dsigmai_dri_v*np.sqrt(1+np.power(epsilon_i_L+epsilon_i_R,2)/12)


#dsigma/depsilon
def dsigmai_loaf_depsiloni(epsiloni, sigmaX_i):
    return np.power(1.0+epsiloni*epsiloni/3,-0.5)*epsiloni*sigmaX_i/3

#dsigma/depsilon
def dsigmai_loaf_depsiloni_as(epsilon_i_L,epsilon_i_R, sigmaX_i):
    return sigmaX_i*np.power(1+np.power(epsilon_i_L+epsilon_i_R,2)/12,-0.5)*(epsilon_i_L+epsilon_i_R)/12


#dsigmaY/dri
def dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_loaf_dri_v):
    tem = np.sum(np.multiply(np.power(D,2),np.power(sigmaX_loaf,2)))
    v = np.power(tem,-0.5)*(D[i]**2)*sigmaX_loaf[i]*dsigmai_loaf_dri_v
    return v

# dsigmaY/depsiloni
def dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v):
    tem = np.sum(np.multiply(np.power(D,2),np.power(sigmaX_loaf,2)))
    v = np.power(tem,-0.5)*(D[i]**2)*sigmaX_loaf[i]*dsigmai_loaf_depsiloni_v
    return v

#Approximat the deriative of an error function. The error function is approximated
def derf_dx(x,n):
    #return (1.0-x**2+x**4/2.0-x**6/6.0)*2.0/math.sqrt(math.pi)
    val = 0.0
    token = 1.0
    for i in range(0,n):
        val += token * np.power(x,2*i) / factorial(i)
        token *= -1 
    val = val * 2 / np.sqrt(np.pi)
    return val

#dU_dri
def dU_dri(USY, miuY,sigmaY,Cprocess,Ccontrol,dsigmaY_dri,dCprocessi_dri_v,Sp):
    t = USY - miuY
    q = t/(np.sqrt(2)*sigmaY)
    tem2 = q/sigmaY
    tem3 = math.pow(erf(q),-2)*derf_dx(q,n)*tem2*dsigmaY_dri
    tem4 = np.sum(Cprocess) + Sp + np.sum(Ccontrol)
    tem5 = dCprocessi_dri_v/erf(q)
    return tem3*tem4 + tem5

def dU_dri_as(Cprocess,Ccontrol,Sp,dCprocessi_dri_v, beta, dbeta_dri_v):
    sum_cost = np.sum(Cprocess) + Sp + np.sum(Ccontrol)
    value = dCprocessi_dri_v/beta - np.power(beta,-2)*sum_cost*dbeta_dri_v 
    return value
    

def dU_depsiloni(USY,miuY,sigmaY,Cprocess,Ccontrol,dsigmaY_depsiloni_v,dCi_depsiloni_v,Sp):
    t = USY - miuY
    q = t/(np.sqrt(2)*sigmaY)
    tem2 = q/sigmaY
    tem3 = math.pow(erf(q),-2)*derf_dx(q,n)*tem2*dsigmaY_depsiloni_v
    tem4 = np.sum(Cprocess) + np.sum(Ccontrol) + Sp
    #tem5 = (dCi_depsiloni_v - Sp*derf_dx(q,n)*tem2*dsigmaY_depsiloni_v)/erf(q)
    tem5 = dCi_depsiloni_v/erf(q)
    return tem3*tem4 + tem5

def dU_depsiloni_as(Cprocess,Ccontrol,Sp,dCcontroli_depsiloni_v, beta, dbeta_depsiloni_v):
    sum_cost = np.sum(Cprocess) + Sp + np.sum(Ccontrol)
    value = dCcontroli_depsiloni_v/beta - np.power(beta,-2)*sum_cost*dbeta_depsiloni_v 
    return value

    
def estimateM(X,E,F,r_opt,epsilon_opt,USY,Y0,Na,Nb,m,clutch=True,double_side=True):
    x = generate_component(E,F,r_opt,epsilon_opt,m,Na,Nb,X,double_side)
    Y = assembly(x,clutch)
    Y_inspect = Y[np.logical_and(Y>=2*Y0-USY,Y<=USY)]
    M = len(Y_inspect)
    return (M, Y, Y_inspect)

def beta_equation(E,F,r,epsilon,m,LSY,USY, miuY,D,double_side=True):
    """
    Estimate satisfaction rate of product by replacing optimal
    values of r and epsilon into equation (13)
    """

    sigmaX_loaf = sigma_loaf_as(E,F,r,epsilon,m,double_side)
    sigmaY_Taylor = sigmaY(sigmaX_loaf,D)
    beta = productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    return beta

# def M_equation():
#     """
#     Estimate the number of satisfactory 
#     """
#     pass

def U_simulation(r,epsilon,para,X,USY,miuY,Sp,Na,Nb,m, clutch=True,reciprocal=True):

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


    Cprocess_v = Cprocess(A,B,r)
    
    Ccontrol_v = np.zeros(m)
    if control_cost:    
        Ccontrol_v = Ccontrol_as(GL,HL,VL,GR,HR,VR,epsilon,m,reciprocal)
    NSample = Na*Nb
    M, Y, _ = estimateM(X,E,F,r,epsilon,USY,miuY,Na,Nb,m,clutch)
    Ct = np.sum(np.multiply(NSample,Cprocess_v+Ccontrol_v)) + Sp*(np.abs(NSample-M))
    ## debug
    # Ct = np.sum(np.multiply(NSample,Cprocess_v)) + Sp*(np.abs(NSample-M))

    U = Ct/M

    return (M,U,Y)

def U_simulation_single_side(r,epsilon,para,X,USY,miuY,Sp,Na,Nb,m, clutch=True,reciprocal=True):

    A = para["A"]
    B = para["B"]
    E = para["E"]
    F = para["F"]
    G = para["G"]
    H = para["H"]
    V = para["V"]
    # D = para["D"]
    control_cost = para["control_cost"]


    Cprocess_v = Cprocess(A,B,r)
    
    Ccontrol_v = np.zeros(m)
    if control_cost:    
        Ccontrol_v = Ccontrol_as_single_side(G,H,V,epsilon,reciprocal)
    NSample = Na*Nb
    M, Y, _ = estimateM(X,E,F,r,epsilon,USY,miuY,Na,Nb,m,clutch,double_side=False)
    Ct = np.sum(np.multiply(NSample,Cprocess_v+Ccontrol_v)) + Sp*(np.abs(NSample-M))
    ## debug
    # Ct = np.sum(np.multiply(NSample,Cprocess_v)) + Sp*(np.abs(NSample-M))

    U = Ct/M

    return (M,U,Y)

def compare_SigmaY(Y_samples,r,epsilon,D,E,F,m,double_side=True):
    sigmaX_loaf = sigma_loaf_as(E,F,r, epsilon,m,double_side)
    sigmaY_equation = sigmaY(sigmaX_loaf,D)
    _,sigmaY_simulation = stat_parameters_simulation(Y_samples)
    print("sigmaY simulation = ", sigmaY_simulation)
    print("sigmaY equation = ", sigmaY_equation)
    error = (sigmaY_simulation-sigmaY_equation)/sigmaY_simulation
    print(f'error = {error * 100} %')
    return sigmaY_equation, sigmaY_simulation
    
def generate_component(E,F,r_opt,epsilon_opt,m,Na,Nb,X, double_side=True):
    if double_side:
        sigmaX = sigma(E,F,r_opt)
        ## Generate normal distribution, the mean of which deviate.
        x_vec = np.zeros((m,Na,Nb))
        for i in range(m): ## For each type of component
            for j in range(Na): ## The mean deviates Na times
                left = epsilon_opt[i]*sigmaX[i]
                right = epsilon_opt[i+m]*sigmaX[i]
                miu_low = X[i] - left
                miu_high = X[i] + right
                miu = np.random.uniform(miu_low,miu_high,1)
                x_vec[i][j][:] = np.random.normal(miu,sigmaX[i],Nb)
        
        x_vec = x_vec.reshape(m,-1)
        return x_vec
    else:
        sigmaX = sigma(E,F,r_opt)
        ## Generate normal distribution, the mean of which deviate.
        x_vec = np.zeros((m,Na,Nb))
        for i in range(m): ## For each type of component
            for j in range(Na): ## The mean deviates Na times
                left = 0
                right = epsilon_opt[i]*sigmaX[i]
                miu_low = X[i] - left
                miu_high = X[i] + right
                miu = np.random.uniform(miu_low,miu_high,1)
                x_vec[i][j][:] = np.random.normal(miu,sigmaX[i],Nb)
        
        x_vec = x_vec.reshape(m,-1)
        return x_vec

def plot(r_opt, epsilon_opt, E,F,m,Na,Nb,X,clutch=True):
    x_sample = generate_component(E,F,r_opt,epsilon_opt,m,Na,Nb,X)
    y_sample = assembly(x_sample,clutch)
    width = 12
    height = 21
    fig, axs = plt.subplots(m+1, 1,figsize=(width,height))

    n_bins = 50
    for i in range(m):
        axs[i].hist(x_sample[i], bins=n_bins)
        title = 'Component '+str(i+1)
        axs[i].set_title(title)
    axs[m].hist(y_sample, bins=n_bins)
    axs[m].set_title('Product')
    
    fig.savefig(fname='hist_component_product.tif',dpi=300)
    plt.show()

def plot_test(r_opt, epsilon_opt, E,F,m,Na,Nb,X,clutch=True,double_side=True):
    x_sample = generate_component(E,F,r_opt,epsilon_opt,m,Na,Nb,X,double_side)
    y_sample = assembly(x_sample,clutch)
    width = 12
    height = 21
    # fig, axs = plt.subplots(m+1, 1,figsize=(width,height))

    n_bins = 50
    for i in range(m):
        plt.figure()
        plt.hist(x_sample[i], bins=n_bins,density=False, edgecolor="yellow", facecolor='green')
        title = f'$X_{i+1}$' 
        plt.title(title)
        plt.xlabel("Dimension (mm)")
        plt.ylabel("Frequency")
        plt.savefig(fname='hist_component_' + str(i+1) + ".tif", dpi=300)
    plt.figure()
    plt.hist(y_sample, bins=n_bins,density=False, edgecolor="yellow", facecolor='green')
    plt.xlabel("Dimension (mm)")
    plt.ylabel("Frequency")
    plt.title('$Y$')
    plt.savefig(fname='hist_Y.tif',dpi=300)
    plt.show()

def phi(x):
    return 0.5*(1+erf(x/np.sqrt(2)))
    
def productPassRate(LS,US,miu_y,sigma_y):
    return phi((US-miu_y)/sigma_y) - phi((LS-miu_y)/sigma_y)
    
def dphi_dx(x,n):
    return 0.5*derf_dx(x/np.sqrt(2),n)/np.sqrt(2)
    
def dbeta_dri_as(LS, US, miuY, sigma_y, dmiuy_dri_v, dsigmay_dri_v):
    q1 = (US-miuY)/sigma_y
    q2 = (LS-miuY)/sigma_y
    dphi_dz_q1 = dphi_dx(q1,n)
    dphi_dz_q2 = dphi_dx(q2,n)
    tem1 = dphi_dz_q1 * (-dmiuy_dri_v*sigma_y-(US-miuY)*dsigmay_dri_v)/(sigma_y*sigma_y)
    tem2 = dphi_dz_q2 * (-dmiuy_dri_v*sigma_y-(LS-miuY)*dsigmay_dri_v)/(sigma_y*sigma_y)
    return tem1 - tem2

def get_D(miuX):
    D1 = dy_dx1_crank(*miuX)
    D2 = dy_dx2_crank(*miuX)
    D3 = dy_dx3_crank(*miuX)
    D4 = dy_dx4_crank(*miuX)
    D5 = dy_dx5_crank(*miuX)
    D = np.array([D1,D2,D3,D4,D5])
    return D
def dbeta_depsiloni_as(LS, US, miuY, sigma_y, dmiuy_depsiloni_v, dsigmay_depsiloni_v):
    q1 = (US-miuY)/sigma_y
    q2 = (LS-miuY)/sigma_y
    dphi_dz_q1 = dphi_dx(q1,n)
    dphi_dz_q2 = dphi_dx(q2,n)
    tem1 = dphi_dz_q1 * (-dmiuy_depsiloni_v*sigma_y-(US-miuY)*dsigmay_depsiloni_v)/(sigma_y*sigma_y)
    tem2 = dphi_dz_q2 * (-dmiuy_depsiloni_v*sigma_y-(LS-miuY)*dsigmay_depsiloni_v)/(sigma_y*sigma_y)
    return tem1 - tem2

def df_dmiu_i(miuX,i):
    ## if clutch
    if len(miuX) == 3:
        if i==0:
            return dy_dx1_clutch(miuX[0],miuX[1],miuX[2])
        elif i==1:
            return dy_dx2_clutch(miuX[0],miuX[1],miuX[2])
        elif i==2:
            return dy_dx3_clutch(miuX[0],miuX[1],miuX[2])
        else:
            return None
    ## if crank
    else:
        if i==0:
            return dy_dx1_crank(miuX[0],miuX[1],miuX[2],miuX[3],miuX[4])
        elif i==1:
            return dy_dx2_crank(miuX[0],miuX[1],miuX[2],miuX[3],miuX[4])
        elif i==2:
            return dy_dx3_crank(miuX[0],miuX[1],miuX[2],miuX[3],miuX[4])
        elif i==3:
            return dy_dx4_crank(miuX[0],miuX[1],miuX[2],miuX[3],miuX[4])
        elif i==4:
            return dy_dx5_crank(miuX[0],miuX[1],miuX[2],miuX[3],miuX[4])
        else:
            return None
def miu_x_as(X, epsilon, sigmaX, m, double_side=True):
    if double_side:
        epsilon_L_vec = epsilon[0:m]
        epsilon_R_vec = epsilon[m:]
        return X + np.multiply((epsilon_R_vec-epsilon_L_vec)/2,sigmaX)
    else:
        return X + np.multiply(epsilon/2,sigmaX)

def dmiuY_dri(i,miuX,epsilonLi,epsilonRi,dsigmai_dr):
    df_dmiu_i_v = df_dmiu_i(miuX,i)
    return df_dmiu_i_v*dsigmai_dr*(epsilonRi-epsilonLi)/2
    
def dmiuY_depsiloni(i,miuX,sigmai,left):
    df_dmiu_i_v = df_dmiu_i(miuX,i)
    if left:
        return -df_dmiu_i_v*sigmai/2
    else:
        return df_dmiu_i_v*sigmai/2
    
    
    
    