# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:53:00 2021

@author: wyue
"""

#import packages
import numpy as np
#import helper functions
import helpers as hp
#from scipy.stats import norm 
from scipy.spatial import distance

#number of components
m = 5
RECIPROCAL = False
small_number = 1e-7

A = np.array([0.98, 0.52, 1.23, 0.98, 0.52])
B = np.array([2.136, 1.8, 2.568,2.136, 1.8])

E = np.array([0.31, 0.23, 0.12, 0.15, 0.1])
F = np.array([0.01023333, 0.00603333 ,0.00713333, 0.01183333, 0.00733333])

## Control cost
GL = np.array([0.98, 0.52, 1.23, 0.98, 0.52])
HL = np.array([2.136, 1.8, 2.568, 2.136, 1.8])
VL = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
GR = np.array([0.98, 0.52, 1.23, 0.98, 0.52])
HR = np.array([2.136, 1.8, 2.568, 2.136, 1.8])
VR = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

#Scrap cost of a product
Sp = np.sum(A)/10.0

# #Nominal value of Y
# miuY = np.radians(7.0124)
# ##Upper specification limit
# USY = miuY + 0.035
# LSY = miuY - 0.035

X = np.array([500, 300, 375, 750, 125])
D1 = hp.dy_dx1_crank(X[0],X[1],X[2],X[3],X[4])
D2 = hp.dy_dx2_crank(X[0],X[1],X[2],X[3],X[4])
D3 = hp.dy_dx3_crank(X[0],X[1],X[2],X[3],X[4])
D4 = hp.dy_dx4_crank(X[0],X[1],X[2],X[3],X[4])
D5 = hp.dy_dx5_crank(X[0],X[1],X[2],X[3],X[4])

D = np.array([D1,D2,D3,D4,D5])

#r = 10.0 * np.random.rand(5)
#epsilon = 3.0 * np.random.rand(10)
r = 2.0 * np.ones(5)
epsilon = 3.0 * np.random.uniform(0.8, 1.1, 10)
    
sigmaX = hp.sigma(E,F,r)
sigmaX_loaf = hp.sigma_loaf_as(E,F,r,epsilon,m)


Cprocess = hp.Cprocess(A,B,r)
Ccontrol = hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsilon,m,RECIPROCAL)

sigmaY_Taylor = hp.sigmaY(sigmaX_loaf,D)

grad_numerical_r = np.zeros(m)
grad_equation_r = np.zeros(m)
grad_numerical_epsilon_L = np.zeros(m)
grad_equation_epsilon_L = np.zeros(m)
grad_numerical_epsilon_R = np.zeros(m)
grad_equation_epsilon_R = np.zeros(m)
miuX = hp.miu_x_as(X, epsilon, sigmaX, m)

# Nominal value of Y is 7.0124, but in this asymmetrical case, miuY should be computed
## by the mean of component using the design function.
miuY = hp.f_crank(miuX)
##Upper specification limit
USY = miuY + 4.0
LSY = miuY - 4.0

for i in range(0,m):  
    ## Inspect derivative of r
    ri_add_small_number = np.copy(r)
    ri_minus_small_number = np.copy(r)
    ri_add_small_number[i] +=  small_number
    ri_minus_small_number[i] -=  small_number
    sigmaX_plus = hp.sigma(E,F,ri_add_small_number)  
    sigmaX_minus = hp.sigma(E,F,ri_minus_small_number)  
    sigmaX_loaf_plus = hp.sigma_loaf_as(E,F,ri_add_small_number, epsilon, m)  
    sigmaX_loaf_minus = hp.sigma_loaf_as(E,F,ri_minus_small_number, epsilon, m)  
    miuX_plus = hp.miu_x_as(X, epsilon, sigmaX_plus, m)
    miuX_minus = hp.miu_x_as(X, epsilon, sigmaX_minus, m)
    miuY_plus = hp.f_crank(miuX_plus)
    miuY_minus = hp.f_crank(miuX_minus)
    sigmaY_Taylor_plus = hp.sigmaY(sigmaX_loaf_plus,D)
    sigmaY_Taylor_minus = hp.sigmaY(sigmaX_loaf_minus,D)
    Cprocess_plus = hp.Cprocess(A,B,ri_add_small_number)
    Cprocess_minus = hp.Cprocess(A,B,ri_minus_small_number)
    beta_plus = hp.productPassRate(LSY,USY,miuY_plus,sigmaY_Taylor_plus)
    beta_minus = hp.productPassRate(LSY,USY,miuY_minus,sigmaY_Taylor_minus)
    
    #gradient computed by numerical estimation
    grad_numerical_r[i] = (hp.U_as(Cprocess_plus,Ccontrol,Sp,beta_plus) -
                  hp.U_as(Cprocess_minus,Ccontrol,Sp,beta_minus))/(2*small_number)
    print('Numerical: '+'dr'+str(i),'=',grad_numerical_r[i])
    #gradient computed by equation
    epsilonL_i = epsilon[i]
    epsilonR_i = epsilon[i+m]
    dCprocessi_dri_v = hp.dCiprocess_dri(B[i],r[i])
    dsigmai_dr = hp.dsigmai_dri(F[i],r[i])
    dmiuy_dri_v = hp.dmiuY_dri(i,miuX,epsilonL_i,epsilonR_i,dsigmai_dr)
    dsigmai_loaf_dri_v = hp.dsigmai_loaf_dri_as(F[i],r[i],epsilonL_i,epsilonR_i)
    dsigmay_dri_v = hp.dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_loaf_dri_v)
    dbeta_dri_v = hp.dbeta_dri_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_dri_v, dsigmay_dri_v)
    beta = hp.productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    grad_equation_r[i] = hp.dU_dri_as(Cprocess,Ccontrol,Sp,dCprocessi_dri_v, beta, dbeta_dri_v)
    print('Equation: '+'dr'+str(i),'=',grad_equation_r[i])    
    

    ## Inspect epsilon
    ## Gradient computed by mumeracal estimation: left
    epsiloni_add_small_number_L = np.copy(epsilon[:])
    epsiloni_minus_small_number_L = np.copy(epsilon[:])
    epsiloni_add_small_number_L[i] +=  small_number
    epsiloni_minus_small_number_L[i] -=  small_number
    Ccontrol_plus_L = hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsiloni_add_small_number_L,m,RECIPROCAL)
    Ccontrol_minus_L = hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsiloni_minus_small_number_L,m,RECIPROCAL)
    sigmaX_loaf_plus_L = hp.sigma_loaf_as(E,F,r, epsiloni_add_small_number_L,m)
    sigmaX_loaf_minus_L = hp.sigma_loaf_as(E,F,r, epsiloni_minus_small_number_L,m)
    miuX_plus_L = hp.miu_x_as(X, epsiloni_add_small_number_L, sigmaX, m)
    miuX_minus_L = hp.miu_x_as(X, epsiloni_minus_small_number_L, sigmaX, m)
    miuY_plus_L = hp.f_crank(miuX_plus_L)
    miuY_minus_L = hp.f_crank(miuX_minus_L)
    sigmaY_Taylor_plus_L = hp.sigmaY(sigmaX_loaf_plus_L,D)
    sigmaY_Taylor_minus_L = hp.sigmaY(sigmaX_loaf_minus_L,D)
    beta_plus_L = hp.productPassRate(LSY,USY,miuY_plus_L,sigmaY_Taylor_plus_L)
    beta_minus_L = hp.productPassRate(LSY,USY,miuY_minus_L,sigmaY_Taylor_minus_L)
    grad_numerical_epsilon_L[i] = (hp.U_as(Cprocess,Ccontrol_plus_L,Sp,beta_plus_L) -
                  hp.U_as(Cprocess,Ccontrol_minus_L,Sp,beta_minus_L))/(2*small_number)
    print('Numerical: '+'depsilon (left)'+str(i),'=',grad_numerical_epsilon_L[i])

    ## Gradient by equation: left
    epsilonL_i = epsilon[i]
    epsilonR_i = epsilon[i+m]
    dCcontrol_depsiloni_L_v = hp.dCcontrol_depsiloni(HL[i],VL[i],epsilonL_i,symmetry=False,reciprocal=RECIPROCAL)
    dmiuy_depsiloni_L_v = hp.dmiuY_depsiloni(i,miuX,sigmaX[i],left=True)
    dsigmai_loaf_depsiloni_v = hp.dsigmai_loaf_depsiloni_as(epsilonL_i,epsilonR_i,sigmaX[i])
    beta = hp.productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    dsigmaY_depsiloni_v = hp.dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v)
    dbeta_depsiloni_L_v = hp.dbeta_depsiloni_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_depsiloni_L_v, dsigmaY_depsiloni_v)
    grad_equation_epsilon_L[i] = hp.dU_depsiloni_as(Cprocess,Ccontrol,Sp,dCcontrol_depsiloni_L_v, beta, dbeta_depsiloni_L_v)
    print('Equation: '+'depsilon (left)'+str(i),'=',grad_equation_epsilon_L[i])    
    
    ## Gradient computed by mumeracal estimation: right
    epsiloni_add_small_number_R = np.copy(epsilon[:])
    epsiloni_minus_small_number_R = np.copy(epsilon[:])
    epsiloni_add_small_number_R[i+m] +=  small_number ## Attention, do not forget "+m"
    epsiloni_minus_small_number_R[i+m] -=  small_number
    Ccontrol_plus_R = hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsiloni_add_small_number_R,m,RECIPROCAL)
    Ccontrol_minus_R = hp.Ccontrol_as(GL,HL,VL,GR,HR,VR,epsiloni_minus_small_number_R,m,RECIPROCAL)
    sigmaX_loaf_plus_R = hp.sigma_loaf_as(E,F,r, epsiloni_add_small_number_R,m)
    sigmaX_loaf_minus_R = hp.sigma_loaf_as(E,F,r, epsiloni_minus_small_number_R,m)
    miuX_plus_R = hp.miu_x_as(X, epsiloni_add_small_number_R, sigmaX, m)
    miuX_minus_R = hp.miu_x_as(X, epsiloni_minus_small_number_R, sigmaX, m)
    miuY_plus_R = hp.f_crank(miuX_plus_R)
    miuY_minus_R = hp.f_crank(miuX_minus_R)
    sigmaY_Taylor_plus_R = hp.sigmaY(sigmaX_loaf_plus_R,D)
    sigmaY_Taylor_minus_R = hp.sigmaY(sigmaX_loaf_minus_R,D)
    beta_plus_R = hp.productPassRate(LSY,USY,miuY_plus_R,sigmaY_Taylor_plus_R)
    beta_minus_R = hp.productPassRate(LSY,USY,miuY_minus_R,sigmaY_Taylor_minus_R)
    grad_numerical_epsilon_R[i] = (hp.U_as(Cprocess,Ccontrol_plus_R,Sp,beta_plus_R) -
                  hp.U_as(Cprocess,Ccontrol_minus_R,Sp,beta_minus_R))/(2*small_number)
    print('Numerical: '+'depsilon (right)'+str(i),'=',grad_numerical_epsilon_R[i])

    ## Gradient by equation: right
    epsilonL_i = epsilon[i]
    epsilonR_i = epsilon[i+m]
    dCcontrol_depsiloni_R_v = hp.dCcontrol_depsiloni(HR[i],VR[i],epsilonR_i,symmetry=False,reciprocal=RECIPROCAL)
    dmiuy_depsiloni_R_v = hp.dmiuY_depsiloni(i,miuX,sigmaX[i],left=False)
    dsigmai_loaf_depsiloni_v = hp.dsigmai_loaf_depsiloni_as(epsilonL_i,epsilonR_i,sigmaX[i])
    beta = hp.productPassRate(LSY,USY,miuY,sigmaY_Taylor)
    dsigmaY_depsiloni_v = hp.dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v)
    dbeta_depsiloni_R_v = hp.dbeta_depsiloni_as(LSY, USY, miuY, sigmaY_Taylor, dmiuy_depsiloni_R_v, dsigmaY_depsiloni_v)
    grad_equation_epsilon_R[i] = hp.dU_depsiloni_as(Cprocess,Ccontrol,Sp,dCcontrol_depsiloni_R_v, beta, dbeta_depsiloni_R_v)
    print('Equation: '+'depsilon (right)'+str(i),'=',grad_equation_epsilon_R[i])    



distance12_r =  distance.euclidean(grad_equation_r,grad_numerical_r)
length1_r = distance.euclidean(grad_equation_r,np.zeros_like(grad_equation_r))
length2_r = distance.euclidean(grad_numerical_r,np.zeros_like(grad_numerical_r))
graderror_r = distance12_r/(length1_r + length2_r)
print('error of dr=',graderror_r)

## Epsilon (left)
distance12_epsilon_L =  distance.euclidean(grad_equation_epsilon_L,grad_numerical_epsilon_L)
length1_epsilon_L = distance.euclidean(grad_equation_epsilon_L,np.zeros_like(grad_equation_epsilon_L))
length2_epsilon_L = distance.euclidean(grad_numerical_epsilon_L,np.zeros_like(grad_numerical_epsilon_L))
graderror_epsilon_L = distance12_epsilon_L/(length1_epsilon_L + length2_epsilon_L)
print('error of depsilon (left)=',graderror_epsilon_L)


## Epsilon (right)
distance12_epsilon_R=  distance.euclidean(grad_equation_epsilon_R,grad_numerical_epsilon_R)
length1_epsilon_R = distance.euclidean(grad_equation_epsilon_R,np.zeros_like(grad_equation_epsilon_R))
length2_epsilon_R = distance.euclidean(grad_numerical_epsilon_R,np.zeros_like(grad_numerical_epsilon_R))
graderror_epsilon_R = distance12_epsilon_R/(length1_epsilon_R + length2_epsilon_R)
print('error of depsilon (right)=',graderror_epsilon_R)

