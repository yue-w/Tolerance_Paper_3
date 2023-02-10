8# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:54:11 2019
Check the gradient of ri and ki bomputed by equation and by numerical method
@author: wyue
"""

#import packages
import numpy as np
#import helper functions
import helpers as hp
#from scipy.stats import norm 
from scipy.spatial import distance

#number of components
m = 3

small_number = 1e-7

A = np.array([0.98, 0.52, 1.23])
B = np.array([2.136, 1.8, 2.568])

E = np.array([0.032, 0.02136, 0.05336])
F = np.array([0.0004, 0.0002, 0.0006]) 

## Control cost
G = np.array([0.98, 0.52, 1.23])
H = np.array([2.136, 1.8, 2.568])
V = np.array([0.001, 0.001, 0.001])

#Scrap cost of a product
Sp = np.sum(A)/10.0

#Nominal value of Y
miuY = np.radians(7.0124)
##Upper specification limit
USY = miuY + 0.035

X= np.array([55.291, 22.86, 101.6])
D1 = hp.dy_dx1_clutch(X[0],X[1],X[2])
D2 = hp.dy_dx2_clutch(X[0],X[1],X[2])
D3 = hp.dy_dx3_clutch(X[0],X[1],X[2])

D = np.array([D1,D2,D3])


#r = 10.0 * np.random.rand(3)
r = np.ones(3)
epsilon = 3.0 * np.random.rand(3)

sigmaX = hp.sigma(E,F,r)
sigmaX_loaf = hp.sigma_loaf(E,F,r,epsilon)


Cprocess = hp.Cprocess(A,B,r)
Ccontrol = hp.Ccontrol(G,H,V,epsilon)

sigmaY_Taylor = hp.sigmaY(sigmaX_loaf,D)


grad_numerical_r = np.zeros(m)
grad_equation_r = np.zeros(m)
grad_numerical_epsilon = np.zeros(m)
grad_equation_epsilon = np.zeros(m)

for i in range(0,m):  
    ## Inspect derivate of r
    ri_add_small_number = np.copy(r)
    ri_minus_small_number = np.copy(r)
    ri_add_small_number[i] +=  small_number
    ri_minus_small_number[i] -=  small_number
    sigmaX_loaf_plus = hp.sigma_loaf(E,F,ri_add_small_number, epsilon)  
    sigmaX_loaf_minus = hp.sigma_loaf(E,F,ri_minus_small_number, epsilon)  
    sigmaY_Taylor_plus = hp.sigmaY(sigmaX_loaf_plus,D)
    sigmaY_Taylor_minus = hp.sigmaY(sigmaX_loaf_minus,D)
    Cprocess_plus = hp.Cprocess(A,B,ri_add_small_number)
    Cprocess_minus = hp.Cprocess(A,B,ri_minus_small_number)

    #gradient computed by numerical estimation
    grad_numerical_r[i] = (hp.U(Cprocess_plus,Ccontrol,USY,miuY,sigmaY_Taylor_plus,Sp) -
                  hp.U(Cprocess_minus,Ccontrol,USY,miuY,sigmaY_Taylor_minus,Sp))/(2*small_number)
    print('Numerical: '+'dr'+str(i),'=',grad_numerical_r[i])
    
    #gradient computed by equation
    dCi_dri_v = hp.dCiprocess_dri(B[i],r[i])
    dsigmai_dri_v = hp.dsigmai_loaf_dri(F[i],r[i],epsilon[i])
    dsigmaY_dri_v = hp.dsigmaY_dri(D,sigmaX_loaf,i,dsigmai_dri_v)        
    grad_equation_r[i] = hp.dU_dri(USY,miuY,sigmaY_Taylor,Cprocess,Ccontrol,dsigmaY_dri_v,dCi_dri_v,Sp)
    print('Equation: '+'dr'+str(i),'=',grad_equation_r[i])    
    
    
    ## Inspect derivate of epsilon
    ##varify epsilon        
    epsilon_add_small_number = np.copy(epsilon)
    epsilon_minus_small_number = np.copy(epsilon)
    epsilon_add_small_number[i] += small_number
    epsilon_minus_small_number[i] -= small_number  
    sigmaX_loaf_plus = hp.sigma_loaf(E,F,r, epsilon_add_small_number)  
    sigmaX_loaf_minus = hp.sigma_loaf(E,F,r, epsilon_minus_small_number)  
    sigmaY_Taylor_plus = hp.sigmaY(sigmaX_loaf_plus,D)
    sigmaY_Taylor_minus = hp.sigmaY(sigmaX_loaf_minus,D)
    Ccontrol_plus = hp.Ccontrol(G,H,V, epsilon_add_small_number)
    Ccontrol_minus = hp.Ccontrol(G,H,V, epsilon_minus_small_number)
    grad_numerical_epsilon[i] = (hp.U(Cprocess,Ccontrol_plus,USY,miuY,sigmaY_Taylor_plus,Sp)
    - hp.U(Cprocess,Ccontrol_minus, USY,miuY,sigmaY_Taylor_minus,Sp))/(2*small_number)
    print('Numerical: '+'depsilon'+str(i),'=',grad_numerical_epsilon[i])
    
    ##gradient computed by equation
    dCi_depsiloni_v = hp.dCcontrol_depsiloni(H[i],V[i],epsilon[i],symmetry=True)
    dsigmai_loaf_depsiloni_v = hp.dsigmai_loaf_depsiloni(epsilon[i], sigmaX[i])
    dsigmaY_depsiloni_v = hp.dsigmaY_depsiloni(D,sigmaX_loaf,i,dsigmai_loaf_depsiloni_v)
    grad_equation_epsilon[i] = hp.dU_depsiloni(USY, miuY,sigmaY_Taylor,Cprocess,
                                               Ccontrol,dsigmaY_depsiloni_v,dCi_depsiloni_v,Sp)
    print('Equation: '+'depsilon'+str(i),'=',grad_equation_epsilon[i])        
    


distance12_r =  distance.euclidean(grad_equation_r,grad_numerical_r)
length1_r = distance.euclidean(grad_equation_r,np.zeros_like(grad_equation_r))
length2_r = distance.euclidean(grad_numerical_r,np.zeros_like(grad_numerical_r))
graderror_r = distance12_r/(length1_r + length2_r)
print('error of dr=',graderror_r)

distance12_epsilon =  distance.euclidean(grad_equation_epsilon,grad_numerical_epsilon)
length1_epsilon = distance.euclidean(grad_equation_epsilon,np.zeros_like(grad_equation_epsilon))
length2_epsilon = distance.euclidean(grad_numerical_epsilon,np.zeros_like(grad_numerical_epsilon))
graderror_epsilon = distance12_epsilon/(length1_epsilon + length2_epsilon)
print('error of depsilon=',graderror_epsilon)    


