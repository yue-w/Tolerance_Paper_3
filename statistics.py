# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:16:13 2021

@author: wyue
"""

from scipy.stats import norm 
from scipy.special import erf
import numpy as np


#Percent point function (ppf, inverse of cdf — percentiles).
## miu = 0. Solve zScore so that 95% is between (-zScore, zScore), or 2.5% smaller than -zScore and 2.5% larger than zScore 
zScore = norm.ppf(0.975)
print(zScore)

#Cumulative distribution function.
p = norm.cdf(0,loc=0,scale=1)
print(p)

##Use the Gauss error function to approximate the cumulative distribution function
z = 2
p = norm.cdf(z,loc=0,scale=1)
p_error = 0.5*(1+erf(z/np.sqrt(2)))
print(p-p_error)
