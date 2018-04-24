from scipy.stats import norm
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

#Task 5 Arithmetic Mean Basket call/put options
#Implement Asian basket options w/wo variates
    # S1 = Stock price #1
    # S2 = Stock price #2
    # sigma1 = volatility of stock #1
    # sigma2 = volatility of stock #2
    # r = risk free interest rate
    # T = time to maturity
    # K = strike price
    # p = correlation coefficient
    # callputswitch = 'c' or 'p' ; call/put options.
    # M = number of paths of stock price
    # sigma = volatility
    # control = 1 ; use control variate

def ArithmeticBasketOption (S1, S2, sigma1, sigma2, r, T, K, p, CallPutSwitch, M, Control):
    #Geometric Basket Call/Put option
    sigmaBg =  np.sqrt(sigma1*sigma1 + sigma1*sigma2*p + sigma2*sigma1*p + sigma2*sigma2) /2
    uBg = r - 0.5 * (sigma1*sigma1 + sigma2*sigma2)/2 + 0.5*sigmaBg*sigmaBg
    Bg0 = np.sqrt(S1 * S2)
    
    d1 = (np.log(Bg0/K) + (uBg + 0.5*sigmaBg*sigmaBg)*T) / (sigmaBg * sqrt(T))
    d2 = d1 - sigmaBg * np.sqrt(T)

    if CallPutSwitch == 'c':
            geoBasket = np.exp(-r * T) * (Bg0*np.exp(uBg*T)*norm.cdf(d1) - K*norm.cdf(d2))
    else:
            geoBasket = np.exp(-r * T) * (K*norm.cdf(-d2) - (Bg0*np.exp(uBg*T)*norm.cdf(-d1)))
            
    #Monte Carlo simulation on Arithmetic Basket options
    N = 1
    dt = T * 1.0 / N
    Spath1 = np.empty(N, dtype=float)
    Spath2 = np.empty(N, dtype=float)
    SpathArith = np.empty(N, dtype=float)
    SpathGeo = np.empty(N, dtype=float)
    arithPayOff = np.empty(M, dtype=float)
    geoPayOff = np.empty(M, dtype=float)

    #Calcuting S path
    drift1 = exp((r-0.5*sigma1*sigma1) * dt)
    drift2 = exp((r-0.5*sigma2*sigma2) * dt)

    np.random.seed(seed=2334)

    for i in range (0,M,1):
        # Random generated variable required to have a correlation of p
        randnX = np.random.randn(1)
        randnY = np.random.randn(1)

        growthFactor1 = drift1 * exp(sigma1 * sqrt(dt) * randnX)
        growthFactor2 = drift2 * exp(sigma2 * sqrt(dt) * (p*randnX + np.sqrt(1-p*p)*randnY))
        Spath1[0] = S1 * growthFactor1
        Spath2[0] = S2 * growthFactor2
        
        for j in range (1,N,1):
            randnX = np.random.randn(1)
            randnY = np.random.randn(1)

            growthFactor1 = drift1 * exp(sigma1 * sqrt(dt) * randnX)
            growthFactor2 = drift2 * exp(sigma2 * sqrt(dt) * (p*randnX + np.sqrt(1-p*p)*randnY))
            Spath1[j] = Spath1[j-1] * growthFactor1
            Spath2[j] = Spath2[j-1] * growthFactor2
        
        #Basket options (Sum mean for arithmetic and product mean for geometric)
        SpathArith = (Spath1+Spath2)/2
        SpathGeo = np.sqrt(Spath1*Spath2)

        #arithmetic mean
        arithMean = np.mean(SpathArith)
        #geometric mean
        geoMean = np.exp((1/(N)) * np.sum(np.log(SpathGeo)))

        if CallPutSwitch == 'c':
            arithPayOff[i] = exp(-r * T) * max(arithMean - K, 0)
            geoPayOff[i] = exp(-r * T) * max(geoMean - K, 0)
        else:
            arithPayOff[i] = exp(-r * T) * max(K - arithMean, 0)
            geoPayOff[i] = exp(-r * T) * max(K - geoMean, 0)

    #standard Monte Carlo (Arithmetic)
    Pmean = np.mean(arithPayOff)
    Pstd = np.std(arithPayOff)
    confmc = [Pmean-1.96*Pstd/sqrt(M), Pmean+1.96*Pstd/sqrt(M)]
    #print("Arithmetic MC payoff:", confmc)

    #standard Monte Carlo (Geometric)
    #Pmean = np.mean(geoPayOff)
    #Pstd = np.std(geoPayOff)
    #confmc = [Pmean-1.96*Pstd/sqrt(M), Pmean+1.96*Pstd/sqrt(M)]
    #print("Geometric MC payoff:", confmc)

    #control variate
    covXY = np.mean(arithPayOff * geoPayOff) - np.mean(arithPayOff) * np.mean(geoPayOff)
    theta = covXY / np.var(geoPayOff)

    #control variate version
    Z = arithPayOff + theta * (geoBasket - geoPayOff)
    Zmean = np.mean(Z)
    Zstd = np.std(Z)
    confcv = [Zmean-1.96*Zstd /sqrt(M), Zmean+1.96*Zstd /sqrt(M)]

    #print ("No Control  : " + str(confmc))
    #print ("With Control: " + str(confcv))

    if Control == 1:
        return np.mean(confcv)
    return np.mean(confmc)

# Sample Test Cases
'''
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 10000, Control = 0)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 10000, Control = 0))
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 10000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 10000, Control = 1))

print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 10000, Control = 0)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 10000, Control = 0))
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 10000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 10000, Control = 1))
'''

# Required Test Cases
# Put Options #1

print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1))
# Put Options #2
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.9, CallPutSwitch = 'p', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.9, CallPutSwitch = 'p', M = 100000, Control = 1))
# Put Options #3
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.1, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.1, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1))
# Put Options #4
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 80, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 80, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1))
# Put Options #5
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 120, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 120, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1))
# Put Options #6
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.5, sigma2 = 0.5, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.5, sigma2 = 0.5, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p', M = 100000, Control = 1))

# Call Options #1
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1))
# Call Options #2
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.9, CallPutSwitch = 'c', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.9, CallPutSwitch = 'c', M = 100000, Control = 1))
# Call Options #3
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.1, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.1, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1))
# Call Options #4
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 80, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 80, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1))
# Call Options #5
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 120, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 120, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1))
# Call Options #6
print ("ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.5, sigma2 = 0.5, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1)")
print (ArithmeticBasketOption(S1 = 100, S2 = 100, sigma1 = 0.5, sigma2 = 0.5, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c', M = 100000, Control = 1))

