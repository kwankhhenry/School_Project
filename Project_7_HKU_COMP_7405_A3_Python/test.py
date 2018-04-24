from scipy.stats import norm
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

def ArithmeticAsianGeoOptionWithControlVariates (CallPutSwitch,S1,S2,K,sigma1,sigma2,r,T,p,M,Control):
    dt = T
    drift1 = exp((r - 0.5 * sigma1 * sigma1) * dt)
    drift2 = exp((r - 0.5 * sigma2 * sigma2) * dt)
    S1next = 0.0
    S2next = 0.0
    arithPayOff = np.empty(M, dtype=float)

    np.random.seed(seed=2334)

    for i in range(0, M, 1):
        Rand1 = np.random.randn(1)
        Rand2 = np.random.randn(1)
        growthFactor1 = drift1 * exp(sigma1 * sqrt(dt) * Rand1)
        S1next = S1 * growthFactor1
        growthFactor2 = drift2 * exp(sigma2 * sqrt(dt) * (p * Rand1 + sqrt(1-p*p) * Rand2))
        S2next = S2 * growthFactor2

    # Arithmetic mean
        arithMean = 0.5 * (S1next + S2next)
        if CallPutSwitch == 'c':
            arithPayOff[i] = exp(-r * T) * max((arithMean - K), 0)
        else:
            arithPayOff[i] = exp(-r * T) * max((K - arithMean), 0)

# Standard monte carlo
    Pmean = np.mean(arithPayOff)
    Pstd = np.std(arithPayOff)

    confmc = [Pmean - 1.96 * Pstd / sqrt(M), Pmean + 1.96 * Pstd / sqrt(M)]
    return np.mean(confmc)

print (ArithmeticAsianGeoOptionWithControlVariates('c',S1=100,S2=100,K=100,sigma1=0.30,sigma2=0.30,r=0.05,T=3.0,p=0.50,M=10000,Control=0))
print (ArithmeticAsianGeoOptionWithControlVariates('p',S1=100,S2=100,K=100,sigma1=0.30,sigma2=0.30,r=0.05,T=3.0,p=0.50,M=10000,Control=0))
