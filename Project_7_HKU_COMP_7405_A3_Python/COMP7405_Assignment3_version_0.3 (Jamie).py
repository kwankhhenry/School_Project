from scipy.stats import norm
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#O = option price
#S = stock price
#K = strike price
#T = time to maturity
#r = risk free interest rate
#q = repo rate

#Task 1 Implement Black-Scholes Formulas for European call/put options
#European call/put option: the spot price of asset S(0), the volatility σ, risk-free interest
#rate r, repo rate q, time to maturity (in years) T, strike K, and option type (call or put).
def BlackScholes(S,sigma,r,q,T,K,CallPutSwitch):
    #np.log is equivalent to ln
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
    #S = spot price of asset S(0)
    #sigma = volatility
    d1 = (np.log(S/K) + (r - q) * (T))/(np.float64(sigma) * np.sqrt(T)) + ((1/2) * np.float64(sigma) * np.sqrt(T))
    d2 = d1 - (np.float64(sigma) * np.sqrt(T))
    if CallPutSwitch=='c':
        return ((S * np.exp((-q)*(T)) * norm.cdf(d1)) - (K * np.exp((-r)*(T)) * norm.cdf(d2)))
    elif CallPutSwitch=='p':
        return ((K * np.exp((-r)*(T)) * norm.cdf(-d2)) - (S * np.exp((-q)*(T)) * norm.cdf(-d1)))
    else:
        #vega
        return (S * np.exp((-q) * (T)) * np.sqrt(T) * norm.pdf(d1))

#Test case
#print ("\nBlack-Scholes Formulas for European call/put options")
#print ("\nCase 1: S = 100;σ = 0.2; r = 0.05; q = 0.02; T = 3; K = 100; Call")
#print (BlackScholes(100, 0.2, 0.05, 0.02, 3, 100, 'c'))
#print ("\nCase 2: S = 100;σ = 0.3; r = 0.05; q = 0.02; T = 3; K = 100; Call")
#print (BlackScholes(100, 0.3, 0.05, 0.02, 3, 100, 'c'))
#print ("\nCase 3: S = 100;σ = 0.2; r = 0.08; q = 0.02; T = 3; K = 100; Call")
#print (BlackScholes(100, 0.2, 0.08, 0.02, 3, 100, 'c'))
#print ("\nCase 4: S = 100;σ = 0.2; r = 0.05; q = 0.04; T = 3; K = 100; Call")
#print (BlackScholes(100, 0.2, 0.05, 0.04, 3, 100, 'c'))
#print ("\nCase 5: S = 100;σ = 0.2; r = 0.05; q = 0.02; T = 5; K = 100; Call")
#print (BlackScholes(100, 0.2, 0.05, 0.02, 5, 100, 'c'))
#print ("\nCase 6: S = 100;σ = 0.2; r = 0.05; q = 0.02; T = 3; K = 120; Call")
#print (BlackScholes(100, 0.2, 0.05, 0.02, 3, 120, 'c'))

#print ("\nCase 7: S = 100;σ = 0.2; r = 0.05; q = 0.02; T = 3; K = 100; Put")
#print (BlackScholes(100, 0.2, 0.05, 0.02, 3, 100, 'p'))
#print ("\nCase 8: S = 100;σ = 0.3; r = 0.05; q = 0.02; T = 3; K = 100; Put")
#print (BlackScholes(100, 0.3, 0.05, 0.02, 3, 100, 'p'))
#print ("\nCase 9: S = 100;σ = 0.2; r = 0.08; q = 0.02; T = 3; K = 100; Put")
#print (BlackScholes(100, 0.2, 0.08, 0.02, 3, 100, 'p'))
#print ("\nCase 10: S = 100;σ = 0.2; r = 0.05; q = 0.04; T = 3; K = 100; Put")
#print (BlackScholes(100, 0.2, 0.05, 0.04, 3, 100, 'p'))
#print ("\nCase 11: S = 100;σ = 0.2; r = 0.05; q = 0.02; T = 5; K = 100; Put")
#print (BlackScholes(100, 0.2, 0.05, 0.02, 5, 100, 'p'))
#print ("\nCase 12: S = 100;σ = 0.2; r = 0.05; q = 0.02; T = 3; K = 120; Put")
#print (BlackScholes(100, 0.2, 0.05, 0.02, 3, 120, 'p'))

###########################################################################################

#Task 2: Implied volatility calculations using NewtonRaphson

def newtonRaphson1(S,r,q,T,K,optionPremium,CallPutSwitch):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5

    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        BSprice = BlackScholes(S,sigma,r,q,T,K,CallPutSwitch) 
        vega = BlackScholes(S,sigma,r,q,T,K,'v')

        if vega == 0.0:
                print ("divide by zero")
                return "NAN"
            
        diff = optionPremium - BSprice  # our root


        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma

print (newtonRaphson1(100,0.05,0.02,3,100,17.5,'c'))
#http://www.option-price.com/implied-volatility.php


###########################################################################################
#Task 3
#Implement closed-form formulas for geometric Asian and geometric basket
    # callputswitch = 'c' or 'p' ; call/put options.
    # S0 = stock price
    # K = strike price
    # sigma = volatility
    # r = risk free interest rate
    # T = time to maturity
    # n = number of observation times for the geometric average n

#Closed-form formulas for geometric Asian
    
def ClosedFormGeometricAsianOption (S0, sigma, r, T, K, n, CallPutSwitch):

    sigsqT = sigma * sigma * T * (n + 1) * (2 * n + 1) / (6 * n * n)
    aT = (r - 0.5 * sigma * sigma) * T * (n + 1) / (2 * n) + 0.5 * sigsqT

    d1 = (np.log(S0/K) + (aT + 0.5 * sigsqT)) / (sqrt(sigsqT))
    d2 = d1 - sqrt(sigsqT)

    if CallPutSwitch == 'c':
        geoAsian = np.exp((-r) * T) * (S0 * np.exp(aT) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        geoAsian = np.exp((-r) * T) * (K * norm.cdf(-d2) - S0 * np.exp(aT) * norm.cdf(-d1))
    return geoAsian


#Test case
#print ("\nClosed Form Geometric Asian")
#print ("\nσ = 0.3; K = 100; n = 50; Put")
#print (ClosedFormGeometricAsianOption(S0 = 100, sigma = 0.3, r = 0.05, T = 3.0 ,K = 100, n = 50, CallPutSwitch = 'p'))
#print ("\nσ = 0.3; K = 100; n = 100; Put")
#print (ClosedFormGeometricAsianOption(S0 = 100, sigma = 0.3, r = 0.05, T = 3.0 ,K = 100, n = 100, CallPutSwitch = 'p'))
#print ("\nσ = 0.4; K = 100; n = 50; Put")
#print (ClosedFormGeometricAsianOption(S0 = 100, sigma = 0.4, r = 0.05, T = 3.0 ,K = 100, n = 50, CallPutSwitch = 'p'))
#print ("\nσ = 0.3; K = 100; n = 50; Call")
#print (ClosedFormGeometricAsianOption(S0 = 100, sigma = 0.3, r = 0.05, T = 3.0 ,K = 100, n = 50, CallPutSwitch = 'c'))
#print ("\nσ = 0.3; K = 100; n = 100; Call")
#print (ClosedFormGeometricAsianOption(S0 = 100, sigma = 0.3, r = 0.05, T = 3.0 ,K = 100, n = 100, CallPutSwitch = 'c'))
#print ("\nσ = 0.4; K = 100; n = 50; Call")
#print (ClosedFormGeometricAsianOption(S0 = 100, sigma = 0.4, r = 0.05, T = 3.0 ,K = 100, n = 50, CallPutSwitch = 'c'))



# Closed-form formulas for geometric basket
    # callputswitch = 'c' or 'p' ; call/put options.
    # S1 = spot price of asset 1
    # S2 = spot price of asset 2
    # K = strike price
    # sigma1 = volatility of asset 1
    # sigma2 = volatility of asset 2
    # r = risk free interest rate
    # T = time to maturity
    # p = correlation
    # sigmaBg =volatility of geometric Brownian motion
    # uBg = drift
    # Bg0 = multiple of all spot price at time 0
    
def ClosedFormGeometricBasketOption (S1, S2, sigma1, sigma2, r, T, K, p, CallPutSwitch):


    sigmaBg =  np.sqrt(sigma1*sigma1 + sigma1*sigma2*p + sigma2*sigma1*p + sigma2*sigma2) /2
    uBg = r - 0.5 * (sigma1*sigma1 + sigma2*sigma2)/2 + 0.5*sigmaBg*sigmaBg
    Bg0 = np.sqrt(S1 * S2)
    
    d1 = (np.log(Bg0/K) + (uBg + 0.5*sigmaBg*sigmaBg)*T) / (sigmaBg * sqrt(T))
    d2 = d1 - sigmaBg * np.sqrt(T)

    if CallPutSwitch == 'c':
            geoBasket = np.exp(-r * T) * (Bg0*np.exp(uBg*T)*norm.cdf(d1) - K*norm.cdf(d2))
    else:
            geoBasket = np.exp(-r * T) * (K*norm.cdf(-d2) - (Bg0*np.exp(uBg*T)*norm.cdf(-d1)))
            
    return geoBasket

#Test case
#print ("\nClosed Form Geometric Basket")
#print ("\nS1(0) = 100; S2(0) = 100; K = 100; σ1 = 0.3; σ2 = 0.3; p = 0.5; Put")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 100; σ1 = 0.3; σ2 = 0.3; p = 0.9; Put")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.9, CallPutSwitch = 'p'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 100; σ1 = 0.1; σ2 = 0.3; p = 0.5; Put")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.1, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 80; σ1 = 0.3; σ2 = 0.3; p = 0.5; Put")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 80, p = 0.5, CallPutSwitch = 'p'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 120; σ1 = 0.3; σ2 = 0.3; p = 0.5; Put")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 120, p = 0.5, CallPutSwitch = 'p'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 100; σ1 = 0.5; σ2 = 0.5; p = 0.5; Put")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.5, sigma2 = 0.5, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'p'))

#print ("\nS1(0) = 100; S2(0) = 100; K = 100; σ1 = 0.3; σ2 = 0.3; p = 0.5; Call")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 100; σ1 = 0.3; σ2 = 0.3; p = 0.9; Call")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.9, CallPutSwitch = 'c'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 100; σ1 = 0.1; σ2 = 0.3; p = 0.5; Call")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.1, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 80; σ1 = 0.3; σ2 = 0.3; p = 0.5; Call")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 80, p = 0.5, CallPutSwitch = 'c'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 120; σ1 = 0.3; σ2 = 0.3; p = 0.5; Call")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.3, sigma2 = 0.3, r = 0.05, T = 3.0 ,K = 120, p = 0.5, CallPutSwitch = 'c'))
#print ("\nS1(0) = 100; S2(0) = 100; K = 100; σ1 = 0.5; σ2 = 0.5; p = 0.5; Call")
#print (ClosedFormGeometricBasketOption(S1 = 100, S2 = 100, sigma1 = 0.5, sigma2 = 0.5, r = 0.05, T = 3.0 ,K = 100, p = 0.5, CallPutSwitch = 'c'))



###########################################################################################
#Task 4
#Implement the Monte Carlo method with control variate technique for arithmetic Asian
    # callputswitch = 'c' or 'p' ; call/put options.
    # S = stock price
    # K = strike price
    # sigma = volatility
    # r = risk free interest rate
    # T = time to maturity
    # N = binomial steps
    # M = Path
    # control = 1 ; use control variate
    # dt = size of time step
def ArithmeticAsianOptionWithControlVariates (CallPutSwitch,S, K, sigma, r, T, N, M, Control):
    dt = T * 1.0 / N
    Spath = np.empty(N, dtype=float)
    arithPayOff = np.empty(M, dtype=float)
    geoPayOff = np.empty(M, dtype=float)

    #Geom Asian Exact Mean
    sigsqT = sigma * sigma * T * (N + 1) * (2 * N + 1) / (6 * N * N)
    muT = (r - 0.5 * sigma * sigma) * T * (N + 1) / (2 * N) + 0.5 * sigsqT

    d1 = (np.log(S/K) + (muT + 0.5 * sigsqT)) / (sqrt(sigsqT))
    d2 = d1 - sqrt(sigsqT)

    drift = exp((r - 0.5 * sigma * sigma) * dt)
    np.random.seed(seed=2334)
    for i in range (0,M,1):
        temp = np.random.randn(1)
        #print (temp)
        growthFactor = drift * exp(sigma * sqrt(dt) * temp)
        Spath[0] = S * growthFactor
        #print(growthFactor)

        for j in range (1,N,1):
            temp = np.random.randn(1)
            growthFactor = drift * exp(sigma * sqrt(dt) * temp)
            Spath[j] = Spath[j - 1] * growthFactor

        #arithmetic mean
        arithMean = np.mean(Spath)

        #geometric mean
        geoMean = np.exp((1/N) * np.sum(np.log(Spath)))

        if CallPutSwitch == 'c':
            arithPayOff[i] = exp(-r * T) * max(arithMean - K, 0)
            geoPayOff[i] = exp(-r * T) * max(geoMean - K, 0)
        else:
            arithPayOff[i] = exp(-r * T) * max(K - arithMean, 0)
            geoPayOff[i] = exp(-r * T) * max(K - geoMean, 0)

    #standard Monte Carlo
    Pmean = np.mean(arithPayOff)
    Pstd = np.std(arithPayOff)

    confmc = [Pmean-1.96*Pstd/sqrt(M), Pmean+1.96*Pstd/sqrt(M)]

    #control variate
    covXY = np.mean(arithPayOff * geoPayOff) - np.mean(arithPayOff) * np.mean(geoPayOff)
    theta = covXY / np.var(geoPayOff)

    #geo = geometric asian call/put option value
    if CallPutSwitch == 'c':
        geo = np.exp((-r) * T) * (S * np.exp(muT) * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        geo = np.exp((-r) * T) * (K * norm.cdf(-d2) - S * np.exp(muT) * norm.cdf(-d1))

    #control variate version
    Z = arithPayOff + theta * (geo - geoPayOff)
    Zmean = np.mean(Z)
    Zstd = np.std(Z)

    confcv = [Zmean-1.96*Zstd /sqrt(M), Zmean+1.96*Zstd /sqrt(M)]

    #print ("No Control  : " + str(confmc))
    #print ("With Control: " + str(confcv))

    if Control == 1:
        return confcv
    return confmc

#Test case
#print (ArithmeticAsianOptionWithControlVariates(CallPutSwitch = 'c',S = 100, K = 100, sigma = 0.3, r = 0.05, T = 3.0 , N = 50, M = 10000, Control=1))


##########################################################################
#Task 6
#Cox Ross and Rubinstein method
def BinomialTree(CallPutSwitch,S,K,T,sigma,r,N):
    # S = stock price
    # K = strike price
    # T = time to maturity
    # r = risk free interest rate
    # sigma = volatility
    # N = binomial steps
    # dt = size of time step

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r * dt) - d) / (u - d)

    stepNumber = N + 1

    stockValue = np.zeros((stepNumber,stepNumber))
    optionValue = np.zeros((stepNumber, stepNumber))

    #initialise Stock value
    stockValue [0 , 0] = S

    #Loop over each loop to calculate the stock value
    for i in range(1, stepNumber):
        stockValue[i,0] = stockValue[i-1,0] * u
        for j in range(1, i+1):
            stockValue[i,j] = stockValue[i-1,j-1] * d

    #Backward recursion for option price

    #calculate the optionvalue at the leaf nodes
    for j in range(0, stepNumber):
        if CallPutSwitch == 'c':
            optionValue[N,j] = max(0,(stockValue[N,j] - K))
        else:
            optionValue[N,j] = max(0, (K - stockValue[N,j]))

    for i in range(N-1,-1,-1):
        for j in range(0,i+1):
            if CallPutSwitch == 'c':
                optionValue[i,j] = max((p * optionValue[i+1,j]+(1 - p) * optionValue[i+1,j+1]), stockValue[i,j]-K)
            else:
                optionValue[i, j] = max((p * optionValue[i + 1, j] + (1 - p) * optionValue[i + 1, j + 1]),K-stockValue[i, j])

    return optionValue[0,0]

#Test case
#print (BinomialTree("c",100,100,1.0,0.2,0.03,1000))
#print (BinomialTree("p",100,100,1.0,0.2,0.03,1000))
