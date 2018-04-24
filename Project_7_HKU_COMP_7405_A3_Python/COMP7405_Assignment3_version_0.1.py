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

#Task 1
#European call/put option: the spot price of asset S(0), the volatility σ, risk-free interest
#rate r, repo rate q, time to maturity (in years) T, strike K, and option type (call or put).
def BlackScholes(CallPutSwitch,S,K,T,sigma,r,q):
    #np.log is equivalent to ln
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
    d1 = (np.log(S/K) + (r - q) * (T))/(np.float64(sigma) * np.sqrt(T)) + ((1/2) * np.float64(sigma) * np.sqrt(T))
    d2 = d1 - (np.float64(sigma) * np.sqrt(T))
    if CallPutSwitch=='c':
        return ((S * np.exp((-q)*(T)) * norm.cdf(d1)) - (K * np.exp((-r)*(T)) * norm.cdf(d2)))
    elif CallPutSwitch=='p':
        return ((K * np.exp((-r)*(T)) * norm.cdf(-d2)) - (S * np.exp((-q)*(T)) * norm.cdf(-d1)))
    else:
        #vega
        return (S * np.exp((-q) * (T)) * np.sqrt(T) * norm.pdf(d1))

#Task 2: Implied volatility calculations using NewtonRaphson
#option premium ??
def newtonRaphson(CallPutSwitch,price,S,K,T,r,q):
    #initial guess
    sigmahat = np.sqrt(2 * np.absolute( (np.log(S/K) + (r - q) * (T))/(T)))
    #print (sigmahat)

    tolerant = 0.000001
    tolerant2 = 0.001
    torerant3 = 0
    sigma = sigmahat
    sigmadiff = 1.0
    n = 1
    nmax = 200
    previous_price = 0.0

    if CallPutSwitch=='c':
        while (sigmadiff >= tolerant and n < nmax) :
            call_price = BlackScholes('c',S,K,T,sigma,r,q)
            vega = BlackScholes('v',S,K,T,sigma,r,q)
            #print("Call Price: " + str(call_price) + " Vega: " + str(vega) + " Sigma: " + str(sigma))
            if vega == 0.0:
                print ("divide by zero")
                return "NAN"
            increment = (call_price - price)/vega
            #print("Price: " + str(price) + " Vega: " + str(vega) + " Increment: " + str(increment))
            sigma = sigma - increment
            #print("Call Price: " + str(call_price) + " Sigma: " + str(sigma))
            n = n + 1
            sigmadiff = np.absolute(call_price - price)
            #print("SigmaDiff: " + str(sigmadiff))
            #print(n)
            if n == nmax and sigmadiff > tolerant2:
                print("Likely that convergent failed "+ str(sigmadiff))
                return "NAN"
            if previous_price == call_price:
                torerant3 = 1 + torerant3
                if torerant3 >= 5:
                    #print("Convergent failed: Calculated Price: " + str(call_price) + " True Price: " + str(price) + " Sigma: " + str(sigma))
                    return sigma
            previous_price = call_price
        return sigma
    else:
        while (sigmadiff >= tolerant and n < nmax) :
            put_price = BlackScholes('p', S, K, T, sigma, r, q)
            vega = BlackScholes('v', S, K, T, sigma, r, q)
            if vega == 0.0:
                print ("divide by zero")
                return "NAN"
            #print("Put Price: " + str(put_price) + " Vega: " + str(vega) + " Sigma: " + str(sigma))
            increment = (put_price - price) / vega
            #print("Price: " + str(price) + " Vega: " + str(vega) + " Increment: " + str(increment))
            sigma = sigma - increment
            #print("Sigma: " + str(sigma))
            n = n + 1
            sigmadiff = np.absolute(put_price - price)
            #print("SigmaDiff: " + str(sigmadiff))
            #print(sigmadiff)
            #print(n)
            if n == nmax and sigmadiff > tolerant2:
                print("Likely that convergent failed " + str(sigmadiff))
                return "NAN"
            if previous_price == put_price:
                torerant3 = 1 + torerant3
                if torerant3 >= 5:
                    #print("Convergent failed: Calculated Price: " + str(put_price) + " True Price: " + str(price))
                    return sigma
            previous_price = put_price
        return sigma

#print (newtonRaphson('c',price=1.0,S=100.0,K=120.0,T=8.0 / 365,r=0.04,q=0.2))
#print (BlackScholes('c',100,100,1.0,0.45,0.04,0.2))




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
