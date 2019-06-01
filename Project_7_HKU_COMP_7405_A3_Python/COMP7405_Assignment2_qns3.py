from scipy.stats import norm
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mibian import BS
import pandas as pd

#O = option price
#S = stock price
#K = strike price
#T = time to maturity
#t = current time ?? is this needed? Shouldn't this be 0
#r = risk free interest rate
#q = repo rate

def BlackScholes(CallPutSwitch,S,K,t,T,sigma,r,q):
    #np.log is equivalent to ln
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
    d1 = (np.log(S/K) + (r - q) * (T - t))/(np.float64(sigma) * np.sqrt(T - t)) + ((1/2) * np.float64(sigma) * np.sqrt(T -t))
    d2 = d1 - (np.float64(sigma) * np.sqrt(T - t))
    if CallPutSwitch=='c':
        return ((S * np.exp((-q)*(T-t)) * norm.cdf(d1)) - (K * np.exp((-r)*(T-t)) * norm.cdf(d2)))
    elif CallPutSwitch=='p':
        return ((K * np.exp((-r)*(T-t)) * norm.cdf(-d2)) - (S * np.exp((-q)*(T-t)) * norm.cdf(-d1)))
    else:
        #vega
        return (S * np.exp((-q) * (T - t)) * np.sqrt(T - t) * norm.pdf(d1))

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except ValueError:
        return False
    return True

def hasArbitrate(S,K,t,T,call_sigma,put_sigma,r,q):
    tolerant = 0.001
    if is_number(call_sigma) and is_number(put_sigma):
        lhs = BlackScholes('c',S,K,t,T,call_sigma,r,q) - BlackScholes('p',S,K,t,T,put_sigma,r,q)
        rhs = S * np.exp((-q)*(T-t)) - K * np.exp((-r) * (T-t))

        if (np.absolute(lhs - rhs) < tolerant ):
            #no arbitrate opportunity
            return "false"
        #has arbitrate opportunity
        value = np.absolute(lhs - rhs) * 10000
        if value > 3.3:
            return " Has arbitrage regardless of Transaction Cost: " + str(K) + " lhs: " + str(round(lhs,4))\
                   + " rhs: " + str(round(rhs,4)) + " Value of Arbitrage: $" + str(round(value,2))
        else:
            return " Has arbitrage only with no Transaction cost: " + str(K) + " lhs: " + str(round(lhs,4))\
                   + " rhs: " + str(round(rhs,4)) + " Value of Arbitrage: $" + str(round(value,2))
    return "NA"

def newtonRaphson(CallPutSwitch,price,S,K,t,T,r,q):
    #initial guess
    sigmahat = np.sqrt(2 * np.absolute( (np.log(S/K) + (r - q) * (T - t))/(T - t)))
    #print (sigmahat)

    tolerant = 0.000001
    tolerant2 = 0.001
    torerant3 = 0
    sigma = sigmahat
    sigmadiff = 1.0
    n = 1
    nmax = 200
    previous_price = 0.0

    if CallPutSwitch=='C':
        while (sigmadiff >= tolerant and n < nmax) :
            call_price = BlackScholes('c',S,K,t,T,sigma,r,q)
            vega = BlackScholes('v',S,K,t,T,sigma,r,q)
            #print("Call Price: " + str(call_price) + " Vega: " + str(vega) + " Sigma: " + str(sigma))
            if vega == 0.0:
                #print ("divide by zero")
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
                #print("Likely that convergent failed "+ str(sigmadiff))
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
            put_price = BlackScholes('p', S, K, t, T, sigma, r, q)
            vega = BlackScholes('v', S, K, t, T, sigma, r, q)
            if vega == 0.0:
                #print ("divide by zero")
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
                #print("Likely that convergent failed " + str(sigmadiff))
                return "NAN"
            if previous_price == put_price:
                torerant3 = 1 + torerant3
                if torerant3 >= 5:
                    #print("Convergent failed: Calculated Price: " + str(put_price) + " True Price: " + str(price))
                    return sigma
            previous_price = put_price
        return sigma

#print (newtonRaphson('c',price=100.0,S=100.0,K=100.0,t=0,T=1.0,r=0.04,q=0.2))
#print (BlackScholes('c',100,100,0,1,0.45,0.04,0.2))
#(cp=1, price = 1.52, s=23.95, k=24, t=71.0/365, rf=0.05)

instruments = pd.read_csv('Techniques in computational finance/instruments.csv')
marketdata = pd.read_csv('Techniques in computational finance/marketdata.csv')

stock_price = 0.0
option_type = 'C'
strike_price = 0.0
symbol = ''
bid_price = 0.0
ask_price = 0.0
time = 0
time_cutoff = pd.to_datetime('2016-02-16 09:33:00.000000')

for index, row_instruments in instruments.iterrows():
    if row_instruments['Type'] == 'Option':
        #print (row_instruments['Symbol'])
        symbol = row_instruments['Symbol']
        option_type = row_instruments['OptionType']
        strike_price = row_instruments['Strike']

        #For each symbol
        #Get sum of bid/Ask price & quantity
        #Perform newtonRaphson on it
        n = 0
        bid_price = 0
        ask_price = 0
        stock_price = 0
    #else:
        #Equity

    for index, row_marketdata in marketdata.iterrows():
        if row_marketdata['Symbol'] == symbol:
            option_time = pd.to_datetime(row_marketdata['LocalTime'])

            if option_time < time_cutoff:
                n = n + 1.0
                bid_price = row_marketdata['Bid1']
                ask_price = row_marketdata['Ask1']
                #stock_price = stock_price + row_marketdata['Last']
                stock_price_bid_31 = 1.95825
                stock_price_ask_31 = 1.95941
                stock_price_bid_32 = 1.95741
                stock_price_ask_32 = 1.958625
                stock_price_bid = 1.9574
                stock_price_ask = 1.9586

    #if n == 0 then that means that there are no instrument data for this time period
    if n > 0:
        #stock_price = stock_price / n
        stock_price = (bid_price + ask_price) / 2.0
        bid_sigma = newtonRaphson(option_type, price=bid_price, S=stock_price_bid, K=strike_price, t=0, T=8 / 365.0, r=0.04,q=0.2)
        ask_sigma = newtonRaphson(option_type, price=ask_price, S=stock_price_ask, K=strike_price, t=0, T=8 / 365.0, r=0.04,q=0.2)
        #print (hasArbitrate(S=stock_price, K=strike_price, t=0.0, T= 8 /365.0, sigma=bid_sigma, r=0.04, q=0.2))
        if option_type == 'C':
            print(str(symbol) + "," + str(strike_price) + "," + "NA,NA" + "," + str(bid_sigma) + "," + str(ask_sigma))
        else:
            print(str(symbol) + "," + str(strike_price) + "," + str(bid_sigma) + "," + str(ask_sigma) + ",NA,NA")
        #print ("Processed: " + str(symbol) + " OptionType: " + option_type + " Strike: " + str(strike_price) + " Bid: " + str(bid_price) + " Ask: " + str(ask_price) + " Last: " + str(stock_price))
        #print ("Processed: " + str(symbol) + "  Bid Volatility: " + str(newtonRaphson(option_type,price=bid_price,S=stock_price,K=strike_price,t=0,T=8/365.0,r=0.04,q=0.2)))
        #print("Processed: " + str(symbol) + "  Ask Volatility: " + str(newtonRaphson(option_type, price=ask_price, S=stock_price,K=strike_price, t=0, T=8/365.0, r=0.04,q=0.2)))
    #for debugging
    #if symbol == 10000566:
    #    print ("Processed: " + str(symbol) + " OptionType: " + option_type + " Strike: " + str(strike_price) + " Bid: " + str(bid_price) + " Ask: " + str(ask_price) + " Stock: " + str(stock_price))
    #    print(str(symbol) + "," + str(strike_price) + "," + "NA,NA" + "," + str(newtonRaphson(option_type,price=bid_price,S=stock_price,K=strike_price,t=0,T=8/365.0,r=0.04,q=0.2)) + "," + str(newtonRaphson(option_type, price=ask_price, S=stock_price,K=strike_price, t=0, T=8/365.0, r=0.04,q=0.2)))

        #print("Processed: " + str(symbol) + "  Bid Volatility: " + str(newtonRaphson(option_type, price=ask_price, S=stock_price, K=strike_price, t=0, T=8 / 365.0, r=0.04,q=0.2)))
    else:
        print (str(symbol) + "," + str(strike_price) + "," + "NA,NA,NA,NA")
#print("Processed: " + "Ask Volatility: " + str(newtonRaphson('C', price=0.1562, S=0.1555,K=1.8, t=0, T=8.0 / 365.0, r=0.04,q=0.2)))
#print (BlackScholes(CallPutSwitch='p',S=0.113,K=2.05,t=0,T=0.021,sigma=0.2,r=0.04,q=0.2))

stock_price_bid_31 = 1.95825
stock_price_ask_31 = 1.95941
stock_price_bid_32 = 1.95741
stock_price_ask_32 = 1.958625
stock_price_bid_33 = 1.9574
stock_price_ask_33 = 1.9586

csv = ['Techniques in computational finance/assignment 2/31.csv', 'Techniques in computational finance/assignment 2/32.csv', 'Techniques in computational finance/assignment 2/33.csv']
stock_price_bid = [stock_price_bid_31,stock_price_bid_32,stock_price_bid_33]
stock_price_ask = [stock_price_ask_31,stock_price_ask_32,stock_price_ask_33]

for i in range(0,3):
    print ("Checking for Arbitrage from: " + csv[i])
    processed_data = pd.read_csv(csv[i])

    for index, pd_instruments in processed_data.iterrows():
        put_bid_sigma = pd_instruments['BidVolP']
        call_bid_sigma = pd_instruments['AskVolC']
        strike_price = pd_instruments['Strike']
        if put_bid_sigma == "NAN" or call_bid_sigma == "NAN":
            next
            #print ("NAN")
        else:
            results = hasArbitrate(S=stock_price_bid[i], K=strike_price, t=0.0, T= 8 /365.0, call_sigma=call_bid_sigma,put_sigma=put_bid_sigma, r=0.04, q=0.2)
            if (results == "false"):
                next
                #print ("Has no arbitrage at: " + str(strike_price))
            else:
                print ("Bid:" + results)
                #next

    for index, pd_instruments in processed_data.iterrows():
        put_ask_sigma = pd_instruments['BidVolC']
        call_ask_sigma = pd_instruments['AskVolC']
        strike_price = pd_instruments['Strike']
        if put_ask_sigma == "NAN" or call_ask_sigma == "NAN":
            next
            #print ("NAN")
        else:
            results = hasArbitrate(S=stock_price_ask[i], K=strike_price, t=0.0, T= 8 /365.0, call_sigma=call_ask_sigma,put_sigma=put_ask_sigma, r=0.04, q=0.2)
            if (results == "false"):
                next
                #print ("Has no arbitrage at: " + str(strike_price))
            else:
                #next
                print ("Ask:" + results)
