#include <iostream>
#include <iomanip>
#include <math.h>
#include "NormalDistribution.h"

using namespace std;

enum InstType{ CALL, PUT };
double price_call(double, double, double, double, double, double, double, double, NormalDistribution);
double price_put(double, double, double, double, double, double, double, double, NormalDistribution);
double vega(double, double, double, double, double, NormalDistribution);
void newtonMethod(InstType, double, double, double, double, double, double, NormalDistribution, double);

int main()
{
	// Methods to compute volatility sigma and option prices
	NormalDistribution N(0., 1.);
	InstType OPTION_TYPE;
	double S, K, t, T, r, q, C_true, P_true;
	
	// Parameters to be changed
	S = 1.8; K = 1.95; t = 0; T = 3; r = 0.04; q = 0.2; C_true = 0.103532; P_true = 0.845166;
	cout << "Testing parameters: S = 1.8; K = 1.95; t = 0; T = 3; r = 0.04; q = 0.2; C_true = 0.103532; P_true = 0.845166\n";
	cout << "Goal: Expect implied volatility = 40% in both cases.\n";

	// Newton Iteration based on call price
	OPTION_TYPE = CALL;
	newtonMethod(OPTION_TYPE, S, q, T, t, K, r, N, C_true);

	// Newton Iteration based on put price
	OPTION_TYPE = PUT;
	newtonMethod(OPTION_TYPE, S, q, T, t, K, r, N, P_true);

	return 0;
}

double price_call(double S, double q, double T, double t, double K, double r, double d1, double d2, NormalDistribution N)
{
	return S*exp(-q*(T-t))*N.cdf(d1) - K*exp(-r*(T-t))*N.cdf(d2);
}

double price_put(double S, double q, double T, double t, double K, double r, double d1, double d2, NormalDistribution N)
{
	return K*exp(-r*(T-t))*N.cdf(-1*d2) - S*exp(-q*(T-t))*N.cdf(-1*d1); 
}

double vega(double S, double q, double T, double t, double d1, NormalDistribution N)
{
	return S*exp(-q*(T-t))*sqrt(T-t)*N.pdf(d1);
}

void newtonMethod(InstType OPTION, double S, double q, double T, double t, double K, double r, NormalDistribution N, double Option_true)
{
	double sigmadiff = 1;
	int n = 1;
	int nmax = 1000;
	const double tol = 1e-8;

	double d1, d2;
	double OptionPrice = 0.0;
	double OptionVega = 0.0;
	double increment = 0.0;

	// Initial sigma guess
	double sigmahat = sqrt(2*fabs((log(S/K)+(r-q)*(T-t))/(T-t)));
	double sigma = sigmahat;

	while( sigmadiff >= tol && n < nmax)
	{
		cout << "\nIterated " << n << " times. \n";
		d1 = (log(S/K)+(r-q)*(T-t))/(sigma*sqrt(T-t))+(1./2.)*sigma*sqrt(T-t);
		d2 = (log(S/K)+(r-q)*(T-t))/(sigma*sqrt(T-t))-(1./2.)*sigma*sqrt(T-t);

		switch (OPTION){
			case CALL: OptionPrice = price_call(S,q,T,t,K,r,d1,d2,N);
					   cout << "Calculated CALL option price is: " << OptionPrice << "\n";
					   break;
			case PUT: OptionPrice = price_put(S,q,T,t,K,r,d1,d2,N);
					  cout << "Calculated PUT option price is: " << OptionPrice << "\n";
					  break;
		}

		OptionVega = vega(S,q,T,t,d1,N);
		increment = (OptionPrice-Option_true)/OptionVega;
		sigma = sigma - increment;

		cout << "Sigma for option is: " << sigma << endl; 

		n++;
		sigmadiff = fabs(increment);
	}
}
