#include <iostream>
#include <string>
#include <math.h>
#include "Evaluation.h"

using namespace std;

Evaluation::Evaluation(double S, double K, double t, double T, double r, double q, double Option_True, std::string Option_Type)
{
	this->S = S;
	this->K = K;
	this->t = t;
	this->T = T;
	this->r = r;
	this->q = q;
	this->Option_True = Option_True;
	this->Option_Type = Option_Type;
	N.setDist(0.,1.);
}

void Evaluation::set_option(double S, double K, double t, double T, double r, double q, double Option_True, std::string Option_Type)
{
	this->S = S;
	this->K = K;
	this->t = t;
	this->T = T;
	this->r = r;
	this->q = q;
	this->Option_True = Option_True;
	this->Option_Type = Option_Type;
}

double Evaluation::option_price(double d1, double d2)
{
	if(Option_Type == "C")
			return S*exp(-q*(T-t))*N.cdf(d1) - K*exp(-r*(T-t))*N.cdf(d2);
	else if(Option_Type == "P")
			return K*exp(-r*(T-t))*N.cdf(-1*d2) - S*exp(-q*(T-t))*N.cdf(-1*d1);
	else
			return 0;
}

double Evaluation::option_vega(double d1)
{
	return S*exp(-q*(T-t))*sqrt(T-t)*N.pdf(d1);
}

double Evaluation::newtonMethod()
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

	//cout << "Intial sigmahat is: " << sigmahat << endl;

	while( sigmadiff >= tol && n < nmax)
	{
		d1 = (log(S/K)+(r-q)*(T-t))/(sigma*sqrt(T-t))+(1./2.)*sigma*sqrt(T-t);
		d2 = (log(S/K)+(r-q)*(T-t))/(sigma*sqrt(T-t))-(1./2.)*sigma*sqrt(T-t);

		OptionPrice = option_price(d1,d2);
		OptionVega = option_vega(d1);

		increment = (OptionPrice-Option_True)/OptionVega;
		sigma = sigma - increment;

		/*	
		cout << "\nIterated " << n << " times. \n";
		cout << "D1: " << d1 << "\n";
		cout << "D2: " << d2 << "\n";
		cout << "Calculated option price is: " << OptionPrice << "\n";
		cout << "Calculated option vega is: " << OptionVega << "\n";
		cout << "Sigma for option is: " << sigma << endl; */

		n++;
		sigmadiff = fabs(increment);
	}
	return sigma;
}
