#include <iostream>
#include <iomanip>
#include <math.h>
#include "NormalDistribution.h"

using namespace std;

double d1(double, double, double, double, double, double);
double d2(double, double, double, double, double, double);
double call_option(double, double, double, double, double, double, double, double);
double put_option(double, double, double, double, double, double, double, double);
void debug(double, double, double, double, double, double, double, double, double, double);

int main()
{
	NormalDistribution Gauss(0., 1.);
	double S, K, t, T, sigma, r;
	double D1, D2;

	cout << "Question 1->(3.1) Paramemters are : S=100, K=100, t=0, T=0.5, sigma=20%, r=1% " << endl;
	S = 100., K = 100., t = 0.; T = 0.5, sigma = 0.2,  r = 0.01;
	D1 = d1(S,K,r,T,t,sigma);
	D2 = d2(S,K,r,T,t,sigma);
	//debug(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2),Gauss.cdf(-1*D1),Gauss.cdf(-1*D2));
	cout << "Call option price is: " << call_option(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2)) << endl;
	cout << "Put option price is: " << put_option(S,K,r,T,t,sigma,Gauss.cdf(-1*D1),Gauss.cdf(-1*D2)) << endl;
	cout << "\n";

	cout << "Question 1->(3.2) Paramemters are : S=100, K=120, t=0, T=0.5, sigma=20%, r=1% " << endl;
	S = 100., K = 120., t = 0.; T = 0.5, sigma = 0.2,  r = 0.01;
	D1 = d1(S,K,r,T,t,sigma);
	D2 = d2(S,K,r,T,t,sigma);
	//debug(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2),Gauss.cdf(-1*D1),Gauss.cdf(-1*D2));
	cout << "Call option price is: " << call_option(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2)) << endl;
	cout << "Put option price is: " << put_option(S,K,r,T,t,sigma,Gauss.cdf(-1*D1),Gauss.cdf(-1*D2)) << endl;
	cout << "\n";

	cout << "Question 1->(3.3) Paramemters are : S=100, K=100, t=0, T=1.0, sigma=20%, r=1% " << endl;
	S = 100., K = 100., t = 0.; T = 1.0, sigma = 0.2,  r = 0.01;
	D1 = d1(S,K,r,T,t,sigma);
	D2 = d2(S,K,r,T,t,sigma);
	//debug(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2),Gauss.cdf(-1*D1),Gauss.cdf(-1*D2));
	cout << "Call option price is: " << call_option(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2)) << endl;
	cout << "Put option price is: " << put_option(S,K,r,T,t,sigma,Gauss.cdf(-1*D1),Gauss.cdf(-1*D2)) << endl;
	cout << "\n";

	cout << "Question 1->(3.4) Paramemters are : S=100, K=100, t=0, T=0.5, sigma=30%, r=1% " << endl;
	S = 100., K = 100., t = 0.; T = 0.5, sigma = 0.3,  r = 0.01;
	D1 = d1(S,K,r,T,t,sigma);
	D2 = d2(S,K,r,T,t,sigma);
	//debug(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2),Gauss.cdf(-1*D1),Gauss.cdf(-1*D2));
	cout << "Call option price is: " << call_option(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2)) << endl;
	cout << "Put option price is: " << put_option(S,K,r,T,t,sigma,Gauss.cdf(-1*D1),Gauss.cdf(-1*D2)) << endl;
	cout << "\n";

	cout << "Question 1->(3.5) Paramemters are : S=100, K=100, t=0, T=0.5, sigma=20%, r=2% " << endl;
	S = 100., K = 100., t = 0.; T = 0.5, sigma = 0.2,  r = 0.02;
	D1 = d1(S,K,r,T,t,sigma);
	D2 = d2(S,K,r,T,t,sigma);
	//debug(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2)),Gauss.cdf(-1*D1),Gauss.cdf(-1*D2));
	cout << "Call option price is: " << call_option(S,K,r,T,t,sigma,Gauss.cdf(D1),Gauss.cdf(D2)) << endl;
	cout << "Put option price is: " << put_option(S,K,r,T,t,sigma,Gauss.cdf(-1*D1),Gauss.cdf(-1*D2)) << endl;

	cout << "\nComments on Q1: Impacts on option value by each of the following parameters:\n"
		 << "a) Strike: Determines if the option has any intrinsic value, where the intrinsic value is the difference between the strike price (K) and the price of underlying asset (S).  Premium increases as the option becomes further in the money; Likewise, the premium of the option decreases when the option becomes further out-of-money.\n"
		 << "\tCall Option: Given asset S=100, the call option with strike price at K=100 is at-the-money and just begin to intrinsic value for S > 100.  On the other hand, option at K=120 is far out-of-money and therfore has no intrinsic value and only time value at its price. The price of option is expected to be lower at higher strike price given the same value of underlying asset.\n"
		 << "\tPut Option: Put option on the other hand, would have a lower value at K=100, because it is just at-the-money, compared to K=120 being deep in-the-money.  At K=120, an underlying asset with S=100 would be deep in the money and have positive P&L (positive intrinsic value) and therefore must be much more expensive than put option with strike K=100. \n"

		 << "\nb) Maturity: The longer an option has until expiration (Greater T), the greater chance it will end up in-the-money(profitable), so the value of the option is higher with greater maturity period.  Both call and put options will have higher price at T=1.0 compared to base case with T=0.5. \n"
		 << "\tCall Option: The price of call option is expected to be higher with longer period of maturity.\n"
		 << "\tPut Option: Similar to call option, the price of put option is also higher with greater T.\n"

		 << "\nc) Volatility: It is a measure of speed and magnitude of which an underlying asset price changes.  Therefore at higher volatility, we would expect the price of both call and put options to be higher.\n"
		 << "\tCall Option: Call option price at sigma=30% is higher than price with sigma=20%.\n"
		 << "\tPut Option: Put option price at sigma=30% is also higher than the price with sigma=20%.\n"

		 << "\nd) Risk Free Rate: Interest rates have small effects on option prices.  In general, as interest rate rises, call premiums increase and put premiums decrease.  Costs in interest will be incurred to buy the options.\n"
 		 << "\tCall Option: The price of call option will be higher at r=2%, compared to price at r=1%.  The magnitude of price changes is not as much compared to other parameter changes such as strike price, maturity or volatility.\n"
		 << "\tPut Option: The price of put option will be lower at r=2% versus the price when r=1%.\n";

	return 0;
}

double d1(double S, double K, double r, double T, double t, double sigma)
{
	return (log(S/K)+r*(T-t))/(sigma*sqrt(T-t))+(1./2.)*sigma*sqrt(T-t);
}

double d2(double S, double K, double r, double T, double t, double sigma)
{
	return (log(S/K)+r*(T-t))/(sigma*sqrt(T-t))-(1./2.)*sigma*sqrt(T-t);
}

double call_option(double S, double K, double r, double T, double t, double sigma, double Nd1, double Nd2)
{
	return S*Nd1 - K*exp(-r*(T-t))*Nd2;
}

double put_option(double S, double K, double r, double T, double t, double sigma, double Nd1, double Nd2)
{
	return K*exp(-r*(T-t))*Nd2 - S*Nd1;
}

void debug(double S, double K, double r, double T, double t, double sigma, double Nd1, double Nd2, double Nd3, double Nd4)
{
	cout << setw(50) << setfill('*') << "" << endl;
	cout << "Z-score First part is: " << (log(S/K)+r*(T-t))/(sigma*sqrt(T-t)) << endl;
	cout << "Z-score Second part is: " << (1./2.)*sigma*sqrt(T-t) << endl;
	cout << "Calculated D1 is: " << d1(S,K,r,T,t,sigma) << endl;
	cout << "Calculated D2 is: " << d2(S,K,r,T,t,sigma) << endl;
	cout << "N(d1) yields: " << Nd1 << endl;
	cout << "N(d2) yields: " << Nd2 << endl;
	cout << "N(-d1) yields: " << Nd3 << endl;
	cout << "N(-d2) yields: " << Nd4 << endl;

	cout << setw(50) << setfill('*') << "" << endl;
}

