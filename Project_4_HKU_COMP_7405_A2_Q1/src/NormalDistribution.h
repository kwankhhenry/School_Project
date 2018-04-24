#ifndef NORMALDISTRIBUTION_H
#define NORMALDISTRIBUTION_H

const double pi = 3.14159265;

class NormalDistribution
{
	public:
		NormalDistribution(double _mu, double _sigma) : mu(_mu), sigma(_sigma){}
		inline double pdf(double x);
		double cdf(double x);
	private:
		double mu;
		double sigma;
};
#endif
