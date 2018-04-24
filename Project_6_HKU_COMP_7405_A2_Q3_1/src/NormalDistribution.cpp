#include <math.h>
#include "NormalDistribution.h"

double NormalDistribution::pdf(double x)
{
	return exp( -1*(x-mu)*(x-mu)/(2*sigma*sigma)) / (sigma*sqrt(2*pi));
}

double NormalDistribution::cdf(double x)
{
	const double ninf = mu - 10 * sigma;
	double sum = 0;
	double n = 1e6;
	double c = (x - ninf) / n;

	for(double k = 1.; k < n-1; k++)
		sum += pdf(ninf + k*c);

	return c * ((pdf(x) + pdf(ninf))/2 + sum);
}
