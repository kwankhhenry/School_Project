#ifndef EVALUATION_H
#define EVALUATION_H

#include <string>
#include "NormalDistribution.h"

class Evaluation
{
	public:
		Evaluation(): S(0.0), K(0.0), t(0.0), T(0.0), r(0.0), q(0.0), Option_True(0.0), N(0.,1.){};
		Evaluation(double, double, double, double, double, double, double, std::string);
		void set_option(double, double, double, double, double, double, double, std::string);
		double option_price(double, double);
		double option_vega(double);
		double newtonMethod();
	private:
		std::string Option_Type;
		double S, K, t, T, r, q, Option_True;
		double d1, d2;
		NormalDistribution N;
	
};
#endif

