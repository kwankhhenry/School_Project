#ifndef STATISTIC_H
#define STATISTIC_H

#include <vector>

using namespace std;

class Statistic
{
	public:
		Statistic(int);
		void randomGenerate(double, double);
		double getMean();
		double getVariance();
		double getItem(int);
		void setItem(double);
	private:
		vector<double> data;
		int sampleSize;

};
#endif
