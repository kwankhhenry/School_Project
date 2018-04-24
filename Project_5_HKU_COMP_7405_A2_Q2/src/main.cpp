#include <iostream>
#include <iomanip>
#include <math.h>
#include <vector>
#include "Statistic.h"

using namespace std;

int main()
{
	const int nSamples = 200;
	double mean = 0.0;
	double stdDev = 1.0;
	double rho = 0.5;
	double calNum = 0.0;
	double Correlation = 0.0;

	Statistic X(nSamples);
	Statistic Y(nSamples);
	Statistic Z(nSamples);
	Statistic Covariance(nSamples);

	// Standard normal distribution of X and Y
	X.randomGenerate(mean, stdDev);
	Y.randomGenerate(mean, stdDev);

	// Obtain random variable Z based on random generation of X and Y
	for(int i = 0; i < nSamples; i++)
	{
		calNum = rho*X.getItem(i)+sqrt(1-rho*rho)*Y.getItem(i);
		Z.setItem(calNum);
	}

	// Calculate covariance vector
	for(int i = 0; i < nSamples; i++)
	{
		calNum = (X.getItem(i)-X.getMean())*(Z.getItem(i)-Z.getMean());
		Covariance.setItem(calNum);
	}

	// Printing X, Y, Z data 
	cout << "Random Variable Tables\n";
	cout << setw(15) << right << "X" << setw(15) << "Y" << setw(15) << "Z" << endl;
	cout << setfill('-') << setw(60) << "-" <<  endl; 

	cout << setfill(' ');
	for(int j = 0; j < nSamples; j++)
	{
		cout << right << setw(15) << X.getItem(j);
		cout << right << setw(15) << Y.getItem(j);
		cout << right << setw(15) << Z.getItem(j) << endl;
	}

	// Output statistics
	cout << "\nOutput Statistics: " << endl;
	cout << "X Mean is " << X.getMean() << endl;
	cout << "X Variance is " << X.getVariance() << endl;
	cout << "Y Mean is " << Y.getMean() << endl;
	cout << "Y Variance is " << Y.getVariance() << endl;
	cout << "Z Mean is " << Z.getMean() << endl;
	cout << "Z Variance is " << Z.getVariance() << endl;

	// Note: Covariance is the expected value of covariance vector
	cout << "Covariance(X,Z) is " << Covariance.getMean() << endl;

	// Calculate and return correlation coefficient
	Correlation = Covariance.getMean()/(sqrt(X.getVariance())*sqrt(Z.getVariance()));
	cout << "Correlation coefficient is " << Correlation << endl;

	return 0;
}


