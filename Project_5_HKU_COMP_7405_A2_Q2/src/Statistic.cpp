#include <iostream>
#include <random>
#include "Statistic.h"

using namespace std;

Statistic::Statistic(int sampleNum)
{
	this->sampleSize = sampleNum;
}

void Statistic::randomGenerate(double mean, double stdDev)
{
	static mt19937 generator;
	normal_distribution<double> distribution(mean, stdDev);

	for(int i = 0; i < sampleSize; i++)
	{
		double number = distribution(generator);
		data.push_back(number);
	}
}

double Statistic::getMean()
{
	double sum = 0.0;
	vector<double>::iterator it;
	for(it = data.begin(); it != data.end(); it++)
		sum += *it;
	
	return sum/sampleSize;
}

double Statistic::getVariance()
{
	double mean = getMean();
	double sum = 0.0;

	vector<double>::iterator it;
	for(it = data.begin(); it != data.end(); it++)
		sum += (*it - mean)*(*it - mean);

	return sum/sampleSize;
}

double Statistic::getItem(int position)
{
	return data.at(position); 
}

void Statistic::setItem(double value)
{
	data.push_back(value);
}
