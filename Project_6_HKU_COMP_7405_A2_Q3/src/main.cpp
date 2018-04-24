#include <iostream>
#include <iomanip>
#include <math.h>
#include "NormalDistribution.h"
#include "FileMapIO.h"
#include "Instrument.h"

using namespace std;

int main()
{
	// Read instruments.csv
	FileMapIO reader("instruments.csv");
	reader.readFile();
	//reader.printFile();

	// Read marketdata.csv
	Instrument portfolio(reader);
	portfolio.readMktData("marketdata.csv");
	//portfolio.printMap();

	// Consolidate and compute data for 09:31:00
	portfolio.updateMktData("2016-Feb-16 09:30:59.999999");
	portfolio.calBidAskVolty();
	portfolio.printMap();

	portfolio.mergeStrike();
	portfolio.outputCSV("31.csv");

	// Consolidate and compute data for 09:32:00
	portfolio.updateMktData("2016-Feb-16 09:31:59.999999");
	portfolio.calBidAskVolty();
	portfolio.printMap();

	portfolio.mergeStrike();
	portfolio.outputCSV("32.csv");

	// Consolidate and compute data for 09:33:00
	portfolio.updateMktData("2016-Feb-16 09:32:59.999999");
	portfolio.calBidAskVolty();
	portfolio.printMap();

	portfolio.mergeStrike();
	portfolio.outputCSV("33.csv");

	return 0;
}
