#ifndef INSTRUMENT_H
#define INSTRUMENT_H

#include <string>
#include <map>
#include "FileMapIO.h"

struct portfolio{
	instrument tags;
	double bid_price = 0.0;
	double ask_price = 0.0;
	double bid_volty = 0.0;
	double ask_volty = 0.0;
	std::string last_update = "9999-Dec-31 12:59:00.000000";
};

struct mkt{
	int MktID;
	double MktLast;
	double MktBid;
	int MktBidQty;
	double MktAsk;
	int MktAskQty;
};

struct outVol{
	double bid_volP;
	double ask_volP;
	double bid_volC;
	double ask_volC;
};

class Instrument
{
	public:
		Instrument(FileMapIO &);
		void readMktData(string);
		void updateMktData(string);
		void calBidAskVolty();
		void mergeStrike();
		void outputCSV(std::string);
		void printMap();

	private:
		// Keys for portfolioMap & mktMap
		int optionID;
		std::string date_time;

		// Store latest bid/ask stock price
		double bid_stock;
		double ask_stock;

		mkt mktData;
		map<int, portfolio> portfolioMap;
		map<std::string, mkt> mktMap;
		map<double, outVol> outMap;

};
#endif
