#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include "Instrument.h"
#include "Evaluation.h"

using namespace std;

Instrument::Instrument(FileMapIO &ImportData)
{
	map<int, instrument> dataMap;
	map<int, instrument>::iterator it;
	portfolio portData;

	dataMap = ImportData.getDataSet();

	for(it = dataMap.begin(); it != dataMap.end(); it++)
	{
		optionID = it->first;
		portData.tags.type = (it->second).type;
		portData.tags.expiry = (it->second).expiry;
		portData.tags.strike = (it->second).strike;
		portData.tags.option_type = (it->second).option_type;

		portfolioMap.insert(pair<int, portfolio>(optionID, portData));
	}

	bid_stock = 0.0;
	ask_stock = 0.0;
}

void Instrument::printMap()
{
	map<int, portfolio>::iterator it;

	cout << setw(8) << "ID" << " ";
	cout << setw(6) << "Type" << " ";
	cout << setw(8) << "Expiry" << " ";
	cout << setw(6) << "Strike" << " ";
	cout << setw(1) <<  "T" << " ";
	cout << setw(6) << "Bid" << " ";
	cout << setw(6) << "Ask" << " ";
	cout << setw(8) << "BidVol" << " ";
	cout << setw(8) << "AskVol" << " ";
	cout << setw(18) << "Last Update" << "\n";

	for(it = portfolioMap.begin(); it != portfolioMap.end(); it++)
	{
		cout << setw(8) << (it->first) << " ";
		cout << setw(6) << (it->second).tags.type << " ";
		cout << setw(8) << (it->second).tags.expiry << " ";
		cout << setw(6) << (it->second).tags.strike << " ";
		cout << setw(1) <<  (it->second).tags.option_type << " ";
		cout << setw(6) << (it->second).bid_price << " ";
		cout << setw(6) << (it->second).ask_price << " ";
		cout << setw(8) << (it->second).bid_volty << " ";
		cout << setw(8) << (it->second).ask_volty << " ";
		cout << setw(27) << (it->second).last_update << "\n";
	}
}

void Instrument::readMktData(string FileName)
{
	string line;
	ifstream ifile;

	ifile.open(FileName.c_str());
	if(ifile)
	{
		cout << "File " << FileName << " is read successfully!\n";
		getline(ifile, line);
		while(getline(ifile, line))
		{
			stringstream lineStream;
			lineStream << line;
			string token;

			getline(lineStream, token, ',');
			date_time = token;

			getline(lineStream, token, ',');
			if (token == "")
				mktData.MktID = 0;
			else
				mktData.MktID = stoi(token);

			getline(lineStream, token, ',');
			if (token == "")
				mktData.MktLast = 0.0;
			else
				mktData.MktLast = stod(token);

			getline(lineStream, token, ',');
			if (token == "")
				mktData.MktBid = 0.0;
			else
				mktData.MktBid = stod(token);

			getline(lineStream, token, ',');
			if (token == "")
				mktData.MktBidQty = 0;
			else
				mktData.MktBidQty = stoi(token);

			getline(lineStream, token, ',');
			if (token == "")
				mktData.MktAsk = 0.0;
			else
				mktData.MktAsk = stod(token);

			getline(lineStream, token, '\r');
			if (token == "")
				mktData.MktAskQty = 0;
			else
				mktData.MktAskQty = stoi(token);

			mktMap.insert(pair<string, mkt>(date_time, mktData));
		}
		ifile.close();
	}
	else
	{
		cerr << "Failed to read " << FileName << "\n";
	}
}

void Instrument::updateMktData(string maxDateTime)
{
	map<string, mkt>::iterator it;
	
	for(it = mktMap.begin(); it != mktMap.end(); it++)
	{
		if(it->first <= maxDateTime)
		{
			portfolioMap.find((it->second).MktID)->second.bid_price = (it->second).MktBid;
			portfolioMap.find((it->second).MktID)->second.ask_price = (it->second).MktAsk;
			portfolioMap.find((it->second).MktID)->second.last_update = (it->first);

			// Additionally update bid/ask stock variables 
			if(portfolioMap.find((it->second).MktID)->second.tags.option_type == "S")
			{
				bid_stock = (it->second).MktBid;
				ask_stock = (it->second).MktAsk;
			}
		}
	}
}

void Instrument::calBidAskVolty()
{
	// Market data parameters
	double S, K, t, T, r, q, Option_true;
	std::string Option_type;

	// Variables for calculations
	Evaluation analysis;
	map<int, portfolio>::iterator it;

	// Initialize some fixed parameters
	t = 0.;
	T = 8./365.;
	r = 0.04;
	q = 0.2;

	for(it = portfolioMap.begin(); it != portfolioMap.end(); it++)
	{
		K = (it->second).tags.strike;
		Option_type = (it->second).tags.option_type;

		if((it->second).tags.option_type != "S")
		{
			cout << "Evaluating " << Option_type << " option ID: " << (it->first) << "...";
			
			// Calculate Bid volatility
			S = bid_stock;
			Option_true = (it->second).bid_price;

			analysis.set_option(S,K,t,T,r,q,Option_true,Option_type);
			(it->second).bid_volty = analysis.newtonMethod();
			cout << "Bid volatility = " << (it->second).bid_volty << " ";

			// Calculate Ask volatility
			S = ask_stock;
			Option_true = (it->second).ask_price;

			analysis.set_option(S,K,t,T,r,q,Option_true,Option_type);
			(it->second).ask_volty = analysis.newtonMethod();
			cout << "Ask volatility = " << (it->second).ask_volty << "\n";
		}
	}
}

void Instrument::mergeStrike()
{
	// Obtain a list of existing strikes
	map<int, portfolio>::iterator it;
	std::vector<double> strikeList;

	// Create a temp Map using strike as key
	struct temp
	{
		string OptType;
		double BidVol;
		double AskVol;
	} tempStruct;
	std::map<double, temp> tempCall, tempPut;

	// Insert portfolioMap into Call and Put table
	for(it = portfolioMap.begin(); it != portfolioMap.end(); it++)
	{
		if((it->second).tags.option_type == "C")
		{
			strikeList.push_back((it->second).tags.strike);

			tempStruct.OptType = (it->second).tags.option_type;
			tempStruct.BidVol = (it->second).bid_volty;
			tempStruct.AskVol = (it->second).ask_volty;
			tempCall.insert(pair<double, temp> ((it->second).tags.strike, tempStruct));
		}
		else if((it->second).tags.option_type == "P")
		{
			strikeList.push_back((it->second).tags.strike);

			tempStruct.OptType = (it->second).tags.option_type;
			tempStruct.BidVol = (it->second).bid_volty;
			tempStruct.AskVol = (it->second).ask_volty;
			tempPut.insert(pair<double, temp> ((it->second).tags.strike, tempStruct));
		}
	}

	// Get distinct vector of strikes
	vector<double>::iterator vt;
	
	std::sort(strikeList.begin(), strikeList.end());
	vt = std::unique(strikeList.begin(), strikeList.end());
	strikeList.erase(vt, strikeList.end());

	// Insert into a merge table
	outVol volTbl;
	map<double, temp>::iterator ti;
	outMap.clear();

	for(vt = strikeList.begin(); vt != strikeList.end(); vt++)
	{
		volTbl.bid_volP = tempPut.find(*vt)->second.BidVol;
		volTbl.ask_volP = tempPut.find(*vt)->second.AskVol;
		volTbl.bid_volC = tempCall.find(*vt)->second.BidVol;
		volTbl.ask_volC = tempCall.find(*vt)->second.AskVol;

		outMap.insert(pair<double, outVol> (*vt, volTbl));
	}
}

void Instrument::outputCSV(std::string filename)
{
	ofstream outFile;
	outFile.open(filename);

	map<double, outVol>::iterator it;

	// Print column headers
	outFile << "Strike,BidVolP,AskVolP,BidVolC,AskVolC\n";

	for(it = outMap.begin(); it != outMap.end(); it++)
	{
		outFile << it->first << "," << (it->second).bid_volP << "," << (it->second).ask_volP << "," << (it->second).bid_volC << "," << (it->second).ask_volC << "\n";
	}
	outFile.close();
}
