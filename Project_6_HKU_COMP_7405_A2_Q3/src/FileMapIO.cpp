#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include "FileMapIO.h"

using namespace std;

void FileMapIO::readFile()
{
	string line;
	ifstream ifile;
	
	ifile.open(filename.c_str());
	if(ifile)
	{
		cout << "File " << filename << " is read successfully!\n"; 
		getline(ifile, line);
		while(getline(ifile, line))
		{
			stringstream lineStream;
			lineStream << line;
			string token;

			if(getline(lineStream, token, ','))
				dataSet.type = token;

			getline(lineStream, token, ',');
			if (token == "")
				optionID = 0;
			else
				optionID = stoi(token);

			getline(lineStream, token, ',');
			if (token == "")
				dataSet.expiry = 0;
			else
				dataSet.expiry = stoi(token);
			
			getline(lineStream, token, ',');
			if (token == "")
				dataSet.strike = 0.0;
			else
				dataSet.strike = stod(token);
	
			getline(lineStream, token, '\r');
			if (token == "")
				dataSet.option_type = "S";
			else
				dataSet.option_type = token;

		dataMap.insert(pair<int, instrument>(optionID, dataSet));
		}
		ifile.close();
	}
	else
	{
		cerr << "Failed to read " << filename << "\n";
	}
}

void FileMapIO::printFile()
{
	map<int, instrument>::iterator it;

	cout << setw(8) << "Key" << " ";
	cout << setw(6) << "Type" << " ";
	cout << setw(8) << "Expiry" << " ";
	cout << setw(6) << "Strike" << " ";
	cout << setw(1) << "T" << "\n";

	for(it = dataMap.begin(); it != dataMap.end(); it++)
	{
		cout << setw(8) << it->first << " ";
		cout << setw(6) << (it->second).type << " ";
		cout << setw(8) << (it->second).expiry << " ";
		cout << setw(6) << (it->second).strike << " ";
		cout << setw(1) << (it->second).option_type << "\n";
	}
}

map<int, instrument> FileMapIO::getDataSet()
{
	return dataMap;
}
