#ifndef FILEMAPIO_H
#define FILEMAPIO_H

#include <string>
#include <vector>
#include <map>

using namespace std;

struct instrument{
	string type;
	int expiry;
	double strike;
	string option_type;
}; 

class FileMapIO
{
	public:
		FileMapIO(string fname, char delm = ','):
			filename(fname), delimeter(delm){};
		map<int, instrument> getDataSet();
		void readFile();
		void printFile();

	private:
		string filename;
		char delimeter;
		instrument dataSet;
		int optionID;

		map<int, instrument> dataMap;
};
#endif
