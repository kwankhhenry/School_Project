#include <iostream>
#include <iomanip>
#include "BusinessCalendar.h"

using namespace QuantLib;

void CurveBuilding(BusinessCalendar&, bool);

int main(int argc, char* argv[])
{
	// Beginning of programs
	BusinessCalendar myBusCalendar("Holiday.csv");

	// Program interface
	Size width = 89;
	Size textwidth = width-2;
    std::string separator = " | ";
    std::string rule(width, '-'), dblrule(width, '=');
    std::string tab(8, ' ');

	// Parameters to initialize
	std::string myInput;
	int Opts;
	Date inDate;
	bool prtFlag;

	while(1)
	{
	std::cout << dblrule << std::endl;
	std::cout << "Program interface for COMP7406 QuantLib Assignment 3\n";
	std::cout << dblrule << std::endl;
	std::cout << std::setw(textwidth) << "" << separator << std::endl;
	std::cout << std::setw(textwidth) << left << "Please select following options:" << separator << "\n";
	std::cout << std::setw(textwidth) << tab + "1) Print imported calendar. (Raw data)" << separator << "\n";
	std::cout << std::setw(textwidth) << tab + "2) Apply calendar of input market. (Will override calendar)" << separator << "\n";
	std::cout << std::setw(textwidth) << tab + "3) Print added holiday list." << separator << "\n";
	std::cout << std::setw(textwidth) << tab + "4) Print removed holiday list." << separator << "\n";
	std::cout << std::setw(textwidth) << tab + "5) Test case#1 Add holiday to calendar." << separator << "\n";
	std::cout << std::setw(textwidth) << tab + "6) Test case#2 Remove holiday from calendar." << separator << "\n";
	std::cout << std::setw(textwidth) << tab + "7) Test case#3 Curve Building. (Print key dates only)" << separator << "\n";
	std::cout << std::setw(textwidth) << tab + "8) Test case#4 Curve Building. (Print all dates)" << separator << "\n";
	std::cout << std::setw(textwidth) << "" << separator << std::endl;
	std::cout << dblrule << std::endl;

	std::cout << "Your input (option 1-8): ";
	std::cin >> Opts;

	switch(Opts)
		{
		case 1:
			std::cout << "Printing raw calendar file...\n";
			myBusCalendar.printHolidayFile();
			break;
		case 2:
			std::cout << "Please enter your market: ";
			std::cin >> myInput;
			myBusCalendar.applyMktHoliday(myInput);
			break;
		case 3:
			myBusCalendar.printAddedHoliday();
			break;
		case 4:
			myBusCalendar.printRemovedHoliday();
			break;
		case 5:
			std::cout << "Enter holiday to be added. (Format: yyyymmdd)\n";
			std::cin >> myInput;
			inDate = DateParser::parseFormatted(myInput, "%Y%m%d");
			myBusCalendar.addHoli(inDate);
			myBusCalendar.printAddedHoliday();		
			break;
		case 6:
			std::cout << "Enter holiday to be removed. (Format: yyyymmdd)\n";
			std::cin >> myInput;
			inDate = DateParser::parseFormatted(myInput, "%Y%m%d");
			myBusCalendar.removeHoli(inDate);
			myBusCalendar.printAddedHoliday();		
			break;
		case 7:
			prtFlag = true;
			CurveBuilding(myBusCalendar, prtFlag);
			break;
		case 8:
			prtFlag = false;
			CurveBuilding(myBusCalendar, prtFlag);
			break;
		default:
			std::cout << "Unknown input.  Please enter again.\n";
		}
	}
	return 0;
}
