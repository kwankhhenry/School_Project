#ifndef BUSINESS_CALENDAR_H
#define BUSINESS_CALENDAR_H

#include <iostream>
#include <fstream>

#include <map>
#include <string>

#include <ql/utilities/dataparsers.hpp>
#include <ql/time/calendar.hpp>
#include <ql/time/date.hpp>

using namespace std;

class BusinessCalendar : public QuantLib::Calendar
{
	public:
		BusinessCalendar(std::string holidayfile, std::string = "ALL");
		void applyMktHoliday(std::string = "ALL");

		void addHoli(const QuantLib::Date&);
		void removeHoli(const QuantLib::Date&);

		void printHolidayFile(std::string = "ALL");
		void printAddedHoliday();
		void printRemovedHoliday();
		
		std::string getMyMarket();

	private:
		std::multimap<std::string, std::string> holStrMap;
		std::string loadMkt;
        
		class Impl : public QuantLib::Calendar::Impl{
			public:
				std::string name() const { return "My Calendar"; }
				bool isWeekend(QuantLib::Weekday) const;
				bool isBusinessDay(const QuantLib::Date&) const;
		};
	public:
		BusinessCalendar();
};
#endif
