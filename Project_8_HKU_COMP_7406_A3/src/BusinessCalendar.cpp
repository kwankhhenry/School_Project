#include "BusinessCalendar.h"

BusinessCalendar::BusinessCalendar()
{
	impl_ = boost::shared_ptr<QuantLib::Calendar::Impl> (new Impl);
}

BusinessCalendar::BusinessCalendar(std::string holidayfile, std::string market)
{
	this->loadMkt = market;
	ifstream holFile;
	holFile.open(holidayfile.c_str());
	
	if(holFile.is_open())
	{
		std::string exchangeMkt;
		std::string dateStr;
		
		while(!holFile.eof())
		{
			getline(holFile,exchangeMkt,',');
			getline(holFile,dateStr,'\n');

			if(market == "ALL")
			{
				holStrMap.insert( std::pair<std::string, std::string>(exchangeMkt, dateStr));
			}
			else
			{
				if(exchangeMkt == market)
					holStrMap.insert( std::pair<std::string, std::string>(exchangeMkt, dateStr));
			}
		}
		holFile.close();

		cout << "Read holiday file complete.\n";
	}
	else
		cout << "Failed to open file!.\n";

	impl_ = boost::shared_ptr<QuantLib::Calendar::Impl> (new Impl);
}

bool BusinessCalendar::Impl::isWeekend(QuantLib::Weekday myWeekday) const
{
	if (myWeekday == 1 || myWeekday == 7)
		return true;
	else
		return false;
}

bool BusinessCalendar::Impl::isBusinessDay(const QuantLib::Date& myDate) const
{
	if (myDate.weekday() == 1 || myDate.weekday() == 7)
	{
		return false;
	}
	else
	{
        if (addedHolidays.find(myDate) != addedHolidays.end())
            return false;
		else
			return true;
	}
}

void BusinessCalendar::applyMktHoliday(std::string Mkt)
{
	std::multimap<std::string, std::string>::iterator it;

	if (holStrMap.empty())
		std::cout << "Holiday file is empty!\n";
	else if (Mkt == "ALL")
		std::cout << "A market needs to be specified!\n";
	else if ((it = holStrMap.find(Mkt)) != holStrMap.end())
	{
		std::cout << "Market = " << Mkt << " found. Applying holiday.\n";
		std::cout << "Reinitialize holiday buffer...\n";
		impl_->addedHolidays.clear();
		impl_->removedHolidays.clear();

		for(it = holStrMap.begin(); it != holStrMap.end(); it++)
		{
			if ((*it).first == Mkt)
			{
				// Parse date string format
				QuantLib::Date holiDate = QuantLib::DateParser::parseFormatted((*it).second, "%Y%m%d");
				addHoliday(holiDate);
			}
		}
		std::cout << "Add holiday completed.\n";
	}
	else
		std::cout << "Selected market is not found in file!\n";
}

void BusinessCalendar::printHolidayFile(std::string myMkt)
{
	std::multimap<std::string, std::string>::iterator it;

	for(it = holStrMap.begin(); it != holStrMap.end(); it++)
	{
		if(myMkt == "ALL")
			std::cout << (*it).first << "," << (*it).second << "\n";
		else if(myMkt == (*it).first)
			std::cout << (*it).first << "," << (*it).second << "\n";
		else
		{
			std::cout << "Unknown market input.\n";
			break;
		}
	}
}

void BusinessCalendar::addHoli(const QuantLib::Date& myDate)
{
	addHoliday(myDate);

	// Find if exist on removedHoliday list
	std::set<QuantLib::Date>::iterator it;

	if((it = impl_->removedHolidays.find(myDate)) != impl_->removedHolidays.end())
		impl_->removedHolidays.erase(myDate);
}

void BusinessCalendar::removeHoli(const QuantLib::Date& myDate)
{
	removeHoliday(myDate);
	impl_->removedHolidays.insert(myDate);
}

void BusinessCalendar::printAddedHoliday()
{
	std::set<QuantLib::Date>::iterator it;

	std::cout << "Printing holiday list:\n";
	if(impl_->addedHolidays.empty() == false)
	{
		for(it = impl_->addedHolidays.begin(); it != impl_->addedHolidays.end(); it++)
		{
			std::cout << (*it) << "\n";
		}
	}
	else
		std::cout << "Added holiday list is empty!\n";
}

void BusinessCalendar::printRemovedHoliday()
{
	std::cout << "Printing removed holiday list:\n";
	std::set<QuantLib::Date>::iterator it;
	if(impl_->removedHolidays.empty() == false)
	{
		for(it = impl_->removedHolidays.begin(); it != impl_->removedHolidays.end(); it++)
		{
			std::cout << (*it) << "\n";
		}
	}
	else
		std::cout << "Removed holiday list is empty!\n";

}

std::string BusinessCalendar::getMyMarket()
{
	if(loadMkt.empty() == true)
		return "Empty";
	else if(loadMkt == "ALL")
		return "Error holiday list";
	else
		return loadMkt;
}
