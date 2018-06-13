#include <iostream>
#include <map>
#include <string>

#include "BusinessCalendar.h"

#include <ql/time/calendars/all.hpp>
#include <ql/termstructures/yield/ratehelpers.hpp>
#include <ql/termstructures/yield/piecewiseyieldcurve.hpp>

#include <ql/indexes/ibor/jpylibor.hpp>
#include <ql/indexes/ibor/tibor.hpp>
#include <ql/time/imm.hpp>

#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>

using namespace std;
using namespace QuantLib;

void CurveBuilding(BusinessCalendar& myCalendar, bool prtFlag)
{
	try
	{
	// Beginning of programs
	Calendar calendar = myCalendar;  // Provide My Calendar

	/**************************** BEGINNING OF CURVE BUILDING ******************************/
	// Define spot date and make sure it is a business day
	
	std::cout << "Markets of exchange = " << myCalendar.getMyMarket() << endl;
	Date settlementDate(12, March, 2007);
	settlementDate = calendar.adjust(settlementDate);

	// Find base date
	Integer fixingDays = 2;
	Date baseDate = calendar.advance(settlementDate, -fixingDays, Days);

	// Set up trade date for future valuation
	Settings::instance().evaluationDate() = baseDate;
	baseDate = Settings::instance().evaluationDate();

	std::cout << "Base Date = " << baseDate.weekday() << ", " << baseDate << "\n";
	std::cout << "Spot Date = " << settlementDate.weekday() << ", " << settlementDate << "\n";

    /*********************
    ***  MARKET DATA  ***
    *********************/
	// Money Deposit
	Rate ONQuote=0.0057000;
	Rate TNQuote=0.0057000;
	Rate d3mQuote=0.0070625;
	// Futures
    Real fut1Quote=99.3070;
    Real fut2Quote=99.3082;
    Real fut3Quote=99.2235;
    Real fut4Quote=99.1369;
    Real fut5Quote=99.0504;
    Real fut6Quote=98.9680;
    Real fut7Quote=98.8927;
    Real fut8Quote=98.8175;
    // Swaps
    Rate s3yQuote=0.0105625;
    Rate s4yQuote=0.0118750;
    Rate s5yQuote=0.0130750;
    Rate s6yQuote=0.0141375;
    Rate s7yQuote=0.0150875;
    Rate s8yQuote=0.0159750;
    Rate s9yQuote=0.0168000;
    Rate s10yQuote=0.0175750;
    Rate s12yQuote=0.0189125;
    Rate s15yQuote=0.0204875;
    Rate s20yQuote=0.0223250;
    Rate s25yQuote=0.0234525;
    Rate s30yQuote=0.0241375;
    Rate s40yQuote=0.0249375;

    /********************
    ***    QUOTES    ***
    ********************/
	// Money Deposit
    boost::shared_ptr<Quote> ONRate(new SimpleQuote(ONQuote));
    boost::shared_ptr<Quote> TNRate(new SimpleQuote(TNQuote));
    boost::shared_ptr<Quote> d3mRate(new SimpleQuote(d3mQuote));
    // Futures
    boost::shared_ptr<Quote> fut1Price(new SimpleQuote(fut1Quote));
    boost::shared_ptr<Quote> fut2Price(new SimpleQuote(fut2Quote));
    boost::shared_ptr<Quote> fut3Price(new SimpleQuote(fut3Quote));
    boost::shared_ptr<Quote> fut4Price(new SimpleQuote(fut4Quote));
    boost::shared_ptr<Quote> fut5Price(new SimpleQuote(fut5Quote));
    boost::shared_ptr<Quote> fut6Price(new SimpleQuote(fut6Quote));
    boost::shared_ptr<Quote> fut7Price(new SimpleQuote(fut7Quote));
    boost::shared_ptr<Quote> fut8Price(new SimpleQuote(fut8Quote));
    // Swaps
    boost::shared_ptr<Quote> s3yRate(new SimpleQuote(s3yQuote));
    boost::shared_ptr<Quote> s4yRate(new SimpleQuote(s4yQuote));
    boost::shared_ptr<Quote> s5yRate(new SimpleQuote(s5yQuote));
    boost::shared_ptr<Quote> s6yRate(new SimpleQuote(s6yQuote));
    boost::shared_ptr<Quote> s7yRate(new SimpleQuote(s7yQuote));
    boost::shared_ptr<Quote> s8yRate(new SimpleQuote(s8yQuote));
    boost::shared_ptr<Quote> s9yRate(new SimpleQuote(s9yQuote));
    boost::shared_ptr<Quote> s10yRate(new SimpleQuote(s10yQuote));
    boost::shared_ptr<Quote> s12yRate(new SimpleQuote(s12yQuote));
    boost::shared_ptr<Quote> s15yRate(new SimpleQuote(s15yQuote));
    boost::shared_ptr<Quote> s20yRate(new SimpleQuote(s20yQuote));
    boost::shared_ptr<Quote> s25yRate(new SimpleQuote(s25yQuote));
    boost::shared_ptr<Quote> s30yRate(new SimpleQuote(s30yQuote));
    boost::shared_ptr<Quote> s40yRate(new SimpleQuote(s40yQuote));

    /*********************
    ***  RATE HELPERS ***
    *********************/
    // Money Deposit (ACT/360)
    DayCounter depositDayCounter = Actual360();

    boost::shared_ptr<RateHelper> dON(new DepositRateHelper(
        Handle<Quote>(ONRate),
        1*Days, 0,
        calendar, ModifiedFollowing,
        true, depositDayCounter));
    boost::shared_ptr<RateHelper> dTN(new DepositRateHelper(
        Handle<Quote>(TNRate),
        2*Days, 0,
        calendar, ModifiedFollowing,
        true, depositDayCounter));
    boost::shared_ptr<RateHelper> d3m(new DepositRateHelper(
        Handle<Quote>(d3mRate),
        3*Months, fixingDays,
        calendar, ModifiedFollowing,
        true, depositDayCounter));

    // Futures (ACT/360)
    // Rate convexityAdjustment = 0.0;
    Integer futMonths = 3;
    Date imm = IMM::nextDate(settlementDate); // Identify the four quarterly dates of each year for maturity date
    boost::shared_ptr<RateHelper> fut1(new FuturesRateHelper(
        Handle<Quote>(fut1Price),
        imm,
        futMonths, calendar, ModifiedFollowing,
        true, depositDayCounter));
        imm = IMM::nextDate(imm+1);
    boost::shared_ptr<RateHelper> fut2(new FuturesRateHelper(
        Handle<Quote>(fut2Price),
        imm,
        futMonths, calendar, ModifiedFollowing,
        true, depositDayCounter));
        imm = IMM::nextDate(imm+1);
    boost::shared_ptr<RateHelper> fut3(new FuturesRateHelper(
        Handle<Quote>(fut3Price),
        imm,
        futMonths, calendar, ModifiedFollowing,
        true, depositDayCounter));
        imm = IMM::nextDate(imm+1);
    boost::shared_ptr<RateHelper> fut4(new FuturesRateHelper(
        Handle<Quote>(fut4Price),
        imm,
        futMonths, calendar, ModifiedFollowing,
        true, depositDayCounter));
        imm = IMM::nextDate(imm+1);
    boost::shared_ptr<RateHelper> fut5(new FuturesRateHelper(
        Handle<Quote>(fut5Price),
        imm,
        futMonths, calendar, ModifiedFollowing,
        true, depositDayCounter));
        imm = IMM::nextDate(imm+1);
    boost::shared_ptr<RateHelper> fut6(new FuturesRateHelper(
        Handle<Quote>(fut6Price),
        imm,
        futMonths, calendar, ModifiedFollowing,
        true, depositDayCounter));
        imm = IMM::nextDate(imm+1);
    boost::shared_ptr<RateHelper> fut7(new FuturesRateHelper(
        Handle<Quote>(fut7Price),
        imm,
        futMonths, calendar, ModifiedFollowing,
        true, depositDayCounter));
        imm = IMM::nextDate(imm+1);
    boost::shared_ptr<RateHelper> fut8(new FuturesRateHelper(
        Handle<Quote>(fut8Price),
        imm,
        futMonths, calendar, ModifiedFollowing,
        true, depositDayCounter));

    // Swaps (ACT/365)
    Frequency swFixedLegFrequency = Semiannual;
    BusinessDayConvention swFixedLegConvention = ModifiedFollowing;//Unadjusted;
    DayCounter swFixedLegDayCounter = Actual365Fixed();//Actual365Fixed::Standard
    boost::shared_ptr<IborIndex> swFloatingLegIndex(new JPYLibor(Period(6, Months)));

    boost::shared_ptr<RateHelper> s3y(new SwapRateHelper(
        Handle<Quote>(s3yRate), 3*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s4y(new SwapRateHelper(
        Handle<Quote>(s4yRate), 4*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s5y(new SwapRateHelper(
        Handle<Quote>(s5yRate), 5*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s6y(new SwapRateHelper(
        Handle<Quote>(s6yRate), 6*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s7y(new SwapRateHelper(
        Handle<Quote>(s7yRate), 7*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s8y(new SwapRateHelper(
        Handle<Quote>(s8yRate), 8*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s9y(new SwapRateHelper(
        Handle<Quote>(s9yRate), 9*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s10y(new SwapRateHelper(
        Handle<Quote>(s10yRate), 10*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s12y(new SwapRateHelper(
        Handle<Quote>(s12yRate), 12*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s15y(new SwapRateHelper(
        Handle<Quote>(s15yRate), 15*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s20y(new SwapRateHelper(
        Handle<Quote>(s20yRate), 20*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s25y(new SwapRateHelper(
        Handle<Quote>(s25yRate), 25*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s30y(new SwapRateHelper(
        Handle<Quote>(s30yRate), 30*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));
    boost::shared_ptr<RateHelper> s40y(new SwapRateHelper(
        Handle<Quote>(s40yRate), 40*Years,
        calendar, swFixedLegFrequency,
        swFixedLegConvention, swFixedLegDayCounter,
        swFloatingLegIndex));

    /*********************
    **  CURVE BUILDING **
    *********************/

	// Collection of interest rates for all different terms
    // Any DayCounter would be fine.
    // ActualActual::ISDA ensures that 30 years is 30.0
    DayCounter termStructureDayCounter =
    ActualActual(ActualActual::ISDA);

    double tolerance = 1.0e-15;

    // A depo-futures-swap curve
    std::vector<boost::shared_ptr<RateHelper> > depoFutSwapInstruments;
    depoFutSwapInstruments.push_back(dON);
    depoFutSwapInstruments.push_back(dTN);
    depoFutSwapInstruments.push_back(d3m);
    depoFutSwapInstruments.push_back(fut1);
    depoFutSwapInstruments.push_back(fut2);
    depoFutSwapInstruments.push_back(fut3);
    depoFutSwapInstruments.push_back(fut4);
    depoFutSwapInstruments.push_back(fut5);
    depoFutSwapInstruments.push_back(fut6);
    depoFutSwapInstruments.push_back(fut7);
    depoFutSwapInstruments.push_back(fut8);
    depoFutSwapInstruments.push_back(s3y);
    depoFutSwapInstruments.push_back(s4y);
    depoFutSwapInstruments.push_back(s5y);
    depoFutSwapInstruments.push_back(s6y);
    depoFutSwapInstruments.push_back(s7y);
    depoFutSwapInstruments.push_back(s8y);
    depoFutSwapInstruments.push_back(s9y);
    depoFutSwapInstruments.push_back(s10y);
    depoFutSwapInstruments.push_back(s12y);
    depoFutSwapInstruments.push_back(s15y);
    depoFutSwapInstruments.push_back(s20y);
    depoFutSwapInstruments.push_back(s25y);
    depoFutSwapInstruments.push_back(s30y);
    depoFutSwapInstruments.push_back(s40y);

    boost::shared_ptr<YieldTermStructure> depoFutSwapTermStructure(
        new PiecewiseYieldCurve<Discount,LogLinear>(
                                    baseDate, depoFutSwapInstruments,
                                    termStructureDayCounter,
                                    tolerance));

    /*****************
    * CURVE BUILDING *
    ******************/
	// Utilities for reporting
    std::vector<std::string> headers(5);
    headers[0] = "Day";
    headers[1] = std::string(3, ' ') + "Date" + std::string(3, ' ');
	headers[2] = "Day of Week";
    headers[3] = "Discount Factor";
	headers[4] = "Key Dates Description" + std::string(15, ' ');

    std::string separator = " | ";
    Size width = headers[0].size() + separator.size()
               + headers[1].size() + separator.size()
               + headers[2].size() + separator.size()
			   + headers[3].size() + separator.size()
               + headers[4].size() + separator.size() - 1;
    std::string rule(width, '-'), dblrule(width, '=');
    std::string tab(8, ' ');

	// Provide key dates description
	std::map<Date, std::string> keyDates;

	//Date Date1(8, March, 2007);
	keyDates.insert(std::pair<Date, std::string>(baseDate, "Base Date"));
	//Date Date2(9, March, 2007);
	keyDates.insert(std::pair<Date, std::string>(dON->maturityDate(), "ON Date"));
	//Date Date3(12, March, 2007);
	keyDates.insert(std::pair<Date, std::string>(dTN->maturityDate(), "TN Date or Spot Date"));
	Date Date4(21, March, 2007);
	keyDates.insert(std::pair<Date, std::string>(Date4, "1st Futures"));
	//Date Date5(12, June, 2007);
	keyDates.insert(std::pair<Date, std::string>(d3m->maturityDate(), "3M"));
	Date Date6(20, June, 2007);
	keyDates.insert(std::pair<Date, std::string>(Date6, "2nd Futures"));
	Date Date7(21, June, 2007);
	keyDates.insert(std::pair<Date, std::string>(Date7, "Maturity 1st Futures"));
	Date Date8(19, September, 2007);
	keyDates.insert(std::pair<Date, std::string>(Date8, "3rd Futures"));
	Date Date9(20, September, 2007);
	keyDates.insert(std::pair<Date, std::string>(Date9, "Maturity 2nd Futures"));
	Date Date10(19, December, 2007);
	keyDates.insert(std::pair<Date, std::string>(Date10, "Maturity 3rd Futures/ 4th Futures"));
	Date Date11(19, March, 2008);
	keyDates.insert(std::pair<Date, std::string>(Date11, "Maturity 4th Futures/ 5th Futures"));
	Date Date12(18, June, 2008);
	keyDates.insert(std::pair<Date, std::string>(Date12, "6th Futures"));
	Date Date13(19, June, 2008);
	keyDates.insert(std::pair<Date, std::string>(Date13, "Maturity of 5th Futures"));
	Date Date14(17, September, 2008);
	keyDates.insert(std::pair<Date, std::string>(Date14, "7th Futures"));
	Date Date15(18, September, 2008);
	keyDates.insert(std::pair<Date, std::string>(Date15, "Maturity of 6th Futures"));
	Date Date16(17, December, 2008);
	keyDates.insert(std::pair<Date, std::string>(Date16, "Maturity of 7th Futures/ 8th Futures"));
	Date Date17(17, March, 2009);
	keyDates.insert(std::pair<Date, std::string>(Date17, "Maturity of 8th Futures"));
	Date Date18(14, September, 2009);
	keyDates.insert(std::pair<Date, std::string>(Date18, "One period before 3Y SWAP Maturity"));
	keyDates.insert(std::pair<Date, std::string>(s3y->maturityDate(), "DF for 3Y SWAP"));
	Date Date19(13, September, 2010);
	keyDates.insert(std::pair<Date, std::string>(Date19, "One period before 4Y SWAP Maturity"));
	keyDates.insert(std::pair<Date, std::string>(s4y->maturityDate(), "DF for 4Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s5y->maturityDate(), "DF for 5Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s6y->maturityDate(), "DF for 6Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s7y->maturityDate(), "DF for 7Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s8y->maturityDate(), "DF for 8Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s9y->maturityDate(), "DF for 9Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s10y->maturityDate(), "DF for 10Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s12y->maturityDate(), "DF for 12Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s15y->maturityDate(), "DF for 15Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s20y->maturityDate(), "DF for 20Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s25y->maturityDate(), "DF for 25Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s30y->maturityDate(), "DF for 30Y SWAP"));
	keyDates.insert(std::pair<Date, std::string>(s40y->maturityDate(), "DF for 40Y SWAP"));

	// Print columns
    std::cout << dblrule << std::endl;
    std::cout <<  "Discount Factor by Dates" << std::endl;
    std::cout << dblrule << std::endl;

    std::cout << headers[0] << separator
              << headers[1] << separator
			  << headers[2] << separator
			  << headers[3] << separator
			  << headers[4] << separator << std::endl;
    std::cout << rule << std::endl;

	// Calculations
	bool prtAllFlag = prtFlag; // Flag to determine if range dates should be printed
	Date startDate(8, March, 2007);
	Date endDate(12, March, 2047);

	DayCounter dayDiff = Actual360();

	if ( prtAllFlag == 0 )
	{
		for(BigInteger i = 0; i <= dayDiff.dayCount(startDate, endDate); i++)
		{
			map<Date, std::string>::iterator it = keyDates.find((baseDate+i));

			std::cout << std::setw(headers[0].size())
					  << i << separator;
			std::cout << std::setw(headers[1].size())
					  << io::short_date(baseDate+i) << separator;
			std::cout << std::setw(headers[2].size())
					  << (baseDate+i).weekday() << separator;
			std::cout << std::setw(headers[3].size())
					  << std::fixed << std::setprecision(9) << depoFutSwapTermStructure->discount(baseDate+i) << separator;

			if( it != keyDates.end())
			{
			std::cout << std::setw(headers[4].size())
					  << ((*it).second) << separator;
			}
			else
			{
			std::cout << std::setw(headers[4].size())
					  << "" << separator;
			}
			std::cout << std::endl;
		}
	}
	else
	{
		for(std::map<Date, std::string>::iterator it = keyDates.begin(); it != keyDates.end(); it++)
		{
			std::cout << std::setw(headers[0].size())
					  << distance(keyDates.begin(), it) << separator;
			std::cout << std::setw(headers[1].size())
					  << io::short_date((*it).first) << separator;
			std::cout << std::setw(headers[2].size())
					  << ((*it).first).weekday() << separator;
			std::cout << std::setw(headers[3].size())
					  << std::fixed << std::setprecision(9) << depoFutSwapTermStructure->discount((*it).first) << separator;
			std::cout << std::setw(headers[4].size())
					  << ((*it).second) << separator;
			std::cout << std::endl;
		}
	}
    std::cout << rule << std::endl;

    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
        //return 1;
    } catch (...) {
        std::cerr << "unknown error" << std::endl;
        //return 1;
    }
}
