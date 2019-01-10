#
#
# Import Required Libraries
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import os
import seaborn as sns ; sns.set()
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as sco
import time
from datetime import datetime
from dateutil.parser import parse
#%matplotlib inline
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
#
# In[9]:
# SET THE PATH TO THE EXCEL INPUT FILES!!!
HomeFolder = "D:/Boris/LEM_Strategy/Software/Data_Download"
OutPutFolder = HomeFolder+"/DataOutPut"
os.chdir(HomeFolder)   # Windows 10
# os.chdir('//Volumes/Data/Exchange')  # MAc OS X
# File Name with EM FX Currency Data, Benchmark Index/Weights, Matrix Shrinkage Factors.
FX_Sheet = "_FX_Rates.xlsx"
FX_Tab = "FX_Rates"
# Factors
Factor_Sheet = "_Factors.xlsx"
FactorTab = "Factors"
# BenchMark Portfolio Data
Benchmark_Sheet = "_Benchmark.xlsx"
BenchmarkTab = "Benchmark"
# Local Currency Interest Index
LocCurr_Sheet = "_LocCurr_Index.xlsx"
LocCurrTab = "LocCurr_Index"
##
# USE THIS TO WORK WITH ALL THE DATA OR A SUBSET
# Using SMALL dataset vastly Improves Execution Time when testing the SW
#Dataset = 'Large'
#Dataset = 'Small'
#
#
# In[9]:
# Global Variables Never Change
#
# SET THE DATE RANGES OF THE TOTAL DATASET
Start = '01-02-1999'
End = '01-09-2019'
#
Days_in_Trading_Year = 252
#
ReturnPeriod = 5  # Use weekly Returns assuming data without weekends, ie a 5 day week
ReturnFrequency = 'W-WED'
#
Pie_Dims = (10, 10)
Bar_Dims = (10, 16)

# In[]
# Get the Return Period (in days) for the potential frequency of returns in Python
def getreturnperiod (f):
    #
    Months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    Days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
    # Generate Lists of Frequency Types in Python
    Daily = ['D', 'B', ]
    Monthly = ['M', 'BM', 'MS', 'BMS']
    #
    Weekly =  []; Quarterly = []; Annualy = []
    #
    W = 'W-'
    for d in Days:
        Weekly.append(W+d)
    #
    Q0 = 'Q-' ; Q1 = 'QS-'; Q2 = 'BQ-'; Q3 = 'BQS-'
    for m in Months:
        Quarterly.append(Q0+m)
        Quarterly.append(Q1+m)
        Quarterly.append(Q2+m)
        Quarterly.append(Q3+m)
    #
    A0 = 'A-'; A1 = 'AS-'; A2 = 'BA-'; A3 = 'BAS-'
    for m in Months:
        Annualy.append(A0+m)
        Annualy.append(A1+m)
        Annualy.append(A2+m)
        Annualy.append(A3+m)
    #
    if f in Daily:
        return 1
    elif f in Weekly:
        return round (Days_in_Trading_Year / 52, 0)
    elif f in Monthly:
        return round (Days_in_Trading_Year / 12, 0)
    elif f in Quarterly:
        return round (Days_in_Trading_Year / 4, 0)
    elif f in Annualy:
        return Days_in_Trading_Year
    else:
        print ("ERROR IN FREQUENCY: GETRETURNPERIOD ")
        return 0
    
# FACTORS TO ANNUALIZE Log Return/Var
Ret_Factor =  Days_in_Trading_Year / getreturnperiod (ReturnFrequency)
Std_Factor = np.sqrt ( Ret_Factor )
Var_Factor = Ret_Factor #
#
#
# In[]
#
# Clean The Time Series of Bad Data and Weekend Dates
#

def CleanDataFrame (frame):   # REMOVES BAD DATA AND WEEKENDS FROM TIME SERIES
    frame.dropna(inplace = True) # Drop Grabage
    return frame[frame.index.dayofweek < 5] # get only weekdays


# Upload Excel Data to Panda DataFrames, Clean the Data
#
def ProcessExcelFile (File, Tab, Ind, Col):
    Raw = pd.read_excel(File, sheet_name = Tab, index_col = Ind, header = Col, parse_dates = True)
    return CleanDataFrame (Raw)
#
# In[]
# 
%time ALL_Factor = ProcessExcelFile(Factor_Sheet, FactorTab,  0,  3)
%time ALL_FX = ProcessExcelFile(FX_Sheet,  FX_Tab,  0,  3) 
%time ALL_LocCurr = ProcessExcelFile(LocCurr_Sheet, LocCurrTab, 0, 3)
%time ALL_BenchMark = ProcessExcelFile(Benchmark_Sheet, BenchmarkTab,  0, 3)
#
# In[]

#     SPLIT THE DATA into Prices, Factors, BenchMarks and BenchMarkWeights
#
# Get Money Market Index Data
MM_Header = ['xUSD']
MoneyMarket = pd.DataFrame(ALL_LocCurr[MM_Header]).copy()
#
Factor_Headers = ['MXWD','DXY','VIX','CRB','US_5Y/5Y_CPI','US_LBR_1Y/1Y','US-EU_1Y_Sprd','US-JY_1Y_Sprd']
Factors = pd.DataFrame(ALL_Factor[Factor_Headers]).copy()
#
BenchMark_Headers = ['EM22_FX','BBRG_8EM','BBRG_ASIA','BBRG_EMEA','BBRG_G10','BBRG_Latam']
BenchMark =  pd.DataFrame(ALL_BenchMark[BenchMark_Headers]).copy()
#
#
# USE Large or Small Currency Dataset
FX_Headers = ['ARS','CNH','RUB','SGD','TRY','ZAR','BRL','CLP','COP','IDR','INR','KRW','MYR','PEN','PHP','THB','TWD','MXN','EURHUF','EURCZK','EURPLN','EURRON']
#
BMW1 = ['_ARS','_CNH','_CZK','_PLN','_RON','_RUB','_SGD','_TRY','_ZAR','_BRL','_CLP']   # Spyder can't handle long lines of code??? Jupyter notebook worked fine...
BMW2 = ['_COP','_IDR','_INR','_KRW','_MYR','_PEN','_PHP','_THB','_TWD','_MXN','_HUF']
BMW  = BMW1 + BMW2
#
LCH1 = ['xARS','xCNH','xRUB','xSGD','xTRY','xZAR','xBRL','xCLP','xCOP','xIDR','xINR']
LCH2 = ['xKRW','xMYR','xPEN','xPHP','xTHB','xTWD','xMXN','xHUF','xCZK','xPLN','xRON']
LCH  = LCH1 + LCH2
#
#if Dataset == "Small"  : # Spyder: what's wrong with this IF?? Jupyter notebook worked fine
#    FX_Headers = ['ARS','TRY','BRL','INR','MXN']
#    #
#    BMW = ['_ARS','_TRY','_BRL','_INR','_MXN']
#    #
#   LCH = ['xARS','xTRY','xBRL','xINR','xMXN']
#        
Prices = pd.DataFrame(ALL_FX[FX_Headers]).copy()
BenchMarkWeights = pd.DataFrame(ALL_BenchMark[BMW]).copy()
LocCurrs = pd.DataFrame(ALL_LocCurr[LCH]).copy()
#
#  Normalize Financial Time Series
#
MoneyMarket = MoneyMarket / MoneyMarket.iloc[0]
Factors = Factors / Factors.iloc[0]
BenchMark = BenchMark / BenchMark.iloc[0]
Prices = Prices / Prices.iloc[0]



# In[]:

#  Functions to Calculate Stats for All Data Series including prices, factors and benchmarks.

#  Discount Price Series using a Numeraire
def discountprices (p, num):
    Numeraire = num.iloc[:,0]  # Get the Numeraire Data
    Matrix =  np.zeros(len(p.iloc[:,1])) # Create an output Matrix with one Dummy Col
    ColHead = [*p.columns]
    Dummy = ['Dummy']
    for i in range(len(ColHead)):
        Matrix = np.c_[Matrix , p.iloc[:,i] / Numeraire ]
        # Build DataFrame with the Matrix, adding Index and columns
        #   Take only the FX_Headers, ie remove the dummy col used to create the original Matrix
    return  (pd.DataFrame(Matrix, index=num.index, columns = Dummy+ColHead))[ColHead]

#  Get Subset of Data and Get Returns on consecutive prices
def extractdata (p, start, end, frequency):
    Dates = pd.date_range(start,end, freq = frequency)
    prices = p.copy()
    prices = prices.reindex(Dates)
    return CleanDataFrame (prices)   # Return a clean dataframe

def getreturns (prices):
    # Get log Returns
    returns = np.log ( prices / prices.shift(1))
    return CleanDataFrame (returns)   # Return a clean dataframe
#
#
    #
def descriptive (sheet,returns, frequency):
    # Descriptive Stats for Return Series: returns ending date of the series, summary stats, covariance matrix and correlation matrix
    
    Index = returns.index
    End = Index[-1].strftime('%Y-%m-%d')
    Start = Index[0].strftime('%Y-%m-%d')
    
    mean = returns.mean(axis=0)
    var = np.var(returns,axis=0)
    stdev = np.sqrt (var)
    skew = returns.skew(axis=0)
    kurtosis = returns.kurtosis(axis=0)
    covar = returns.cov()
    corr = returns.corr()
    sharpe = mean / stdev
    descrip = pd.concat([mean,stdev,var,skew,kurtosis,sharpe],axis=1)
    descrip.columns = ["Mean","StdDev","VAR","Skew","Kurtosis","Mean/StdDev"]
    # Save Data Statistics to an Excel Sheet
    if sheet != '':
        os.chdir(OutPutFolder)  
        writer=pd.ExcelWriter('Descriptive'+'_'+sheet+'_'+Start+'_'+End+'.xlsx', engine='xlsxwriter')
        descrip.to_excel(writer, 'Statistics')
        covar.to_excel(writer, 'Covariance')
        corr.to_excel(writer, 'Correlation')
        writer.save()
    #
    # Returns the Date of the last Return, Summary Stas, Cov and Corr Matricies
    return [returns.index[-1], descrip, covar, corr]
#
# In[]
#  CHARTING TOOLS 
#
def plotheatmap (corr, start_date, end_date, title):
    #S = start_date.strftime('%Y-%m-%d')
    #E = end_date.strftime('%Y-%m-%d')
    # Plot Correlation Matrix Heatmap
    sns.set(font_scale=0.75)
    Plot_Dims = (18.0, 10)
    plt.subplots(figsize = Plot_Dims)
    ax = sns.heatmap(corr, annot = True,   cmap = "coolwarm" )
    ax.set_title(title, fontsize=20)
    fig = ax.get_figure()
    FileName = "HeatMapCorrelations_"+start_date+"-"+end_date+"_"+".png"
    os.chdir(OutPutFolder)
    fig.savefig(FileName)
    # plt.close()
    return
#
def plotlogreturns (P, title):
    # Plot Chart of Price Log Returns 
    Chart = np.log (P).plot(figsize=(18,14), title="Cummulative Log Returns of "+title)
    fig = Chart.get_figure()
    FileName = "Log_Returns_"+title+".png"
    os.chdir(OutPutFolder)
    fig.savefig(FileName)
    #plt.close()
    return
#
def plothistograms (R, title):
    # Plot Histogram of Log Returns 
    plt.figure(figsize=(36,36))
    i=1
    n = int(np.sqrt(len(R.columns)))+1
    for r in R.columns:
        # Set up the plot
        ax =  plt.subplot(n, n, i)
        ax.hist(R[r], bins = 75, color = 'blue', edgecolor = 'black')
        # Title and labels
        ax.set_title(title+': '+r , size = 20)
        ax.set_xlabel('', size = 15)
        ax.set_ylabel('Frequency', size= 15)
        i+=1
    #
    FileName = "Histogram_"+title+'.png'
    os.chdir(OutPutFolder)
    plt.savefig(FileName)
    #plt.close()
    return
#
#
def plotseries (T, S, Title):
    # Plot Histogram of Log Returns 
    plt.figure(figsize=(36,36))
    i=1
    n = int(np.sqrt(len(S.columns)))+1
    for r in S.columns:
        # Set up the plot
        ax =  plt.subplot(n, n, i)
        ax.plot(S[r],  color = 'blue')
        # Title and labels
        ax.set_title(Title+' '+T+'--> '+r , size = 20)
        ax.set_xlabel('Date', size = 15)
        ax.set_ylabel(r, size= 15)
        i+=1
    #
    FileName = Title+'_'+T+'.png'
    os.chdir(OutPutFolder)
    plt.savefig(FileName)
    #plt.close()
    return
#


# In[]:

# Extract The Data For the Start / End Range, Take Weekly Data on Wednesdays (Avoid Long Weekends, Short Weeks)
Weekly_Prices = extractdata (Prices, Start, End, ReturnFrequency)
#  This plots the log performance of all the FX Rates
%time plotlogreturns (Weekly_Prices,"Weekly FX Rates")

#  Get Log Returns
FX_Returns = getreturns (Weekly_Prices)
# Plot Histogram of Log Returns: notice the Fat Tails (indicated by the width of the x-axis, as bars of tails are very small to notice at first sight!)
%time plothistograms (FX_Returns, "Weekly FX Returns")



# In[]:
# Calculate Basic Statistics for Weekly Returns
FX_Stats = descriptive('FX_Returns', FX_Returns, ReturnFrequency)
Historical_Stats = FX_Stats[1]  # View Summary Stats
Historical_Stats


# In[]:
# Plot Correlation Heatmap
plotheatmap (FX_Stats[3], Start, End, 'Discounted EM FX Return Correlation Heatmap')


# In[]:
# Process Factors: View Correlation Heatmap and  Covariance Matrix
Weekly_Factors = extractdata (Factors, Start, End, ReturnFrequency)
Factor_Returns = getreturns (Weekly_Factors)
FactorStats = descriptive('Factor_Returns', Factor_Returns, ReturnFrequency)
plotheatmap (FactorStats[3], Start, End, 'Discounted Portfolio Factors Return Correlation Heatmap')
FactorStats[2]


# In[]:

# Process BenchMarks: Correlation Matrix and Plot  Correlation Heatmap
Weekly_BenchMark = extractdata (BenchMark, Start, End, ReturnFrequency)
BenchMark_Returns = getreturns (Weekly_BenchMark)
BenchMarkStats = descriptive('BenchMark_Returns', BenchMark_Returns, ReturnFrequency)
plotheatmap (BenchMarkStats[3], Start, End, 'Discounted EM BenchMarks Return Correlation Heatmap')
BenchMarkStats[3]  # View Correlation Matrix



# In[]:

# Build a rolling window of data to plot the evolution of price stats.
# Given that we have a long data set, portfolio statistics are not expected to be stable over time. Given that we prefer to work with weekly data (to avoid the 
# highly noisy daily data), while still allowing us to generate long series of windows.
# We will work with windows of 2 years, which provide with around 100 weekly returns.
# We will roll the window from the start of our dataset to be able to trak the evolution over time of each and everyone of the stats calculated (mean returns, correlations, skew....)
#
def WindowStats (returns, freq, WindowInYears):
    #
    Window = int(WindowInYears*Days_in_Trading_Year/ getreturnperiod (freq))
    NumberOfWindows = len(returns.index) - Window
    if NumberOfWindows <= 0:
        print ('WindowStats: Not enough dates for year window' )
        return []
    else:
        Dates = []
        Stats = []
        CoVar = []
        Corr  = []
        for i in range(NumberOfWindows):
            start = returns.index[i].strftime('%Y-%m-%d')
            end = returns.index[i+Window].strftime('%Y-%m-%d')
            WindowData = extractdata (returns, start, end, freq)
            one_run = descriptive('', WindowData, freq)
            Dates = Dates + [one_run[0]]
            Stats = Stats + [one_run[1]]
            CoVar = CoVar + [one_run[2]]
            Corr = Corr + [one_run[3]]
        return  pd.date_range(Dates[0], periods=len(Dates), freq=ReturnFrequency), Stats, CoVar, Corr
#


# In[]:

# This Take some time to build this
%time WindowEndDates, Stats, CoVar, Corr = WindowStats(FX_Returns, ReturnFrequency, 2)


# In[]:
#  Rolling Statistical Data Per Currency
def GetStatSeries (ticker, dates, statdata):
    # Check that stat and stock are in the Stat Table
    if ticker not in statdata[1].index:
        print ('error')
        return
    output = pd.DataFrame(index=dates)
    for stat  in statdata[1].columns:
        values = []
        for i in range(len(statdata)):
            matrix = statdata[i]
            values.append(matrix.loc[ticker,stat])
        newcol = pd.DataFrame (values, index=dates, columns=[stat])
        output = pd.merge(output, newcol, left_index=True, right_index=True)
    output.index.name = ticker+': Roll Stats'
    return output
# 
#  Rolling Correlations Per Currency
def GetCorrSeries (ticker, dates, corrdata):
    # Check that stat and stock are in the Stat Table
    if ticker not in corrdata[1].index:
        print ('error')
        return
    output = pd.DataFrame(index=dates)
    for tick  in corrdata[1].index:
        values = []
        if tick != ticker:
            for i in range(len(dates)):
                matrix = corrdata[i]
                values.append(matrix.loc[ticker,tick])
            newcol = pd.DataFrame (values, index=dates, columns=[tick])
            output = pd.merge(output, newcol, left_index=True, right_index=True)
    output.index.name = ticker+' Rolling Corr'
    return output



# In[]:
#
# Generate Rolling Statistics and Rolling Correlations for All TICKERS
for Tick in FX_Returns.columns:# ###
    writer=pd.ExcelWriter('Rolling_'+Tick+".xlsx", engine='xlsxwriter')
    StatSeries = GetStatSeries (Tick, WindowEndDates, Stats)
    StatSeries.to_excel(writer, 'Rolling_Statistics')
    plotseries (Tick, StatSeries, "Rolling_Statistics")
    #
    CorrSeries = GetCorrSeries (Tick, WindowEndDates, Corr)
    CorrSeries.to_excel(writer, 'Rolling_Correlations')
    plotseries (Tick, CorrSeries, "Rolling_Correlations")
    plothistograms (CorrSeries, Tick+"_-_Rolling_Correlation")
    os.chdir(OutPutFolder)
    writer.save()
#

# In[]:


Charts = StatSeries.plot(figsize=(6,12), subplots=True, title=Tick+": Rolling Statistics")


# In[]:


Chart = StatSeries.hist(figsize=(10,10),bins=50)


#  Rolling Correlation
# Set ticker to the FX of your choice to get the  rolling correlations vs other currencies in the portfolio.


# In[]:


Tick = 'BRL'



# In[]:


Chart = CorrSeries.plot(figsize=(10,36), subplots=True, title=Tick+": Rolling Correlations")


#  There seems to be strong evidence of correlation regimes in these histograms!

# In[]:


Chart = CorrSeries.hist(figsize=(12,12),bins=50)


# Implementation of Covariance Matrix Shrinkage

#  Build Covariance Matrix of Weekly Returns

# In[]:


# Return the Begining/Ending dates for a lookback date range over a dataframe with a Time Index
def GetStartEndDates(P,Y):
    Index = P.index
    LookBack = int(Y * Days_in_Trading_Year) # Number of days to look back from most recent data in number of Years
    E = Index[-1]
    S = Index[-LookBack]
    return S, E

# Define a Historical Window to Get the Data For the Martix: Choose historical data for the past Return_Years, in weekly
# up to lastest observation
def GetDescriptive(Prices, Years, Freq ):
    Win_Start, Win_End =  GetStartEndDates(Prices, Years)
    Data = extractdata (Prices, Win_Start, Win_End, Freq)  # Get the price data in weekly dates
    # Get Log Returns
    Returns = getreturns (Data)
    return descriptive('', Returns, Freq)
#
#


# Work with the subdata of the number of years of weekly retruns for the FX, Factor and Benchmark data

# In[]:


Return_Years = 5.0
Win_Start, Win_End =    GetStartEndDates(Prices,Return_Years)
DiscountedPrices = discountprices (Prices, MoneyMarket)
# Get the WEEKLY FX prices from the total PRICE Series
FX_Prices = extractdata (DiscountedPrices, Win_Start, Win_End, ReturnFrequency)
# Get Weekly Returns
FX_Returns = getreturns (FX_Prices)
# Save the Covariance Matrix from the sample of  returns
Stats = GetDescriptive (DiscountedPrices, Return_Years, ReturnFrequency)
FX_MeanReturns = Stats[1]["Mean"].copy()
FX_StdDevReturns = Stats[1]["StdDev"].copy()
FX_VARReturns = Stats[1]["VAR"].copy()
SampleCov = Stats[2]
SampleCorr = Stats[3]
SampleCorr


#  Which Factors to Use?

# In[]:


plotheatmap (FactorStats[3], Start, End, 'Factors Return Correlation Heatmap')


# We have downloaded a set 8 potential factors to build the structured estimator for the Covariance Shrinkage. These are broadly broken down into 2 large categories: global market factors (MSCI AllCountry World Free stock index, DYX USD index, VIX and the CRB commodity index) and global interest rate factors (the USD 5 year forward inflation breakevens, the 1Y forward 1Y libor USD rate, the USD/EUR 1Y spread and the USD/JPY 1Y spread)

# From the Factor Correlation Heatmap above, we can see that interest rate factors are all highly correlated, which leads us to look to choose only 2 factors from this set. We whoose among those which seem to be the least correlated with the global market factors, US_5Y/5Y_CPI and US-JY_1Y_Sprd, both of which show a combination of low positive and some negative correlations with the global market factors. I'm inclined to use the USD 5Y CPI breakevens which is traditionally considered a leading factor for 1Y rates (despite that this needs to que quantitatively measured). Also, the USD is the funding rate for all the NDF local currency rates which are the large drivers of the carry returns. Finally, US_LBR_1Y/1Y and the US-EU_1Y_Sprd are also lowly correlated with the global market factors, with the caveat that there is a very strong correlation between the USD 5Y CPI  Breakevens  (already in our Factor sample)and the US-EU_1Y_Sprd, reason why we will also drop the latter (keeping the US_LBR_1Y/1Y forward rate).
#
# Among the global factors, we notive the high negative correlation between the MXWD (global stocks) and the VIX, which suggest that we should use one or the other. Given that the MSCI is only published with a lag after the market close, I prefer to use to VIX, which can be observed in real time, along with the FX/interest rates in the relevant EM currencies and the interest rate factors.
#
# The final list of factors will be: VIX, DXY, CRB, US_LBR_1Y/1Y forward rate and USD 5Y/5Y CPI Breakevens.

# In[]:


FactorList = ['VIX','DXY', 'CRB','US_5Y/5Y_CPI','US_LBR_1Y/1Y']
Estimator = Factors[FactorList].copy()
# Get Weekly Sample to Factors
Win_Start, Win_End =    GetStartEndDates(Estimator,Return_Years)
Estimators = extractdata (Estimator, Win_Start, Win_End, ReturnFrequency)
# Get Weekly Returns from Factors
Factor_Returns = getreturns (Estimators)
# Get Factor Statistics
Factor_Stats = descriptive('Factor_Returns', Factor_Returns, ReturnFrequency)


#  Build the Estimator Covariance Matrix of the FX vs Factors
# As explained by R. Diamond at Project Workshop Video: THIS IS NOT THE SAME AS THE METHOD DESCRIBED IN THE "Honey I Shrunk the Covariance Matrix" paper.
#

# In[]:


# Returns a DataFrame Matrix of zeros with Ticker Row/Columms
def ZeroMatrix (TickerList):
    NumTickers = len(TickerList)
    return pd.DataFrame (data=np.zeros(shape=(NumTickers, NumTickers),dtype='float64'),
                         index=TickerList,
                         columns=TickerList)
#


# In[]:


#
NumFactors = len(FactorList)
NumReturns = len(Factor_Returns.index)
TickerList = SampleCov.index.copy()
NumTickers = len(TickerList)
#
#
Structured_Estimator = []  # List of Matrices
# Fill the Matrices
for Fact in FactorList:
    Cov = pd.Series({symbol: Factor_Returns[Fact].cov(FX_Returns[symbol])
               for symbol in FX_Returns })
    New = ZeroMatrix(TickerList)
    New.index.name   = 'Covar vs'+Fact
    for Tick in TickerList:
        New[Tick][Tick] = Cov[Tick]
    Structured_Estimator.append(New)


#  Structured estimator for 3rd Factor: zeros matrix with the covariance between each currency and the Factor on the Diagonal

# In[]:


Structured_Estimator[2]  #And so on for all the Factors.... 0, 1, 2, 3...


#  Add Weightings to Each Matrix
# Set Total Delta at 70% and split evently among the Factor Matrices

# In[]:


Delta = 0.7
FactorDelta = Delta / len(FactorList)
#
# Build Shrunk Matrix
Shrunk_Sigma = (1 - Delta)* SampleCov
# Add the Factor Covariances
for i in range(len(Structured_Estimator)):
    Shrunk_Sigma +=  FactorDelta * Structured_Estimator[i]
#


#  This is the shrunk covariance matrix using 30% of the sample covariance matrix and adding the covariance with the factor list

# In[]:


Shrunk_Sigma


# Implementation of the Ledoit/Wolf 's "Honey, I Shrunk the Covariance Matrix"
# In the paper, the Structured Estimator (Shirnkage Target in the paper parlance) is a Covariance Matrix using the average Sample Correlation to compute the Structured Matrix. This matrix is then used to optimize the shrinkage intensity using a quadratic measure of the distance between the measured and the true covariance matrices based on the Frobenius norm. Using this norm its possible to find estimates for a shrinkage factor that minimize the a quadratic loss function.

#  SHRINKAGE TARGET (Appendix A)

# The Shrinkage Target is a Covariance Matrix where all the correlations have been replaced by the average correlation of the Sample Correlation Matrix.

# In[]:


# Get the average correlation from a Sample Covariance Matrix, exclude diagonal
def AvgRho (Cov):
    Average = 0
    NumTickers = len(Cov.index)
    #
    for i in range(NumTickers-1):
        for j in range(i+1,NumTickers):
            Average += Cov.iloc[i,j] /np.sqrt(Cov.iloc[i,i]*Cov.iloc[j,j])
    #
    return Average * (2 / ((NumTickers - 1) * NumTickers))
#
print('Average Correlation estimated from the Sample Covariance Matrix=', AvgRho(SampleCov))


# In[]:


# Build the Structured Covariance Matrix Estimator: returns the Matrix
def GetStructuredEstimator (SampleCov):
    TickerList = SampleCov.columns
    Structured_Estimator =  ZeroMatrix(TickerList)
    NumTickers = len(SampleCov.index)
    AvgR = AvgRho (SampleCov)
    for i in range(NumTickers):
        for j in range(NumTickers):
            Structured_Estimator.iloc[i,j] = AvgR * np.sqrt(SampleCov.iloc[i,i]*SampleCov.iloc[j,j])
    return Structured_Estimator
#
Structured_Estimator_LedoitWolf = GetStructuredEstimator (SampleCov)
Structured_Estimator_LedoitWolf


#  SHRINKAGE INTENSITY (Delta) (Appendix B)

# The Shinkage intensity (Delta in the original paper) is estimated as $max(0, min(1, k/T))$, where $T$ is the number of sample returns and a constant $k$, which is estimated as $(Pi - Rho)/Gamma$.
#
# $Pi$ is the estimate for the sum of asymptotic variances of the entries in the sample covariance matrix.
#
# $Rho$ is the estimate for the sum of asymptotic covariances of the entries in the shrinkage target matrix.
#
# $Gamma$ is the estimate for the measure of the misrepresentation of the population shrinkage target.
#
#

# In[]:

#  $Pi$ is the estimate for the sum of asymptotic variances of the entries in the sample covariance matrix.


# Returns the Matrix of Pi and the total sum of Pi values in the Matrix
def GetPi (Returns, MeanReturns, SampleCov):
    Pi_LedoitWolf = 0
    NumReturns = len(Returns)
    TickerList = Returns.columns
    Pi = ZeroMatrix(TickerList)
    for i in TickerList:
        for j in TickerList:
            for t in Returns.index:
                 Pi[i][j] += np.square((Returns[i][t] - MeanReturns[i]) *
                                    (Returns[j][t] - MeanReturns[j])
                                    - SampleCov[i][j])
    Pi = Pi * (1/NumReturns)
    #
    for i in TickerList:
        for j in TickerList:
            Pi_LedoitWolf += Pi[i][j]
    return Pi_LedoitWolf, Pi
#
#



# In[]:
#  $Rho$ is the estimate for the sum of asymptotic covariances of the entries in the shrinkage target matrix.
#  This is computationally very intensive.
# Each AsyCov matrix takes around 7 seconds to compute for every 2 years of weekly return data, times the number of FX Tickers (22)..... this is the time-consuming bit of this procedure.


# Get AsyCov Matrix for Ticker
def AsyCov (Ticker, Returns, MeanReturns, SampleCov):
    NumReturns = len(Returns)
    # Save the Ticker's VAR for use in nested for loops
    Var = SampleCov.loc[Ticker][Ticker]
    TickerList = Returns.columns
    ACov = ZeroMatrix(TickerList)
    for i in TickerList:
        for j in TickerList:
            for t in Returns.index:
                ACov[i][j] +=( (np.square(Returns[i][t] - MeanReturns[i]) - Var) *
                                ((Returns[i][t] - MeanReturns[i])*(Returns[j][t] - MeanReturns[j])
                                 - SampleCov[i][j])
                             )
    return ACov * (1/NumReturns)
#


# In[]:


# Build List of AsyCov Matrices, one for each Ticker, in the same order as the tickers are on the Returns
def AsyCovArray (Returns, MeanReturns, Cov):
    Array = []
    TickerList = Returns.columns
    for i in TickerList:
        Array.append( AsyCov( i, Returns, MeanReturns, Cov))
    return Array
#
#


# In[]:


# Get The Rho for a Set of returns, mean returns and Sample Covariance Matrix
def GetRho (Returns, MeanReturns, Cov):
    Rho_LedoitWolf = 0
    Average_Rho = AvgRho(Cov)
    TickerList = Returns.columns
    Pi_Value, Pi_Matrix = GetPi (Returns, MeanReturns, Cov)
    # Add up all the Pi's along the diagonal of the Pi Matrix
    for i in TickerList:
        Rho_LedoitWolf += Pi_Matrix [i][i]
    # Build Array with the AsyCov Matricies for each Ticker
    AsyCov_Array = AsyCovArray (Returns, MeanReturns, Cov)
    # Add up all the AsyCov Factors
    AsyCovSum = 0
    index = Cov.index
    for i in TickerList:
        Asy_ii = AsyCov_Array[index.get_loc(i)]
        for j in TickerList:
            if i != j:
                Asy_jj = AsyCov_Array[index.get_loc(j)]
                AsyCovSum += ( Asy_ii[i][j] * np.sqrt(Cov[j][j]/Cov[i][i]) +
                               Asy_jj[i][j] * np.sqrt(Cov[i][i]/Cov[j][j])
                             )
    #
    Rho_LedoitWolf = Rho_LedoitWolf + (Average_Rho/2)*AsyCovSum
    return Rho_LedoitWolf
#



# In[]:
#  $Gamma$ is the estimate for the measure of the misrepresentation of the population shrinkage target.
#
# Get the Gamma Factor
def GetGamma (Cov):
    Gamma_LedoitWolf = 0
    TickerList = Cov.columns
    S_E = GetStructuredEstimator (Cov)
    for i in TickerList:
        for j in TickerList:
            Gamma_LedoitWolf += np.square(S_E[i][j] - Cov[i][j])
    return Gamma_LedoitWolf
#


# In[]:


def GetDelta (Returns, MeanReturns, Cov):
    NumReturns = len(Returns)
    Pi, Pi_Matrix = GetPi (Returns, MeanReturns, Cov)
    Rho = GetRho (Returns, MeanReturns, Cov)
    Gamma = GetGamma(Cov)
    k = (Pi - Rho) / Gamma
    return max(0,min(1, k / NumReturns))
#


#  Get the Delta of the Sample Covariance Matrix, Around 5 mins on my machine....

# In[]:


print('Start:',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
Delta_LedoitWolf = GetDelta (FX_Returns, FX_MeanReturns, SampleCov)
print("Ledoit-Wolf Shrinkage Constant (Delta) =",Delta_LedoitWolf)
print('End:',datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



# In[]:
#   SHRUNK COVARIANCE MATRIX - Ledoit Wolf Paper
#

Shrunk_Sigma_LedoitWolf = (1-Delta_LedoitWolf)*SampleCov + Delta_LedoitWolf*Structured_Estimator_LedoitWolf
Shrunk_Sigma_LedoitWolf


# The Delta factor will depend on the estimate error in the calculation of the Sample Covariance Matrix, which depends on the length of the returns sample, the number of currencies/tickers. Depending on these factors, the Delta estimate of the shrinkage intensity may be 1 (100% use of the Shrinkage Target Matrix, i.e. the sample covariance matrix is too noisy) or 0 (100% use of the sample covariance matrix). As the size of the return sample grows, relative to the number of currencies, the Delta factor computed will tend to 0.  We will try to see who these factors affect the estimate of Delta...

#  Build a Matrix of Deltas (DO NOT RUN THIS!!!)
# Takes a very long time, skip to the output where we load the excel output to plot the chart.
#
#  By varying the number of Currencies and the number of years of weekly returns to use to calculate the covariance matrix.
#
#
#  Iterate gradually increasing the  lookback period of returns, reducing 1-by-1 the number of currencies. Calculate Delta for each combination
#  To test for the 22 currencies and annual increments of 20 years of returns, with annual increments  took 18 hrs...
# Below the code does a very short run: 5 currencies, 4 years of returns with annual increments. Runs fast (to see chart skip the
#

# In[]:


# Copy all the FX prices from start to end
Initial_Data = Prices.copy()
# Save the Complete Ticker List
TickerList = Initial_Data.columns
First_Date = Initial_Data.index[0]
Last_Date = Initial_Data.index[-1]
Max_Years = Last_Date.year - First_Date.year - 1
#
# Select How Many Currencies to take into account
Max_Curr = 5  # Limit the number of returns from the total: Range from 3-22 (min not 2, because of cubic interpolation)
Pop_Curr= 0 # Which Return Series to eliminate: 0 first, -1  last
#
# Select How long to look back into history to
Base_Year =  1  # Number of years for first run up to...
Max_Years = 4  # Maximum number of years of historical returns Range: 1.0 - 20
Year_Increment = 1 # increasing the lookback period by this number of years
#
Delta_Matrix = []
#
while len(Initial_Data.columns) > Max_Curr:
    Initial_Data.pop(Initial_Data.columns[0])
Sample = Initial_Data.copy()
timestamp = datetime.now()
print('Start:',timestamp.strftime('%Y-%m-%d %H:%M:%S'))
laststamp = timestamp


# In[]:


# Main Loop:  Reduce The Number of Currencies 1 by 1 until we are left with a 2xY Matrix
while len(Sample.columns) >= 2:
    Y = Base_Year
    while Y <= Max_Years:
        Win_S, Win_E = GetStartEndDates(Sample,Y)
        Window_Prices = extractdata (Sample, Win_S, Win_E, ReturnFrequency)
        Returns = getreturns (Window_Prices)
        Window_Stats = GetDescriptive (Sample, Y, ReturnFrequency)
        MeanReturns = Window_Stats[1]["Mean"].copy()  # Save column of mean returns per currency
        Cov = Window_Stats[2]                         # The Covariance Matrix
        Delt = GetDelta (Returns, MeanReturns, Cov)   # Get the Delta factor
        timestamp = datetime.now()
        gap = timestamp - laststamp
        #print(gap,len(Sample.columns), Y, Delt)
        Delta_Matrix.append([gap,len(Sample.columns), Y, Delt])
        Y += Year_Increment
        laststamp = timestamp
    if Pop_Curr == 0:
        Initial_Data.pop(Sample.columns[0])  # Drop one currency from the Returns Data
    else:
        Initial_Data.pop(Sample.columns[-1])  # Drop one currency from the Returns Data
    Sample = Initial_Data.copy()
#


# In[]:


# Save the parameters of the run to create excel and chart names.
DeltaRunParameters = "_S"+str(int(Base_Year))+"_E"+str(int(Max_Years))+"_d"+str(int(Year_Increment*100))+"_FX"+str(Max_Curr)
if Pop_Curr == 0:
    DeltaRunParameters = DeltaRunParameters+'_First'
else:
    DeltaRunParameters = DeltaRunParameters+'_Last'
#
Delta_Frame = pd.DataFrame(data=Delta_Matrix, columns=['Time','Num Tickers','Years','Delta'])
Chart_Frame = Delta_Frame
print('End:',timestamp.strftime('%Y-%m-%d %H:%M:%S'))
#
# LET'S SAVE THIS TO PLOT LATER... Just in case.
os.chdir(HomeFolder)
filename = 'Ledoit-Wolf_Delta'+DeltaRunParameters+'.xlsx'
writer=pd.ExcelWriter(filename, engine='xlsxwriter')
Delta_Frame.to_excel(writer, 'Delta')
writer.save()
Chart_Frame = Delta_Frame.copy()
#


#  Upload Data from LONG RUN: 22 currencies, 20 years with annual increments

# In[59]:


# UPLOAD EXCEL FILE WITH DELTA DATA TO PLOT
os.chdir(HomeFolder)
file = 'Ledoit-Wolf_Delta_S1_E20_d100_FX22_First.xlsx'  # Try any of the Delta Run Excel sheets saved
DeltaRunParameters = file.replace('Ledoit-Wolf_Delta','')
Chart_Frame = pd.read_excel(file, sheet_name = "Delta", index_col = 0, header=0, parse_dates = True)


#  OUTPUT:  Plot Delta Chart

# In[60]:


# This is the clumsyest chart I've ever made....
Time = ['Time']
Del = ['Delta']
Yr = ['Years']
Num = ['Num Tickers']
Deltas = Chart_Frame['Delta']
Years = Chart_Frame['Years']
Nums = Chart_Frame['Num Tickers']

#
# Set Up 3D Surface Plot of Delta vs # of Currencies and # of Return (in years of weekly returns)
x1 = np.linspace(Chart_Frame['Num Tickers'].min(),
                 Chart_Frame['Num Tickers'].max(),
                 len(Chart_Frame['Num Tickers'].unique()))
y1 = np.linspace(Chart_Frame['Years'].min(),
                 Chart_Frame['Years'].max(),

                 Chart_Frame['Years'].max()-Chart_Frame['Years'].min()+1)

x2, y2 = np.meshgrid(x1, y1)
z2 = griddata((Chart_Frame['Num Tickers'], Chart_Frame['Years']), Chart_Frame['Delta'], (x2, y2), method='cubic')
fig = plt.figure(figsize=(16,8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap="coolwarm" ,
    linewidth=0, antialiased=False)
ax.set_zlim(0.0, 1.1*Chart_Frame['Delta'].max())
ax.set_xlabel('Number of Currencies')
ax.set_ylabel('Years of Weekly Returns')
ax.set_zlabel('Delta')
ax.view_init(30,60)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Ledoit-Wolf Shrinkage Intensity Factor')
os.chdir(OutPutFolder)
FileName = 'Ledoit-Wolf_Delta'+DeltaRunParameters+'.png'
fig.savefig(FileName)


# From the plot above we can conclude that, as expected, the Ledoit-Wolf shrinkage factor decreases as we increase the lenght of the return sample period, indicating that the need to shrink the covariance matrix decreases as the size of the sample grows. Notice also that at around year 10 the Delta jumps as the covariance matrix starts to incorporate the volatility from the financial crisis.
#
# The Delta also increases as we increase the number of currencies in the sample covariance matrix. The increase is not smooth, as there are currencies which increase/decrease the size of the error estiamtor (Delta). An extreme case of this phenomenon is found in the middle of the chart, a (or a set of) currences gets added to the returns sample: the Delta factor spikes to close to 15% (for short returns samples),increasing the need to shrink the covariance matrix.
#
# Also, notice that the total ammount of "shrinkage" is very low. Let's assume that we use 5 years of historical returns, with the asset universe of 22 currencies, we would apply only apply around 5% shrinkage towards the structured matrix.  It seems that this procedure would be more useful when the universe of investable assets is very large (S&P or Russell), given that, for our universe of investable assets, the resulting covariance matrix will be strongly tilted towards the sample matrix, with all the problems this entails.
#
# At this point we should consider alternative methods to deal with the covariance matrix for our project.

#  A quick aside on computational efficiency
#

#  This problem is linear in time and cuadratic over the number of currencies.
# This is a challenge, as the estimation of the Rho constant to calculate the shrinkage factor involves significant computing power. The calculation of the Delta plot above (on a Mac with 2 CPUs/4 cores each, 4GHz) took 17:45 hours of total runtime, despite that the process never consumed more than 15% of total computing power. This suggests that there is a need of specialized programing to breakdown this problem into tasks that can run in parallel if we are ever going to perform these types of computations on a regular basis.

# In[61]:


x2, y2 = np.meshgrid(x1, y1)
z2 = griddata((Chart_Frame['Num Tickers'], Chart_Frame['Years']), Chart_Frame['Time'], (x2, y2), method='cubic')
fig = plt.figure(figsize=(16,8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap="coolwarm" ,
    linewidth=0, antialiased=False)
ax.set_zlim(0.0, 1.1*Chart_Frame['Time'].max())
ax.set_xlabel('Number of Currencies')
ax.set_ylabel('Years of Weekly Returns')
ax.set_zlabel('Execution Time')
ax.view_init(30,265)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.04f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Time to Compute Each DELTA (Days)')
FileName = 'Ledoit-Wolf_Time'+DeltaRunParameters+'.png'
os.chdir(OutPutFolder)
fig.savefig(FileName)


#  Check to see how different the two matricies are
# Sample covariance matrix vs the Shrunk Matrix (per Diamond lecture) vs the Ledoit/Wold: expresses as a ratio of the resulting covariances.

# In[62]:


Shrunk_Sigma / SampleCov


# The matrix shrunk with the Factors is very similar to Structured Estimator: remember this estimator is 30% weighted towards the original sample covariance matrix, with the diagonal terms modified by sum of the covariances between each currency against each factor.  The deviation on the diagonal terms from the 30% level represents the ammoung of "shrinkage" that has been applied.  However, this method has a critical flaw: notice  that, on the diagonal, we now find currency variance terms which are negative, this makes this Matrix useles.

# In[63]:


Shrunk_Sigma_LedoitWolf / SampleCov


# Here, the ratio of the matrices shows that the Ledoit-Wolf matix is very similar to the original sample covariance matrix, with most of the terms above close to 100% (1.00), however, we can detect some trully outlyers, such as the ARS/COP currency pair, which has a covariance term which is 2.8x the oritinal covariance. Also, notice that there are no changes in sign, as the signs of the structured estimators are all positive, given that its estimated using the average correlation, which is positive in this case.

#    # Classic Portfolio Theory
#    Lets apply a basic Markowitz mean variance optimization to the Sample Covariance Matrix and the Ledoit-Wolf shrunk covariance matrix... are there any differences?
#

# In[64]:


# Returns the Annualized Portfolio Returns according to Weights
def PortfolioReturn (W ,Ret):
    return np.dot(Ret*Ret_Factor, W)
def NegPortfolioReturn (W ,Ret): # For Return Maximization with a minimization optimization
    return  -1*PortfolioReturn (W ,Ret)

# Returns the Annualized Portfolio Variance according to Weights and the Covariance Matrix
def PortfolioVar (W,CoVar):
    return np.dot(W, np.dot(CoVar*Var_Factor, W))

# Returns the Annualized Portfolio Satandard Deviation according to Weights and the Covariance Matrix
def PortfolioStdDev (W,CoVar):
    return  PortfolioVar (W,CoVar) ** 0.5

# Returns the Negative Value of the Function and Arguments Passed
def NegFun (Fun, *args):
    return -1*Fun(*args)


# In[65]:


# Returns Minimum Variance Portfolio Weights given the Covariance Matrix, capacity to Short, Position Limist and Leverage
def PortOptMinRisk (W, CoVar, Shorting, MaxPos, Budget):
    #Set Number of Tickers in the problem
    NumTickers = len(CoVar.columns)
    # Set the Investment Boundaries for the Optimization
    if Shorting:
        PosRange = (-MaxPos, MaxPos)
    else:
        PosRange = (0, MaxPos)
    bnds = tuple( PosRange for FX in range(NumTickers))
    # Set the Budget Function for the Optimization: Budget Parameter stakes how much leverage 1, 1.5, 2....
    cons = ({'type': 'eq', 'fun': lambda W: sum(W) - Budget })
    return sco.minimize(PortfolioStdDev, W, (CoVar), constraints=cons, bounds=bnds)

# Returns Maximum Return Portfolio Weights given the Returns, capacity to Short, Position Limist and Leverage
def PortOptMaxRet (W, Ret, Shorting, MaxPos, Budget):
    #Set Number of Tickers in the problem
    NumTickers = len(Ret)
    # Set the Investment Boundaries for the Optimization
    if Shorting:
        PosRange = (-MaxPos, MaxPos)
    else:
        PosRange = (0, MaxPos)
    bnds = tuple( PosRange for FX in range(NumTickers))
    # Set the Budget Function for the Optimization: Budget Parameter stakes how much leverage 1, 1.5, 2....
    cons = ({'type': 'eq', 'fun': lambda W: sum(W) - Budget })
    return sco.minimize(NegPortfolioReturn, W, (Ret), constraints=cons, bounds=bnds)

# Returns Maximum Return Portfolio Weights given a target level of RISK
def PortOptMaxRiskTgt (W, Ret, CoVar, Shorting, MaxPos, Budget, RiskTgt):
    #Set Number of Tickers in the problem
    NumTickers = len(Ret)
    # Set the Investment Boundaries for the Optimization
    if Shorting:
        PosRange = (-MaxPos, MaxPos)
    else:
        PosRange = (0, MaxPos)
    bnds = tuple( PosRange for FX in range(NumTickers))
    # Set the Budget Function for the Optimization: Budget Parameter stakes how much leverage 1, 1.5, 2....
    cons = ({'type': 'eq', 'fun': lambda W: sum(W) - Budget },
            {'type': 'eq', 'fun': lambda W: PortfolioStdDev(W,CoVar) - RiskTgt })
    return sco.minimize(NegPortfolioReturn, W, (Ret), constraints=cons, bounds=bnds)


# In[66]:


def PiePlot (title,labels,data):
    fig = plt.figure(figsize=Pie_Dims)
    plt.pie(data, labels=labels, autopct='%1.1f%%', shadow=False, startangle=140,pctdistance=1.1, labeldistance=1.3) #, explode=explode, colors=colors)
    plt.axis('equal')
    plt.title(title, fontsize = 15)
    FileName = 'Pie_'+title+'.png'
    os.chdir(OutPutFolder)
    fig.savefig(FileName)
    plt.show()
    return

def BarPlot (title,labels,data):
    fig = plt.figure(figsize=Pie_Dims)
    y_pos = np.arange(len(labels))
    # figsize= Bar_Dims
    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos, labels, rotation='vertical')
    plt.ylabel('Portfolio Allocation')
    plt.title(title, fontsize = 15)
    FileName = 'Bar_'+title+'.png'
    os.chdir(OutPutFolder)
    fig.savefig(FileName)
    plt.show()
    return



#
#
#
#  Let's take a look at the BenchMark Weights

# In[67]:


TickerList = BenchMarkWeights.columns
labels = tuple(TickerList)
Weights =  BenchMarkWeights[TickerList][-2:-1].values
Weights = list(Weights[0])
BarPlot ("BenchMark Portfolio Weightings", labels, Weights)
print("Return =", PortfolioReturn (Weights,FX_MeanReturns))
print("StdDev =", PortfolioStdDev (Weights,SampleCov))


#  We have negative returns, remember we used 5 years of historical data to get the mean returns.
# Lets take the longest timeprod (Historical) to sample returns...

# In[]:


Short_MeanReturns = Historical_Stats["Mean"].copy()


# In[]:


TickerList = BenchMarkWeights.columns
labels = tuple(TickerList)
Weights =  BenchMarkWeights[TickerList][-2:-1].values
Weights = list(Weights[0])
BarPlot ("BenchMark Portfolio Weightings", labels, Weights)
print("Return =", PortfolioReturn (Weights,Short_MeanReturns))
print("StdDev =", PortfolioStdDev (Weights,SampleCov))


#  Ok, our starting point is a portfolio with a positive average return... important for the optimization.

#
#  Lets see the Naive Minimum Risk Portfolio:
# With NO Shorting, no Max % per currency, 1x leverage

# In[]:


OptResult =PortOptMinRisk (Weights,  SampleCov, False, 1 ,1 )
OptWeights = OptResult['x']
BarPlot ("Minimum Risk Portfolio Weightings", labels, OptWeights)
MinRiskRet = PortfolioReturn (OptWeights,Short_MeanReturns)
MinRiskRsk = PortfolioStdDev (OptWeights,SampleCov)
print("Return =", MinRiskRet)
print("StdDev =", MinRiskRsk)


#  What Happens When we use the Ledoit-Wolf Shrunk Matrix

# In[]:


ShrunkResult =PortOptMinRisk (Weights,  Shrunk_Sigma_LedoitWolf, False, 1 ,1 )
ShrunkWeights = ShrunkResult['x']
BarPlot ("Min Risk Weightings (Ledoit-Wolf Shrinkage Covariance Matrix)", labels, ShrunkWeights)
ShrMinRiskRet = PortfolioReturn (ShrunkWeights,Short_MeanReturns)
ShrMinRiskRsk = PortfolioStdDev (ShrunkWeights,Shrunk_Sigma_LedoitWolf)
print("Return =", ShrMinRiskRet)
print("StdDev =", ShrMinRiskRsk)
NaiveMinRisk = ShrunkWeights


# Not much of a difference? Let's check...
#

# In[]:


BarPlot("Change in Min Risk Weightings (Shrunk - Sample Cov)", labels,  (ShrunkWeights - OptWeights))


# There are some small movements (0-30bps) around in allocation... Let's see what's the difference in Portfolio Return and Risk

# In[]:


print('Shrunk Return =', ShrMinRiskRet, ' Sample Return', MinRiskRet,'Difference =',ShrMinRiskRet-MinRiskRet)
print('Shrunk StdDev =', ShrMinRiskRsk, ' Sample StdDev', MinRiskRsk,'Difference =',ShrMinRiskRsk-MinRiskRsk)


# Well, the gains are insignificant in terms of return and risk improvement.  This is because the shrinkage intensity factor is very low (5%), which means that the shrunk matrix and the sample covariance matrix are too similar to make a difference.

#
#  Lets see the Naive Maximum Return Portfolio:
# With NO Shorting, no Max % per currency, 1x leverage

# In[]:


OptResult =PortOptMaxRet (ShrunkWeights,  Short_MeanReturns, False, 1 ,1 )
OptWeights = OptResult['x']
BarPlot ("Maximum Return Portfolio Weightings", labels, OptWeights)
ShrMaxRetRet = PortfolioReturn (OptWeights,Short_MeanReturns)
ShrMaxRetRsk = PortfolioStdDev (OptWeights,Shrunk_Sigma_LedoitWolf)
print("Return =", ShrMaxRetRet)
print("StdDev =", ShrMaxRetRsk)


# This is the classical "corner" solution... Go all in to Argentina.
#
#  Let's relax some constraints: still no leverage, shorting with 25% position limits.

# In[]:


PosLimit = 0.25


# In[]:


OptResult =PortOptMaxRet (ShrunkWeights,  Short_MeanReturns, True, PosLimit ,1)
OptWeights = OptResult['x']
BarPlot ("Maximum Return Portfolio Weightings", labels, OptWeights)
ShrMaxRetRet = PortfolioReturn (OptWeights,Short_MeanReturns)
ShrMaxRetRsk = PortfolioStdDev (OptWeights,Shrunk_Sigma_LedoitWolf)
print("Return =", ShrMaxRetRet)
print("StdDev =", ShrMaxRetRsk)


# Ok, we got the same returns.... but with much lower risk, apparently...
# Yes, its only achieved by potentially assuming massive leverage in the long/short positions.

# In[]:


# Total_Investment = np.abs(OptWeights)
Leverage = np.sum(np.abs(OptWeights))
print("Total Leverage = ", Leverage)


# So we are leveraged here around 4.2 : 1!!
#
#


# In[]:
#
#  Let's test the optimizer for formal leverage, with varying degrees of position limits and with short selling


def Plot3DSurf (x,y,z,xLab,yLab,zLab):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    Z = z.reshape(X.shape)
    surf = ax.plot_surface(X, Y, Z, cmap='RdYlGn')
    ax.set_xlabel(xLab)
    ax.set_ylabel(yLab)
    ax.set_zlabel(zLab)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()
    return


# In[]:


spacesize = 25
pos = np.linspace(0.01,1,spacesize)
lev = np.linspace(1,2,spacesize)


# In[]:


def TestPortfolio (Weights, Returns, Sigma, Short):
    pos = np.linspace(0.01,1,spacesize)
    lev = np.linspace(1,2,spacesize)
    ii = len(pos)
    jj = len(lev)
    Ret = np.zeros((ii,jj))
    Rsk = np.zeros((ii,jj))
    Sharpe = np.zeros((ii,jj))
    Leverage = np.zeros((ii,jj))
    for i in range(len(pos)):
        for j in range(len(lev)):
            Result = PortOptMaxRet (Weights,  Returns, Short, pos[i] ,lev[j] )
            if Result['success']:
                W = Result['x']
                if np.count_nonzero(W)>1:
                    Ret[i,j] = PortfolioReturn (W, Returns)
                    Rsk[i,j] = PortfolioStdDev (W ,Sigma)
                    Leverage[i,j] = np.sum(np.abs(W))
                else: # Corner Solution
                    Ret[i,j] = -0.0001
                    Rsk[i,j] = 1
            else: # No Solution
                    Ret[i,j] = -0.0002
                    Rsk[i,j] = 1
    Sharpe = Ret/Rsk
    return Ret, Rsk, Sharpe, Leverage
#


# In[]:


Short = False
%time Test1 = TestPortfolio (ShrunkWeights, Short_MeanReturns, Shrunk_Sigma_LedoitWolf, Short)


# In[]:


Plot3DSurf(pos,lev,Test1[0],"Position Limit",'Leverage', 'Return' )
Plot3DSurf(pos,lev,Test1[1],"Position Limit",'Leverage', 'StdDev')
Plot3DSurf(pos,lev,Test1[2],"Position Limit",'Leverage', 'Sharpe Ratio')
Plot3DSurf(pos,lev,Test1[3],"Position Limit",'Leverage', 'Total Leverage')


#  This looks like an extremely hard market to make money in a  passive strategy (no Shorting w/Leverage):
# 1) Can't make money unless the strategy assumes leverage.
#
# 2) Risk piles up faster than leverrage. As shown by the downward sloping Sharpe Ratios as leverage increases and as position limits decrease.
#
# 3) There seems to be some merit (in terms of Sharpe Ratios) to limit leverage at low levels.
#
# 4) Corner Solutions(1 asset) or optimizer failure, both denoted by 100% Risk (corner solution) or 200% (no solution) are prevalent in the cases of no leverage, particularly when large position limits are allowed.
#

# In[]:


Short = True
%time Test2 = TestPortfolio (ShrunkWeights, Short_MeanReturns, Shrunk_Sigma_LedoitWolf, Short)

# In[]:


Plot3DSurf(pos,lev,Test2[0],"Position Limit",'Leverage', 'Return')
Plot3DSurf(pos,lev,Test2[1],"Position Limit",'Leverage', 'StdDev')
Plot3DSurf(pos,lev,Test2[2],"Position Limit",'Leverage', 'Sharpe Ratio')
Plot3DSurf(pos,lev,Test2[3],"Position Limit",'Leverage', 'Total Leverage')


#  Leverage seems to help... go bankrupt faster.
# The prevalence of corner solutions diminishes only marginally and are,again, present when we have no leverage.
#
# Sharpe ratios are generaly larger than in the no leverage test (no surprises), but we are not measuring the hidden leverage in the long/shorts: this can be seen in the total leverage plot, which explodes to 20:1 (as opposed to 2:1 in the prior case).
#
# This can make the portfolio blow up pretty quickly as soon as real world co-variances deviate from the historical (shrunk) values fed to the optimization, as long/short positions add more implicit leverage than the explicit budget constraint.
#
# Maybe we should add a new constraint to the optimizer regarding the total allowed long/short leverage.

# In[]:

# Let's Get the Efficient Frontier
# Method 1: brute force Monte Carlo Simulation of Random Portfolios.
# SET NUM_SIMULATIONS !!  It takes 7mins to generate 150k simulations on my machine
np.random.seed(655321)
Total_Simulations = 150000
# Generate Random Portfolios
NumTickers = len(Shrunk_Sigma_LedoitWolf.columns)
RP = np.random.random((Total_Simulations,NumTickers)) # Get random Variables Matrix
# ADD CONCENTRATED PORTFOLIOS: for 25% of portfolios 6 or more positions (rest set to zero)
Skips = 2  # 1/Skips will have a highly concentrated portfolio
MinZeros = NumTickers- 6  # Minimum number of zero entries
print('Start:',(datetime.now()).strftime('%Y-%m-%d %H:%M:%S'))
for i in range(2,int(Total_Simulations/Skips)):
    numzeros = int(np.random.uniform(MinZeros,NumTickers))
    for j in range (numzeros):
        RP[:i,int(np.random.uniform(0,NumTickers-1))] = 0
# Add The Minimum RIsk Portfolio from the Optimizer for this strategy... just to add the dot to the plot.
RP[:1,:] = NaiveMinRisk
# Divide each row by the sum of the row to get weights adding to 100%, no negative weights
RP = (RP.T / RP.sum(axis=1)).T
print('End:',(datetime.now()).strftime('%Y-%m-%d %H:%M:%S'))


# In[]:


# These are the stats for a Regular Portfolio (no leverage, no shorting, no Position limits)
%time PortStats = [(PortfolioStdDev (w ,Shrunk_Sigma_LedoitWolf), PortfolioReturn (w, Short_MeanReturns))for w in RP]
# Turn tuple into an np.array
PortStats = np.array(PortStats)


# In[]:


# Get the minimum Risk portfolio for the Regular Portfolio: simple weights from 0,1, etc...
Ret = PortfolioReturn (NaiveMinRisk,Short_MeanReturns)
Rsk = PortfolioStdDev (NaiveMinRisk,Shrunk_Sigma_LedoitWolf)


# In[]:


plt.figure(figsize=(16,10))
plt.plot(PortStats[:,0], PortStats[:,1], 'ro')
plt.ylabel('Portfolio Return')
plt.xlabel('Portfolio StdDev')
plt.title('Portfolio Return vs StdDev')
plt.axvline(Rsk, c='b', ls='--')
plt.axhline(Ret, c='b', ls='--')
plt.show()


#  Despite running a significant number of simulations and adding a high number to concentrated portfolios, the Minimum Risk portfolio still looks like an outlier
# The Mersenne Twister implemented in Python to generate random numbers may be not so good after all.  By adding the Naive Minimum Risk portfolio to the Random Sample, it shows that it's a posible portfolio, but it's a highly concentrated one (it has  6 non-zero weight allocations). For my peace of mind, lets just count how many portfolios in the random sample have 14 or less non-zero-weight positions.

# In[]:


print(np.count_nonzero(NaiveMinRisk)==14)
NumZero = [np.count_nonzero(w)<=14 for w in RP]
print(np.sum(NumZero))


#  So we need to make a dramatically larger Simulation to get the outliers in the efficient frontier
# This would be hihgly inneficient. So, brute force to get a glimpse at the efficient frontier is not an option when we have mulitple assets in the mix.

#  Method 2: Use Optimizer to get the maximum return portfolio subject to a level of risk

# In[]:


Max_Risk = np.max(PortStats[:,0])
Ports_on_Frontier = 150
RiskLevel = np.linspace(Rsk,Max_Risk,Ports_on_Frontier)
Eff = []
Ports = []


# In[]:


for RskskTgt in  RiskLevel:
    Opti = PortOptMaxRiskTgt (NaiveMinRisk, Short_MeanReturns, Shrunk_Sigma_LedoitWolf, False, 1 , 1, RskskTgt)
    OptWeights = Opti['x']
    Eff.append((PortfolioStdDev (OptWeights,Shrunk_Sigma_LedoitWolf),PortfolioReturn (OptWeights,Short_MeanReturns)))
    Ports.append(OptWeights)
Frontier = np.array(Eff)
FrontPorts = np.array(Ports)
Frontier[:3]


# In[]:


plt.figure(figsize=(16,10))
plt.plot(PortStats[:,0], PortStats[:,1], 'ro')
plt.plot(Frontier[:,0], Frontier[:,1], 'r', label='No Short No Lev No PosLims')
plt.ylabel('Portfolio Return')
plt.xlabel('Portfolio StdDev')
plt.title('Portfolio Efficient Frontier')
plt.axvline(Rsk, c='b', ls='--')
plt.axhline(Ret, c='b', ls='--')
plt.legend(loc='upper left')
plt.show()


# In[]:


NumZero = [np.count_nonzero(w)<=14 for w in FrontPorts]
print('Number of Portfolios on the Frontier with 6 or more Zero-Weight Positions =',np.sum(NumZero))
print('As a % of the total Portfolios = ', 100*np.sum(NumZero)/Ports_on_Frontier )


#  So we are back to square one: there are a lot of highly concentrated portfolios on the efficient frontier
# and generating random concentrated portfolios doesn't help. Also, it seems I can't escape the conclusion that it only makes sense to invest in  hihgly conmcentrated portfolios if there is any hope to improve the performance of the portfolio with clasical portfolio theory..

#  Let's get the efficient frontier for the portfolio with Leverage, position limits and shorting

# In[]:


PosLimit = 0.25 # Set a Position Limit as before
LevResult =PortOptMinRisk (NaiveMinRisk,  Shrunk_Sigma_LedoitWolf, True,  PosLimit, 1.4)
MinRiskLev = LevResult['x']
BarPlot ("Min Risk Weightings (Shorting, Leverage, Position Limits)", labels, MinRiskLev)
MinRiskLevRet = PortfolioReturn (MinRiskLev,Short_MeanReturns)
MinRiskLevRsk = PortfolioStdDev (MinRiskLev,Shrunk_Sigma_LedoitWolf)


# In[]:


RiskLevelLev = np.linspace(MinRiskLevRsk,Max_Risk,Ports_on_Frontier)
#
EffLev = []
PortsLev = []
#
for RskskTgt in  RiskLevelLev:
    Opti = PortOptMaxRiskTgt (Weights, Short_MeanReturns, Shrunk_Sigma_LedoitWolf, True,  PosLimit, 1.4, RskskTgt)
    OptWeights = Opti['x']
    EffLev.append((PortfolioStdDev (OptWeights,Shrunk_Sigma_LedoitWolf),PortfolioReturn (OptWeights,Short_MeanReturns)))
    PortsLev.append(OptWeights)
FrontierLev = np.array(EffLev)
FrontPortsLev = np.array(PortsLev)


# In[]:


plt.figure(figsize=(16,10))
plt.plot(PortStats[:,0], PortStats[:,1], 'ro')
plt.plot(Frontier[:,0], Frontier[:,1], 'r', label='No Short No Lev No PosLims')
plt.plot(FrontierLev[:,0], FrontierLev[:,1], 'b', label='Shorts Lev 1.4x  PosLims 25%')
plt.ylabel('Portfolio Return')
plt.xlabel('Portfolio StdDev')
plt.title('Portfolio Return vs StdDev')
plt.axvline(Rsk, c='r', ls='--')
plt.axhline(Ret, c='r', ls='--')
plt.axvline(MinRiskLevRsk, c='b', ls='--')
plt.axhline(MinRiskLevRet, c='b', ls='--')
plt.legend(loc='upper left')
plt.show()


#  Again, let's test for the level of concentration of portfolios on the Leveraged Efficient Frontier
#
#

# In[]:


NumZero = [np.count_nonzero(w)<=1 for w in FrontPortsLev]
print('Number of Portfolios on the Frontier with 11 or less Positions',np.sum(NumZero))
print('As a % of the total Portfolios = ', 100*np.sum(NumZero)/Ports_on_Frontier )


#  Here we can see no portfolios on the efficient frontier have one zero-weighted entry once we relax assumptions on leverage and shorting.
# Let's get the stats for the long/short leverage on the efficient frontier.

# In[]:


Average_LS_Leverage = [np.sum(np.abs(w)) for w in FrontPortsLev]
Average_LS_Leverage =np.array(Average_LS_Leverage )
print("Average Long/Short Leverage on Efficient Frontier = ", np.average(Average_LS_Leverage))
print("MAX Long/Short Leverage on Efficient Frontier = ", np.max(Average_LS_Leverage))
print("MIN Long/Short Leverage on Efficient Frontier = ", np.min(Average_LS_Leverage))


#  Let's get the efficient frontier for the portfolio with Modest Leverage, Low Position limits and Shorting

# In[]:


PosLimit = 0.10 # Set a Position Limit as before
LowLevResult =PortOptMinRisk (NaiveMinRisk,  Shrunk_Sigma_LedoitWolf, True,  PosLimit, 1.15)
MinRiskLowLev = LowLevResult['x']
BarPlot ("Min Risk Weightings (Shorting, Low Leverage, Small Position Limits)", labels, MinRiskLowLev)
MinRiskLowLevRet = PortfolioReturn (MinRiskLowLev,Short_MeanReturns)
MinRiskLowLevRsk = PortfolioStdDev (MinRiskLowLev,Shrunk_Sigma_LedoitWolf)


# In[]:


RiskLevelLev = np.linspace(MinRiskLevRsk,Max_Risk,Ports_on_Frontier)
#
EffLowLev = []
PortsLowLev = []
#
for RskskTgt in  RiskLevelLev:
    Opti = PortOptMaxRiskTgt (Weights, Short_MeanReturns, Shrunk_Sigma_LedoitWolf, True,  PosLimit, 1.15, RskskTgt)
    OptWeights = Opti['x']
    EffLowLev.append((PortfolioStdDev (OptWeights,Shrunk_Sigma_LedoitWolf),PortfolioReturn (OptWeights,Short_MeanReturns)))
    PortsLowLev.append(OptWeights)
FrontierLowLev = np.array(EffLowLev)
FrontPortsLowLev = np.array(PortsLowLev)


# In[]:


plt.figure(figsize=(16,10))
plt.plot(PortStats[:,0], PortStats[:,1], 'ro')
plt.plot(Frontier[:,0], Frontier[:,1], 'r', label='No Short No Lev No PosLims')
plt.plot(FrontierLowLev[:,0], FrontierLowLev[:,1], 'g', label='Shorts Lev 1.2x  PosLims 15%')
plt.plot(FrontierLev[:,0], FrontierLev[:,1], 'b', label='Shorts Lev 1.4x  PosLims 25%')
#
plt.ylabel('Portfolio Return')
plt.xlabel('Portfolio StdDev')
plt.title('Portfolio Return vs StdDev')
plt.axvline(Rsk, c='r', ls='--')
plt.axhline(Ret, c='r', ls='--')
plt.axvline(MinRiskLevRsk, c='b', ls='--')
plt.axhline(MinRiskLevRet, c='b', ls='--')
plt.axvline(MinRiskLowLevRsk, c='g', ls='--')
plt.axhline(MinRiskLowLevRet, c='g', ls='--')
plt.legend(loc='upper left')
plt.show()


#  Some Observations:
#
# 1) The Naive Portfolio (RED), produces corner solutions over the efficient frontier, which are at great risk of mispecification of expected returns given the low level of diversification.
#
# 2) The Modestly Levererd Portfolio (GREEN) underperforms by adding risk and redicing returns
#
# 3) The Highly Levered Portfolio (BLUE) manages to outperform the Naive Portfolio, but at the price of assuming an inordinate ammount of leverage which leaves the portfolio exposed to divergence between the realized correlations/covariances and the ones implicit in the shrunk matrix used in the optimization.
#
# The conclusion seems to be that, to effectively manage a portfolio in these markets we need to apply leverage consciously. What does this mean? Well, leverge seems to be a must, the covariance matrix needs a lot more working (as we have seen) and the return/volatility outlook needs to be enhanced by applying expert views on a Black-Litterman approach, particularly incorporating confidence intervals adjusted for "expert bias."

#
