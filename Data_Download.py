#
#
# Import Required Libraries
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as sco
import time
from datetime import datetime
from dateutil.parser import parse
# Show Plots in Python console screen 
#%matplotlib inline
# Show Plots on separate windows
# %matplotlib qt5

# Clean The Time Series of Bad Data and Weekend Dates
def CleanDataFrame (frame):   # REMOVES BAD DATA AND WEEKENDS FROM TIME SERIES
    frame.dropna(inplace = True) # Drop Grabage
    return frame[frame.index.dayofweek < 5] # get only weekdays

# Upload Excel Data to Panda DataFrames, Clean the Data
def ProcessExcelFile (File, Tab, Ind, Col):
    Raw = pd.read_excel(File, sheet_name = Tab, index_col = Ind, header = Col, parse_dates = True)
    return CleanDataFrame (Raw)


# In[]
################################################################
#  UPLOAD DATA FROM EXCEL
#  SET THE PATH TO THE EXCEL INPUT FILES!!!
HomeFolder = "D:/Boris/LEM_Strategy/Software/Data_Download"
OutPutFolder = HomeFolder+"/DataOutPut"
InPutFolder = HomeFolder+"/DataInPut"
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
    
os.chdir(InPutFolder)
%time ALL_Factor = ProcessExcelFile(Factor_Sheet, FactorTab,  0,  3)
%time ALL_FX = ProcessExcelFile(FX_Sheet,  FX_Tab,  0,  3) 
%time ALL_LocCurr = ProcessExcelFile(LocCurr_Sheet, LocCurrTab, 0, 3)
# %time ALL_BenchMark = ProcessExcelFile(Benchmark_Sheet, BenchmarkTab,  0, 3)


# In[]
################################################################# 

# USE THIS TO WORK WITH ALL THE DATA OR A SUBSET
# Using SMALL dataset vastly Improves Execution Time when testing the SW
#Dataset = 'Large'
#Dataset = 'Small'

#
################################################################
# Global Variables Never Change
#
# SET THE DATE RANGES OF THE TOTAL DATASET
Start = '01-01-2008'
End = '12-31-2018'
#
Days_in_Trading_Year = 252
#
ReturnPeriod = 5  # Use weekly Returns assuming data without weekends, ie a 5 day week
ReturnFrequency = 'W-WED'  #   D, W-WED... 
ReturnFrequencyText = '-'

# Calculate Log Returns?
Log_Return = False
#
################################################################
# Generate Lists of Frequency Types in Python
Months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
Days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
Daily = ['D', 'B', ]
Monthly = ['M', 'BM', 'MS', 'BMS']
Weekly =  []; Quarterly = []; Annualy = []
W = 'W-'
for d in Days:
    Weekly.append(W+d)
Q0 = 'Q-' ; Q1 = 'QS-'; Q2 = 'BQ-'; Q3 = 'BQS-'
for m in Months:
    Quarterly.append(Q0+m)
    Quarterly.append(Q1+m)
    Quarterly.append(Q2+m)
    Quarterly.append(Q3+m)
A0 = 'A-'; A1 = 'AS-'; A2 = 'BA-'; A3 = 'BAS-'
for m in Months:
    Annualy.append(A0+m)
    Annualy.append(A1+m)
    Annualy.append(A2+m)
    Annualy.append(A3+m)

#
################################################################
# Set Chart Parameters
Pie_Dims = (10, 10)
Bar_Dims = (10, 16)
Hist_Dims = (10,15)
LogRet_Dims = (18,14)
Series_Dims = (18,14)
HeatMap_Dims = (18, 10)
Show_Plot = False

# Show Plots on Screen 
def showplot():
    if Show_Plot:
        plt.show()
    return



################################################################
# FUNCTIONS TO PROCESS INPUT DATA

# Get the Return Period (in days) for the potential frequency of returns in Python
def getreturnperiod (f):
    #
    if f in Daily:
        FrequencyText = 'Daily'
        n = 1
    elif f in Weekly:
        FrequencyText = 'Weekly'
        n = round (Days_in_Trading_Year / 52, 0)
    elif f in Monthly:
        FrequencyText = 'Monthly'
        n= round (Days_in_Trading_Year / 12, 0)
    elif f in Quarterly:
        FrequencyText = 'Quarterly'
        n= round (Days_in_Trading_Year / 4, 0)
    elif f in Annualy:
        FrequencyText = 'Annual'
        n= Days_in_Trading_Year
    else:
        print ("ERROR IN FREQUENCY: GETRETURNPERIOD ")
        n= 0
    return n , FrequencyText
# FACTORS TO ANNUALIZE Log Return/Var
n , ReturnFrequencyText = getreturnperiod (ReturnFrequency)
Ret_Factor =  Days_in_Trading_Year / n
Std_Factor = np.sqrt ( Ret_Factor )
Var_Factor = Ret_Factor #
#

# Invert FX Quotes 
def invertFX (p):
    return (1/p)

#  Normalize timeseries to re-base start at 1 (great for plottong Log return)
def normalize (P):
    CleanDataFrame (P)
    return CleanDataFrame (P / P.iloc[0])

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
    x , f = getreturnperiod (frequency) # To set the frequency text for plots....
    Dates = pd.date_range(start,end, freq = frequency)
    prices = p.copy()
    prices = prices.reindex(Dates)
    return CleanDataFrame (prices)   # Return a clean dataframe

def getreturns (prices):
    # Get log Returns
    if Log_Return:
        returns = np.log ( prices / prices.shift(1))
    else:
        returns = ( prices / prices.shift(1))-1
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


#  CHARTING TOOLS 
#
def plotheatmap (corr, start_date, end_date, title):
    #S = start_date.strftime('%Y-%m-%d')
    #E = end_date.strftime('%Y-%m-%d')
    # Plot Correlation Matrix Heatmap
    sns.set(font_scale=0.75)
    plt.subplots(figsize = HeatMap_Dims)
    ax = sns.heatmap(corr, annot = True,   cmap = "coolwarm" )
    ax.set_title(title, fontsize=20)
    fig = ax.get_figure()
    FileName = "HeatMapCorrelations_"+start_date+"-"+end_date+"_"+".png"
    os.chdir(OutPutFolder)
    fig.savefig(FileName)
    showplot()
    return


def plotlogreturns (P, title):
    # Plot Chart of Price Log Returns 
    Chart = np.log (P).plot(figsize=LogRet_Dims , title="Cummulative Log Returns of "+title)
    fig = Chart.get_figure()
    FileName = "Log_Returns_"+title+".png"
    os.chdir(OutPutFolder)
    fig.savefig(FileName)
    showplot()
    return


def plothistograms (R, title):
    # Plot Histogram of Log Returns 
    plt.figure(figsize=Hist_Dims)
    i=1
    n = int(np.sqrt(len(R.columns)))+1
    for r in R.columns:
        # Set up the plot
        bins = 50 # int(len(R[r].unique())/5)
        ax =  plt.subplot(n, n, i)
        ax.hist(R[r], bins = bins, color = 'blue', edgecolor = 'blue')
        # Title and labels
        ax.set_title(title+': '+r , size = 10)
        ax.set_xlabel('Observations', size = 8)
        plt.xticks(rotation='vertical')
        ax.set_ylabel('Frequency', size= 8)
        i+=1
    #
    FileName = "Histogram_"+title+'.png'
    os.chdir(OutPutFolder)
    plt.savefig(FileName)
    showplot()
    return
#
def plotseries (T, S, Title):
    # Plot Histogram of Log Returns 
    plt.figure(figsize=Series_Dims)
    i=1
    n = int(np.sqrt(len(S.columns)))+1
    for r in S.columns:
        # Set up the plot
        ax =  plt.subplot(n, n, i)
        ax.plot(S[r],  color = 'blue')
        # Title and labels
        ax.set_title(Title+' '+T+'--> '+r , size = 14)
        ax.set_xlabel('Date', size = 15)
        ax.set_ylabel(r, size= 15)
        i+=1
    #
    FileName = Title+'_'+T+'.png'
    os.chdir(OutPutFolder)
    plt.savefig(FileName)
    showplot()
    return
#
################################################################
#     SPLIT THE DATA into Prices, Factors, BenchMarks and BenchMarkWeights

# Get Money Market Index Data
MM_Header = ['xUSD']
MoneyMarket = pd.DataFrame(ALL_LocCurr[MM_Header]).copy()
#
Factor_Headers = ['MXWD','DXY','VIX','CRB','US_5Y/5Y_CPI','US_LBR_1Y/1Y','US-EU_1Y_Sprd','US-JY_1Y_Sprd']
Factors = pd.DataFrame(ALL_Factor[Factor_Headers]).copy()
#
#BenchMark_Headers = ['EM22_FX','BBRG_8EM','BBRG_ASIA','BBRG_EMEA','BBRG_G10','BBRG_Latam']
#BenchMark =  pd.DataFrame(ALL_BenchMark[BenchMark_Headers]).copy()
#
#
# USE Large or Small Currency Dataset
#FX_Headers = ['ARS','CNH','RUB','SGD','TRY','ZAR','BRL','CLP','COP','IDR','INR','KRW','MYR','PEN','PHP','THB','TWD','MXN','EURHUF','EURCZK','EURPLN','EURRON']
#
#BMW1 = ['_ARS','_CNH','_CZK','_PLN','_RON','_RUB','_SGD','_TRY','_ZAR','_BRL','_CLP']   # Spyder can't handle long lines of code??? Jupyter notebook worked fine...
#BMW2 = ['_COP','_IDR','_INR','_KRW','_MYR','_PEN','_PHP','_THB','_TWD','_MXN','_HUF']
#BMW  = BMW1 + BMW2
#
#LCH1 = ['xARS','xCNH','xRUB','xSGD','xTRY','xZAR','xBRL','xCLP','xCOP','xIDR','xINR']
#LCH2 = ['xKRW','xMYR','xPEN','xPHP','xTHB','xTWD','xMXN','xHUF','xCZK','xPLN','xRON']
#LCH  = LCH1 + LCH2
#
#if Dataset == "Small"  : # Spyder: what's wrong with this IF?? Jupyter notebook worked fine
FX_Headers = ['TRY','BRL','EURPLN','EUR']
BMW = ['_TRY','_BRL'] #,'_EURPLN','_EUR']
LCH = ['xTRY','xBRL'] #,'xPLN'],'xEUR']
#        
Prices = pd.DataFrame(ALL_FX[FX_Headers]).copy()
#BenchMarkWeights = pd.DataFrame(ALL_BenchMark[BMW]).copy()
LocCurrs = pd.DataFrame(ALL_LocCurr[LCH]).copy()


# Extract The Data For the Start / End Range, Take Weekly Data on Wednesdays (Avoid Long Weekends, Short Weeks)
MoneyMarket = extractdata (MoneyMarket, Start, End, ReturnFrequency)
Prices = extractdata (Prices, Start, End, ReturnFrequency)
Factors = extractdata (Factors, Start, End, ReturnFrequency)
LocCurrs = extractdata (LocCurrs, Start, End, ReturnFrequency)
#Weekly_BenchMark = extractdata (BenchMark, Start, End, ReturnFrequency)

# Normalize the Timeseries
MoneyMarket = normalize(MoneyMarket)
Factors = normalize(Factors)
#BenchMark = normalize(BenchMark)
Prices = normalize(Prices)
LocCurrs = normalize(LocCurrs)

# CHANGE QUOTE TYPE (1/p)
# Prices = invertFX(Prices)

#  This plots the log performance of all the FX Rates
%time plotlogreturns (Prices,(ReturnFrequencyText+" FX Rates"))

#  Get Log Returns
FX_Returns = getreturns (Prices)
# Plot Histogram of Log Returns: notice the Fat Tails (indicated by the width of the x-axis, as bars of tails are very small to notice at first sight!)
%time plothistograms (FX_Returns, ReturnFrequencyText+" FX Returns")


################################################################
# Calculate Basic Statistics for Weekly Returns
FX_Stats = descriptive('FX_Returns', FX_Returns, ReturnFrequency)
Historical_Stats = FX_Stats[1]  # View Summary Stats


# Plot Correlation Heatmap
plotheatmap (FX_Stats[3], Start, End, ReturnFrequencyText+' EM FX Return Correlation Heatmap')


# In[]:

################################################################
# Process Factors: View Correlation Heatmap and  Covariance Matrix
#Factor_Returns = getreturns (Weekly_Factors)
#FactorStats = descriptive('Factor_Returns', Factor_Returns, ReturnFrequency)
#plotheatmap (FactorStats[3], Start, End, 'Discounted Portfolio Factors Return Correlation Heatmap')
#FactorStats[2]

# Process BenchMarks: Correlation Matrix and Plot  Correlation Heatmap
#BenchMark_Returns = getreturns (Weekly_BenchMark)
#BenchMarkStats = descriptive('BenchMark_Returns', BenchMark_Returns, ReturnFrequency)
#plotheatmap (BenchMarkStats[3], Start, End, 'Discounted EM BenchMarks Return Correlation Heatmap')
#BenchMarkStats[3]  # View Correlation Matrix



# In[]:

################################################################
# Build a rolling window of data to plot the evolution of price stats.
# Given that we have a long data set, portfolio statistics are not expected to be stable over time. Given that we prefer to work with weekly data (to avoid the 
# highly noisy daily data), while still allowing us to generate long series of windows.
# We will work with windows of 2 years, which provide with around 100 weekly returns.
# We will roll the window from the start of our dataset to be able to trak the evolution over time of each and everyone of the stats calculated (mean returns, correlations, skew....)

def WindowStats (returns, freq, WindowInYears):
    #
    # Get length of window (in days units) to figure out how many rolling windows to run
    n , ReturnFrequencyText = getreturnperiod (freq)
    Window = int(WindowInYears*Days_in_Trading_Year/ n)
    NumberOfWindows = len(returns.index) - Window
    if NumberOfWindows <= 0:
        print ('WindowStats: Not enough dates for year window' )
        return []
    else:
        # Run windows and save tuple of descriptive stats, cov, corr
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
        # Return the end date of the windows, and the series to stat matrices
        return  pd.date_range(Dates[0], periods=len(Dates), freq=ReturnFrequency), Stats, CoVar, Corr

#  Rolling Statistical Data Per Currency
def GetStatSeries (ticker, dates, statdata):
    # Check that stat and stock are in the Stat Table
    if ticker not in statdata[1].index:
        print ('GetStatSeries: ticker error')
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
################################################################
#  Build the Historical Timeseries
%time WindowEndDates, Stats, CoVar, Corr = WindowStats(FX_Returns, ReturnFrequency, 2)


# Generate Rolling Statistics and Rolling Correlations for All TICKERS
for Tick in FX_Returns.columns:# ###
    writer=pd.ExcelWriter('Rolling_'+Tick+".xlsx", engine='xlsxwriter')
    StatSeries = GetStatSeries (Tick, WindowEndDates, Stats)
    StatSeries.to_excel(writer, ReturnFrequencyText+' Rolling_Statistics')
    plotseries (Tick, StatSeries, ReturnFrequencyText+" Rolling_Statistics")
    #
    CorrSeries = GetCorrSeries (Tick, WindowEndDates, Corr)
    CorrSeries.to_excel(writer, ReturnFrequencyText+'_Rolling_Correlations')
    plotseries (Tick, CorrSeries, ReturnFrequencyText+"_Rolling_Correlations")
    plothistograms (CorrSeries, Tick+"_-_"+ReturnFrequencyText+"_Rolling_Correlation")
    os.chdir(OutPutFolder)
    writer.save()
#

# In[]:

################################################################
## Show the data for the Brazilian Real
Tick = 'BRL'
Charts = StatSeries.plot(figsize=(6,12), subplots=True, title=Tick+": Rolling Statistics")
Chart = StatSeries.hist(figsize=(10,10),bins=50)
Chart = CorrSeries.plot(figsize=(10,36), subplots=True, title=Tick+": Rolling Correlations")
Chart = CorrSeries.hist(figsize=(12,12),bins=50)


