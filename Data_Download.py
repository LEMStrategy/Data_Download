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

#############################################################################
# Upload Excel Data to Panda DataFrames, Clean the Data

def LoadExcel (File, Tab, Ind, Col):
    Raw = pd.read_excel(File, sheet_name = Tab, index_col = Ind, header = Col, parse_dates = True)
    return Raw

# Save an Excel Sheete named Name, each tab/data is a list of keyword args
def SaveExcel (Name, **kwargs):
    #
    os.chdir(OutPutFolder)  
    writer=pd.ExcelWriter(Name+'.xlsx', engine='xlsxwriter')
    for tab, values in kwargs.items(): 
        values.to_excel(writer, tab)
    writer.save()
    return

# Clean The Time Series of Bad Data and Weekend Dates
def CleanDataFrame (frame):   # REMOVES BAD DATA AND WEEKENDS FROM TIME SERIES
    frame.dropna(inplace = True) # Drop Grabage
    return frame   #[frame.index.dayofweek < 5] # get only weekdays

#  Get Subset of Data and Get Returns on consecutive prices
def extractdata (p, start, end, freq):
    output = p.loc[start:end].copy()
    if freq in Daily:
        if freq=="B":
            output = output[output.index.dayofweek < 5]
    elif freq in Weekly:
        D = freq[-3:] # Get the day of the week, last 3 chars in freq
        NumDay = Days.index(D)# Get the number of the day of the week to re-index
        output = output[output.index.dayofweek == NumDay]
    else:
        print ("extractdata: Invalid Frequency")
    return output   # Return a clean dataframe

#############################################################################
#  CHARTING TOOLS 

def plotheatmap (corr, title):
    #S = start_date.strftime('%Y-%m-%d')
    #E = end_date.strftime('%Y-%m-%d')
    # Plot Correlation Matrix Heatmap
    sns.set(font_scale=0.75)
    plt.subplots(figsize = HeatMap_Dims)
    ax = sns.heatmap(corr, annot = True,   cmap = "coolwarm" )
    ax.set_title(title, fontsize=20)
    fig = ax.get_figure()
    FileName = title+"_CorrelationHeatMap"+".png"
    os.chdir(OutPutFolder)
    fig.savefig(FileName)
    showplot()
    plt.close(fig)
    return


def plotlogreturns (P, title):
    # Plot Chart of Price Log Returns 
    Chart = np.log (P).plot(figsize=LogRet_Dims , title="Cummulative Log Returns of "+title)
    fig = Chart.get_figure()
    FileName = title+"_Log_Returns"+".png"
    os.chdir(OutPutFolder)
    fig.savefig(FileName)
    showplot()
    plt.close(fig)
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
    FileName = title+"_Histogram"+'.png'
    os.chdir(OutPutFolder)
    plt.savefig(FileName)
    showplot()
    plt.close()
    return
#
def plotseries (S, Title):
    # Plot Histogram of Log Returns 
    plt.figure(figsize=Series_Dims)
    i=1
    n = int(np.sqrt(len(S.columns)))+1
    for r in S.columns:
        # Set up the plot
        ax =  plt.subplot(n, n, i)
        ax.plot(S[r],  color = 'blue')
        # Title and labels
        ax.set_title(Title+' '+'--> '+r , size = 14)
        ax.set_xlabel('Date', size = 15)
        ax.set_ylabel(r, size= 15)
        i+=1
    #
    FileName = Title+'.png'
    os.chdir(OutPutFolder)
    plt.savefig(FileName)
    showplot()
    plt.close()
    return

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

# Invert FX Quotes 
def invert (p):
    return (100/p)  # Invert FX quotes, multiply by 100 to avoid penny pricing... 

#  Normalize timeseries to re-base start at 1 (great for plottong Log return)
def normalize (P):
    CleanDataFrame (P)  # Clean just in case first data is NaN, Zero
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

def getreturns (prices):
    # Get log Returns
    if Log_Return:
        returns = np.log ( prices / prices.shift(1))
    else:
        returns = ( prices / prices.shift(1))-1
    return CleanDataFrame (returns)   # Return a clean dataframe

# Descriptive Stats for Return of a set of Prices
def descriptive (Prices, Start, End, Frequency):
    # Pre-process the inputs
    Tickers = Prices.columns
    Stats = ["Mean","StdDev","VAR","Skew","Kurtosis","Mean/StdDev"]
    Dim = len(Tickers)
    # Create output data structures
    Covar = pd.DataFrame (np.zeros( (Dim,Dim), dtype='float64'), index = Tickers, columns = Tickers)
    Correl = pd.DataFrame (np.zeros( (Dim,Dim), dtype='float64'), index = Tickers, columns = Tickers)
    descrip = pd.DataFrame(np.zeros((Dim, len(Stats)), dtype='float64'), index = Tickers, columns = Stats)
    # Build Decriptive Statistics for each ticker on the date range desired 
    for Tick in Tickers:
        Tick_Data = Prices[Tick].copy()
        Tick_Returns = getreturns(CleanDataFrame(Tick_Data))
        Returns = extractdata (Tick_Returns, Start, End, Frequency)
        mean = Returns.mean()
        Var = np.var(Returns)
        descrip ["Mean"][Tick] = mean
        descrip ["StdDev"][Tick] = np.var(Returns)
        descrip ["StdDev"][Tick] = np.sqrt(Var)
        descrip ["VAR"][Tick] = Var
        descrip ["Skew"][Tick] = Returns.skew()
        descrip ["Kurtosis"][Tick] = Returns.kurtosis()
        descrip ["Mean/StdDev"][Tick] = mean/np.sqrt(Var)
        for Tock in Tickers:
            if Tick == Tock:
                Data = Prices[Tick].copy()
                Returns = getreturns(CleanDataFrame(Data))
                Returns = extractdata (Returns, Start, End, Frequency)
                Correl[Tick][Tock] = 1
                Covar[Tick][Tock] = np.var(Returns)
            else:
                Pair = [Tick]
                Pair.append(Tock)
                Data = Prices[Pair].copy()
                Returns = getreturns(CleanDataFrame(Data))
                Returns = extractdata (Returns, Start, End, Frequency)
                Correl[Tick][Tock] = Returns[Tick].corr(Returns[Tock]) 
                Covar [Tick][Tock] = Returns[Tick].cov (Returns[Tock]) 
                # Returns the Dates of first/last Return, Summary Stas, Cov and Corr Matricies
    return [Start, End, descrip, Correl, Covar]

# Run Descriptive Stats for Each Type of Price data. Price Data uses **kwargs (Keyword Arguments)
def PriceStats (Start, End, Frequency, **PriceData):
    for File, Prices in PriceData.items(): 
        Stats = descriptive(Prices, Start, End, Frequency)
        SaveExcel (File+"_"+Frequency+"_"+Start+'_'+End, Stats = Stats[2], Corr = Stats[3],CoVar= Stats[4]) 
        # Plot the Evolution of Log Prices
        NormPrices = normalize(Prices.copy())
        plotlogreturns (NormPrices,(File+"_"+Frequency+"_"+Start+"_"+End))
        # Plot Histogram of Log Returns
        Returns = getreturns (NormPrices)
        plothistograms (Returns, File+"_"+Frequency+"_"+Start+"_"+End)
        plotheatmap (Stats[3], File+"_"+Frequency+"_"+Start+"_"+End)
    return


################################################################
#  MAIN PROGRAM STARTS HERE

##################################
# Set Up the RunTime Environemnt

# 
#  SET THE PATH TO THE EXCEL INPUT FILES!!!
HomeFolder = "D:/Boris/LEM_Strategy/Software/Data_Download"
OutPutFolder = HomeFolder+"/DataOutPut"
InPutFolder = HomeFolder+"/DataInPut"
os.chdir(HomeFolder)   # Windows 10
# os.chdir('//Volumes/Data/Exchange')  # MAc OS X
# File Name with EM FX Currency Data, Benchmark Index/Weights, Matrix Shrinkage Factors.
FX_Sheet = "_Alt_FX_Rates.xlsx"
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

######################################################
#SET THE DATE RANGES OF THE TOTAL DATASET
Start = '1-1-2017'
End = '12-31-2018'
ReturnFrequency = 'D'  #   D, W-WED... 
#Dataset = "Large"
Dataset = "Small"

Days_in_Trading_Year = 252
ReturnPeriod = 5  # Use weekly Returns assuming data without weekends, ie a 5 day week
# Calculate Log Returns?
Log_Return = False
#
# Generate Lists of Frequency Types in Python
Months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
Days = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT','SUN']
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

# Set Chart Parameters
Pie_Dims = (10, 10)
Bar_Dims = (10, 16)
Hist_Dims = (10,15)
LogRet_Dims = (18,14)
Series_Dims = (18,14)
HeatMap_Dims = (18, 10)
Show_Plot = False
# FACTORS TO ANNUALIZE Log Return/Var
n , ReturnFrequencyText = getreturnperiod (ReturnFrequency)
Ret_Factor =  Days_in_Trading_Year / n
Std_Factor = np.sqrt ( Ret_Factor )
Var_Factor = Ret_Factor 

##############################################################################
# UPLOAD DATA FROM EXCEL
print("Loading Bloomberg/Excel Inputs")
os.chdir(InPutFolder)
%time ALL_Factor = LoadExcel (Factor_Sheet, FactorTab,  0,  3)
%time ALL_FX = LoadExcel (FX_Sheet,  FX_Tab,  0,  3) 
%time ALL_LocCurr = LoadExcel (LocCurr_Sheet, LocCurrTab, 0, 3)
%time ALL_BenchMark = LoadExcel (Benchmark_Sheet, BenchmarkTab,  0, 3)

ALL_FX= invert(ALL_FX) # Switch from indirect to direct quotes

# In[]

################################################################
#     SPLIT THE DATA into Prices, Factors, BenchMarks and BenchMarkWeights

# Get Money Market Index Data
MM_Header = ['xUSD']
MoneyMarket = pd.DataFrame(ALL_LocCurr[MM_Header]).copy()

Factor_Headers = ['MXWD','DXY','VIX','CRB','US_5Y/5Y_CPI','US_LBR_1Y/1Y','US-EU_1Y_Sprd','US-JY_1Y_Sprd']
Factor_Prices = pd.DataFrame(ALL_Factor[Factor_Headers]).copy()

BenchMark_Headers = ['EM22_FX','BBRG_8EM','BBRG_ASIA','BBRG_EMEA','BBRG_G10','BBRG_Latam']
BenchMark_Prices =  pd.DataFrame(ALL_BenchMark[BenchMark_Headers]).copy()

# USE Large or Small Currency Dataset
if Dataset == "Large" :     
    FX_Headers = ['ARS','CNH','RUB','SGD','TRY',
                  'ZAR','BRL','CLP','COP','IDR',
                  'INR','KRW','MYR','PEN','PHP',
                  'THB','TWD','MXN',
                  'EURHUF','EURCZK','EURPLN','EURRON']
    BMW1 = ['_ARS','_CNH','_CZK','_PLN','_RON','_RUB',
            '_SGD','_TRY','_ZAR','_BRL','_CLP']
    BMW2 = ['_COP','_IDR','_INR','_KRW','_MYR','_PEN',
            '_PHP','_THB','_TWD','_MXN','_HUF']
    BMW  = BMW1 + BMW2
    
    LCH1 = ['xARS','xCNH','xRUB','xSGD','xTRY','xZAR',
            'xBRL','xCLP','xCOP','xIDR','xINR']
    LCH2 = ['xKRW','xMYR','xPEN','xPHP','xTHB','xTWD',
            'xMXN','xHUF','xCZK','xPLN','xRON']
    LCH  = LCH1 + LCH2
elif Dataset == "Small": 
        FX_Headers = ['TRY','BRL','MXN','ZAR','EUR','GBP']
        BMW = ['_TRY','_BRL','_MXN','_ZAR']
        LCH = ['xTRY','xBRL','xMXN','xZAR']
else:
        print("DATASET ERROR.")
    
FX_Prices = pd.DataFrame(ALL_FX[FX_Headers]).copy()
LocCurrs = pd.DataFrame(ALL_LocCurr[LCH]).copy()
#BenchMarkWeights = pd.DataFrame(ALL_BenchMark[BMW]).copy()

# In[]

Start = '1-1-2017'
End = '12-31-2018'
ReturnFrequency = 'D'  #   D, W-WED... 
n , ReturnFrequencyText = getreturnperiod (ReturnFrequency)
print("Processing Price Data:", Start, End, ReturnFrequency)
%time PriceStats (Start, End, ReturnFrequency, FX = FX_Prices, Factors = Factor_Prices, Benchmarks = BenchMark_Prices, L_Cur = LocCurrs)


# In[]
#Start = '1-1-2008'
#End = '12-31-2018'
#ReturnFrequency = 'W-WED'  #   D, W-WED 
#n , ReturnFrequencyText = getreturnperiod (ReturnFrequency)
#print("Processing Price Data:", Start, End, ReturnFrequency)
#%time PriceStats (Start, End, ReturnFrequency, FX_Currencies = FX_Prices, Factors = Factor_Prices, Benchmarks = BenchMark_Prices, LocalCurrency = LocCurrs)

# In[]:

################################################################
# Build a rolling window of data to plot the evolution of price stats.
# Given that we have a long data set, portfolio statistics are not expected to be stable over time. Given that we prefer to work with weekly data (to avoid the 
# highly noisy daily data), while still allowing us to generate long series of windows.
# We will work with windows of 2 years, which provide with around 100 weekly returns.
# We will roll the window from the start of our dataset to be able to trak the evolution over time of each and everyone of the stats calculated (mean returns, correlations, skew....)

def WindowStats (Prices, Start, End, Freq, WindowInYears):
    #
    # Get length of window (in days units) to figure out how many rolling windows to run
    n , ReturnFrequencyText = getreturnperiod (Freq)
    Window = int(WindowInYears*Days_in_Trading_Year/ n)
    RangePrices = extractdata (Prices, Start, End, Freq)
    NumberOfWindows = len(RangePrices.index) - Window
    if NumberOfWindows <= 0:
        print ('WindowStats: Not enough dates for year window' )
        return []
    else:
        # Run windows and save tuples of descriptive start/end dates, stats, cov, corr
        Start_Dates = []
        End_Dates = []
        Stats = []
        CoVar = []
        Corr  = []
        for i in range(NumberOfWindows):
            # Set the Date Ranges of the Window
            win_start = RangePrices.index[i].strftime('%Y-%m-%d')
            win_end = RangePrices.index[i+Window].strftime('%Y-%m-%d')
            # From the whole price series, extract the data in the window range at the specified frequency.
            WindowData = extractdata (RangePrices.copy(), win_start, win_end, Freq)
            # Get the descriptive stats for the returns in the window
            one_run = descriptive(WindowData, win_start, win_end, Freq)
            # Add output of the run to the list of outputs
            Start_Dates = Start_Dates + [one_run[0]]
            End_Dates = End_Dates + [one_run[1]]
            Stats = Stats + [one_run[2]]
            Corr = Corr + [one_run[3]]
            CoVar = CoVar + [one_run[4]]
        # Return the Start/End dates of the window, the series to stat matrices
        Out1 = Start_Dates[0]
        Out2 = End_Dates[0]
        Out3 = pd.date_range(start=End_Dates[0], end=End_Dates[-1], 
                             periods = len(End_Dates))
        Out4 = Stats
        Out5 = Corr
        Out6 = CoVar
        return  Out1, Out2, Out3, Out4, Out5, Out6

#  Rolling Statistical Data Per Currency
def GetStatSeries (ticker, dates, statdata):
    # Check that stat and stock are in the Stat Table
    if ticker not in statdata[1].index:
        print ('GetStatSeries: ticker error')
        return
    output = pd.DataFrame(index=dates.strftime('%Y-%m-%d'))
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

#   NumZero = [np.count_nonzero(w)<=1 for w in FrontPortsLev]


# In[]:
################################################################
#  Build the Historical Timeseries
Start = '1-1-2016'
End = '12-31-2018'
ReturnFrequency = 'D'  #   D, W-WED... 
n , ReturnFrequencyText = getreturnperiod (ReturnFrequency)
WindowDays = 120
WindowSize = np.round(WindowDays /252, decimals=2) #Years
%time Start_Dates,End_Dates, Dates_Index, Stats, Correls, CoVars = WindowStats(FX_Prices, Start, End, ReturnFrequency, WindowSize)

# In[]:
################################################################
# Tick = 'BRL'
# Generate Rolling Statistics and Rolling Correlations for All TICKERS
for Tick in FX_Prices.columns: 
    writer=pd.ExcelWriter(Tick+'_Rolling_Statistics_'+str(WindowSize)+"Y_"+ReturnFrequencyText+"_"+"_"+Start+"_"+End+".xlsx", engine='xlsxwriter')
    #
    StatSeries = GetStatSeries (Tick, Dates_Index, Stats)
    StatSeries.to_excel(writer, "Rolling Statistics")
    plothistograms (StatSeries, Tick+"_"+ReturnFrequency+str(WindowSize)+'Y'+" Rolling Statistics")
    plotseries (StatSeries, Tick+"_"+ReturnFrequency+str(WindowSize)+"Y"+" Rolling Statistics")
    #
    CorrSeries = GetCorrSeries (Tick, Dates_Index, Correls)
    CorrSeries.to_excel(writer, "Rolling Correlations")
    plotseries (CorrSeries, Tick+"_"+ReturnFrequency+str(WindowSize)+"Y"+" Rolling Correlations")
    plothistograms (CorrSeries, Tick+"_"+ReturnFrequency+str(WindowSize)+"Y"+" Rolling Correlations")
    os.chdir(OutPutFolder)
    writer.save()
#


