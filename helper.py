'''
File with constants defined and few common functions

'''
# Import Pandas 
import pandas as pd
from csv import reader
import numpy as np
from datetime import date
# import the Path function from pathlib
from pathlib import Path
from sklearn.linear_model import LinearRegression


#import API
import requests
import json

# Third Party
import yfinance as yf
import matplotlib.pyplot as plt

################## Data Structues #########################################
# Dictionary with ng comsumption settings, weather file names, and storage column names by US and states

region_info =  { "US" :
                {
                   "consumption" :  {"residential" : "NG.N3010US2.M", "industrial" : "NG.N3035US2.M" },
                   "weather_file" : r"",
                    "storage" :  "48 States"
                },  
    
                'PA' : 
                {
                   "consumption" :  {"residential" : "NG.N3010PA2.M", "industrial" : "NG.N3035PA2.M"},
                   "weather_file" : r"Data\Pittsburg_Area_Temp_2010-2014.csv",
                    "storage" :  r"East Region"
                },  
                 'CT' : {
                   "consumption" :  {"residential" : "NG.N3010CT2.M", "industrial" : "NG.N3035CT2.M"},
                    "weather_file": r"Data\Hartford_Area_Temp_2010-2014.csv",
                    "storage" : r"East Region"
                },
                 'IL' : {
                   "consumption" :  {"residential" : "NG.N3010IL2.M", "industrial" : "NG.N3035IL2.M"},
                    "weather_file": r"Data\Chicago_Area_Temp_2010-2014.csv",
                    "storage" : r"Midwest Region"
                },
                 'TX' : {
                   "consumption" :  {"residential" : "NG.N3010TX2.M", "industrial" : "NG.N3035TX2.M"},
                    "weather_file": r"Data\Dallas_Area_Temp_2010-2014.csv",
                    "storage" : r"South Central"
                },
                 'CA' : {
                   "consumption" :  {"residential" : "NG.N3010CA2.M", "industrial" : "NG.N3035CA2.M"},
                    "weather_file": r"Data\LosAngelus_Area_Temp_2010-2014.csv",
                    "storage" : r"Pacific Region"
                }                   
                
}
# init list of states 
states_list = list(region_info.keys())[-(len(region_info.keys()) - 1):]
# init list of regions 
region_list = list(region_info.keys())
str_res = "Residential Consumption"

################## Functions #########################################



# Get data for a ticker from yFinance for last 10 years
# Input: 
#    ticker : the ticker to get data for
# Output: A data frame with date as index and Adj Close prices as the ticker name
def yfinance_tickers_data(ticker, start_date, end_date, drop_extra_cols = True):
    try:
        # get ticker object
        ticker_data = yf.Ticker(ticker)
        # fetch data between start date ane end date
        df = ticker_data.history(start=start_date, end=end_date)
        # Drop unnecessary columns
        if len(df) > 0 and drop_extra_cols:
            df = df.drop(["Open", "High", "Low", "Volume", "Dividends", "Stock Splits" ], axis=1)
        return df
    
    except Exception as e:
        print(f"ERROR: An Error occurred while fetching and formatting data for {ticker} from yfinance  \n DETAILS: {repr(e)}")
        raise
        
    
    
  
# request series data from EIA API
# Input: 
#    api_key : EIA key to load data
#    series_id : the series to fetch the data for
# Output: json object with data
def eia_consumption_data_by_series(api_key, series_id):
    try:
    
        # Create variable to hold request url
        api_url = f"http://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
        # Execute GET request and store response
        response_data = requests.get(api_url)
        # Formatting as json
        data = response_data.json()
        # convert json to string
        data_str = json.dumps(data)
        #print(data_str)
        # check if an error was returned in json
        if  data_str.find("error") != -1 :
            raise RuntimeError(data["data"])
        return data      
    except Exception as e:
        print(f"ERROR:An  Error Occured while fetching data for series from EIA API  \n DETAILS: {str(e)}") 
        raise
        
   
# Get series data from EIA and fetch only between start_date and end_date
# Input: 
#    api_key : EIA key to load data
#    series_id : the series_id to get data for
#    stype : type of consumption, used to name columns
#    start_date: the start date 
#    end_date: the end date 
# Output: dataframe with relevant data
def eia_consumption_data_by_series_df(api_key, series_id, stype, start_date, end_date):
    try:
        data = eia_consumption_data_by_series(api_key, series_id);

        # create a data frame from the series of date and prices
        df_comsumption  = pd.DataFrame(list(data["series"][0]["data"]))

        str_type = f'{stype} Consumption'
        #Rename the columns from 0 & 1 to YearMonth and Consumption
        df_comsumption.rename(columns={0: "YearMonth", 1: str_type}, inplace = True)

        #Create a date column to select relevant data
        df_comsumption['Date'] = pd.to_datetime(df_comsumption["YearMonth"], format="%Y%m")

        # Set datetype as datetime
        df_comsumption["Date"].astype('datetime64', copy=False)

        # create mask to select only dates in our range
        mask = (df_comsumption['Date'] >= start_date) & (df_comsumption['Date'] <= end_date)

        # Apply mask and get relevant data 
        df_comsumption = df_comsumption.loc[mask]

        # Sort values 
        df_comsumption= df_comsumption.sort_values(by="Date", ascending = True)

        # Set Index
        df_comsumption.set_index("Date", inplace = True)
        #if drop_date:
        # Drop date 
        df_comsumption.drop(columns="YearMonth", inplace = True)

        return df_comsumption
    except Exception as e:
        print(f"ERROR: An Error occurred while fetching data for series from EIA API into dataframe \n DETAILS: {repr(e)}") 
        raise

        
        
        
# Get weather data form datafile
# Input: 
#    state : state for which we need to load data fiile
#    file_path : Path of file to load
# Output: dataframe with relevant weather data
def weather_data(state, file_path):
    try:
    # get OS independent file_path
        weather_path = Path(file_path)
        # Read CSV with date as index
        weather_df = pd.read_csv(
        weather_path, index_col="Date", infer_datetime_format=True, parse_dates=True)
        # Sort index
        weather_df = weather_df.sort_index()
        # Drop unnecessary Columns
        weather_df = weather_df.drop(["Departure", "HDD", "CDD", "Precipitation", "New Snow", "Snow Depth" ], axis=1)
        #weather_df["state"] = state
        # Rename Column for clarity
        weather_df.rename(columns = {"Average": "Avg Temp"}, inplace = True)

        return weather_df
    except FileNotFoundError as e:
        print(f"ERROR: Weather File not found {file_path} \n DETAILS: ")
        raise
    except ValueError as e:
        print(f"ERROR: An error occurred while loading file {file_path} \n DETAILS: {repr(e)}") 
        raise
    except Exception as e:
        print(f"ERROR: An error occurred while loading file {file_path} \n DETAILS: {repr(e)}") 
        raise
    

# Convert daily closing price to monthly closing price
# Input: 
#    df_price : daily closing price data
# Output: dataframe with monthly closing price data
def agg_stock_closing_price_monthly(df_price):
    try:
        # Group by year and then month and get mean
        df_avg_price = df_price.groupby(by=[df_price.index.year, df_price.index.month]).mean()

        #rename the new multi index
        df_avg_price.index.rename("Year", level=0, inplace = True)
        df_avg_price.index.rename("Month", level=1, inplace = True)

        # Convert index to columns
        df_avg_price = pd.DataFrame(df_avg_price.to_records()) 

        # Add a 0 to months that are single digit
        df_avg_price["Month"] = df_avg_price.Month.map("{:02}".format)

        # Create a single new column to save the year and month
        df_avg_price['YearMonth'] = df_avg_price['Year'].apply(str) + df_avg_price['Month'].apply(str)

        # Create an Date column to convert all dates with the first day of the month
        df_avg_price['Date'] = pd.to_datetime(df_avg_price["YearMonth"], format="%Y%m")

        # Set index to the Date column
        df_avg_price = df_avg_price.set_index("Date")

        # Drop the year and month Columns
        df_avg_price = df_avg_price.drop(["Year", "Month", "YearMonth"], axis=1)

        #return data
        return df_avg_price
    
    except Exception as e:
        print(f"ERROR: An error occurred while converting the daily Closing Price to monthly \n DETAILS: {repr(e)}") 
        raise

    except Exception as e:
        print(f"ERROR: An error occurred while converting the daily Closing Price to monthly \n DETAILS: {repr(e)}")         
        raise

        
# Convert daily closing price and temeprature to monthly
# Input: 
#    df_price : daily closing price and temeprature dataframe
# Output: dataframe with monthly closing price  and temeprature data
def agg_price_temperature_monthly(df_price_temp):
    try:

        # Group by year and then month and get mean
        df_avg_price_weather = df_price_temp.groupby(by=[df_price_temp.index.year, df_price_temp.index.month]).mean()

        #rename the new multi index
        df_avg_price_weather.index.rename("Year", level=0, inplace = True)
        df_avg_price_weather.index.rename("Month", level=1, inplace = True)

        # Convert index to columns
        df_avg_price_weather = df_avg_price_weather.reset_index() 

        #df_avg_price_weather = pd.DataFrame(df_avg_price_weather.to_records()) 


        # Add a 0 to months that are single digit
        df_avg_price_weather["Month"] = df_avg_price_weather.Month.map("{:02}".format)

        # Create a single new column to save the year and month
        df_avg_price_weather['YearMonth'] = df_avg_price_weather['Year'].apply(str)  + df_avg_price_weather['Month'].apply(str)


        # Create an Date column to convert all dates with the first day of the month
        df_avg_price_weather['Date'] = pd.to_datetime(df_avg_price_weather["YearMonth"], format="%Y%m")

        # Set index to the YearMonth column
        df_avg_price_weather = df_avg_price_weather.set_index("Date")

        # Drop the year and month Columns
        df_avg_price_weather = df_avg_price_weather.drop(["Year", "Month", "YearMonth"], axis=1)

        #return data
        return df_avg_price_weather
    
    except ValueError as e:
        print(f"ERROR: An error occurred while converting the daily Closing Price and Temperature to monthly \n DETAILS: {repr(e)}")        
        raise
    except Exception as e:
        print(f"ERROR: An error occurred while converting the daily Closing Price and Temperature to monthly \n DETAILS: {repr(e)}")       
        raise

# Convert weekly storage to monthly
# Input: 
#    df_storage_data : weekly Storage dataframe
# Output: dataframe with monthly storage
def format_strorage_monthly(df_storage_data):
    try:
        # Group by year and then month and get mean
        df_storage_data_monthly = df_storage_data.groupby(by=[df_storage_data.index.year, df_storage_data.index.month]).sum()

        #rename the new multi index
        df_storage_data_monthly.index.rename("Year", level=0, inplace = True)
        df_storage_data_monthly.index.rename("Month", level=1, inplace = True)

        # Convert index to columns
        df_storage_data_monthly = pd.DataFrame(df_storage_data_monthly.to_records()) 


        # Add a 0 to months that are single digit
        df_storage_data_monthly["Month"] = df_storage_data_monthly.Month.map("{:02}".format)

        # Create a single new column to save the year and month
        df_storage_data_monthly['YearMonth'] = df_storage_data_monthly['Year'].apply(str) + df_storage_data_monthly['Month'].apply(str)


        # Create an Date column to convert all dates with the first day of the month
        df_storage_data_monthly['Date'] = pd.to_datetime(df_storage_data_monthly["YearMonth"], format="%Y%m")

        # Set index to the YearMonth column
        df_storage_data_monthly = df_storage_data_monthly.set_index("Date")

        # Drop the year and month Columns
        df_storage_data_monthly = df_storage_data_monthly.drop(["Year", "Month", "YearMonth"], axis=1)

        #return data
        return df_storage_data_monthly
    
    except ValueError as e:
        print(f"ERROR: An error occurred while converting the weekly Storage to monthly \n DETAILS: {repr(e)}")       
        raise
    except Exception as e:
        print(f"ERROR: An error occurred while converting the weekly Storage to monthly \n DETAILS: {repr(e)}")  
        raise

# Convert daily temperature to monthly
# Input: 
#    df_temp : daily temeparture dataframe
# Output: dataframe with monthly temeprature
def agg_temperature_monthly(df_temp):

    try:
        # Group by year and then month and get mean
        df_avg_weather = df_temp.groupby(by=[df_temp.index.year, df_temp.index.month]).mean()


        #rename the new multi index
        df_avg_weather.index.rename("Year", level=0, inplace = True)
        df_avg_weather.index.rename("Month", level=1, inplace = True)

        # Convert index to columns
        df_avg_weather = pd.DataFrame(df_avg_weather.to_records()) 


        # Add a 0 to months that are single digit
        df_avg_weather["Month"] = df_avg_weather.Month.map("{:02}".format)

        # Create a single new column to save the year and month
        df_avg_weather['YearMonth'] = df_avg_weather['Year'].apply(str) + df_avg_weather['Month'].apply(str)


        # Create an Date column to convert all dates with the first day of the month
        df_avg_weather['Date'] = pd.to_datetime(df_avg_weather["YearMonth"], format="%Y%m")

        # Set index to the YearMonth column
        df_avg_weather = df_avg_weather.set_index("Date")

        # Drop the year and month Columns
        df_avg_weather = df_avg_weather.drop(["Year", "Month", "YearMonth"], axis=1)

        #return data
        return df_avg_weather

    except ValueError as e:
        print(f"ERROR: An error occurred while converting the daily Temperature to monthly  \n DETAILS: {repr(e)}")          
        raise
    except Exception as e:
        print(f"ERROR: An error occurred while converting the daily Temperature to monthly  \n DETAILS: {repr(e)}")          
        raise
    


# load all data (weather, storage & consumption)  for each state in region_info
# Input: 
#    df_temp : daily temeparture dataframe
# Output: dataframe with monthly temeprature

# input : 
#      region_info : dictionary with settings on how/what data to load for each state
#      eia_api_key: EIA API key to load consumption data
#      start_date : start_date of data
#      end_date : end_date of data
#      df_storage_monthly : monthly storage dataframe
#      df_historic:  historic natural gas stock daily closing price dataframe
def load_data(region_info, eia_api_key, start_date, end_date, df_storage_monthly, df_historic):
    try:

        # init dictionary to store dataframes needed in this notebook and dashboard
        dfs = {}
        # Loop through the region_info dictionary and fetch data, cleanup and format data into various dataframes    
        for key, value in region_info.items():
            # init a dictionary for each key
            dfs[key] = {}

            # Fetch comumption data per region (key) both industrial and residential from EIA

            # Set residential series id
            series_id = region_info[key]["consumption"]["residential"]
            # Fetch residential data for region(key) from EIA
            df_res = eia_consumption_data_by_series_df(eia_api_key, series_id, "Residential", 
                                                       start_date, end_date)

            # Set Industrial series id
            series_id = region_info[key]["consumption"]["industrial"]
            # Fetch Industrial data for region(key) from EIA
            df_industrial = eia_consumption_data_by_series_df(eia_api_key, series_id, "Industrial",
                                                              start_date, end_date)

            # Concat both data into a comsumption data frame
            df_comsumption = pd.concat((df_industrial, df_res), join="inner", axis=1 , sort=True).dropna()


            # Init the Storage data frame
            df_storage = pd.DataFrame()
            # Slice the regional storage data
            df_storage["Storage"] = df_storage_monthly[region_info[key]["storage"]]


            # If region is US
            if key == "US":
                # Aggregrate the daily stock closing prices only as US has no weather into into monthly 
                df_avg_price = agg_stock_closing_price_monthly(df_historic)
                # Concat the stock price monly, conspumtion only and  storage monthly
                dfs[key]["combined"]  = pd.concat( [df_avg_price, df_comsumption, df_storage] ,
                                                    join="inner", axis=1 , sort=True).dropna()

            else:
                # Get weather for region (using a city to represent a region/state)
                df_weather = weather_data(key, region_info[key]["weather_file"])

                # Set the weather data  for region
                #dfs[key]["weather"] = df_weather

                dfs[key]["price_temperature"] = pd.concat((df_historic, df_weather), join="inner", axis=1 , sort=True).dropna()
                # Aggegrate the stock prices and tempretaure into monthly values
                df_avg_price_temp =  agg_price_temperature_monthly(dfs[key]["price_temperature"].copy())

                # Set the combined aggregated dataframes into one
                dfs[key]["combined"]  = pd.concat( [df_avg_price_temp, df_comsumption, df_storage ] ,
                                                    join="inner", axis=1 , sort=True).dropna()

        return dfs
    except ValueError as e:
        print(f"ERROR: An error occurred while loading data for weather, consumption & storage \n DETAILS: {repr(e)}")
        raise
    except KeyError as e:
        print(f"ERROR: An error occurred while loading data for weather, consumption & storage \n DETAILS: {repr(e)}") 
        raise
    except Exception as e:
        print(f"ERROR: An error occurred while loading data for weather, consumption & storage  \n DETAILS: {repr(e)}") 
        raise



 
   
# Compute Linear regresssion between x & y variable, shifting y by shift_period
# input:
#      x : the x time series varaible, usually a data column
#      y : the y time series varaible, usually a data column
#  shift_period : period to shift y
# Output : 
#      r_sq:
#      model: the regression model
#      y_pred: predicted values for y
def linear_regression(x, y, shift_period = 90):

    try:
        
        # shift period and drop any nulls
        if shift_period != 0:
            x = x.shift(periods=shift_period).dropna()

        # Set the length of y same as x's length
        y = y.iloc[0: len(x)]

        #drop the index from both so the arrays are 1-dimensional
        x.reset_index(drop=True, inplace=True)
        x.reset_index(drop=True, inplace=True)
        #assign independent variable (x) and dependent variable (y) to the proper dataframe

        #reshape the data frames to pandas arrays
        x = np.array(x).reshape(-1,1)
        y = np.array(y)

        #create the model and fit it to the data
        model = LinearRegression()
        model.fit(x,y)
        r_sq = model.score(x,y)
        y_pred = model.predict(x)


        return r_sq, model, y_pred
    
    except ValueError as e:
        print(f"ERROR: An error occurred while computing Linear Regression \n DETAILS: {repr(e)}")         
        raise
    except Exception as e:
        print(f"ERROR: An error occurred while computing Linear Regression \n DETAILS: {repr(e)}")         
        raise

# Compute Linear regresssion between all relevant variables and load data 
# input:
#      dfs : Dataframe with region/state wise data, used to read and append data
# Output : 
#      dfs: Updated Dataframe with  all linear regression data loaded
def compute_linear_regression(dfs):
    # specify that dfs is global
    #global dfs
    # loop over for each state/region
    try:
        df_linear_regression = {}

        for key in dfs.keys():
            # if region is not equal to US
            if key != "US":

                df_linear_regression[key] = {}


                # CLOSING PRICE AND TEMPERATURE
                # compute and store the linear regression between temepature and closing price
                # Pull closing price and temepature data for region
                df = dfs[key]["price_temperature"]
                #print(f"\n----------{key} - Avg Temp - Close ------------------")

                # compute the linear regression between temepature and closing price
                r_sq, linear_model, y_pred = linear_regression(df["Avg Temp"], df["Close"], shift_period =0)

                # create a dataframe to store values
                df_linear_regression[key]["temp_close_lin_reg"] = pd.DataFrame()

                # Set values
                df_linear_regression[key]["temp_close_lin_reg"]["Slope"] = linear_model.coef_
                df_linear_regression[key]["temp_close_lin_reg"]["Intercept"] = linear_model.intercept_
                df_linear_regression[key]["temp_close_lin_reg"]["Coefficient  of determination"] =  r_sq
                # Re-arrange columns
                df_linear_regression[key]["temp_close_lin_reg"]  = df_linear_regression[key]["temp_close_lin_reg"][['Coefficient  of determination', 'Intercept', 'Slope']]

                # % CHANGE CLOSING PRICE AND TEMPERATURE            
                # compute and store the linear regression between n % change of temepature and closing price

                # Pull closing price and temepature data for region
                df_pct_change = dfs[key]["price_temperature"].copy()
                # Compute % change and drop nulls
                df_pct_change = df_pct_change.pct_change().dropna()
                # Get Linear Regression between % change of price and temperature
                r_sq_pct, linear_model_pct, y_pred_pct = linear_regression(df_pct_change["Avg Temp"], 
                                                                           df_pct_change["Close"], 
                                                                           shift_period =-90)

                # Create DataFrame to store values

                df_linear_regression[key]["pct_temp_close_lin_reg"] = pd.DataFrame()

                # Set values
                df_linear_regression[key]["pct_temp_close_lin_reg"]["Slope"] = linear_model_pct.coef_
                df_linear_regression[key]["pct_temp_close_lin_reg"]["Intercept"] = linear_model_pct.intercept_
                df_linear_regression[key]["pct_temp_close_lin_reg"]["Coefficient  of determination"] =  r_sq_pct
                # Re-arrange columns
                df_linear_regression[key]["pct_temp_close_lin_reg"]  = df_linear_regression[key]["pct_temp_close_lin_reg"][['Coefficient  of determination', 'Intercept', 'Slope']]




                # AVG TEMPERATURE AND RESIDENTIAL CONSUMPTION            
                # Get Linear Regression between temperarure and consumption
                #Pull a copy of the data relevant to the state
                df_combined = dfs[key]["combined"].copy()
                #Compute the linear regression between Avg. Temp and residential consumption
                r_sq_temp_com, linear_model_temp_con, y_pred_temp_com = linear_regression(df_combined["Avg Temp"], 
                                                                              df_combined[str_res], 
                                                                              shift_period =0)

                # Create DataFrame to store values

                df_linear_regression[key]["temp_res_consump_lin_reg"] = pd.DataFrame()

                # Set values
                df_linear_regression[key]["temp_res_consump_lin_reg"]["Slope"] = linear_model_temp_con.coef_
                df_linear_regression[key]["temp_res_consump_lin_reg"]["Intercept"] = linear_model_temp_con.intercept_
                df_linear_regression[key]["temp_res_consump_lin_reg"]["Coefficient  of determination"] =  r_sq_temp_com
                # Re-arrange columns
                df_linear_regression[key]["temp_res_consump_lin_reg"]  = df_linear_regression[key]["temp_res_consump_lin_reg"][['Coefficient  of determination', 'Intercept', 'Slope']]



                # STORAGE AND AVG TEMPERATURE
                #pull storage for relevant region
                r_sq_storage_temp, linear_model_storage_temp, y_pred_storage_temp = linear_regression(
                                                                                        df_combined["Storage"].copy(), 
                                                                                        df_combined["Avg Temp"].copy(), 
                                                                                        shift_period = -3)

                # Create DataFrame to store values

                df_linear_regression[key]["storage_temp_lin_reg"] = pd.DataFrame()

                # Set values
                df_linear_regression[key]["storage_temp_lin_reg"]["Slope"] = linear_model_storage_temp.coef_
                df_linear_regression[key]["storage_temp_lin_reg"]["Intercept"] = linear_model_storage_temp.intercept_
                df_linear_regression[key]["storage_temp_lin_reg"]["Coefficient  of determination"] =  r_sq_storage_temp
                # Re-arrange columns
                df_linear_regression[key]["storage_temp_lin_reg"]  = df_linear_regression[key]["storage_temp_lin_reg"][['Coefficient  of determination', 'Intercept', 'Slope']]


               # STORAGE AND  Consumption
                #pull storage for relevant region
                r_sq_storage_con, linear_model_storage_con, y_pred_storage_con = linear_regression(
                                                                                        df_combined["Storage"].copy(),
                                                                                        df_combined[str_res].copy(),
                                                                                        shift_period = -3)

                # Create DataFrame to store values

                df_linear_regression[key]["storage_con_lin_reg"] = pd.DataFrame()

                # Set values
                df_linear_regression[key]["storage_con_lin_reg"]["Slope"] = linear_model_storage_con.coef_
                df_linear_regression[key]["storage_con_lin_reg"]["Intercept"] = linear_model_storage_con.intercept_
                df_linear_regression[key]["storage_con_lin_reg"]["Coefficient  of determination"] =  r_sq_storage_con
                # Re-arrange columns
                df_linear_regression[key]["storage_con_lin_reg"]  = df_linear_regression[key]["storage_con_lin_reg"][['Coefficient  of determination', 'Intercept', 'Slope']]

        return df_linear_regression

    except ValueError as e:
        print(f"ERROR: An error occurred while computing  and loading all needed Linear Regression  \n DETAILS: {repr(e)}")          
        raise
    except Exception as e:
        print(f"ERROR: An error occurred while computing and loading all needed  Linear Regression \n DETAILS: {repr(e)}")                   
        raise
    

#Test the Pittsburg model with out of sample data

#Test the Pittsburg model with out of sample data
# input : 
#      state: state for which the linear regrssion computed, default PA
#      Output : Dataframe with relevant linear regresion values for state, styled
def predictied_linear_regression_temp_comsumption(region_dfs, eia_api_key, state = "PA"):
    
    try:
        #Import Pittsburg model with Pittsburg temp data 2015-2019 
        df_non_sample_temp = weather_data("PA", r"Data\Pittsburg_Temp_2015_2019.csv")
        # Aggregrate to monthly format    
        df_non_sample_temp = agg_temperature_monthly(df_non_sample_temp)
        # Pull Avg.Temp
        df_avg_non_sample_temp = df_non_sample_temp["Avg Temp"]
        # Reset  and drop index
        df_non_sample_temp.reset_index(drop=True, inplace=True)

        # Init new_date
        new_data = df_non_sample_temp
        #display(df_non_sample_temp.head())
        # Pull and Copy PA data
        df = region_dfs["PA"]["combined"].copy()
        # Compute Linear Regression between Avg.Temp and Residential Consumption
        r_sq, linear_model, y_pred = linear_regression(df["Avg Temp"], df[str_res], shift_period =0)
        # reshape
        new_x = np.array(new_data["Avg Temp"]).reshape(-1,1)
        #display(new_x)
        # get/pull predicted value from the out of sample temperature values
        y_new_pred = linear_model.predict(new_x)
        # Create dataframe with predicted consumption values based upon the new out of sample temperature values 
        future_predic_response = pd.DataFrame(y_new_pred, columns=["Predicted Consumption"])
        #future_predic_response.head()

        # Set out of sample dates
        start_date_2 = "2015-01-01"
        end_date_2 = "2019-12-30"

        # Set PA Residentail Series Data to
        pa_series_id = region_info["PA"]["consumption"]["residential"] #"NG.N3010PA2.M"
        #  fetch consumption data from EIA
        df_out_sample_comsumption = eia_consumption_data_by_series_df(eia_api_key, pa_series_id, "Residential", start_date_2, end_date_2)
        # Sort Values
        df_out_sample_comsumption= df_out_sample_comsumption.sort_values(by="Date", ascending = True)

        #df_out_sample_comsumption.drop(columns="Date", inplace = True)
        # Reset Index
        df_out_sample_comsumption =  df_out_sample_comsumption.reset_index()
        #display(df_out_sample_comsumption)
        # Concatinate out of sample consumption and predicted comsumption
        df_out_sample_combined = pd.concat((df_out_sample_comsumption, future_predic_response), join="inner", axis=1, sort=True)
        #df_out_sample_combined.head()
        #Set Index to Date
        df_out_sample_combined.set_index("Date", inplace = True)

        return df_out_sample_combined
    except ValueError as e:
        print(f"ERROR: An error occurred while computing Linear Reession for out of Sample for Consumption and Temeparure for PA  \n DETAILS: {repr(e)}")          
        raise
    except Exception as e:
        print(f"ERROR: Error occurred while computing Linear Reession for out of Sample for Consumption and Temeparure for PA  \n DETAILS: {repr(e)}")            
        raise


def read_text_file(file_path):
    try:
        with open(Path(file_path), "r", encoding='utf-8') as f:
            text= f.read()
        return text

    except FileNotFoundError as e:
        print(f"ERROR: File not found {file_path} ")           
        raise
    except Exception as e:
        print(f"ERROR: An Error occurred file loading file {file_path} \n DETAILS: {repr(e)}")                   
        raise
            


