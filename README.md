# Natural Gas Analysis
## Overview
Analysis of Natural Gas Stock Closing price with storage, consumption and temperature.

The purpose of this project was to determine if there's a relationship between temperature and the price of natural gas. We initially started with the hope of identifying a strong correlation between temperature and natural gas (NG) price. But the analysis we performed did not show the relationship we expected. We also analyzed NG storage and consumption to determine if temperature impacted either and if either had an impact on NG prices.  We found a strong relationship between temperature, NG storage, and NG consumption but not with NG price. As part of our analysis we created a regression model that predicted residential natural gas consumption as a function of temperature. When testing the model with out of sample data it performed reasonably well.

## Prerequisites:
- .env file located in the analysis notebook folder with EIA_API_KEY=<<EIA_API_KEY>>. Click the hyperlink to obtain your own [EIA_API_KEY](https://www.eia.gov/opendata/register.php) 
- Anaconda environment: A combination of the both dev and Pyviz (as installed in class). Clone dev environment and install Pyviz on it. It needs Pandas, Numpy, sklearn, panel, requests, json, dotenv, yfinance, plotly, holoviews, hvplot, json etc.


## Files:
- Natural_Gas_Analysis_and_Dashboard.ipynb : File that contains the Analysis and Dashboard code.
- helper.py : Python script with helper functions to help keep code clean and uncluttered.
- Third Party Files 
    - MCForecastTools.py : for the Monte Carlo Simulation
## Folders
- Data: contains the data Weather and Storage data files. Files in the folder are as follows:
    - Pittsburg_Area_Temp_2010-2014.csv and Pittsburg_Temp_2015_2019.csv
    - Hartford_Area_Temp_2010-2014.csv 
    - Chicago_Area_Temp_2010-2014.csv
    - Dallas_Area_Temp_2010-2014.csv
    - LosAngelus_Area_Temp_2010-2014.csv
    - NG_STOR_WKLY_S1_W.csv
- Dashbord_Text: contains text files for the overview of project and  each analysis performed. Files in the folder are as follows:
    - Overview.txt: Overview of the Project
    - Closing_Price.txt: Overview of the storage analysis
    - Storage.txt: Overview of the Storage analysis
    - Consumption.txt: Overview of the Consumption analysis
    - Correlation.txt: Overview of the Correlation
    - Monte_Carlo.txt: Overview of the Monte_Carlo


