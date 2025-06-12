# U.S. Election Impact on Equities 
As a part of my Capstone project at Duke, I worked on a project that analyzed the impact of 2024 US election events upon stock returns. The project entailed:

1. Building a Python-based model to observe market volatility during US elections using NASDAQ and S&P 500 returns, political news headlines, and Google Trends. I experimented with different data sets from headlines to search trends to observe political sentiment.
2. I identified important election events as a part of the event study and then observed 3 day and 5 day rolling windows. Ultimately the Ridge regression model based on news headline sentiments yielded the best explanatory power. 
3. Trained GARCH, XGBoost, and NLP models to forecast returns. I also optimized the model with regularization and clustering (R² = 0.87). 
4. Ultimately, I visualized forecasts in Tableau and Python, highlighting macro shifts and political catalysts to support strategic asset allocation for investors.

**Contribution:**

* Focusing on NASDAQ stock performance and comparing against S\&P 500 returns.   
* Data cleaning of news headlines by economic tones  
* Cleaning and merging google trends data with NASAQ with keywords such as gold, tariffs, Project 2025\.   
* Conducting an event study from January 2024 to December 2024 to observe events prior to the election and how they impacted returns and volatility.   
* Modeling sentiment analysis using VADER and TF\_IDF.  
* Developed XGBoost and OLS regression models to predict abnormal returns around election events.  
* I also experimented with NASDAQ returns against Trump tweet sentiments, gold prices, 10 year treasury yields, NASDAQ VIX etc, Bitcoin, DXY. The cleaned data can be found as the 'formatted_trump_tweets (0ct 1 - 4 Nov)' file.  
* Analyzing volatility trends by calculating returns to deviation across event dates.

**Project information:**

* **NASDAQ Event Study:** Main event study analysis on NASDAQ100 returns (see NASDAQ100 Event Study Code).  
* **Sentiment Analysis:** Built Ridge regression models and developed associated visualizations (it’s at Sentiment Analysis \- Ridge Regression Model and Visuals).  
* **Exploratory Data analysis:** Included analysis with alternative regressors (Gold, VIX, DXY, etc.) using the merged NASDAQ returns and Google Trends dataset. This model performed well by getting 87% R squared but Brian also wanted us to focus on aspects that didn’t rely on VIX. So I prioritized the other models more.   
* The data files I cleaned for this analysis are:   
  NEWS headlines (Aug 2024 to Apr 2025\) \- It has headlines about US elections from Reuters, bloomberg etc. along with short summaries and links.   
  Merged\_NASDAQ\_returns data (with Google trends).  

**Running code:**

* The files relevant to running the sentiment analysis code is:  
  \-NEWS headlines (Aug 2024 to Apr 2025\)  
  \-NASDAQ100 daily   
  \-S\&P500 returns   
* While the event study code relies on the NASDAQ100 daily code.   
* The ‘NASDAQ100 data exploration and initially modeling’ required the ‘Merged\_NASDAQ\_returns data (with Google trends)’ to be uploaded.  
