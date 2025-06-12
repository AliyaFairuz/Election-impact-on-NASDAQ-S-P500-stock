# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 18:28:31 2025
@author: af377
"""
#Purpose of this file is to provide the code used for sentiment analysis. This includes the model and the visuals
#an HTML report of this files output excluding the images is also provided. 
#All visuals are on the report and the final slides submitted

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from datetime import datetime
import nltk
import os
nltk.download('vader_lexicon') #sentiment

#All files can be found in the folder 
#NEWS headlines file is the NEWS headlines (Aug 2024 to Apr 2025) file
Tk().withdraw()
print("\U0001F4E4 Select 'NEWS headlines.xlsx'")
news_df = pd.read_excel(askopenfilename())

#file for NASDAQ and S&P is also available on the file
#NASDAQ100 daily returns extracted, volatility calculated in terms of returns
print("\U0001F4E4 Select 'NASDAQ100 daily.xlsx'")
nasdaq_df = pd.read_excel(askopenfilename(), skiprows=1)
nasdaq_df.columns = ['Date', 'Close']
nasdaq_df['Date'] = pd.to_datetime(nasdaq_df['Date'])
nasdaq_df['Return'] = nasdaq_df['Close'].pct_change()
nasdaq_df['EWMA_Vol'] = nasdaq_df['Return'].ewm(span=5, adjust=False).std()

#S&P500 daily returns extracted, volatility calculated in terms of returns
print("\U0001F4E4 Select 'S&P500 returns.xlsx'")
sp500_df = pd.read_excel(askopenfilename(), skiprows=1)
sp500_df.columns = ['Date', 'Close']
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
sp500_df['Return'] = sp500_df['Close'].pct_change()
sp500_df['EWMA_Vol'] = sp500_df['Return'].ewm(span=5, adjust=False).std()

# headlines & topics are categorized using keywords generated from summaries
news_df['Published date'] = pd.to_datetime(news_df['Published date'])
news_df['Date'] = news_df['Published date'] + pd.Timedelta(days=1)

sid = SentimentIntensityAnalyzer()
news_df['Sentiment'] = news_df['Summary'].fillna("").astype(str).apply(lambda x: sid.polarity_scores(x)['compound'])

keywords = [
    'economic', 'election', 'trade', 'policies', 'potential', 'trump', 'market', 'growth', 'inflation',
    'investors', 'policy', 'economy', 'impact', 'stocks', 'global', 'uncertainty', 'volatility', 'victory'
]
tfidf = TfidfVectorizer(vocabulary=keywords)
tfidf_matrix = tfidf.fit_transform(news_df['Summary'].fillna("").astype(str))

svd = TruncatedSVD(n_components=7, random_state=42)
topics = svd.fit_transform(tfidf_matrix)
weighted_topics = topics * news_df['Sentiment'].values.reshape(-1, 1)
weighted_df = pd.DataFrame(weighted_topics, columns=[f'WTopic{i+1}' for i in range(7)])
weighted_df['Date'] = news_df['Date'].values
weighted_df.set_index('Date', inplace=True)
smoothed_weighted = weighted_df.rolling(window=3).mean().reset_index()

# Ridge regression
alphas = np.logspace(-4, 4, 50)

def run_ridge_model(price_df, label):
    data = pd.merge(price_df[['Date', 'EWMA_Vol']], smoothed_weighted, on='Date').dropna()
    data['Lag_Vol'] = data['EWMA_Vol'].shift(1)
    data['Interaction1'] = data['WTopic3'] * data['Lag_Vol']
    data['Interaction2'] = data['WTopic1'] * data['WTopic5']
    data.dropna(inplace=True)

    X_cols = [f'WTopic{i+1}' for i in range(7)] + ['Lag_Vol', 'Interaction1', 'Interaction2']
    X = data[X_cols]
    y = data['EWMA_Vol']

    pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=5))
    model = pipeline.fit(X, y)
    preds = model.predict(X)

    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    coefs = model.named_steps['ridgecv'].coef_

    print(f"\n\U0001F4C8 {label} Ridge Regression Results")
    print(f"R² Score: {r2:.3f}\nRMSE: {rmse:.5f}\nOptimal Alpha: {model.named_steps['ridgecv'].alpha_}")
    print(pd.Series(coefs, index=X.columns))

    # Actual vs Predicted plot
    plt.figure(figsize=(12, 5))
    plt.plot(data['Date'], y, '--o', label='Actual', color='purple')
    plt.plot(data['Date'], preds, '-x', label='Predicted', color='blue')
    plt.title(f"{label}: Actual vs Predicted EWMA Volatility")
    plt.xlabel("Date")
    plt.ylabel("EWMA Volatility")
    plt.legend()
    plt.grid(True, linestyle='dotted')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Accuracy scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y, preds, alpha=0.7, edgecolors='k')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='gray')
    plt.title(f"{label}: Accuracy Plot")
    plt.xlabel("Actual Volatility")
    plt.ylabel("Predicted Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return data, model, pd.Series(coefs, index=X.columns)

nasdaq_results, nasdaq_model, nasdaq_betas = run_ridge_model(nasdaq_df, 'NASDAQ')
sp500_results, sp500_model, sp500_betas = run_ridge_model(sp500_df, 'S&P500')

# Correlation Heatmap to detect how topics are correlated
nasdaq_corr = nasdaq_results[[col for col in nasdaq_results.columns if 'WTopic' in col] + ['EWMA_Vol']].corr()
sp500_corr = sp500_results[[col for col in sp500_results.columns if 'WTopic' in col] + ['EWMA_Vol']].corr()

fig, axs = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(nasdaq_corr, annot=True, cmap="RdBu_r", center=0, ax=axs[0])
axs[0].set_title("📈 NASDAQ Topic Correlation Matrix")
sns.heatmap(sp500_corr, annot=True, cmap="RdBu_r", center=0, ax=axs[1])
axs[1].set_title("📉 S&P500 Topic Correlation Matrix")
plt.tight_layout()
plt.show()

# Importing again
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
news_df['Sentiment'] = news_df['Summary'].fillna("").astype(str).apply(lambda x: sid.polarity_scores(x)['compound'])

nasdaq_df['Volatility'] = nasdaq_df['Return'].rolling(3).std()
sp500_df['Volatility'] = sp500_df['Return'].rolling(3).std()

# Merge headlines with market data
merged_nasdaq = pd.merge(news_df, nasdaq_df, on='Date', how='inner')
merged_sp500 = pd.merge(news_df, sp500_df, on='Date', how='inner')

# Extract top volatility days to get the headlines with the highest volatility
top_nasdaq = merged_nasdaq.sort_values(by='Volatility', ascending=False).head(5)
top_sp500 = merged_sp500.sort_values(by='Volatility', ascending=False).head(5)

# Combine top movers of return volatility
top_combined = pd.concat([
    top_nasdaq.assign(Index='NASDAQ'),
    top_sp500.assign(Index='S&P500')
])[['Date', 'Headline', 'Summary', 'Sentiment', 'Return', 'Volatility', 'Index']]

# Print the most impactful headlines in terms of volatility
print("\n📊 Top 5 Headlines Causing Highest Volatility:\n")
print(top_combined.to_string(index=False))

#MORE VISUALS
# Scatter plots for: Sentiment vs Return & Volatility for NASDAQ and S&P500

# Filtering market+news merged data for the same timeline i.e. Sep 2024 to Dec 2024
merged_nasdaq = pd.merge(news_df, nasdaq_df[['Date', 'Return', 'EWMA_Vol']], on='Date', how='inner')
merged_sp500 = pd.merge(news_df, sp500_df[['Date', 'Return', 'EWMA_Vol']], on='Date', how='inner')

start, end = pd.to_datetime("2024-09-01"), pd.to_datetime("2024-12-31")
merged_nasdaq = merged_nasdaq[(merged_nasdaq['Date'] >= start) & (merged_nasdaq['Date'] <= end)]
merged_sp500 = merged_sp500[(merged_sp500['Date'] >= start) & (merged_sp500['Date'] <= end)]

# Scatter Plot Function (Editing chart to have no headline text below chart)
def plot_sentiment_scatter(data, metric, index_name, color_map, ylabel):
    fig, ax = plt.subplots(figsize=(14, 5))
    scatter = ax.scatter(
        data['Date'], data[metric],
        c=data['Sentiment'], cmap=color_map, edgecolors='black', s=120, alpha=0.9
    )
    ax.set_title(f"{index_name}: {ylabel} vs News Sentiment (Sep–Dec 2024)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.xticks(rotation=45)
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Sentiment Score", fontsize=10)
    plt.tight_layout()
    plt.show()

# Plot
plot_sentiment_scatter(merged_nasdaq, 'Return', 'NASDAQ', 'YlGn', 'Return')
plot_sentiment_scatter(merged_sp500, 'Return', 'S&P500', 'YlGn', 'Return')
plot_sentiment_scatter(merged_nasdaq, 'EWMA_Vol', 'NASDAQ', 'OrRd', 'Volatility')
plot_sentiment_scatter(merged_sp500, 'EWMA_Vol', 'S&P500', 'OrRd', 'Volatility')

# For slides and information, headlines that are the most impactful are printed
def print_headline_summary(data, label):
    print(f"\n--- {label} HEADLINES (Sep–Dec 2024) ---")
    summary = data[['Date', 'Sentiment', 'Headline']].copy()
    summary['Date'] = summary['Date'].dt.strftime('%Y-%m-%d')
    summary['Sentiment'] = summary['Sentiment'].round(2)
    print(summary.to_string(index=False))

print_headline_summary(merged_nasdaq, "NASDAQ")
print_headline_summary(merged_sp500, "S&P500")

# Compute absolute returns to see the highest change, whether negative or positive
merged_nasdaq['AbsReturn'] = merged_nasdaq['Return'].abs()
merged_sp500['AbsReturn'] = merged_sp500['Return'].abs()

# Top 10 by absolute return
top_return_nasdaq = merged_nasdaq.sort_values(by='AbsReturn', ascending=False).head(10)
top_return_sp500 = merged_sp500.sort_values(by='AbsReturn', ascending=False).head(10)

# Top 10 S&P500 by volatility
top_vol_sp500 = merged_sp500.sort_values(by='EWMA_Vol', ascending=False).head(10)

# Formatted text for report of top 10 movers
def display_table(df, label, sort_col):
    display_df = df[['Date', 'Headline', 'Sentiment', 'Return', 'EWMA_Vol']].copy()
    display_df['Sentiment'] = display_df['Sentiment'].round(2)
    display_df['Return'] = (display_df['Return'] * 100).round(2).astype(str) + '%'
    display_df['Volatility'] = (display_df['EWMA_Vol'] * 100).round(2).astype(str) + '%'
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df.drop(columns='EWMA_Vol', inplace=True)
    print(f"\n--- {label} (Top 10 by {sort_col}) ---")
    print(display_df.to_string(index=False))

# Print
display_table(top_return_nasdaq, "NASDAQ Headlines", "Absolute Return Change")
display_table(top_return_sp500, "S&P500 Headlines", "Absolute Return Change")
display_table(top_vol_sp500, "S&P500 Headlines", "Volatility")

# TOP 10 by positive and negative returns to get exclusive changes
# Highest positive return with headlines and dates
top_positive_return_nasdaq = merged_nasdaq.sort_values(by='Return', ascending=False).head(10)
top_positive_return_sp500 = merged_sp500.sort_values(by='Return', ascending=False).head(10)

# Most negative return days with headlines and dates
top_negative_return_nasdaq = merged_nasdaq.sort_values(by='Return', ascending=True).head(10)
top_negative_return_sp500 = merged_sp500.sort_values(by='Return', ascending=True).head(10)

# Display function
def display_raw_return_table(df, label, sort_col):
    display_df = df[['Date', 'Headline', 'Sentiment', 'Return', 'EWMA_Vol']].copy()
    display_df['Sentiment'] = display_df['Sentiment'].round(2)
    display_df['Return'] = (display_df['Return'] * 100).round(2).astype(str) + '%'
    display_df['Volatility'] = (display_df['EWMA_Vol'] * 100).round(2).astype(str) + '%'
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df.drop(columns='EWMA_Vol', inplace=True)
    print(f"\n--- {label} (Top 10 by {sort_col}) ---")
    print(display_df.to_string(index=False))

# Print top positive and negative return change days
display_raw_return_table(top_positive_return_nasdaq, "NASDAQ Headlines", "Positive Return Change")
display_raw_return_table(top_positive_return_sp500, "S&P500 Headlines", "Positive Return Change")

display_raw_return_table(top_negative_return_nasdaq, "NASDAQ Headlines", "Negative Return Change")
display_raw_return_table(top_negative_return_sp500, "S&P500 Headlines", "Negative Return Change")


def plot_positive_return_trends(df, index_name, color_return='green', color_sentiment='blue'):
    # Filter only positive returns
    df = df[df['Return'] > 0].copy()
    df.sort_values(by='Date', inplace=True)
    df['Sentiment'] = df['Sentiment'].rolling(window=2).mean()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df['Date'], df['Return'] * 100, label='Positive Return (%)', color=color_return, marker='o')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Return (%)', color=color_return)
    ax1.tick_params(axis='y', labelcolor=color_return)
    ax1.set_title(f'{index_name} - Positive Returns and News Sentiment (Election Period)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['Sentiment'], label='Sentiment Score', color=color_sentiment, linestyle='--', marker='x')
    ax2.set_ylabel('Sentiment Score', color=color_sentiment)
    ax2.tick_params(axis='y', labelcolor=color_sentiment)

    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    fig.tight_layout()
    plt.show()

# Plot based on top_positive_return_nasdaq and top_positive_return_s&p500
plot_positive_return_trends(top_positive_return_nasdaq, 'NASDAQ')
plot_positive_return_trends(top_positive_return_sp500, 'S&P500')

def plot_top_headlines_bar(df, index_name):
    df = df.sort_values(by='Return', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        df['Headline'].str.slice(0, 60),  # truncate long headlines
        df['Return'] * 100,               # convert to percentage
        color='#3CB371'                   # mid green
    )
    ax.set_xlabel("Return (%)")
    ax.set_title(f"{index_name} – Top Headlines by Positive Return", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

plot_top_headlines_bar(top_positive_return_nasdaq, 'NASDAQ')
plot_top_headlines_bar(top_positive_return_sp500, 'S&P500')


# Getting top headlines by sentiment
top_positive = news_df.sort_values(by='Sentiment', ascending=False).head(10)
top_negative = news_df.sort_values(by='Sentiment', ascending=True).head(10)

# HTML report 
html_path = os.path.join(os.getcwd(), "market_sentiment_volatility_report.html")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(f"""
    <html>
    <head>
        <title>Market Volatility Sentiment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
                font-size: 14px;
            }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>📊 Market Volatility & Sentiment Analysis</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📈 NASDAQ Ridge Regression Betas</h2>
        <pre>{nasdaq_betas.to_string()}</pre>

        <h2>📉 S&P500 Ridge Regression Betas</h2>
        <pre>{sp500_betas.to_string()}</pre>

        <h3>📰 Top Positive Headlines</h3>
        <ul>
            {''.join(f'<li><strong>{row["Date"].date()}</strong>: {row["Headline"]}</li>' for _, row in top_positive.iterrows())}
        </ul>

        <h3>💥 Top Negative Headlines</h3>
        <ul>
            {''.join(f'<li><strong>{row["Date"].date()}</strong>: {row["Headline"]}</li>' for _, row in top_negative.iterrows())}
        </ul>

        <h3>🚨 Top Market Impact Headlines by Volatility</h3>
        <table>
            <tr>
                <th>Date</th>
                <th>Index</th>
                <th>Headline</th>
                <th>Sentiment</th>
                <th>Return</th>
                <th>Volatility</th>
            </tr>
            {''.join(f"<tr><td>{row['Date'].date()}</td><td>{row['Index']}</td><td>{row['Headline']}</td><td>{row['Sentiment']:.3f}</td><td>{row['Return']:.4f}</td><td>{row['Volatility']:.4f}</td></tr>" for _, row in top_combined.iterrows())}
        </table>

        <h3>🔍 Notes</h3>
        <p>This report analyzes the relationship between election-related news sentiment and market volatility, using a Ridge regression model with interaction effects and topic weighting. The headlines listed above were associated with the highest movements in volatility across NASDAQ and S&P500 during the 2024 U.S. election period.</p>
    </body>
    </html>
    """)

print(f"✅ HTML report saved to: {html_path}")

