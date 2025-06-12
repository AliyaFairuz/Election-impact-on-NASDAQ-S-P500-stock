
#This file has the code for the event study for NASDAQ to see CAR

# Importing needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#UPLOAD NASDAQ100 Daily excel file
Tk().withdraw()  # Hide root window
filename = askopenfilename(title="Select NASDAQ Excel File")

nasdaq_df = pd.read_excel(filename, skiprows=1)
nasdaq_df.columns = ['date', 'close']
nasdaq_df['date'] = pd.to_datetime(nasdaq_df['date'])
nasdaq_df = nasdaq_df.sort_values('date')
nasdaq_df['return'] = nasdaq_df['close'].pct_change()

# 21 event dates that align with the S&P500 event dates
events = [
    ("2024-01-15", "Iowa Caucus", "Primary", "Positive"),
    ("2024-03-04", "Trump Indicted in NY", "Legal", "Negative"),
    ("2024-07-13", "Assassination Attempt at Rally", "Security", "Negative"),
    ("2024-07-15", "Trump Picks VP JD Vance", "Nomination", "Positive"),
    ("2024-07-18", "Accepts GOP Nomination", "Nomination", "Positive"),
    ("2024-09-11", "Debate with Kamala Harris", "Debate", "Neutral"),
    ("2024-09-15", "Shooting at Florida Rally", "Security", "Negative"),
    ("2024-10-27", "MSG Rally Controversy", "Campaign", "Negative"),
    ("2024-11-05", "Wins 2024 Election", "Election", "Positive"),
    ("2024-03-05", "Super Tuesday", "Primary", "Positive"),
    ("2024-03-12", "Biden Clinches Nomination", "Primary", "Positive"),
    ("2024-05-31", "Trump Convicted", "Legal", "Negative"),
    ("2024-06-27", "1st Biden-Trump Debate", "Debate", "Neutral"),
    ("2024-07-11", "Trump Sentencing", "Legal", "Negative"),
    ("2024-07-21", "Biden Withdraws", "Campaign", "Neutral"),
    ("2024-08-02", "Harris Nominated", "Nomination", "Positive"),
    ("2024-08-19", "DNC Begins", "Nomination", "Positive"),
    ("2024-09-10", "2nd Debate (ABC)", "Debate", "Neutral"),
    ("2024-10-01", "VP Debate", "Debate", "Neutral"),
    ("2024-10-07", "Trump Florida Rally", "Campaign", "Positive"),
    ("2024-10-22", "Trump Greensboro Rally", "Campaign", "Positive"),
]

event_df = pd.DataFrame(events, columns=["event_date", "event_name", "category", "sentiment"])
event_df["event_date"] = pd.to_datetime(event_df["event_date"])

# ±3-day CARs for NASDAQ
def compute_features(event_date, df, window=3, est_window=100):
    df = df.sort_values('date').reset_index(drop=True)
    idx = df.index[df['date'] == event_date]
    if len(idx) == 0 or idx[0] < est_window or idx[0] + window >= len(df):
        return None
    idx = idx[0]
    mu = df.iloc[idx - est_window:idx]['return'].mean()
    vol = df.iloc[idx - est_window:idx]['return'].std()
    car = (df.iloc[idx - window:idx + window + 1]['return'] - mu).sum()
    return {'CAR': car, 'volatility': vol}

records = []
for _, row in event_df.iterrows():
    features = compute_features(row['event_date'], nasdaq_df, window=3)
    if features:
        records.append({
            'event_date': row['event_date'],
            'event_name': row['event_name'],
            'category': row['category'],
            'sentiment': row['sentiment'],
            **features
        })

df_model = pd.DataFrame(records)

#Adding model features based on event tone and type e.g. security implies the shooting
df_model['security_event'] = (df_model['category'] == 'Security').astype(int)
df_model['sentiment_positive'] = (df_model['sentiment'] == 'Positive').astype(int)
df_model['interaction'] = df_model['volatility'] * df_model['sentiment_positive']


# Fit OLS Regression
X = df_model[['volatility', 'security_event', 'sentiment_positive', 'interaction']]
X = sm.add_constant(X)
y = df_model['CAR']
model = sm.OLS(y, X).fit()

# For exporting summary as an HTML file
summary_html = model.summary().as_html()

# Plot CARs
df_model['label'] = df_model.apply(
    lambda row: f"{row['event_name']} ({row['event_date'].strftime('%Y-%m-%d')})", axis=1
)
df_sorted = df_model.sort_values(by='CAR', ascending=True)
colors = ['red' if val < 0 else 'green' for val in df_sorted['CAR']]

plt.figure(figsize=(10, 10))
plt.barh(df_sorted['label'], df_sorted['CAR'], color=colors)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.xlabel('Cumulative Abnormal Return (CAR) [±3-Day Window]')
plt.ylabel('Event')
plt.title('NASDAQ100 Reactions to 2024 Election Events')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# Save plot to file
plot_filename = "car_plot.png"
plt.savefig(plot_filename)
plt.close()

# HTML
html_content = f"""
<html>
<head><title>NASDAQ100 Event Study Report</title></head>
<body>
<h1>NASDAQ100 Event Study around 2024 Election Events</h1>

<h2>Regression Results</h2>
{summary_html}

<h2>CAR Plot</h2>
<img src="{plot_filename}" alt="CAR Plot">

</body>
</html>
"""

# Save to HTML file
with open("nasdaq_event_study.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ Output saved to nasdaq_event_study.html and car_plot.png!")


