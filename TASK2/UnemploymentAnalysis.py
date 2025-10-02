import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment in India.csv")
df.columns = df.columns.str.strip()
df.rename(columns={
    "Estimated Unemployment Rate (%)": "Unemployment_rate",
    "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "Labour_Participation_Rate"
}, inplace=True)
df = df.dropna(subset=['Unemployment_rate', 'Region', 'Date'])
df = df.fillna(method='ffill')

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values(by="Date")

# SUMMARY STATISTICS
mean_rate = df['Unemployment_rate'].mean()
print(f"Mean Unemployment Rate: {mean_rate:.2f}%")

# TREND ANALYSIS AND COVID IMPACT
df_avg = df.groupby("Date")['Unemployment_rate'].mean().reset_index()

# Define COVID period
covid_start = pd.to_datetime('2020-03-01')
pre_covid_avg = df_avg[df_avg['Date'] < covid_start]['Unemployment_rate'].mean()
covid_avg = df_avg[df_avg['Date'] >= covid_start]['Unemployment_rate'].mean()

# Calculate % change
percent_change = ((covid_avg - pre_covid_avg) / pre_covid_avg) * 100
print(f"Average Unemployment Pre-COVID: {pre_covid_avg:.2f}%")
print(f"Average Unemployment During COVID: {covid_avg:.2f}%")
print(f"Percentage Increase During COVID: {percent_change:.2f}%")

# Plot trend
plt.figure(figsize=(12,6))
plt.plot(df_avg['Date'], df_avg['Unemployment_rate'], marker="o", label="National Avg Unemployment")
plt.axvline(covid_start, color='red', linestyle='--', label='Start of COVID (Mar 2020)')
plt.title("National Average Unemployment Rate: Pre-COVID vs COVID")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# RURAL VS URBAN COMPARISON
plt.figure(figsize=(8,6))
sns.boxplot(x='Area', y='Unemployment_rate', data=df, palette='viridis')
plt.title("Unemployment Rate Distribution: Rural vs Urban")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
