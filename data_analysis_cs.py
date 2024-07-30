import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Load data from each file
df_cnbc = pd.read_csv('Output_Data/CS/CS_CNBC_prepped_20240514_203705_discrete.csv')
df_ft = pd.read_csv('Output_Data/CS/CS_FT_prepped_20240514_205217_discrete.csv')
df_nzz = pd.read_csv('Output_Data/CS/CS_NZZ_prepped_20240515_010305_discrete.csv')
df_srf = pd.read_csv('Output_Data/CS/CS_SRF_prepped_20240515_050307_discrete.csv')
df_nzz_2 = pd.read_csv('Output_Data/CS/CS_NZZ_prepped_correction_20240516_165837_discrete.csv')

# Function to extract and categorize sentiment
def extract_and_categorize_sentiment(df):
    df['sentiment_score'] = df['Sentiment'].str.extract(r"(-?\d\.?\d*)\.").astype(float)
    return df

# Process each dataframe
df_cnbc_processed = extract_and_categorize_sentiment(df_cnbc)
df_ft_processed = extract_and_categorize_sentiment(df_ft)
df_nzz_processed = extract_and_categorize_sentiment(df_nzz)
df_nzz_2_processed = extract_and_categorize_sentiment(df_nzz_2)
df_srf_processed = extract_and_categorize_sentiment(df_srf)

# Merge df_nzz and df_nzz_2
df_nzz_combined = pd.concat([df_nzz_processed, df_nzz_2_processed], ignore_index=True)

# Add a 'news_source' column to each dataframe
df_cnbc_processed['news_source'] = 'CNBC'
df_ft_processed['news_source'] = 'Financial Times'
df_nzz_combined['news_source'] = 'NZZ'
df_srf_processed['news_source'] = 'SRF'

# Combine all dataframes into one
df_combined = pd.concat([df_cnbc_processed, df_ft_processed, df_nzz_combined, df_srf_processed], ignore_index=True)

# Ensure 'text_date' is in the correct datetime format
df_combined['text_date'] = pd.to_datetime(df_combined['text_date'])

# Filter dates from 01.01.2019 to 17.03.2023
start_date = '2019-01-01'
end_date = '2023-03-17'
df_combined = df_combined[(df_combined['text_date'] >= start_date) & (df_combined['text_date'] <= end_date)]

# Group by news source and date, then calculate the mean sentiment score
df_grouped_by_source = df_combined.groupby(['news_source', pd.Grouper(key='text_date', freq='ME')])['sentiment_score'].mean().unstack(0)

# Function to plot bar chart with specific bins for each score
def plot_sentiment_distribution_specific(df, title):
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_score'].value_counts().sort_index()
    plt.bar(sentiment_counts.index, sentiment_counts.values, color='skyblue', edgecolor='black')
    plt.title(f'Specific Sentiment Score Distribution for {title}')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Function to categorize sentiment scores
def categorize_sentiment(score):
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'

# Function to plot standardized pie chart for sentiment categories
def plot_standardized_sentiment_pie_chart(df, title):
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    sentiment_counts = df['sentiment_category'].value_counts().reindex(['negative', 'neutral', 'positive'],
                                                                       fill_value=0)
    colors = ['red', 'gray', 'green']

    def autopct_func(pct):
        return f'{pct:.1f}%'

    plt.figure(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct=autopct_func,
        startangle=140,
        colors=colors,
        textprops={'fontsize': 16}  # Increase label font size
    )

    for autotext in autotexts:
        autotext.set_fontsize(16)  # Increase percentage font size

    plt.title(f'Sentiment Category Distribution for {title}', fontsize=17, pad=20)
    plt.axis('equal')
    plt.show()

# Plotting the charts
plot_sentiment_distribution_specific(df_cnbc_processed, 'CNBC')
plot_sentiment_distribution_specific(df_ft_processed, 'Financial Times')
plot_sentiment_distribution_specific(df_nzz_processed, 'NZZ')
plot_sentiment_distribution_specific(df_srf_processed, 'SRF')

plot_standardized_sentiment_pie_chart(df_cnbc_processed, 'CNBC')
plot_standardized_sentiment_pie_chart(df_ft_processed, 'Financial Times')
plot_standardized_sentiment_pie_chart(df_nzz_processed, 'NZZ')
plot_standardized_sentiment_pie_chart(df_srf_processed, 'SRF')


# Monthly Average Sentiment Score by News Source with distinct colors and line styles
plt.figure(figsize=(14, 7))
colors = ['blue', 'green', 'red', 'purple']
line_styles = ['-', '--', '-.', ':']
for i, column in enumerate(df_grouped_by_source.columns):
    plt.plot(df_grouped_by_source.index, df_grouped_by_source[column], label=column, color=colors[i], linestyle=line_styles[i])
plt.title('Monthly Average Sentiment Score by News Source')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.legend(title='News Source')
plt.grid(True)
plt.show()

# Vertical stacked plots for each news source
fig, axes = plt.subplots(nrows=len(df_grouped_by_source.columns), ncols=1, figsize=(10, 20), sharex=True)
for i, column in enumerate(df_grouped_by_source.columns):
    axes[i].plot(df_grouped_by_source.index, df_grouped_by_source[column], label=column, color=colors[i], linestyle=line_styles[i])
    axes[i].set_title(f'Monthly Average Sentiment Score for {column}')
    axes[i].set_ylabel('Avg Sentiment Score')
    axes[i].grid(True)
    axes[i].legend(loc='upper right')
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# Combined plot for Daily, Weekly, and Monthly sentiment scores
df_daily = df_combined.groupby(pd.Grouper(key='text_date', freq='D'))['sentiment_score'].mean()
df_weekly = df_combined.groupby(pd.Grouper(key='text_date', freq='W'))['sentiment_score'].mean()
df_monthly = df_combined.groupby(pd.Grouper(key='text_date', freq='ME'))['sentiment_score'].mean()
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18))
axes[0].plot(df_daily.index, df_daily, label='Daily', alpha=0.5)
axes[1].plot(df_weekly.index, df_weekly, label='Weekly', alpha=0.5)
axes[2].plot(df_monthly.index, df_monthly, label='Monthly', alpha=0.5)
for ax in axes:
    ax.legend()
    ax.grid(True)
plt.xlabel('Date')
plt.tight_layout()
plt.show()



# Heatmap of Daily Average Sentiment Scores
df_heatmap_data = df_daily.reset_index()
df_heatmap_data['year'] = df_heatmap_data['text_date'].dt.year
df_heatmap_data['week'] = df_heatmap_data['text_date'].dt.isocalendar().week
df_heatmap_data['day'] = df_heatmap_data['text_date'].dt.day_name()
heatmap_data = df_heatmap_data.pivot_table(index='day', columns=['year', 'week'], values='sentiment_score', aggfunc='mean')
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(day_order)
plt.figure(figsize=(18, 7))
sns.heatmap(heatmap_data, cmap='coolwarm', linewidths=.5, linecolor='black', cbar_kws={'label': 'Average Sentiment Score'})
plt.title('Heatmap of Daily Average Sentiment Scores')
plt.xlabel('Year and Week Number')
plt.ylabel('Day of Week')
plt.show()

# Heatmap of Weekly Average Sentiment Scores
df_weekly_heatmap_data = df_weekly.reset_index()
df_weekly_heatmap_data['year'] = df_weekly_heatmap_data['text_date'].dt.year
df_weekly_heatmap_data['week'] = df_weekly_heatmap_data['text_date'].dt.isocalendar().week
weekly_heatmap_data = df_weekly_heatmap_data.pivot(index='week', columns='year', values='sentiment_score')

plt.figure(figsize=(12, 10))
sns.heatmap(weekly_heatmap_data, cmap='coolwarm', linewidths=.5, linecolor='black', cbar_kws={'label': 'Average Sentiment Score'})
plt.title('Heatmap of Weekly Average Sentiment Scores')
plt.xlabel('Year')
plt.ylabel('Week Number')
plt.show()


# Calculate the 7-day moving average for the daily sentiment scores
df_daily_ma = df_daily.rolling(window=7).mean()

# Moving average plot for Daily Sentiment Scores
plt.figure(figsize=(12, 6))
plt.plot(df_daily.index, df_daily, label='Daily Sentiment', alpha=0.5)
plt.plot(df_daily_ma.index, df_daily_ma, label='7-Day Moving Average', color='red')
plt.title('Daily Sentiment with 7-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the 4-week moving average for the weekly sentiment scores
df_weekly_ma = df_weekly.rolling(window=4).mean()

# Moving average plot for Weekly Sentiment Scores
plt.figure(figsize=(12, 6))
plt.plot(df_weekly.index, df_weekly, label='Weekly Sentiment', alpha=0.5)
plt.plot(df_weekly_ma.index, df_weekly_ma, label='4-Week Moving Average', color='green')
plt.title('Weekly Sentiment with 4-Week Moving Average')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the 3-month moving average for the monthly sentiment scores
df_monthly_ma = df_monthly.rolling(window=3).mean()

# Moving average plot for Monthly Sentiment Scores
plt.figure(figsize=(12, 6))
plt.plot(df_monthly.index, df_monthly, label='Monthly Sentiment', alpha=0.5)
plt.plot(df_monthly_ma.index, df_monthly_ma, label='3-Month Moving Average', color='blue')
plt.title('Monthly Sentiment with 3-Month Moving Average')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.legend()
plt.grid(True)
plt.show()


# Function to plot the distribution of articles over time
def plot_article_distribution_over_time(df, title, freq='D'):
    df = df.copy()  # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df.loc[:, 'text_date'] = pd.to_datetime(df['text_date'])
    article_counts = df.groupby(pd.Grouper(key='text_date', freq=freq)).size()

    plt.figure(figsize=(14, 7))
    plt.plot(article_counts.index, article_counts, marker='o', linestyle='-')
    plt.title(f'Article Distribution Over Time ({title})', fontsize=16, pad=20)
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.show()


# Function to calculate and plot the weekly average number of articles
# Function to plot the distribution of articles over time
def plot_article_distribution_over_time(df, title, freq='D'):
    df = df.copy()  # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df.loc[:, 'text_date'] = pd.to_datetime(df['text_date'])
    article_counts = df.groupby(pd.Grouper(key='text_date', freq=freq)).size()

    plt.figure(figsize=(14, 7))
    plt.plot(article_counts.index, article_counts, marker='o', linestyle='-')
    plt.title(f'Article Distribution Over Time ({title})', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Articles', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.show()


# Function to calculate and plot the weekly average number of articles
def plot_weekly_average_article_count(df, title):
    df = df.copy()  # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df.loc[:, 'text_date'] = pd.to_datetime(df['text_date'])
    weekly_article_counts = df.groupby(pd.Grouper(key='text_date', freq='W')).size()
    weekly_average = weekly_article_counts.mean()

    plt.figure(figsize=(14, 7))
    plt.plot(weekly_article_counts.index, weekly_article_counts, marker='o', linestyle='-')
    plt.axhline(y=weekly_average, color='red', linestyle='--', label=f'Weekly Average: {weekly_average:.2f}',
                linewidth=2)
    plt.title(f'Weekly Article Count and Average for CS ({title})', fontsize=20, pad=20)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Number of Articles', fontsize=18)
    plt.legend(fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.show()

# Plotting the distribution of articles over time
plot_article_distribution_over_time(df_combined, 'All Sources - Daily', freq='D')
plot_article_distribution_over_time(df_combined, 'All Sources - Weekly', freq='W')
plot_article_distribution_over_time(df_combined, 'All Sources - Monthly', freq='ME')

# Plotting the weekly average article count
plot_weekly_average_article_count(df_combined, 'All Sources')

# for individual news sources
news_sources = ['CNBC', 'Financial Times', 'NZZ', 'SRF']
for source in news_sources:
    plot_article_distribution_over_time(df_combined[df_combined['news_source'] == source], f'{source} - Monthly',
                                        freq='ME')

# get number of articles between 01-01-2019 and 17-03-2023
article_counts_per_source = df_combined['news_source'].value_counts()
print(article_counts_per_source)
print(len(df_combined))