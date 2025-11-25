import pandas as pd
import numpy as np
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class FinancialAnalyzer:
    """
    A class for performing financial and sentiment analysis, encapsulating
    data loading, sentiment calculation (using VADER), aggregation, and 
    correlation analysis.
    """
    
    def __init__(self):
        """Initializes the VADER Sentiment Analyzer."""
        print("--- Initializing FinancialAnalyzer and VADER Sentiment Analyzer ---")
        try:
            # Download VADER lexicon data (if not already present)
            nltk.download('vader_lexicon', quiet=True)
            self.analyzer = SentimentIntensityAnalyzer()
            print("VADER Sentiment Analyzer initialized and ready.")
        except Exception as e:
            self.analyzer = None
            print(f"ERROR: Could not initialize VADER analyzer: {e}")

    def calculate_vader_score(self, text):
        """
        Applies the VADER analyzer to a text and returns the compound score.
        
        Args:
            text (str): The news headline text.
        
        Returns:
            float: The compound VADER sentiment score, or 0.0 on error.
        """
        if not self.analyzer:
            return 0.0 # Return 0 if analyzer failed to initialize
            
        try:
            if pd.isna(text) or text is None:
                return 0.0
            return self.analyzer.polarity_scores(str(text))['compound']
        except Exception:
            return 0.0

    def aggregate_sentiment_by_day_and_stock(self, df, date_col='Date', stock_col='stock', sentiment_col='Sentiment_Score'):
        """
        Aggregates news sentiment data to calculate the mean sentiment score 
        for each stock on each day.
        
        Args:
            df (pd.DataFrame): DataFrame containing news and sentiment scores.
            # Other Args define column names
            
        Returns:
            pd.DataFrame: Aggregated DataFrame with 'Date', 'stock', and 'Sentiment_Score'.
        """
        df_agg = df.groupby([date_col, stock_col])[sentiment_col].mean().reset_index()
        df_agg.rename(columns={sentiment_col: 'Sentiment_Score'}, inplace=True)
        return df_agg

    def calculate_daily_returns(self, stock_df, date_col='Date', close_col='Close'):
        """
        Calculates the daily percentage return for a stock price series.
        
        Args:
            stock_df (pd.DataFrame): DataFrame with daily stock data.
            # Other Args define column names
            
        Returns:
            pd.DataFrame: Stock DataFrame with a new 'Daily_Return' column.
        """
        stock_df[date_col] = pd.to_datetime(stock_df[date_col])
        stock_df = stock_df[[date_col, close_col]].sort_values(by=date_col)
        stock_df['Daily_Return'] = stock_df[close_col].pct_change()
        stock_df.dropna(subset=['Daily_Return'], inplace=True)
        return stock_df

    def correlate_sentiment_with_returns(self, stock_ticker, stock_file_path, df_news_agg):
        """
        Performs the full integration and correlation analysis for a single stock.
        
        Args:
            stock_ticker (str): The ticker symbol (e.g., 'AAPL').
            stock_file_path (str): Base file path (e.g., 'AAPL.csv').
            df_news_agg (pd.DataFrame): The pre-aggregated sentiment DataFrame.

        Returns:
            float | None: The correlation coefficient, or None if an error occurs.
        """
        # Robust file path searching logic
        potential_paths = [
            stock_file_path,                                      
            os.path.join('../data', stock_file_path),             
            os.path.join(os.getcwd(), stock_file_path)            
        ]
        
        actual_stock_path = None
        for path in potential_paths:
            if os.path.exists(path):
                actual_stock_path = path
                break
                
        if actual_stock_path is None:
            print(f"[{stock_ticker}] WARNING: Stock file '{stock_file_path}' not found. Skipping.")
            return None

        try:
            # 1. Load Stock Data and Calculate Returns
            stock_df = pd.read_csv(actual_stock_path)
            stock_df_returns = self.calculate_daily_returns(stock_df)
            
            # Ensure dates are uniform for merging (Date objects)
            stock_df_returns['Date'] = pd.to_datetime(stock_df_returns['Date']).dt.date

            # 2. Filter Sentiment Data and ensure uniform dates
            sentiment_filtered_df = df_news_agg.loc[df_news_agg['stock'] == stock_ticker].copy()
            # Ensure the input df_news_agg date column is converted to date objects
            if not pd.api.types.is_object_dtype(sentiment_filtered_df['Date']) or not all(isinstance(d, pd.Timestamp) or isinstance(d, pd.NaT) for d in sentiment_filtered_df['Date']):
                sentiment_filtered_df['Date'] = pd.to_datetime(sentiment_filtered_df['Date']).dt.date
            sentiment_filtered_df = sentiment_filtered_df[['Date', 'Sentiment_Score']]


            # 3. Data Integration (Merging)
            merged_df = pd.merge(stock_df_returns, sentiment_filtered_df, on='Date', how='inner')

            data_points = len(merged_df)
            if data_points < 5:
                # Require at least 5 data points for meaningful correlation
                print(f"[{stock_ticker}] WARNING: Insufficient matching data points for correlation. (Points: {data_points})")
                return None
                
            # 4. Correlation Analysis
            correlation = merged_df['Sentiment_Score'].corr(merged_df['Daily_Return'])
            
            print(f"[{stock_ticker}] Correlation: {correlation:.4f} (Data Points: {data_points})")
            return correlation

        except Exception as e:
            print(f"[{stock_ticker}] An unexpected error occurred during correlation analysis: {e}")
            return None