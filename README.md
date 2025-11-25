# NewSentiment-week1
Financial Sentiment Correlation Analysis

Project Overview

This project performs a quantitative analysis to determine the correlation between daily news sentiment (derived from analyst headlines) and the corresponding daily stock price returns for major tech companies (AAPL, AMZN, GOOG, META, MSFT, NVDA). The goal is to investigate whether collective news sentiment serves as a leading or coincident indicator for market movement.

The entire analysis pipeline is encapsulated within the FinancialAnalyzer class for modularity and reusability, leading to a clean, end-to-end execution script.

Project Structure

.
├── data/
│   ├── AAPL.csv
│   ├── AMZN.csv
│   ├── ... (other stock CSVs)
│   └── raw_analyst_ratings.csv   <-- Source for news headlines
├── Notebooks/
│   ├── 2.0_Quantitative_Analysis.ipynb
│   └── 3.0_End_to_End_Analysis.ipynb <-- Final execution notebook (to be created)
└── financial_analysis.py           <-- Core analysis logic (FinancialAnalyzer Class)


Setup and Dependencies

This project requires Python and the following libraries:

pandas

numpy

nltk (specifically the VADER lexicon)

Installation

You can install the required packages using pip:

pip install pandas numpy nltk


The VADER lexicon will be automatically downloaded by the FinancialAnalyzer class upon initialization.

Running the Analysis

The core analysis is run through the 3.0_End_to_End_Analysis.ipynb notebook (or directly via a Python script if preferred).

Ensure Data is Present: Verify that all required stock .csv files and the raw_analyst_ratings.csv file are located within the data/ directory.

Execute the Notebook: Open and run all cells in 3.0_End_to_End_Analysis.ipynb.

The output will display the final correlation coefficients for each analyzed stock, along with the number of matched data points used in the calculation.

Core Analysis Steps

The analysis follows these steps, all managed by the FinancialAnalyzer class:

Sentiment Scoring: Each headline in raw_analyst_ratings.csv is assigned a sentiment score using the VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon.

Sentiment Aggregation: Sentiment scores are averaged by Date and Stock Ticker to get a single daily sentiment metric.

Return Calculation: Daily percentage returns are calculated for each stock using the historical closing prices.

Data Merge: The daily sentiment data and daily return data are merged on the Date column.

Correlation: The Pearson correlation coefficient is calculated between the Sentiment_Score and the Daily_Return.