import duckdb
import pandas as pd
import numpy as np

class StockSelectionMethod:
    def __init__(self, daily_db_path, universe_db_path, output_file):
        """Initialize the class with the database paths and output file."""
        self.daily_db_path = daily_db_path
        self.universe_db_path = universe_db_path
        self.output_file = output_file
        self.con_daily = duckdb.connect(database=daily_db_path, read_only=True)
        self.con_universe = duckdb.connect(database=universe_db_path, read_only=True)

        # Step 1: List tables in the universe database and filter only date tables for 2024 Q1
        print("Fetching available date tables from January to April 2024 in the universe database:")
        self.universe = self.list_date_tables()

    def main(self):
        all_results = []  # Initialize a list to store all results

        for date in self.universe:  # For each date in the universe
            print(f"Processing date: {date}")
            stocks_df = self.get_universe_stocks_by_date(date)  # Fetch stock list for the date
            
            for index, row in stocks_df.iterrows():  # Iterate through each stock
                isin = row['ISIN']
                symbol = row['SYMBOL']
                
                # Fetch stock data and calculate momentum score
                momentum_result = self.get_stock_data(isin, date)
                if momentum_result is not None:
                    momentum_result['ISIN'] = isin  # Add ISIN to the result
                    momentum_result['SYMBOL'] = symbol  # Add SYMBOL to the result
                    all_results.append(momentum_result)  # Append each result to the list

        # Concatenate all results into a single DataFrame
        results_df = pd.concat(all_results)

        # Sort the results by ISIN (stock) and then by DATE
        results_df['DATE'] = pd.to_datetime(results_df['DATE'])  # Ensure DATE is a datetime object for sorting
        results_df = results_df.sort_values(by=['ISIN', 'DATE'], ascending=[True, True])

        # Apply transition matrix to calculate probabilities of Buy/Sell states
        # results_df = self.apply_transition_matrix(results_df)

        # Save the DataFrame to a CSV file
        results_df.to_csv(self.output_file, index=False)
        print(f"Results saved and sorted by ISIN and DATE to {self.output_file}")

    def list_date_tables(self):
        """List only the tables that represent dates in January 2024."""
        tables_df = self.con_universe.sql("SHOW TABLES;").df()
        # Filter tables to only include those with names matching the pattern of January 2024 (YYYYMMDD format)
        date_tables = tables_df[tables_df['name'].str.startswith('202401')]  # Match tables for January 2024
        return date_tables['name'].tolist()

    def get_universe_stocks_by_date(self, date):
        """Fetch the stock symbols and ISINs from each date table."""
        query = f'SELECT SYMBOL, ISIN FROM "{date}";'
        result_df = self.con_universe.sql(query).df()
        result_df['DATE'] = date  # Add a column for the date (which is the table name)
        return result_df  # Return the result as a DataFrame

    def get_stock_data(self, isin, start_date):
        """Fetch and filter the stock data based on date range and calculate momentum score."""
        
        # Fetch data for the ISIN (assuming the table is named by the ISIN in the daily database)
        query = f"SELECT DATE, CLOSE FROM {isin};"
        table_data = self.con_daily.sql(query).df()

        # Convert 'DATE' column to datetime format
        table_data['DATE'] = pd.to_datetime(table_data['DATE'], errors='coerce')

        # Drop rows with invalid dates
        table_data.dropna(subset=['DATE'], inplace=True)

        # Ensure there are at least 40 days before the given start date
        date_filter = table_data['DATE'] < pd.to_datetime(start_date)
        table_data = table_data[date_filter].sort_values('DATE', ascending=True).tail(40)
        
        if len(table_data) < 40:
            print(f"Not enough data for 40 days from {start_date}. Skipping {isin}.")
            return None  # Return None if insufficient data

        # Filter for the last 24 days of data, excluding the extra 16 days
        table_data = table_data.sort_values('DATE', ascending=True).tail(24)

        # Calculate daily returns
        table_data['Return'] = table_data['CLOSE'].pct_change()

        # Calculate average returns for 6, 12, and 24 days
        avg_6 = table_data['Return'].head(6).mean()
        avg_12 = table_data['Return'].head(12).mean()
        avg_24 = table_data['Return'].mean()

        # Calculate momentum score as the sum of these averages
        momentum_score = avg_6 + avg_12 + avg_24

        # Classify momentum: 1 for "Buy" (positive), -1 for "Sell" (negative)
        momentum_class = 1 if momentum_score > 0 else -1

        # Return the results as a DataFrame
        result = pd.DataFrame({
            'DATE': [start_date],
            'CLOSE': [table_data['CLOSE'].iloc[0]],  # Get the CLOSE value for the start_date
            'avg_6': [avg_6],
            'avg_12': [avg_12],
            'avg_24': [avg_24],
            'Momentum Score': [momentum_score],
            'Momentum Class': [momentum_class]
        })
        
        return result

# Example usage:
daily_db_path = r"C:\Users\SONY\Downloads\files\files\daily.ddb"
universe_db_path = r"C:\Users\SONY\Downloads\files\files\universe_n500.ddb"
output_file = r"C:\Users\SONY\Downloads\files\results.csv"

# Initialize the stock selection method and fetch universe stocks by date
stock_selector = StockSelectionMethod(daily_db_path, universe_db_path, output_file)
stock_selector.main()
