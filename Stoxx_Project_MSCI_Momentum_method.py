"""
Author : Parthasarathi
Start Date : 19 November 2024
Finish Date : 22 November 2024
Porject : Portfolio construction based on the MSCI USA MOMENTUM INDEX methodology 

"""

import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from openpyxl import Workbook

"""
The code class is used to retrive the stocks from the bhavopy and sort the stocks.
The bhavcopy is taken from the data storage or if you dont have it, will retrive 
from the nse website using web scraping.

"""

class StockDataProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def process_and_save_stock_data(self):
        """
        Processes daily stock data from multiple CSV files in the input folder,
        filters for relevant data (e.g., SERIES == 'EQ'), and stores each stock's
        data in separate CSV files. Each stock's data is sorted by 'Date' and uses 'Date' as the index.
        """
        all_data = []

        # Iterate through all CSV files in the input folder
        for file_name in os.listdir(self.input_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.input_folder, file_name)
                try:
                    # Load the CSV into a DataFrame
                    df = pd.read_csv(file_path)

                    # Remove 'Unnamed: 13' column if it exists
                    if 'Unnamed: 13' in df.columns:
                        df = df.drop(columns=['Unnamed: 13'])

                    # Filter for SERIES == 'EQ'
                    if 'SERIES' in df.columns:
                        df = df[df['SERIES'] == 'EQ']

                    # Rename 'TIMESTAMP' to 'Date', convert to datetime, and sort
                    if 'TIMESTAMP' in df.columns:
                        df = df.rename(columns={'TIMESTAMP': 'Date'})
                        df['Date'] = pd.to_datetime(df['Date'])

                    # Append to the list of dataframes
                    all_data.append(df)

                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")

        # Combine all dataframes into one
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)

            # Get unique stock symbols
            stock_symbols = combined_df['SYMBOL'].unique()

            # Save data for each stock into separate CSV files
            for symbol in stock_symbols:
                stock_data = combined_df[combined_df['SYMBOL'] == symbol]

                # Sort data by 'Date' and set 'Date' as the index
                stock_data = stock_data.sort_values(by='Date')
                stock_data = stock_data.set_index('Date')

                # Save stock data to a CSV file in the output folder
                output_file = os.path.join(self.output_folder, f"{symbol}.csv")
                stock_data.to_csv(output_file)
                print(f"Saved data for {symbol} to {output_file}")

        else:
            print("No data found to process.")

#==========================================================================================================================

"""
The code logic going to take the stocks folder which contains daily data and use the corporate actions
csv and find the symbol purpose and date. and adjust the price or store the same close price.

main 
"""

class CorporateActionsProcessor:
    def __init__(self, corporate_actions_file):
        """
        Initialize the processor by loading the corporate actions CSV file.
        """
        self.corporate_actions = pd.read_csv(corporate_actions_file)
        self.corporate_actions['EX-DATE'] = pd.to_datetime(self.corporate_actions['EX-DATE'])

    def get_relevant_actions(self, symbol, date):
        """
        Get relevant corporate actions for the symbol and date.
        """
        return self.corporate_actions[
            (self.corporate_actions['SYMBOL'] == symbol) & 
            (self.corporate_actions['EX-DATE'] <= date)
        ]

    def adjust_close_price_and_volume(self, close_price, volume, purpose):
        """
        Adjust the close price and volume based on the corporate action purpose.
        """
        if 'split' in purpose.lower():
            ratio = self.extract_split_ratio(purpose)
            return close_price / ratio, volume * ratio  # Adjust both price and volume
        elif 'bonus' in purpose.lower():
            ratio = 1 + self.extract_bonus_ratio(purpose)
            return close_price / ratio, volume * ratio  # Adjust both price and volume
        elif 'dividend' in purpose.lower():
            # Dividends affect only the close price
            return close_price - self.extract_dividend_amount(purpose), volume
        return close_price, volume  # If no corporate action, return original values

    @staticmethod
    def extract_split_ratio(purpose):
        try:
            parts = purpose.split('from Rs. ')[1].split(' to Rs. ')
            return float(parts[0]) / float(parts[1])
        except (IndexError, ValueError):
            return 1

    @staticmethod
    def extract_bonus_ratio(purpose):
        try:
            ratio = purpose.split('Bonus Issue of ')[1]
            numerator, denominator = map(int, ratio.split(':'))
            return numerator / denominator
        except (IndexError, ValueError):
            return 0

    @staticmethod
    def extract_dividend_amount(purpose):
        try:
            # Sum up all "Rs X Per Share" values in the purpose string
            dividends = [float(d.split('Rs ')[1].split(' Per Share')[0]) for d in purpose.split('And') if 'Rs' in d]
            return sum(dividends)
        except (IndexError, ValueError):
            return 0


class StockDataAdjuster:
    def __init__(self, stock_folder, corporate_actions_processor, output_folder):
        self.stock_folder = stock_folder
        self.corporate_actions_processor = corporate_actions_processor
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def adjust_stock_files(self):
        """
        Adjust all stock files in the input folder.
        """
        for file_name in os.listdir(self.stock_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.stock_folder, file_name)
                self.adjust_single_stock(file_path)

    def adjust_single_stock(self, file_path):
        """
        Adjust a single stock file based on corporate actions.
        """
        try:
            stock_data = pd.read_csv(file_path)
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])

            if 'ADJ CLOSE' in stock_data.columns:
                stock_data = stock_data.drop(columns=['ADJ CLOSE'])  # Remove the existing 'ADJ CLOSE' column

            stock_data['ADJ CLOSE'] = stock_data['CLOSE']  # Initialize ADJ CLOSE with CLOSE
            stock_data['ADJ VOLUME'] = stock_data['TOTTRDQTY']  # Initialize ADJ VOLUME with TOTTRDQTY

            for index, row in stock_data.iterrows():
                symbol = row['SYMBOL']
                date = row['Date']
                close_price = row['CLOSE']
                volume = row['TOTTRDQTY']

                # Get corporate actions for the symbol and date
                relevant_actions = self.corporate_actions_processor.get_relevant_actions(symbol, date)

                # Apply adjustments if any actions exist
                for _, action in relevant_actions.iterrows():
                    purpose = action['PURPOSE']
                    close_price, volume = self.corporate_actions_processor.adjust_close_price_and_volume(close_price, volume, purpose)

                stock_data.at[index, 'ADJ CLOSE'] = close_price
                stock_data.at[index, 'ADJ VOLUME'] = volume

            # Save the adjusted data
            output_file = os.path.join(self.output_folder, os.path.basename(file_path))
            stock_data.to_csv(output_file, index=False)
            print(f"Adjusted and saved: {output_file}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")



#----------------------------------------------------------------------------------------------------------------------------------------------
"""
The class is used to filter the nse 500 stocks from the folder of 2000 stocks 
from the adjusted daily data
"""

class NSE500StockFilter:
    def __init__(self, input_folder, nse_500_file, output_folder):
        self.input_folder = input_folder
        self.nse_500_file = nse_500_file
        self.output_folder = output_folder

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Load NSE 500 symbols from the provided CSV
        self.nse_500_symbols = self.load_nse_500_symbols()

    def load_nse_500_symbols(self):
        """
        Load the NSE 500 stock symbols from the given CSV file.
        Assumes the CSV has a column named 'SYMBOL'.
        """
        try:
            nse_500_df = pd.read_csv(self.nse_500_file)
            return set(nse_500_df['Symbol'].dropna().unique())
        except Exception as e:
            print(f"Error loading NSE 500 file: {e}")
            return set()

    def filter_and_save_stocks(self):
        """
        Filters stocks from the input folder based on NSE 500 symbols
        and saves them into the output folder.
        """
        for file_name in os.listdir(self.input_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.input_folder, file_name)
                try:
                    # Load the stock data
                    df = pd.read_csv(file_path)

                    # Ensure the file contains the 'SYMBOL' column
                    if 'SYMBOL' in df.columns:
                        # Get the symbol from the data
                        symbol = df['SYMBOL'].iloc[0]

                        # Check if the symbol is in the NSE 500 list
                        if symbol in self.nse_500_symbols:
                            # Save the stock data to the output folder
                            output_file = os.path.join(self.output_folder, f"{symbol}.csv")
                            df.to_csv(output_file, index=False)
                            print(f"Saved {symbol} to {output_file}")
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")


#--------------------------------------------------------------------------------------------------------------------------------------

"""
The code is used to find the momentum calculated for the 500 stocks from the 2000 stocks
"""

class MomentumCalculator:
    def __init__(self, folder_path, risk_free_rate=0):
        """
        Initialize the MomentumCalculator.
        Args:
            folder_path (str): Path to the folder containing stock CSV files.
            risk_free_rate (float): Annualized risk-free rate to adjust momentum calculations.
        """
        self.folder_path = folder_path
        self.risk_free_rate = risk_free_rate / 252  # Convert annualized rate to daily

    def calculate_and_save(self, file):
        """
        Calculate momentum metrics for a single stock and save back to the same file.
        Args:
            file (str): File name of the stock CSV to process.
        """
        file_path = os.path.join(self.folder_path, file)
        df = pd.read_csv(file_path)

        # Ensure required columns are present
        required_columns = ['Date', 'SYMBOL', 'ADJ CLOSE', 'ADJ VOLUME']
        if not all(col in df.columns for col in required_columns):
            print(f"Skipping {file}: Missing required columns.")
            return

        # Clean and prepare data
        df = df[required_columns].copy()
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format
        df.sort_values(by='Date', inplace=True)  # Sort by Date

        # Calculate 6-month and 12-month momentum
        df['6M Momentum'] = (
            df['ADJ CLOSE'].pct_change(126) - self.risk_free_rate * 126
        )
        df['12M Momentum'] = (
            df['ADJ CLOSE'].pct_change(252) - self.risk_free_rate * 252
        )

        # Standardize momentum values into z-scores
        df['6M Z-Score'] = zscore(df['6M Momentum'], nan_policy='omit')
        df['12M Z-Score'] = zscore(df['12M Momentum'], nan_policy='omit')

        # Combine z-scores into a final momentum score
        df['Combined Z-Score'] = 0.5 * df['6M Z-Score'] + 0.5 * df['12M Z-Score']
        
        df.dropna(inplace=True)

        # Save the updated file back to the same location
        df.to_csv(file_path, index=False)
        print(f"Processed and saved: {file}")

    def process_all_files(self):
        """
        Process all stock files in the folder and update them with momentum metrics.
        """
        for file in os.listdir(self.folder_path):
            if file.endswith('.csv'):  # Process only CSV files
                self.calculate_and_save(file)


#---------------------------------------------------------------------------------------------------------------------------
""" 
After the momentum zscore found we have to rank the stocks on the each date

"""

class StockRanker:
    def __init__(self, folder_path, output_file, top_n=100):
        """
        Initialize the StockRanker.
        Args:
            folder_path (str): Path to the folder containing stock CSV files.
            output_file (str): Path to save the consolidated ranked stocks.
            top_n (int): Number of top-ranked stocks to include for each date.
        """
        self.folder_path = folder_path
        self.output_file = output_file
        self.top_n = top_n

    def rank_stocks(self):
        """
        Rank stocks by combined z-score for each date and save the top N stocks.
        """
        all_data = []
    
        for file in os.listdir(self.folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(self.folder_path, file)
                df = pd.read_csv(file_path)
    
                # Ensure necessary columns are present
                required_columns = ['Date', 'SYMBOL', 'Combined Z-Score', 'ADJ CLOSE', 'ADJ VOLUME']
                if all(col in df.columns for col in required_columns):
                    # Append only if the DataFrame is not empty after filtering
                    if not df.empty:
                        all_data.append(df[['Date', 'SYMBOL', 'Combined Z-Score', 'ADJ CLOSE', 'ADJ VOLUME']])
    
        # Combine all stock data into a single DataFrame
        combined_df = pd.concat(all_data, ignore_index=True)
    
        # Convert 'Date' to datetime format for proper sorting
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
        # Group by date and rank stocks
        ranked_data = []
        unique_dates = combined_df['Date'].unique()
    
        for date in unique_dates:
            # Filter data for the current date
            date_data = combined_df[combined_df['Date'] == date].copy()
    
            # Rank stocks by combined z-score (descending)
            date_data['Rank'] = date_data['Combined Z-Score'].rank(ascending=False, method='dense')
    
            # Sort by Rank to ensure ranks are in order
            date_data = date_data.sort_values(by='Rank')
    
            # Select top N stocks
            top_stocks = date_data.nsmallest(self.top_n, 'Rank')
    
            # Append to the ranked data
            ranked_data.append(top_stocks)
    
        # Combine ranked data for all dates
        final_ranking = pd.concat(ranked_data, ignore_index=True)
    
        # Sort final ranking by date and then by rank
        final_ranking = final_ranking.sort_values(by=['Date', 'Rank'])
    
        # Save the ranked data to a single CSV file
        final_ranking.to_csv(self.output_file, index=False)
        print(f"Ranked stocks saved to {self.output_file}")



#----------------------------------------------------------------------------------------------------------------------------

"""
Here we are going to rebalnce the stocks based on the ranks in monthly bases. Find
the returns and cumulative returns. 

"""

class MomentumRebalancer:
    def __init__(self, ranked_csv, daily_data_folder, transaction_cost=0.0005):
        """
        Initialize the MomentumRebalancer.
        Args:
            ranked_csv (str): Path to the CSV file containing ranked stocks by date.
            daily_data_folder (str): Path to the folder containing daily data CSVs for all stocks.
            transaction_cost (float): Transaction cost as a fraction (e.g., 0.0005 for 0.05%).
        """
        self.ranked_csv = ranked_csv
        self.daily_data_folder = daily_data_folder
        self.transaction_cost = transaction_cost
        self.ranked_data = None
        self.monthly_returns = None
        self.cumulative_returns = None

    def load_ranked_data(self):
        """
        Load the ranked stocks CSV file.
        """
        self.ranked_data = pd.read_csv(self.ranked_csv)
        self.ranked_data['Date'] = pd.to_datetime(self.ranked_data['Date'])  # Ensure Date is datetime
        self.ranked_data.sort_values(by='Date', inplace=True)  # Sort by Date for processing

    def get_daily_data(self, stock_symbol):
        """
        Load daily data for a specific stock from the folder.
        Args:
            stock_symbol (str): The stock symbol to load data for.
        Returns:
            pd.DataFrame: Daily data for the stock.
        """
        file_path = os.path.join(self.daily_data_folder, f"{stock_symbol}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Daily data for stock {stock_symbol} not found in {self.daily_data_folder}")
        
        daily_data = pd.read_csv(file_path)
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        return daily_data

    def calculate_monthly_returns(self, output_file_excel):
        """
        Calculate monthly returns using entry and exit prices for top 100 stocks.
        Save detailed stock-level data and monthly summary to separate sheets in an Excel file.
        Args:
            output_file_excel (str): Path to save the Excel file.
        """
        # Add a "Month" column to group data by month
        self.ranked_data['Month'] = self.ranked_data['Date'].dt.to_period('M')
        monthly_groups = self.ranked_data.groupby('Month')
    
        # Store monthly returns and detailed stock data
        monthly_returns = []
        stock_level_data = []
    
        for i, (month, group) in enumerate(monthly_groups):
            # Determine entry and exit dates
            first_day = group['Date'].min()
            if i == len(monthly_groups) - 1:  # Handle the last group
                last_day = group['Date'].max()  # Use the last available date in the current month
                print(f"Final month {month}: Closing all positions at {last_day}.")
            else:
                next_month_group = list(monthly_groups)[i + 1][1]
                last_day = next_month_group['Date'].max()
    
            # Filter stocks for the first trading day of the current month
            first_day_group = group[group['Date'] == first_day]
    
            # Calculate the total z-score for normalization
            total_z_score = first_day_group['Combined Z-Score'].sum()
    
            # Initialize list to store stock returns and weighted returns
            stock_returns = []
            weighted_returns = []
    
            for stock, z_score in zip(first_day_group['SYMBOL'], first_day_group['Combined Z-Score']):
                try:
                    # Load daily data for the stock
                    daily_data = self.get_daily_data(stock)
    
                    # Get entry and exit prices
                    entry_price = daily_data[daily_data['Date'] == first_day]['ADJ CLOSE'].values[0]
                    exit_price = daily_data[daily_data['Date'] == last_day]['ADJ CLOSE'].values[0]
    
                    # Adjust entry and exit prices for transaction costs
                    entry_price_with_cost = entry_price * (1 + self.transaction_cost)  # Add transaction cost to entry price
                    exit_price_with_cost = exit_price * (1 - self.transaction_cost)    # Subtract transaction cost from exit price
                    
                    # Calculate stock return with the adjusted prices
                    stock_return = (exit_price_with_cost - entry_price_with_cost) / entry_price_with_cost
                    stock_returns.append(stock_return)
    
                    # Normalize the z-score to calculate the weight
                    weight = z_score / total_z_score
    
                    # Calculate the weighted return
                    weighted_return = weight * stock_return
                    weighted_returns.append(weighted_return)
    
                    # Append stock-level data
                    stock_level_data.append({
                        'Month': month.start_time,
                        'SYMBOL': stock,
                        'Entry Date': first_day,
                        'Entry Price': entry_price,
                        'Exit Date': last_day,
                        'Exit Price': exit_price,
                        'Stock Return': stock_return,
                        'Weight': weight,
                        'Weighted Return': weighted_return
                    })
    
                except (FileNotFoundError, IndexError):
                    print(f"Data for stock {stock} missing for {first_day} or {last_day}, skipping...")
    
            # Calculate total monthly return as the sum of weighted returns
            total_monthly_return = sum(weighted_returns)
    
            # Append to monthly returns
            monthly_returns.append((month.start_time, total_monthly_return))
    
        # Convert monthly returns to DataFrame
        self.monthly_returns = pd.DataFrame(monthly_returns, columns=['Month', 'Monthly Portfolio Return'])
    
        # Calculate cumulative returns and add as a new column
        self.monthly_returns['Cumulative Portfolio Return'] = (
            (1 + self.monthly_returns['Monthly Portfolio Return']).cumprod() - 1
        )
    
        # Save results to Excel file with separate sheets
        with pd.ExcelWriter(output_file_excel, engine='openpyxl') as writer:
            # Convert stock-level data to a DataFrame
            stock_level_df = pd.DataFrame(stock_level_data)
    
            # Write stock-level data to a sheet
            stock_level_df.to_excel(writer, sheet_name='Stock-Level Data', index=False)
    
            # Write monthly returns and cumulative returns to a separate sheet
            self.monthly_returns.to_excel(writer, sheet_name='Monthly Returns', index=False)
    
        print(f"Results saved to {output_file_excel}")
        
    def calculate_portfolio_performance(self, output_file_excel):
        """
        Calculate portfolio performance metrics based on monthly and cumulative returns.
        Save these metrics as a separate sheet in the Excel file.
        Args:
            output_file_excel (str): Path to save the Excel file.
        """
        # Ensure 'Monthly Portfolio Return' and 'Cumulative Portfolio Return' exist
        if self.monthly_returns is None or 'Monthly Portfolio Return' not in self.monthly_returns.columns:
            raise ValueError("Monthly and cumulative returns are required for performance metrics.")
    
        # Extract monthly returns
        monthly_returns = self.monthly_returns['Monthly Portfolio Return'].values
    
        # 1. Calculate Maximum Drawdown
        cumulative_returns = (1 + self.monthly_returns['Cumulative Portfolio Return']).cumprod()  # Portfolio growth
        rolling_max = cumulative_returns.cummax()  # Rolling maximum value
        drawdown = (cumulative_returns - rolling_max) / rolling_max  # Drawdown
        max_drawdown = drawdown.min()  # Maximum drawdown
    
        # 2. Calculate Sharpe Ratio
        risk_free_rate = 0.0  # Assuming 0% risk-free rate
        excess_returns = monthly_returns - risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(12)  # Annualized Sharpe ratio
    
        # 3. Calculate Sortino Ratio
        downside_returns = monthly_returns[monthly_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(12) if downside_deviation != 0 else np.nan
    
        # 4. Calculate CAGR
        start_value = 1
        end_value = (1 + self.monthly_returns['Cumulative Portfolio Return'].iloc[-1])
        total_years = len(monthly_returns) / 12
        cagr = (end_value / start_value) ** (1 / total_years) - 1
    
        # 5. Calculate Volatility
        volatility = np.std(monthly_returns) * np.sqrt(12)  # Annualized volatility
    
        # 6. Calculate Calmar Ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
        # Create a dictionary for performance metrics
        performance_metrics = {
            'Maximum Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'CAGR': cagr,
            'Volatility': volatility,
            'Calmar Ratio': calmar_ratio
        }
    
        # Convert performance metrics to a DataFrame
        performance_df = pd.DataFrame(performance_metrics, index=['Portfolio Performance'])
    
        # Save to the existing Excel file
        with pd.ExcelWriter(output_file_excel, engine='openpyxl', mode='a') as writer:
            # Save performance metrics to a new sheet
            performance_df.to_excel(writer, sheet_name='Portfolio Performance', index=True)
    
        print(f"Performance metrics saved to 'Portfolio Performance' sheet in {output_file_excel}")


#---------------------------------------------------------------------------------------------------------------

"""
Output of the portfolio will be stored in the csv file

Main function  

"""

def process_stocks_pipeline():
    try:
        # Step 1: Process and Save Stock Data
        input_folder = "C:\\Users\\SONY\\Desktop\\Stoxx company\\2019-2023 Bhavcopy"
        output_folder = "C:\\Users\\SONY\\Desktop\\Stoxx company\\StockWiseData_sorted"
        print("Step 1: Processing and saving stock data...")
        processor = StockDataProcessor(input_folder, output_folder)
        processor.process_and_save_stock_data()
        print("Step 1 completed successfully.\n")
        
        # Step 2: Adjust Stock Data with Corporate Actions
        corporate_actions_file = "C:\\Users\\SONY\\Desktop\\Stoxx company\\CF-CA-equities-01-10-2019-to-31-10-2024.csv"
        stock_folder = "C:\\Users\\SONY\\Desktop\\Stoxx company\\StockWiseData_sorted"
        adjusted_output_folder = "C:\\Users\\SONY\\Desktop\\Stoxx company\\AdjustedStockData"
        print("Step 2: Adjusting stock data with corporate actions...")
        corporate_actions_processor = CorporateActionsProcessor(corporate_actions_file)
        stock_adjuster = StockDataAdjuster(stock_folder, corporate_actions_processor, adjusted_output_folder)
        stock_adjuster.adjust_stock_files()
        print("Step 2 completed successfully.\n")

        # Step 3: Filter NSE500 Stocks
        nse_500_file = "C:\\Users\\SONY\\Desktop\\Stoxx company\\ind_nifty500list (2).csv"
        adjusted_input_folder = "C:\\Users\\SONY\\Desktop\\Stoxx company\\AdjustedStockData"
        filtered_output_folder = "C:\\Users\\SONY\\Desktop\\Stoxx company\\NSE500Stocks_adjusted"
        print("Step 3: Filtering NSE500 stocks...")
        filterer = NSE500StockFilter(adjusted_input_folder, nse_500_file, filtered_output_folder)
        filterer.filter_and_save_stocks()
        print("Step 3 completed successfully.\n")

        # Step 4: Calculate Momentum Metrics
        print("Step 4: Calculating momentum metrics...")
        momentum_calculator = MomentumCalculator(filtered_output_folder, risk_free_rate=0.05)
        momentum_calculator.process_all_files()
        print("Step 4 completed successfully.\n")

        # Step 5: Rank Stocks
        ranked_output_file = "C:\\Users\\SONY\\Desktop\\Stoxx company\\Ranked_Stocks.csv"
        print("Step 5: Ranking stocks...")
        stock_ranker = StockRanker(filtered_output_folder, ranked_output_file, top_n=100)
        stock_ranker.rank_stocks()
        print("Step 5 completed successfully.\n")

        # Step 6: Rebalance Portfolio and Calculate Monthly Returns
        ranked_csv = r'C:\\Users\\SONY\\Desktop\\Stoxx company\\Ranked_Stocks.csv'
        daily_data_folder = r"C:\\Users\\SONY\\Desktop\\Stoxx company\\Codes for differtnt functions\\NSE500Stocks_adjusted"
        print("Step 6: Rebalancing portfolio and calculating monthly returns...")
        rebalancer = MomentumRebalancer(ranked_csv, daily_data_folder, transaction_cost=0.0005)
        rebalancer.load_ranked_data()
        rebalancer.calculate_monthly_returns("output_file.xlsx")
        print("Step 6 completed successfully.\n")

        # Step 7: Calculate Portfolio Performance
        print("Step 7: Calculating portfolio performance...")
        rebalancer.calculate_portfolio_performance("output_file.xlsx")
        print("Step 7 completed successfully.\n")

        print("All steps completed successfully!")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

# Run the pipeline
process_stocks_pipeline()
