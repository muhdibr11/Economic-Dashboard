import pandas as pd
from Misc import *
import requests
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import pearsonr, spearmanr, ttest_1samp, ttest_ind, chi2_contingency
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from functools import lru_cache
import logging
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
NAME: Data retrival
Purpose: This class retieves data from world bank and imf api and removes erroneous/null data
 """

class DataRetrieval:
    """
    A class that retrieves data from World Bank and IMF APIs and handles data cleaning.
    """
    
    def __init__(self, country_name, indicator_name, start_date, end_date):
        if country_name not in country_codes:
            raise ValueError(f"Invalid country name: {country_name}")
        if indicator_name not in indicator_codes:
            raise ValueError(f"Invalid indicator name: {indicator_name}")
            
        self.country_code = country_codes[country_name]
        self.indicator_name = indicator_name
        self.start_date = int(start_date)
        self.end_date = int(end_date)
        self.description = None
        
        # Configure requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @lru_cache(maxsize=128)
    def wb_retrieval(self):
        """Retrieve data from World Bank API with caching and error handling."""
        try:
            wb_url = f"https://api.worldbank.org/v2/country/{self.country_code}/indicator/{indicator_codes[self.indicator_name][0]}"
            params = {
                'format': 'json',
                'date': f"{self.start_date}:{self.end_date}"
            }
            
            response = self.session.get(wb_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if not data or len(data) < 2 or not data[1]:
                logger.warning(f"No data returned from World Bank API for {self.indicator_name}")
                return None
                
            data_points = data[1]
            df = pd.DataFrame([
                {'Year': entry['date'], 'WB': entry['value']}
                for entry in data_points
                if entry['value'] is not None
            ])
            
            return df if not df.empty else None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving World Bank data: {str(e)}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing World Bank data: {str(e)}")
            return None

    @lru_cache(maxsize=128)
    def imf_retrieval(self):
        """Retrieve data from IMF API with caching and error handling."""
        imf_code = indicator_codes[self.indicator_name][1]
        if imf_code is None:
            return self._create_empty_imf_df()
            
        try:
            imf_url = f"https://www.imf.org/external/datamapper/api/v1/{imf_code}"
            dates = list(range(self.start_date, self.end_date + 1))
            params = {
                'periods': ','.join(map(str, dates))
            }
            
            response = self.session.get(f"{imf_url}/{self.country_code}", params=params)
            response.raise_for_status()
            
            data = response.json()
            indicator_data = data["values"].get(imf_code, {})
            country_data = indicator_data.get(self.country_code, {})
            
            if not country_data:
                logger.warning(f"No IMF data available for {self.indicator_name}")
                return self._create_empty_imf_df()
                
            df = pd.DataFrame([
                {'Year': year, 'IMF': value}
                for year, value in country_data.items()
                if value is not None
            ])
            
            return df if not df.empty else self._create_empty_imf_df()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving IMF data: {str(e)}")
            return self._create_empty_imf_df()
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing IMF data: {str(e)}")
            return self._create_empty_imf_df()

    def _create_empty_imf_df(self):
        """Create an empty DataFrame with NaN values for IMF data."""
        years = list(range(self.start_date, self.end_date + 1))
        return pd.DataFrame({'Year': years, 'IMF': np.nan})

    def correct(self, df):
        """Apply corrections to the data based on the indicator type."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        if self.indicator_name == "Population (Millions of people)":
            if 'WB' in df.columns:
                df['WB'] = df['WB'].astype(float) / 1_000_000
                
        elif self.indicator_name == "Gross GDP (Billions of US$)":
            if 'WB' in df.columns:
                df['WB'] = df['WB'].astype(float) / 1_000_000_000
                
        return df

    def merger(self):
        """Merge and process IMF and World Bank data."""
        imf_data = self.imf_retrieval()
        wb_data = self.wb_retrieval()
        
        if imf_data is None and wb_data is None:
            logger.error("Both IMF and World Bank data retrieval failed")
            return pd.DataFrame(columns=['Year', 'IMF', 'WB', 'Values'])
            
        try:
            # Convert Year to integer and handle missing data
            for df in [imf_data, wb_data]:
                if df is not None:
                    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
            
            # Merge available data
            if imf_data is not None and wb_data is not None:
                merged_df = pd.merge(imf_data, wb_data, on='Year', how='outer')
            elif imf_data is not None:
                merged_df = imf_data
            else:
                merged_df = wb_data
            
            # Apply corrections and calculate mean values
            merged_df = self.correct(merged_df)
            value_columns = [col for col in ['IMF', 'WB'] if col in merged_df.columns]
            if value_columns:
                merged_df['Values'] = merged_df[value_columns].mean(axis=1, skipna=True).round(4)
            
            return merged_df.sort_values('Year')
            
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            return pd.DataFrame(columns=['Year', 'IMF', 'WB', 'Values'])

    @staticmethod
    def aggregate_countries(country_data_list):
        """Aggregate data from multiple countries with improved error handling."""
        try:
            all_data = []
            for country_data in country_data_list:
                country, data = country_data
                if data is not None and isinstance(data, pd.DataFrame) and 'Values' in data.columns:
                    df = data.copy()
                    df['Country'] = country
                    all_data.append(df[['Country', 'Values', 'Year']])
            
            if all_data:
                aggregated_df = pd.concat(all_data, ignore_index=True)
                return aggregated_df.sort_values(['Country', 'Year'])
            
            logger.warning("No valid data to aggregate")
            return pd.DataFrame(columns=['Country', 'Values', 'Year'])
            
        except Exception as e:
            logger.error(f"Error aggregating country data: {str(e)}")
            return pd.DataFrame(columns=['Country', 'Values', 'Year'])




class Plot:
    """
    A class for creating visualizations with improved styling and error handling.
    """
    
    @staticmethod
    def _setup_plot_style():
        """Set up common plot styling."""
        plt.style.use('seaborn')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    @staticmethod
    def _format_axis(ax, title, xlabel, ylabel, rotation=45):
        """Apply common axis formatting."""
        ax.set_title(title, pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(True, alpha=0.3)
        
    @staticmethod
    def bar_plot(df, x_column, y_column, ax, title="Bar Plot", xlabel="X-axis", ylabel="Y-axis"):
        """Create a bar plot with error handling and improved styling."""
        try:
            Plot._setup_plot_style()
            
            if df.empty:
                logger.warning("Empty DataFrame provided for bar plot")
                return
                
            # Create bar plot with error handling for numeric conversion
            df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
            bars = ax.bar(df[x_column], df[y_column], color='skyblue', alpha=0.7)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
            
            Plot._format_axis(ax, title, xlabel, ylabel)
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating bar plot: {str(e)}")
            
    @staticmethod
    def line_plot(df, x_column, y_column, ax, title="Line Plot", xlabel="X-axis", ylabel="Y-axis"):
        """Create a line plot with error handling and improved styling."""
        try:
            Plot._setup_plot_style()
            
            if df.empty:
                logger.warning("Empty DataFrame provided for line plot")
                return
                
            # Create line plot with error handling for numeric conversion
            df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
            line = ax.plot(df[x_column], df[y_column], marker='o', color='blue', linewidth=2, markersize=6)
            
            # Add data points labels
            for x, y in zip(df[x_column], df[y_column]):
                ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
            
            Plot._format_axis(ax, title, xlabel, ylabel)
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating line plot: {str(e)}")
            
    @staticmethod
    def scatter_plot(df, x_column, y_column, ax, title="Scatter Plot", xlabel="X-axis", ylabel="Y-axis"):
        """Create a scatter plot with error handling and improved styling."""
        try:
            Plot._setup_plot_style()
            
            if df.empty:
                logger.warning("Empty DataFrame provided for scatter plot")
                return
                
            # Create scatter plot with error handling for numeric conversion
            df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
            df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
            
            scatter = ax.scatter(df[x_column], df[y_column], 
                               color='green', alpha=0.6, s=100)
            
            # Add trend line
            z = np.polyfit(df[x_column], df[y_column], 1)
            p = np.poly1d(z)
            ax.plot(df[x_column], p(df[x_column]), "r--", alpha=0.8)
            
            Plot._format_axis(ax, title, xlabel, ylabel)
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            
    @staticmethod
    def plot_multiple_countries(aggregated_df, title="Multiple Countries Plot", xlabel="Year", ylabel="Values"):
        """Plot multiple countries with improved styling and error handling."""
        try:
            Plot._setup_plot_style()
            
            if aggregated_df.empty:
                logger.warning("Empty DataFrame provided for multiple countries plot")
                return
                
            plt.figure(figsize=(12, 8))
            
            # Get unique countries and create color map
            countries = aggregated_df['Country'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(countries)))
            
            # Plot each country with unique color and style
            for country, color in zip(countries, colors):
                country_data = aggregated_df[aggregated_df['Country'] == country]
                plt.plot(
                    country_data['Year'],
                    country_data['Values'],
                    marker='o',
                    label=country,
                    color=color,
                    linewidth=2,
                    markersize=6
                )
            
            plt.title(title, pad=20, fontsize=14, fontweight='bold')
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.legend(title="Countries", loc='center left', bbox_to_anchor=(1.05, 0.5),
                      fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            
            # Add data point labels for better readability
            for country in countries:
                country_data = aggregated_df[aggregated_df['Country'] == country]
                for x, y in zip(country_data['Year'], country_data['Values']):
                    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points",
                               xytext=(0,10), ha='center', fontsize=8)
            
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Error creating multiple countries plot: {str(e)}")


"""
A class for fitting and predicting with a linear regression model on economic data.
"""

class LinearRegressionModel:
    def __init__(self, data):
        if data.empty or len(data) < 2:
            raise ValueError("Insufficient data for linear regression. At least two data points are required.")
        self.data = data
        self.model = None

    def train(self):
        years = self.data['Year'].values.reshape(-1, 1)
        values = self.data['Values'].values

        self.model = LinearRegression()
        self.model.fit(years, values)

    def predict(self, years_to_predict):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call `train()` first.")

        if isinstance(years_to_predict, int):
            years_to_predict = [years_to_predict]

        years_to_predict = np.array(years_to_predict).reshape(-1, 1)
        return self.model.predict(years_to_predict)


    def visualize(self, years_to_predict):
        years = self.data['Year'].values.reshape(-1, 1)
        values = self.data['Values'].values

        predicted_values = self.predict(years_to_predict)

        plt.figure(figsize=(10, 6))
        plt.scatter(years, values, color='blue', label='Actual Values')
        plt.plot(years, self.model.predict(years), color='red', label='Linear Regression')
        plt.scatter(years_to_predict, predicted_values, color='green', label='Predicted Values')

        for i, year in enumerate(years_to_predict):
            plt.text(year, predicted_values[i], f'{predicted_values[i]:.2f}', fontsize=9)

        plt.xlabel('Year')
        plt.ylabel('Economic Indicator Value')
        plt.title(f'Linear Regression Predictions')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.tight_layout()

        # Save the plot to a file
        image_path = 'regression_plot.png'
        plt.savefig(image_path)
        plt.close()  # Close the plot to prevent it from displaying in a pop-up window

        return dict(zip(years_to_predict, predicted_values)), image_path


"""
A class for performing KMeans clustering on economic data based on specific indicator categories.
"""
class KMeansClustering:
    INDICATOR_CATEGORIES = {
        "economic_development": [
            "GDP per capita (Current US$)",
            "GNI per capita (Current US$)",
            "Unemployment rate (% of total labor force)",
            "Poverty rate (% of population)",
            "Education expenditure (% of GDP)",
            "Health Expenditure (% of GDP)"
        ],
        "environmental_infrastructure": [
            "CO2 emission (kiloton)",
            "Access to electricity (% of population)",
            "Infrastructure investment"
        ],
        "social_indicators": [
            "Poverty rate (% of population)",
            "Income inequality (GINI index)",
            "Gender Equality (% of Woman in parliaments)"
        ]
    }

    def __init__(self, start_date: int, end_date: int, num_clusters: int = 3):
        self.start_date = start_date
        self.end_date = end_date
        self.num_clusters = num_clusters
        self.data_retriever = DataRetrieval

    def get_indicator_data(self, countries: list, indicators: list) -> pd.DataFrame:
        """
        Retrieve and combine indicator data for multiple countries.
        """
        country_data_list = []
        for country in countries:
            country_indicators_data = []
            for indicator in indicators:
                try:
                    retrieval = self.data_retriever(country, indicator, self.start_date, self.end_date)
                    data = retrieval.merger()
                    if not data.empty:
                        country_indicators_data.append(
                            data[['Year', 'Values']].set_index('Year').rename(columns={'Values': indicator})
                        )
                except Exception as e:
                    print(f"Error retrieving data for {country}, {indicator}: {e}")

            if country_indicators_data:
                country_data = pd.concat(country_indicators_data, axis=1)
                country_data['Country'] = country
                country_data_list.append(country_data)

        if country_data_list:
            return pd.concat(country_data_list, axis=0).reset_index()
        return pd.DataFrame()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in numeric columns with column means.
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df

    def cluster_data(self, data: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Perform KMeans clustering and add cluster labels to the DataFrame.
        """
        numeric_data = data.select_dtypes(include=['float64', 'int64'])
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=0)
        data[label] = kmeans.fit_predict(numeric_data)
        return data

    def cluster(self, countries: list, category: str) -> pd.DataFrame:
        """
        General method to cluster countries based on a specified category of indicators.
        """
        if category not in self.INDICATOR_CATEGORIES:
            raise ValueError(f"Invalid category. Choose from {list(self.INDICATOR_CATEGORIES.keys())}")

        indicators = self.INDICATOR_CATEGORIES[category]
        data = self.get_indicator_data(countries, indicators)
        if not data.empty:
            processed_data = self.preprocess_data(data)
            return self.cluster_data(processed_data, f"{category.capitalize()} Cluster")
        else:
            print("No data available for clustering.")
            return pd.DataFrame()

    def visualize_clusters(self, data: pd.DataFrame, x: str, y: str, cluster_label: str):
        """
        Visualize clusters in a scatter plot.
        """
        if x not in data.columns or y not in data.columns or cluster_label not in data.columns:
            raise ValueError(f"Columns {x}, {y}, or {cluster_label} not found in the dataset.")

        plt.figure(figsize=(10, 6))
        for cluster in data[cluster_label].unique():
            cluster_data = data[data[cluster_label] == cluster]
            plt.scatter(cluster_data[x], cluster_data[y], label=f"Cluster {cluster}")

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"Clustering: {cluster_label}")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()


"""
A class for performing classification of economic data (Growth/Recession, Stability) and plotting the results.
"""
class Classification:
    def __init__(self, country, start_date, end_date):
        self.country = country
        self.start_date = start_date
        self.end_date = end_date

    def retrieve_indicator_data(self, indicator_name):
        """Retrieve data for a specific indicator."""
        retrieval = DataRetrieval(self.country, indicator_name, self.start_date, self.end_date)
        return retrieval.merger()

    def prepare_data(self, indicators):
        """Retrieve and merge multiple indicators into a single DataFrame."""
        data_frames = []
        for indicator in indicators:
            df = self.retrieve_indicator_data(indicator)
            if not df.empty:
                df = df[['Year', 'Values']].rename(columns={'Values': indicator})
                data_frames.append(df.set_index('Year'))

        # Merge all indicator data on the 'Year' thing
        if data_frames:
            combined_df = pd.concat(data_frames, axis=1)
            combined_df.dropna(inplace=True)
            return combined_df
        else:
            print("Error: No data retrieved for classification.")
            return pd.DataFrame()

    def _growth_label(self, row):
        """Label rows as 'Growth' or 'Recession' based on GDP growth."""
        return 'Growth' if row["GDP growth (%)"] > 0 else 'Recession'

    def _stability_label(self, row):
        """Label rows as 'Stable' or 'Unstable' based on economic thresholds."""
        return 'Stable' if (row["Central government debt, total (% of GDP)"] < 60 and
                            row["Unemployment rate (% of total labor force)"] < 7 and
                            row["Inflation rate (%)"] < 5) else 'Unstable'

    @staticmethod
    def plot_feature_importance(model, feature_names, title="Feature Importance"):
        """Plot feature importance for a classification model."""
        importance = model.feature_importances_
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importance, y=feature_names)
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_test, y_pred, labels, title="Confusion Matrix"):
        """Plot the confusion matrix for the classification results."""
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues")
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_classification_over_time(data, classification_column, title="Economic Condition Over Time"):
        """Visualize the classification results over time."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=data.index, y=data[classification_column])
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Classification")
        plt.xticks(rotation=45)
        plt.show()

    def classify(self, indicators, target_column, label_function, model=None, test_size=0.2, random_state=42):
        """General method to classify data based on a target column."""
        data = self.prepare_data(indicators)
        if data.empty:
            print("No data available for classification.")
            return

        data[target_column] = data.apply(label_function, axis=1)
        X = data[indicators]
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


        if model is None:
            model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        accuracy = accuracy_score(y_test, y_pred)
        print(f"{target_column} Classification Accuracy: {accuracy:.2f}")


        self.plot_feature_importance(model, X.columns, title=f"Feature Importance for {target_column} Classification")


        self.plot_confusion_matrix(y_test, y_pred, labels=y.unique(), title=f"Confusion Matrix for {target_column} Classification")


        print(f"**Insights on {target_column} Classification**")
        print(f"The model achieved an accuracy of {accuracy:.2f}. Key indicators include:")
        for feature, importance in zip(X.columns, model.feature_importances_):
            print(f"- {feature}: {importance:.2f}")


        self.plot_classification_over_time(data, target_column, title=f"{target_column} Over Time")

    def classify_growth_recession(self):
        """Perform classification for Growth/Recession."""
        indicators = ["Central government debt, total (% of GDP)", "GDP growth (%)",
                      "Unemployment rate (% of total labor force)", "Inflation rate (%)"]
        self.classify(indicators=indicators, target_column="Growth/Recession", label_function=self._growth_label)

    def classify_stability(self):
        """Perform classification for Stability."""
        indicators = ["Central government debt, total (% of GDP)", "Unemployment rate (% of total labor force)",
                      "Inflation rate (%)"]
        self.classify(indicators=indicators, target_column="Stability", label_function=self._stability_label)





class StatisticalTests:
    """
    A class for performing statistical analyses on economic indicators across multiple countries.
    """

    def __init__(self, countries, indicators, start_date, end_date):
        self.countries = countries
        self.indicators = indicators
        self.start_date = start_date
        self.end_date = end_date
        self.prepared_data = None

    def retrieve_data(self, country, indicator):
        """
        Retrieve data for a specific country and indicator using the DataRetrieval class.
        """
        retrieval = DataRetrieval(country, indicator, self.start_date, self.end_date)
        return retrieval.merger()

    def prepare_data(self):
        """
        Prepares and caches combined indicator data for all specified countries.
        """
        if self.prepared_data is None:  # Cache data to avoid recomputation
            country_data = {}
            for country in self.countries:
                indicator_data = []
                for indicator in self.indicators:
                    df = self.retrieve_data(country, indicator)
                    if not df.empty:
                        df = df[['Year', 'Values']].rename(columns={'Values': f'{indicator}'})
                        indicator_data.append(df.set_index('Year'))
                if indicator_data:
                    combined_data = pd.concat(indicator_data, axis=1)
                    country_data[country] = combined_data.dropna()
            self.prepared_data = country_data
        return self.prepared_data

    def validate_inputs(self, country, *indicators):
        """
        Validates that the specified country and indicators exist in the prepared data.
        """
        data = self.prepare_data()
        if country not in data:
            raise ValueError(f"Country '{country}' not found in prepared data.")
        missing_indicators = [ind for ind in indicators if ind not in data[country].columns]
        if missing_indicators:
            raise ValueError(f"Indicators {missing_indicators} not found for country '{country}'.")

    def pearson_correlation(self, country, indicator1, indicator2):
        """
        Computes the Pearson correlation between two indicators for a specified country.
        """
        self.validate_inputs(country, indicator1, indicator2)
        data = self.prepare_data()[country]
        correlation, p_value = pearsonr(data[indicator1], data[indicator2])
        return f"Pearson Correlation between {indicator1} and {indicator2} in {country}: {correlation:.2f} (p-value: {p_value:.2f})"

    def spearman_correlation(self, country, indicator1, indicator2):
        """
        Computes the Spearman correlation between two indicators for a specified country.
        """
        self.validate_inputs(country, indicator1, indicator2)
        data = self.prepare_data()[country]
        correlation, p_value = spearmanr(data[indicator1], data[indicator2])
        return f"Spearman Correlation between {indicator1} and {indicator2} in {country}: {correlation:.2f} (p-value: {p_value:.2f})"

    def one_sample_t_test(self, country, indicator, population_mean):
        """
        Performs a one-sample t-test for an indicator in a specified country.
        """
        self.validate_inputs(country, indicator)
        data = self.prepare_data()[country]
        t_stat, p_value = ttest_1samp(data[indicator], population_mean)
        return f"One-sample T-test for {indicator} in {country} (population mean = {population_mean}): t-statistic = {t_stat:.2f}, p-value = {p_value:.2f}"

    def independent_samples_t_test(self, country1, country2, indicator):
        """
        Performs an independent samples t-test for an indicator between two countries.
        """
        self.validate_inputs(country1, indicator)
        self.validate_inputs(country2, indicator)
        data1 = self.prepare_data()[country1]
        data2 = self.prepare_data()[country2]
        t_stat, p_value = ttest_ind(data1[indicator], data2[indicator])
        return f"Independent Samples T-test for {indicator} between {country1} and {country2}: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f}"

    def chi_square_test(self, country, indicator1, indicator2):
        """
        Performs a chi-square test for independence between two indicators for a specified country.
        """
        self.validate_inputs(country, indicator1, indicator2)
        data = self.prepare_data()[country]
        contingency_table = pd.crosstab(data[indicator1], data[indicator2])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return f"Chi-square Test between {indicator1} and {indicator2} in {country}: chi2 = {chi2:.2f}, p-value = {p_value:.2f}"

    def summary(self):
        """
        Provides a summary of the prepared data.
        """
        return {country: data.describe() for country, data in self.prepare_data().items()}



#
# data_retrieval_uk = Data_Retreival('United Kingdom', 'Gross GDP (Billions of US$)', 2000, 2020)
# data_retrieval_usa = Data_Retreival('United States of America', 'Gross GDP (Billions of US$)', 2000, 2020)
#
# uk_data = ('United Kingdom', data_retrieval_uk.merger())
# usa_data = ('United States', data_retrieval_usa.merger())
#
# # Pass the list of country data to aggregate_countries
# aggregated_df = Data_Retreival.aggregate_countries([uk_data, usa_data])
# # print(aggregated_df)
#
# # Example: Creating a bar plot for a single DataFrame
# uk_data = data_retrieval_uk.merger()
# Plot.bar_plot(uk_data, x_column='Year', y_column='Values',
#                          title="UK GDP (2000-2020)", xlabel="Year", ylabel="GDP (Billions)")
#
#
#
# # Example: Line plot for the aggregated data
# uk_data = ('United Kingdom', data_retrieval_uk.merger())
# usa_data = ('United States', data_retrieval_usa.merger())
# aggregated_df = Data_Retreival.aggregate_countries([uk_data, usa_data])
# Plot.plot_multiple_countries(aggregated_df,
#                                         title="GDP Comparison (2000-2020)",
#                                         xlabel="Year", ylabel="GDP (Billions)")
#
#
# # Example usage: Linear regression
# data_retrieval_uk = Data_Retreival('United Kingdom', 'Gross GDP (Billions of US$)', 2000, 2020)
# uk_data = data_retrieval_uk.merger()
# lr_model = LinearRegressionModel(uk_data)
# lr_model.train()
# predicted_values = lr_model.visualize([2025, 2030])
# print(predicted_values)
#
# Example usage: Clustering
# kmeans = KMeansClustering(start_date=2000, end_date=2020, num_clusters=4)
# social_clusters = kmeans.cluster(countries=["United States of America", "India", "China", "Germany"], category="social_indicators")
# print(social_clusters)
# print(social_clusters.columns)
# kmeans.visualize_clusters(
#     social_clusters,
#     x='Poverty rate (% of population)',
#     y='Income inequality (GINI index)',
#     cluster_label="Social_indicators Cluster"
# )
#
#
# # Example: Classify Growth/Recession
# classifier = Classification(country="United States of America", start_date=2000, end_date=2020)
# classifier.classify_growth_recession()
#
# # Example: Classify Stability
# classifier.classify_stability()
#
#
# # Initialize the StatisticalTests object
# countries = ["United States of America", "Germany"]
# indicators = ["GDP growth (%)", "Inflation rate (%)", "Unemployment rate (% of total labor force)"]
# start_date = 2000
# end_date = 2020
#
# stats_tests = StatisticalTests(countries, indicators, start_date, end_date)
#
# # Prepare data
# stats_tests.prepare_data()
#
# # Pearson correlation
# print(stats_tests.pearson_correlation("United States of America", "GDP growth (%)", "Inflation rate (%)"))
#
# # Spearman correlation
# print(stats_tests.spearman_correlation("Germany", "Unemployment rate (% of total labor force)", "Inflation rate (%)"))
#
# # One-sample t-test
# print(stats_tests.one_sample_t_test("United States of America", "GDP growth (%)", population_mean=2.0))
#
# # Independent samples t-test
# print(stats_tests.independent_samples_t_test("United States of America", "Germany", "Inflation rate (%)"))
#
# # Chi-square test
# print(stats_tests.chi_square_test("Germany", "GDP growth (%)", "Unemployment rate (% of total labor force)"))
#
# # Summary of prepared data
# print(stats_tests.summary())
# #

