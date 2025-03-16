from main import *
from Misc import *
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QListWidget, QAbstractItemView,
    QGridLayout, QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog, QTextEdit, QStackedWidget,QDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize the matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def agupdate_plot(self, data_dict, x_column, y_column, title="Plot", xlabel="X-axis", ylabel="Y-axis"):
        """
        Updates the plot to show data for multiple countries with a color key.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Iterate through the data dictionary to plot each country's data
        for country, df in data_dict.items():
            ax.plot(df[x_column], df[y_column], marker='o', label=country)

        # Add title, labels, and legend
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(title="Countries")
        ax.grid(alpha=0.5)
        self.canvas.draw()

    def update_plot(self, df, x_column, y_column, plot_type, title="Plot", xlabel="X-axis", ylabel="Y-axis"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if plot_type == "Bar":
            ax.bar(df[x_column], df[y_column], color='skyblue')
        elif plot_type == "Line":
            ax.plot(df[x_column], df[y_column], marker='o', color='blue')
        elif plot_type == "Scatter":
            ax.scatter(df[x_column], df[y_column], color='green', alpha=0.7)


        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.5)
        self.canvas.draw()


    def save_plot(self, filepath, file_format="png"):
        """
        Saves the current plot to a file.
        """
        self.figure.savefig(filepath, format=file_format)

class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Economic Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()
        inputs_layout = QHBoxLayout()

        country_label = QLabel("Select Country:")
        self.country_dropdown = QComboBox()
        self.country_dropdown.addItems(country_codes.keys())

        indicator_label = QLabel("Select Indicator:")
        self.indicator_dropdown = QComboBox()
        self.indicator_dropdown.addItems(indicator_codes.keys())

        start_label = QLabel("Start Year:")
        self.start_input = QLineEdit()

        end_label = QLabel("End Year:")
        self.end_input = QLineEdit()

        plot_type_label = QLabel("Select Plot Type:")
        self.plot_dropdown = QComboBox()
        self.plot_dropdown.addItems(["Bar", "Line", "Scatter"])

        inputs_layout.addWidget(country_label)
        inputs_layout.addWidget(self.country_dropdown)
        inputs_layout.addWidget(indicator_label)
        inputs_layout.addWidget(self.indicator_dropdown)
        inputs_layout.addWidget(start_label)
        inputs_layout.addWidget(self.start_input)
        inputs_layout.addWidget(end_label)
        inputs_layout.addWidget(self.end_input)
        inputs_layout.addWidget(plot_type_label)
        inputs_layout.addWidget(self.plot_dropdown)

        buttons_layout = QHBoxLayout()
        fetch_data_button = QPushButton("Fetch Data")
        fetch_data_button.clicked.connect(self.fetch_data)

        plot_button = QPushButton("Plot Data")
        plot_button.clicked.connect(self.plot_data)

        export_button = QPushButton("Export Plot/Data")
        export_button.clicked.connect(self.export_plot)

        buttons_layout.addWidget(fetch_data_button)
        buttons_layout.addWidget(plot_button)
        buttons_layout.addWidget(export_button)

        self.plot_widget = PlotWidget()
        self.data_table = QTableWidget()

        main_layout.addLayout(inputs_layout)
        main_layout.addLayout(buttons_layout)
        main_layout.addWidget(QLabel("Raw Data:"))
        main_layout.addWidget(self.data_table)
        main_layout.addWidget(self.plot_widget)

        self.setLayout(main_layout)
        self.data = None  # To store the retrieved data

    def fetch_data(self):
        try:
            country = self.country_dropdown.currentText()
            indicator = self.indicator_dropdown.currentText()
            start_year = int(self.start_input.text())
            end_year = int(self.end_input.text())

            retriever = Data_Retreival(country, indicator, start_year, end_year)
            self.data = retriever.merger()

            if not self.data.empty:
                self.data_table.setRowCount(len(self.data))
                self.data_table.setColumnCount(len(self.data.columns))
                self.data_table.setHorizontalHeaderLabels(self.data.columns)

                for row_idx, row in self.data.iterrows():
                    for col_idx, value in enumerate(row):
                        self.data_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))
            else:
                QMessageBox.warning(self, "No Data", "No data available for the selected parameters.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def plot_data(self):
        try:
            country = self.country_dropdown.currentText()
            indicator = self.indicator_dropdown.currentText()
            start_year = int(self.start_input.text())
            end_year = int(self.end_input.text())
            plot_type = self.plot_dropdown.currentText()

            if self.data is not None and not self.data.empty:
                self.plot_widget.update_plot(
                    self.data, x_column="Year", y_column="Values", plot_type=plot_type,
                    title=f"{indicator} in {country} ({start_year}-{end_year})",
                    xlabel="Year", ylabel=indicator
                )
            else:
                QMessageBox.warning(self, "No Data", "No data available for plotting.")
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", str(e))

    def export_plot(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Plot/Data", "", "PNG Files (*.png);;JPEG Files (*.jpeg);;CSV Files (*.csv)", options=options
        )
        if filepath:
            file_format = filepath.split(".")[-1]
            if file_format in ["png", "jpeg"]:
                self.plot_widget.save_plot(filepath, file_format=file_format)
            elif file_format == "csv":
                if self.data is not None and not self.data.empty:
                    try:
                        self.data.to_csv(filepath, index=False)
                        QMessageBox.information(self, "Export Successful", f"Data exported to {filepath}")
                    except Exception as e:
                        QMessageBox.critical(self, "Export Error", str(e))
                else:
                    QMessageBox.warning(self, "No Data", "No data to export.")

class Aggregate(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aggregate Data")
        self.setGeometry(100, 100, 1200, 800)

        # Layouts for the aggregate subwindow
        main_layout = QVBoxLayout()
        inputs_layout = QHBoxLayout()

        # Input for selecting countries (comma-separated)
        countries_label = QLabel("Select Countries:")
        self.countries_input = QLineEdit()

        # Dropdown for selecting an indicator
        indicators_label = QLabel("Select Indicator:")
        self.indicator_dropdown = QComboBox()
        self.indicator_dropdown.addItems(indicator_codes.keys())

        # Inputs for start and end years
        start_label = QLabel("Start Year:")
        self.start_input = QLineEdit()

        end_label = QLabel("End Year:")
        self.end_input = QLineEdit()

        # Button for aggregating data
        aggregate_button = QPushButton("Aggregate")
        aggregate_button.clicked.connect(self.aggregate_data)

        # Button for exporting the plot
        export_button = QPushButton("Export Plot")
        export_button.clicked.connect(self.export_plot)

        # Add widgets to the input layout
        inputs_layout.addWidget(countries_label)
        inputs_layout.addWidget(self.countries_input)
        inputs_layout.addWidget(indicators_label)
        inputs_layout.addWidget(self.indicator_dropdown)
        inputs_layout.addWidget(start_label)
        inputs_layout.addWidget(self.start_input)
        inputs_layout.addWidget(end_label)
        inputs_layout.addWidget(self.end_input)
        inputs_layout.addWidget(aggregate_button)
        inputs_layout.addWidget(export_button)

        # Plot widget to display aggregated data
        self.plot_widget = PlotWidget()

        # Add layouts and widgets to the main layout
        main_layout.addLayout(inputs_layout)
        main_layout.addWidget(self.plot_widget)

        self.setLayout(main_layout)

    def aggregate_data(self):
        """
        Aggregates data for multiple countries and displays it.
        """
        try:
            countries = [c.strip() for c in self.countries_input.text().split(",")]
            if len(countries) < 2:
                QMessageBox.warning(self, "Warning", "Please select at least two countries.")
                return

            indicator = self.indicator_dropdown.currentText()
            start_year = int(self.start_input.text())
            end_year = int(self.end_input.text())

            # Retrieve and prepare data for each country
            country_data = {}
            for country in countries:
                retriever = Data_Retreival(country, indicator, start_year, end_year)
                data = retriever.merger()
                if not data.empty:
                    country_data[country] = data

            if country_data:
                # Use the updated plot function to display data for all countries
                self.plot_widget.agupdate_plot(
                    country_data, x_column="Year", y_column="Values",
                    title=f"{indicator} Data ({start_year}-{end_year})",
                    xlabel="Year", ylabel=indicator
                )
            else:
                QMessageBox.warning(self, "Warning", "No data available for the selected countries and indicator.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def export_plot(self):
        """
        Exports the current aggregated plot to a file.
        """
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png);;JPEG Files (*.jpeg);;CSV Files (*.csv)", options=options
        )
        if filepath:
            file_format = filepath.split(".")[-1]
            if file_format in ["png", "jpeg"]:
                self.plot_widget.save_plot(filepath, file_format=file_format)
            elif file_format == "csv":
                QMessageBox.information(self, "Export", "Exporting CSV is not supported for plots.")






class MachineLearning(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # ML Type Selection
        ml_type_label = QLabel("Select Machine Learning Type:")
        self.ml_type_dropdown = QComboBox()
        self.ml_type_dropdown.addItems(["Regression", "Classification", "Clustering"])
        self.ml_type_dropdown.currentIndexChanged.connect(self.update_input_window)
        main_layout.addWidget(ml_type_label)
        main_layout.addWidget(self.ml_type_dropdown)

        # Stacked widgets for dynamic input forms
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Add specific input forms for each ML type
        self.add_regression_inputs()
        self.add_classification_inputs()
        self.add_clustering_inputs()

        # Output display
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        main_layout.addWidget(self.output)

        self.model = None

    def add_regression_inputs(self):
        """Inputs specific to regression models."""
        regression_layout = QVBoxLayout()

        # Country and Indicator Selection
        regression_layout.addWidget(QLabel("Country:"))
        self.country_dropdown = QComboBox()
        self.country_dropdown.addItems(country_codes.keys())
        regression_layout.addWidget(self.country_dropdown)

        regression_layout.addWidget(QLabel("Indicator:"))
        self.indicator_dropdown = QComboBox()
        self.indicator_dropdown.addItems(indicator_codes.keys())
        regression_layout.addWidget(self.indicator_dropdown)

        # Time Period Selection
        regression_layout.addWidget(QLabel("Time Period:"))
        self.start_year_input = QLineEdit()
        self.start_year_input.setPlaceholderText("Start Year")
        self.end_year_input = QLineEdit()
        self.end_year_input.setPlaceholderText("End Year")
        time_period_layout = QHBoxLayout()
        time_period_layout.addWidget(self.start_year_input)
        time_period_layout.addWidget(self.end_year_input)
        regression_layout.addLayout(time_period_layout)



        # Train and Evaluate Buttons
        train_button = QPushButton("Train Regression Model")
        train_button.clicked.connect(self.train_regression_model)
        evaluate_button = QPushButton("Evaluate Regression Model")
        evaluate_button.clicked.connect(self.evaluate_regression_model)

        regression_layout.addWidget(train_button)
        regression_layout.addWidget(evaluate_button)

        regression_widget = QWidget()
        regression_widget.setLayout(regression_layout)
        self.stacked_widget.addWidget(regression_widget)

    def add_classification_inputs(self):
        """Inputs specific to classification models."""
        classification_layout = QVBoxLayout()

        # Classification Options
        classification_layout.addWidget(QLabel("Select Classification Type:"))
        self.classification_dropdown = QComboBox()
        self.classification_dropdown.addItems([
            "Growth vs. Recession",
            "Economic Stability"
        ])
        classification_layout.addWidget(self.classification_dropdown)

        # Train and Evaluate Buttons
        train_button = QPushButton("Train Classification Model")
        train_button.clicked.connect(self.train_classification_model)
        evaluate_button = QPushButton("Evaluate Classification Model")
        evaluate_button.clicked.connect(self.evaluate_classification_model)

        classification_layout.addWidget(train_button)
        classification_layout.addWidget(evaluate_button)

        classification_widget = QWidget()
        classification_widget.setLayout(classification_layout)
        self.stacked_widget.addWidget(classification_widget)

    def add_clustering_inputs(self):
        """Inputs specific to clustering models."""
        clustering_layout = QVBoxLayout()

        # Clustering Options
        clustering_layout.addWidget(QLabel("Select Clustering Type:"))
        self.clustering_dropdown = QComboBox()
        self.clustering_dropdown.addItems([
            "Economic Development",
            "Environmental & Infrastructure",
            "Social Indicators"
        ])
        clustering_layout.addWidget(self.clustering_dropdown)

        # Country Selection
        clustering_layout.addWidget(QLabel("Select Countries (comma-separated):"))
        self.country_input = QLineEdit()
        self.country_input.setPlaceholderText("Enter countries (e.g., US, UK, China)")
        clustering_layout.addWidget(self.country_input)

        # Time Period Selection
        clustering_layout.addWidget(QLabel("Time Period:"))
        self.start_year_input = QLineEdit()
        self.start_year_input.setPlaceholderText("Start Year")
        self.end_year_input = QLineEdit()
        self.end_year_input.setPlaceholderText("End Year")
        time_period_layout = QHBoxLayout()
        time_period_layout.addWidget(self.start_year_input)
        time_period_layout.addWidget(self.end_year_input)
        clustering_layout.addLayout(time_period_layout)

        # Number of Clusters
        clustering_layout.addWidget(QLabel("Number of Clusters:"))
        self.num_clusters_input = QLineEdit()
        self.num_clusters_input.setPlaceholderText("Enter number of clusters (e.g., 3)")
        clustering_layout.addWidget(self.num_clusters_input)

        # Run Clustering Button
        train_button = QPushButton("Run Clustering")
        train_button.clicked.connect(self.run_clustering)
        clustering_layout.addWidget(train_button)

        # Visualize Clusters Button
        self.visualize_button = QPushButton("Visualize Clusters")
        self.visualize_button.setEnabled(False)  # Disable until clustering is completed
        self.visualize_button.clicked.connect(self.display_visualization_dialogCL)
        clustering_layout.addWidget(self.visualize_button)

        clustering_widget = QWidget()
        clustering_widget.setLayout(clustering_layout)
        self.stacked_widget.addWidget(clustering_widget)

    def update_input_window(self):
        """Update the input window based on selected ML type."""
        self.stacked_widget.setCurrentIndex(self.ml_type_dropdown.currentIndex())

    def train_regression_model(self):
        """Train a regression model using Data_Retreival and LinearRegressionModel."""
        try:
            # Get user input
            country = self.country_dropdown.currentText()
            indicator = self.indicator_dropdown.currentText()
            # start_year = int(self.start_year_input.text().strip())
            # end_year = int(self.end_year_input.text().strip())

            start_year = 1990
            end_year = 2020

            # Retrieve data using Data_Retreival
            data_retrieval = Data_Retreival(country, indicator, start_year, end_year)
            data = data_retrieval.merger()
            print(data)

            # Train LinearRegressionModel
            self.model = LinearRegressionModel(data)
            self.model.train()
            self.output.setText(f"Regression model trained successfully for {country}, {indicator}.")
        except Exception as e:
            self.output.setText(f"Error training regression model: {e}")

    def evaluate_regression_model(self):
        """Evaluate the regression model and visualize predictions."""
        try:
            if not self.model:
                self.output.setText("No regression model has been trained.")
                return

            # Get user input for prediction years
            forecast_years_input = QLineEdit()
            forecast_years_input.setPlaceholderText("Enter future years (comma-separated, e.g., 2025, 2030)")
            forecast_dialog = QWidget()
            forecast_dialog.setWindowTitle("Forecast Years")
            forecast_dialog.setGeometry(200, 200, 400, 200)
            forecast_layout = QVBoxLayout()
            forecast_layout.addWidget(forecast_years_input)
            forecast_button = QPushButton("Forecast")
            forecast_layout.addWidget(forecast_button)
            forecast_dialog.setLayout(forecast_layout)

            def make_forecast():
                try:
                    forecast_years = list(map(int, forecast_years_input.text().split(",")))
                    predicted_values, image_path = self.model.visualize(forecast_years)

                    # Display the image in the GUI
                    pixmap = QPixmap(image_path)
                    image_label = QLabel()
                    image_label.setPixmap(pixmap)
                    image_label.setScaledContents(True)

                    # Add the image to the main layout
                    self.output.setText(f"Forecasted values for {forecast_years}:\n{predicted_values}")
                    self.layout().addWidget(image_label)
                    forecast_dialog.close()
                except Exception as e:
                    self.output.setText(f"Error during forecasting: {e}")

            forecast_button.clicked.connect(make_forecast)
            forecast_dialog.show()
        except Exception as e:
            self.output.setText(f"Error evaluating regression model: {e}")

    def run_clustering(self):
        """Run KMeans clustering based on user input and display results."""
        try:
            # Get user inputs
            category = self.clustering_dropdown.currentText().lower().replace(" ", "_")
            countries = [country.strip() for country in self.country_input.text().split(",")]
            start_year = int(self.start_year_input.text())
            end_year = int(self.end_year_input.text())
            num_clusters = int(self.num_clusters_input.text())

            # Perform clustering
            self.clustering_model = KMeansClustering(start_year, end_year, num_clusters)
            self.clustered_data = self.clustering_model.cluster(countries, category)

            if self.clustered_data.empty:
                self.output.setText("No data available for clustering.")
                return

            # Enable visualization button
            self.visualize_button.setEnabled(True)

            # Display clustering summary
            cluster_summary = self.clustered_data.groupby(f"{category.capitalize()} Cluster")['Country'].apply(
                list).to_string()
            self.output.setText(f"Clustering completed successfully. Cluster assignments:\n\n{cluster_summary}")
            # self.output.setWordWrapMode(QTextOption.WrapAnywhere)


        except Exception as e:
            self.output.setText(f"Error during clustering: {e}")

    def display_visualization_dialogCL(self):
        """Open a dialog to select columns for visualizing clusters."""
        if not hasattr(self, 'clustered_data') or self.clustered_data.empty:
            self.output.setText("No clustered data available for visualization.")
            return

        # Dialog for visualization options
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle("Visualize Clusters")
        self.dialog.setGeometry(200, 200, 400, 300)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Select columns for visualization:"))

        # X-axis input
        x_label = QLineEdit()
        x_label.setPlaceholderText("X-axis column (e.g., GDP per capita)")
        layout.addWidget(QLabel("X-axis:"))
        layout.addWidget(x_label)

        # Y-axis input
        y_label = QLineEdit()
        y_label.setPlaceholderText("Y-axis column (e.g., Unemployment rate)")
        layout.addWidget(QLabel("Y-axis:"))
        layout.addWidget(y_label)

        # Cluster label input
        cluster_label = QLineEdit()
        cluster_label.setPlaceholderText("Cluster column (e.g., Economic_Development_Cluster)")
        layout.addWidget(QLabel("Cluster Label:"))
        layout.addWidget(cluster_label)

        # Visualize button
        visualize_button = QPushButton("Visualize")
        layout.addWidget(visualize_button)

        def visualize_clusters():
            try:
                # Get user input
                x = x_label.text()
                y = y_label.text()
                cluster_col = cluster_label.text()

                # Validate columns exist in data
                if x not in self.clustered_data.columns or y not in self.clustered_data.columns or cluster_col not in self.clustered_data.columns:
                    self.output.setText(f"Invalid columns: {x}, {y}, or {cluster_col}.")
                    return

                # Visualize clusters
                self.clustering_model.visualize_clusters(self.clustered_data, x, y, cluster_col)

                # Save and display the plot
                plt_path = "cluster_plot.png"
                plt.savefig(plt_path)
                plt.close()

                pixmap = QPixmap(plt_path)
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.setScaledContents(True)
                self.layout().addWidget(image_label)

                self.output.setText(f"Visualization generated successfully for {cluster_col}.")
            except Exception as e:
                self.output.setText(f"Error during visualization: {e}")

        visualize_button.clicked.connect(visualize_clusters)
        dialog.setLayout(layout)
        dialog.show()


    def add_classification_inputs(self):
        """Inputs specific to classification models."""
        classification_layout = QVBoxLayout()

        # Classification Options
        classification_layout.addWidget(QLabel("Select Classification Type:"))
        self.classification_dropdown = QComboBox()
        self.classification_dropdown.addItems(["Growth vs. Recession", "Economic Stability"])
        classification_layout.addWidget(self.classification_dropdown)

        # Country and Date Selection
        classification_layout.addWidget(QLabel("Country:"))
        self.country_dropdown = QComboBox()
        self.country_dropdown.addItems(country_codes.keys())
        classification_layout.addWidget(self.country_dropdown)

        classification_layout.addWidget(QLabel("Start Year:"))
        self.start_year_input = QLineEdit()
        self.start_year_input.setPlaceholderText("Start Year")
        classification_layout.addWidget(self.start_year_input)

        classification_layout.addWidget(QLabel("End Year:"))
        self.end_year_input = QLineEdit()
        self.end_year_input.setPlaceholderText("End Year")
        classification_layout.addWidget(self.end_year_input)

        # Train and Evaluate Buttons
        train_button = QPushButton("Train Classification Model")
        train_button.clicked.connect(self.train_classification_model)
        evaluate_button = QPushButton("Evaluate Classification Model")
        evaluate_button.clicked.connect(self.evaluate_classification_model)

        classification_layout.addWidget(train_button)
        classification_layout.addWidget(evaluate_button)

        classification_widget = QWidget()
        classification_widget.setLayout(classification_layout)
        self.stacked_widget.addWidget(classification_widget)

    def update_input_window(self):
        """Update the input window based on selected ML type."""
        self.stacked_widget.setCurrentIndex(self.ml_type_dropdown.currentIndex())

    def train_classification_model(self):
        """Train a classification model for Growth vs. Recession or Economic Stability."""
        try:
            # Get user input
            classification_type = self.classification_dropdown.currentText()
            country = self.country_dropdown.currentText()
            # start_year = int(self.start_year_input.text().strip())
            # end_year = int(self.end_year_input.text().strip())

            start_year = 2002
            end_year = 2024

            if classification_type == "Growth vs. Recession":
                task = "Growth/Recession"
                indicators = [
                    "Central government debt, total (% of GDP)",
                    "GDP growth (%)",
                    "Unemployment rate (% of total labor force)",
                    "Inflation rate (%)"
                ]
            elif classification_type == "Economic Stability":
                task = "Stability"
                indicators = [
                    "Central government debt, total (% of GDP)",
                    "Unemployment rate (% of total labor force)",
                    "Inflation rate (%)"
                ]
            else:
                self.output.setText("Invalid classification type selected.")
                return

            # Create a Classification instance
            self.classifier = Classification(country, start_year, end_year)

            # Set indicators as an attribute in the Classification object
            self.classifier.indicators = indicators  # Dynamically assign indicators
            self.classifier.target_column = task  # Dynamically assign the target column

            # Prepare data and train the model
            data = self.classifier.prepare_data(indicators)
            if data.empty:
                self.output.setText("No data available for training.")
                return

            if classification_type == "Growth vs. Recession":
                self.classifier.classify_growth_recession()
            elif classification_type == "Economic Stability":
                self.classifier.classify_stability()

            self.output.setText(f"{classification_type} classification model trained successfully.")

        except ValueError as ve:
            self.output.setText(f"Invalid input: {ve}")
        except Exception as e:
            self.output.setText(f"Error during classification model training: {e}")

    def evaluate_classification_model(self):
        """Display plots and evaluation metrics for the classification model."""
        try:
            if not hasattr(self, 'classifier'):
                self.output.setText("No classification model has been trained.")
                return

            # Ensure that indicators and target column are available
            if not hasattr(self.classifier, 'indicators') or not hasattr(self.classifier, 'target_column'):
                self.output.setText("Missing indicators or target column for evaluation.")
                return

            # Prepare data if not already prepared
            data = self.classifier.prepare_data(self.classifier.indicators)
            if data.empty:
                self.output.setText("No data available for evaluation.")
                return

            # Generate and display Feature Importance plot
            self.show_plot_window(
                lambda: self.classifier.plot_feature_importance(
                    self.classifier.model, self.classifier.indicators, title="Feature Importance"
                ),
                title="Feature Importance",
                description="Feature importance for the trained model."
            )

            # Generate and display Confusion Matrix
            self.show_plot_window(
                lambda: self.classifier.plot_confusion_matrix(
                    self.classifier.y_test, self.classifier.y_pred,
                    labels=self.classifier.y_test.unique(),
                    title="Confusion Matrix"
                ),
                title="Confusion Matrix",
                description="Confusion matrix for model predictions."
            )

            # Generate and display Classification Over Time plot
            self.show_plot_window(
                lambda: self.classifier.plot_classification_over_time(
                    data=data,  # Pass the prepared data
                    classification_column=self.classifier.target_column,
                    title="Classification Over Time"
                ),
                title="Classification Over Time",
                description="Classification results visualized over time."
            )

        except Exception as e:
            self.output.setText(f"Error displaying plot: {e}")

    def show_plot_window(self, plot_function, title, description):
        """Show a plot in a separate window."""
        try:
            # Create a new window for the plot
            plot_window = QWidget()
            plot_window.setWindowTitle(title)
            plot_window.setGeometry(200, 200, 800, 600)
            layout = QVBoxLayout()

            # Add description
            layout.addWidget(QLabel(description))

            # Run the plot function to generate the plot
            figure = plot_function()  # Generate the plot

            # Embed the plot in the window
            canvas = FigureCanvas(figure)
            layout.addWidget(canvas)

            plot_window.setLayout(layout)
            plot_window.show()

        except Exception as e:
            self.output.setText(f"Error displaying plot: {e}")



class Statistics(QWidget):
    """
    A GUI for performing statistical tests using the StatisticalTests class.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statistical Tests")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        self.main_layout = QVBoxLayout()

        # Statistical Test Selection
        self.main_layout.addWidget(QLabel("Select Statistical Test:"))
        self.test_dropdown = QComboBox()
        self.test_dropdown.addItems([
            "Select a test...",
            "Pearson Correlation",
            "Spearman Correlation",
            "One-Sample T-Test",
            "Independent Samples T-Test",
            "Chi-Square Test",
            "Summary"
        ])
        self.test_dropdown.currentIndexChanged.connect(self.update_test_inputs)
        self.main_layout.addWidget(self.test_dropdown)

        # Inputs Layout (Initially Hidden)
        self.inputs_layout = QVBoxLayout()
        self.main_layout.addLayout(self.inputs_layout)

        # Run Test Button (Initially Hidden)
        self.run_test_button = QPushButton("Run Statistical Test")
        self.run_test_button.clicked.connect(self.run_statistical_test)
        self.run_test_button.setVisible(False)  # Initially hidden
        self.main_layout.addWidget(self.run_test_button)

        # Output Display
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.main_layout.addWidget(QLabel("Results:"))
        self.main_layout.addWidget(self.output)

        # Plot Display Area
        self.plot_widget = PlotWidget()
        self.main_layout.addWidget(QLabel("Visualization:"))
        self.main_layout.addWidget(self.plot_widget)

        self.setLayout(self.main_layout)
        self.stats_tests = None  # Placeholder for the StatisticalTests object

    def update_test_inputs(self):
        """
        Show the correct input fields based on the selected statistical test.
        """
        # Clear existing input fields
        for i in reversed(range(self.inputs_layout.count())):
            widget = self.inputs_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Hide run button initially
        self.run_test_button.setVisible(False)

        # Get the selected test
        test_type = self.test_dropdown.currentText()
        if test_type == "Select a test...":
            return  # Do nothing if no test is selected

        # Country Input (Always Required)
        self.inputs_layout.addWidget(QLabel("Enter Country:"))
        self.country_input = QLineEdit()
        self.country_input.setPlaceholderText("e.g., US, Germany")
        self.inputs_layout.addWidget(self.country_input)

        # **Start Year and End Year Inputs**
        self.inputs_layout.addWidget(QLabel("Start Year:"))
        self.start_year_input = QLineEdit()
        self.start_year_input.setPlaceholderText("e.g., 2000")
        self.inputs_layout.addWidget(self.start_year_input)

        self.inputs_layout.addWidget(QLabel("End Year:"))
        self.end_year_input = QLineEdit()
        self.end_year_input.setPlaceholderText("e.g., 2020")
        self.inputs_layout.addWidget(self.end_year_input)

        # Test-Specific Inputs
        if test_type in ["Pearson Correlation", "Spearman Correlation"]:
            self.inputs_layout.addWidget(QLabel("Enter Indicator 1:"))
            self.indicator1_input = QLineEdit()
            self.indicator1_input.setPlaceholderText("e.g., GDP growth (%)")
            self.inputs_layout.addWidget(self.indicator1_input)

            self.inputs_layout.addWidget(QLabel("Enter Indicator 2:"))
            self.indicator2_input = QLineEdit()
            self.indicator2_input.setPlaceholderText("e.g., Inflation rate (%)")
            self.inputs_layout.addWidget(self.indicator2_input)

        elif test_type == "One-Sample T-Test":
            self.inputs_layout.addWidget(QLabel("Enter Indicator:"))
            self.indicator1_input = QLineEdit()
            self.indicator1_input.setPlaceholderText("e.g., GDP growth (%)")
            self.inputs_layout.addWidget(self.indicator1_input)

            self.inputs_layout.addWidget(QLabel("Enter Population Mean:"))
            self.population_mean_input = QLineEdit()
            self.population_mean_input.setPlaceholderText("e.g., 2.0")
            self.inputs_layout.addWidget(self.population_mean_input)

        elif test_type == "Independent Samples T-Test":
            self.inputs_layout.addWidget(QLabel("Enter Second Country:"))
            self.country2_input = QLineEdit()
            self.country2_input.setPlaceholderText("e.g., UK")
            self.inputs_layout.addWidget(self.country2_input)

            self.inputs_layout.addWidget(QLabel("Enter Indicator:"))
            self.indicator1_input = QLineEdit()
            self.indicator1_input.setPlaceholderText("e.g., GDP growth (%)")
            self.inputs_layout.addWidget(self.indicator1_input)

        elif test_type == "Chi-Square Test":
            self.inputs_layout.addWidget(QLabel("Enter Indicator 1:"))
            self.indicator1_input = QLineEdit()
            self.indicator1_input.setPlaceholderText("e.g., GDP growth (%)")
            self.inputs_layout.addWidget(self.indicator1_input)

            self.inputs_layout.addWidget(QLabel("Enter Indicator 2:"))
            self.indicator2_input = QLineEdit()
            self.indicator2_input.setPlaceholderText("e.g., Inflation rate (%)")
            self.inputs_layout.addWidget(self.indicator2_input)

        # Show the "Run Test" button
        self.run_test_button.setVisible(True)

    def run_statistical_test(self):
        """
        Run the selected statistical test and display the results, including relevant plots.
        """
        try:
            # Get user inputs
            countries = [c.strip() for c in self.country_input.text().split(",")]
            start_year = self.start_year_input.text().strip()
            end_year = self.end_year_input.text().strip()

            # Validate year inputs
            if not start_year.isdigit() or not end_year.isdigit():
                self.output.setText("Please enter valid numeric start and end years.")
                return

            start_year = int(start_year)
            end_year = int(end_year)

            if start_year >= end_year:
                self.output.setText("Start Year must be less than End Year.")
                return

            # **Retrieve indicators correctly based on test type**
            test_type = self.test_dropdown.currentText()
            indicator1 = self.indicator1_input.text().strip() if hasattr(self, "indicator1_input") else None
            indicator2 = self.indicator2_input.text().strip() if hasattr(self, "indicator2_input") else None

            # Validate country input
            if not countries:
                self.output.setText("Please enter at least one country.")
                return

            # Initialize StatisticalTests object
            self.stats_tests = StatisticalTests(countries, [indicator1, indicator2], start_year, end_year)
            data = self.stats_tests.prepare_data()

            # Ensure data is prepared
            if not data:
                self.output.setText("No data available. Please check your inputs.")
                return

            # Retrieve available countries from the prepared data
            available_countries = list(data.keys())

            # **Run the selected test**
            if test_type in ["Pearson Correlation", "Spearman Correlation"]:
                if not indicator1 or not indicator2:
                    self.output.setText("Please enter both indicators.")
                    return

                country = countries[0]  # First country for correlation tests

                if test_type == "Pearson Correlation":
                    result = self.stats_tests.pearson_correlation(country, indicator1, indicator2)
                else:
                    result = self.stats_tests.spearman_correlation(country, indicator1, indicator2)

                # Display correlation plot
                self.display_correlation_plot(country, indicator1, indicator2)

            elif test_type == "One-Sample T-Test":
                if not indicator1:
                    self.output.setText("Please enter the indicator.")
                    return

                population_mean = self.population_mean_input.text().strip() if hasattr(self,
                                                                                       "population_mean_input") else None

                if not population_mean:
                    self.output.setText("Please enter the population mean.")
                    return

                country = countries[0]  # First country for one-sample t-test
                result = self.stats_tests.one_sample_t_test(country, indicator1, float(population_mean))

            elif test_type == "Independent Samples T-Test":
                if len(countries) < 2:
                    self.output.setText("Please enter two valid countries.")
                    return

                country1, country2 = countries[:2]

                if country1 not in available_countries or country2 not in available_countries:
                    self.output.setText(
                        f"Error: One or both countries are not found in the dataset.\n\nAvailable countries: {', '.join(available_countries)}"
                    )
                    return

                if not indicator1:
                    self.output.setText("Please enter the indicator.")
                    return

                result = self.stats_tests.independent_samples_t_test(country1, country2, indicator1)

            elif test_type == "Chi-Square Test":
                if not indicator1 or not indicator2:
                    self.output.setText("Please enter both indicators.")
                    return

                country = countries[0]  # First country for chi-square test
                result = self.stats_tests.chi_square_test(country, indicator1, indicator2)

                # Display chi-square plot
                self.display_chi_square_plot(country, indicator1, indicator2)

            elif test_type == "Summary":
                result = self.stats_tests.summary()
                result = "\n".join([f"{country}:\n{desc}" for country, desc in result.items()])

            else:
                self.output.setText("Invalid test type selected.")
                return

            # Display result
            self.output.setText(result)

        except ValueError as ve:
            self.output.setText(f"Value Error: {ve}")
        except Exception as e:
            self.output.setText(f"Error running statistical test: {e}")

    def display_correlation_plot(self, country, indicator1, indicator2):
        """
        Display a scatter plot for correlation analysis.
        """
        try:
            data = self.stats_tests.prepare_data()[country]
            self.plot_widget.figure.clear()
            ax = self.plot_widget.figure.add_subplot(111)
            ax.scatter(data[indicator1], data[indicator2], alpha=0.7)
            ax.set_title(f"Correlation Plot: {indicator1} vs {indicator2} ({country})")
            ax.set_xlabel(indicator1)
            ax.set_ylabel(indicator2)
            self.plot_widget.canvas.draw()
        except Exception as e:
            self.output.setText(f"Error displaying correlation plot: {e}")

    def display_chi_square_plot(self, country, indicator1, indicator2):
        """
        Display a heatmap for the chi-square test.
        """
        try:
            data = self.stats_tests.prepare_data()[country]
            contingency_table = pd.crosstab(data[indicator1], data[indicator2])
            self.plot_widget.figure.clear()
            ax = self.plot_widget.figure.add_subplot(111)
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Chi-Square Contingency Table: {indicator1} vs {indicator2} ({country})")
            ax.set_xlabel(indicator2)
            ax.set_ylabel(indicator1)
            self.plot_widget.canvas.draw()
        except Exception as e:
            self.output.setText(f"Error displaying chi-square plot: {e}")

class MenuScreen(QWidget):
    def __init__(self, open_dashboard, open_aggregate, open_ml, open_stats):
        super().__init__()
        self.setFixedSize(800, 400)

        # Layout for the menu screen
        layout = QGridLayout()

        # Dashboard section
        dashboard_label = QLabel("Dashboard")
        dashboard_label.setAlignment(Qt.AlignCenter)
        dashboard_image = QLabel()
        dashboard_image.setPixmap(QPixmap("dashboard.png").scaled(200, 150, Qt.KeepAspectRatio))
        dashboard_image.setAlignment(Qt.AlignCenter)
        dashboard_image.mousePressEvent = lambda _: open_dashboard()

        # Aggregate section
        aggregate_label = QLabel("Aggregate")
        aggregate_label.setAlignment(Qt.AlignCenter)
        aggregate_image = QLabel()
        aggregate_image.setPixmap(QPixmap("data.png").scaled(200, 150, Qt.KeepAspectRatio))
        aggregate_image.setAlignment(Qt.AlignCenter)
        aggregate_image.mousePressEvent = lambda _: open_aggregate()

        # Machine Learning section
        ml_label = QLabel("Machine Learning")
        ml_label.setAlignment(Qt.AlignCenter)
        ml_image = QLabel()
        ml_image.setPixmap(QPixmap("world.png").scaled(200, 150, Qt.KeepAspectRatio))
        ml_image.setAlignment(Qt.AlignCenter)
        ml_image.mousePressEvent = lambda _: open_ml()

        # Statistics section
        stats_label = QLabel("Statistics")
        stats_label.setAlignment(Qt.AlignCenter)
        stats_image = QLabel()
        stats_image.setPixmap(QPixmap("world.png").scaled(200, 150, Qt.KeepAspectRatio))
        stats_image.setAlignment(Qt.AlignCenter)
        stats_image.mousePressEvent = lambda _: open_stats()

        # Add sections to the layout
        layout.addWidget(dashboard_image, 0, 0)
        layout.addWidget(dashboard_label, 1, 0)
        layout.addWidget(aggregate_image, 0, 1)
        layout.addWidget(aggregate_label, 1, 1)
        layout.addWidget(ml_image, 0, 2)
        layout.addWidget(ml_label, 1, 2)
        layout.addWidget(stats_image, 0, 3)
        layout.addWidget(stats_label, 1, 3)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Economic Dashboard")
        self.setGeometry(100, 100, 800, 400)

        # Set the main menu screen
        self.menu = MenuScreen(self.open_dashboard, self.open_aggregate, self.open_ml, self.open_stats)
        self.setCentralWidget(self.menu)
        self.windows = []

    def open_dashboard(self):
        """
        Opens the Dashboard subwindow.
        """
        dashboard = Dashboard()
        self.windows.append(dashboard)
        dashboard.show()

    def open_aggregate(self):
        """
        Opens the Aggregate subwindow.
        """
        aggregate = Aggregate()
        self.windows.append(aggregate)
        aggregate.show()

    def open_ml(self):
        """
        Opens the Machine Learning subwindow.
        """
        ml = MachineLearning()
        self.windows.append(ml)
        ml.show()

    def open_stats(self):
        """
        Opens the Statistics subwindow.
        """
        stats = Statistics()
        self.windows.append(stats)
        stats.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

