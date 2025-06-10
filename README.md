# Personal Finance Dashboard
A web-based application for analyzing financial transactions, featuring interactive visualizations, budget tracking, and a chatbot for financial insights. Built during my fourth-year Semester 2, this project demonstrates robust data validation and user-friendly financial analysis.
## Overview
The Personal Finance Dashboard allows users to upload CSV files or manually input financial transactions to visualize spending patterns, track budgets, and detect anomalies. Powered by a Gradio interface, Plotly charts, and an Ollama-based chatbot, it provides an intuitive platform for personal finance management.

## Features

- Interactive Visualizations: Dynamic Plotly charts (pie, bar, line, heatmap) for exploring transaction categories, net USD, and balance trends.
- Budget Tracking: Set monthly budgets per category with alerts for overbudget spending.
- Date Range Filtering: Filter transactions by start and end dates for targeted analysis.
- Chatbot Insights: Ask questions about spending trends or anomalies via an AI-powered financial assistant.
- Robust Validation: Pydantic ensures data integrity by validating transaction inputs.
- Anomaly Detection: Identifies unusual transactions based on statistical thresholds.

## Technologies

- Python: Core programming language.
- Gradio: Web-based UI for interactive dashboard.
- Pandas: Data processing and CSV handling.
- Plotly: Interactive data visualizations.
- Pydantic: Type-safe data validation.
- Ollama: LLM for transaction classification and chatbot functionality.
- Matplotlib/Seaborn: Used in exploratory analysis (notebook).

## Setup

- Clone the Repository:
- git clone https://github.com/LengKoemHort/Personal-Finance-Dashboard-with-LLM-A-Solution-for-Automated-Transaction-Analysis.git
- cd Personal_Finance_Dashboard


Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


## Install Dependencies:
pip install -r requirements.txt


## Install Ollama:

Follow instructions at Ollama to set up the LLM.
Pull the Financial-Assistant:latest model:ollama pull Financial-Assistant:latest




## Run the Application:
python app.py


Access the Gradio UI at http://localhost:7860.



## Usage

Upload CSV: Use statement_report.csv or a custom CSV with columns: Date, Transaction Details, Money In, Money Out, Balance (optional: Money In Ccy, Money Out Ccy, Balance Ccy).
Manual Input: Enter transactions via the Gradio DataFrame interface.
Set Budgets: Define monthly budgets for categories (e.g., Dining: $200).
Filter Dates: Select start and end dates to focus on specific periods.
Chatbot: Ask questions like “How much did I spend on groceries?” for insights.

## Example CSV Format:
Date,Transaction Details,Money In,Money Out,Balance
2025-05-01,Supermarket Purchase,0,50,950
2025-05-02,Salary Deposit,1000,0,1950

Repository Structure

app.py: Main application with Gradio UI and dashboard logic.
main.py: Backend processing for transaction classification and anomaly detection.
Classify_transactions.py: Supporting script for transaction classification.
classify_transactions.ipynb: Jupyter Notebook for exploratory data analysis.
statement_report.csv: Sample input dataset.
statement_transactions.csv: Additional sample dataset.
blueprint_financial_model.Modelfile: Configuration for Ollama LLM model.
.gitignore: Excludes temporary files and outputs.

Interactive dashboard with Plotly charts and budget alerts.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for enhancements or bug fixes.
License
MIT License. See LICENSE for details.

## Contact

GitHub: LengKoemHort
Email: lengkoemhort62@gmail.com

