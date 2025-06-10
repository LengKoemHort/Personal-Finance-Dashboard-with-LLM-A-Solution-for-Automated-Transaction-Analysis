import gradio as gr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from main import process_transactions
from langchain_community.llms import Ollama
from pydantic import BaseModel, NonNegativeFloat, constr, ValidationError
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check dependency versions
logging.info(f"Matplotlib version: {matplotlib.__version__}")
logging.info(f"Seaborn version: {sns.__version__}")

# Pydantic model for transaction validation
class Transaction(BaseModel):
    Date: str
    Transaction_Details: constr(min_length=1)
    Money_In: NonNegativeFloat
    Money_Out: NonNegativeFloat
    Balance: float
    Money_In_Ccy: constr(pattern=r'^(USD|EUR|GBP)$') = "USD"
    Money_Out_Ccy: constr(pattern=r'^(USD|EUR|GBP)$') = "USD"
    Balance_Ccy: constr(pattern=r'^(USD|EUR|GBP)$') = "USD"

# Expected CSV columns
REQUIRED_COLUMNS = ['Date', 'Transaction Details', 'Money In', 'Money Out', 'Balance']
OPTIONAL_COLUMNS = ['Money In Ccy', 'Money Out Ccy', 'Balance Ccy']
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Load and validate CSV or DataFrame
def load_data(file):
    try:
        if isinstance(file, str):
            if not os.path.exists(file):
                logging.error(f"CSV file not found: {file}")
                return None, f"Error: CSV file not found at {file}."
            df = pd.read_csv(file)
        else:
            if file is None:
                logging.error("No file uploaded")
                return None, "Error: No file uploaded."
            df = pd.read_csv(file.name)

        # Preprocess NaN values for numeric columns
        for col in ['Money In', 'Money Out', 'Balance']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Validate DataFrame using Pydantic
        try:
            for _, row in df.iterrows():
                row_dict = {k.replace(' ', '_'): v for k, v in row.to_dict().items()}
                Transaction(**row_dict)
        except ValidationError as e:
            error_msg = (
                f"Error: Invalid data format.\n"
                f"Validation errors: {str(e)}\n"
                f"Required columns: {', '.join(REQUIRED_COLUMNS)}\n"
                f"Optional columns: {', '.join(OPTIONAL_COLUMNS)}\n"
                f"Example CSV format:\n"
                f"{','.join(REQUIRED_COLUMNS)}\n"
                f"2025-05-01,Supermarket Purchase,0,50,950"
            )
            logging.error(f"Pydantic validation error: {str(e)}")
            return None, error_msg

        # Add optional columns with default 'USD' if missing
        for col in OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = 'USD'
                logging.info(f"Added missing column {col} with default 'USD'")

        logging.info(f"CSV columns: {list(df.columns)}")
        logging.info("CSV loaded and validated successfully")
        return df, None
    except Exception as e:
        logging.error(f"Error loading CSV: {str(e)}")
        return None, f"Error loading CSV: {str(e)}"

# Visualizations
def create_visualizations(df):
    sns.set_style("whitegrid")

    pie_path = "category_pie.png"
    bar_path = "usd_bar.png"
    line_path = "balance_line.png"
    heatmap_path = "category_heatmap.png"

    try:
        plt.figure(figsize=(10, 8))
        category_counts = df['Normalized Category'].value_counts()
        plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
        plt.title("Transaction Distribution by Category")
        plt.tight_layout()
        plt.savefig(pie_path, dpi=300)
        plt.close()

        plt.figure(figsize=(10, 8))
        usd_by_type = df.groupby('Expense/Income')['USD'].sum()
        sns.barplot(x=usd_by_type.index, y=usd_by_type.values, palette="Set1")
        plt.title("Net USD by Expense/Income")
        plt.ylabel("USD")
        plt.xlabel("Type")
        plt.tight_layout()
        plt.savefig(bar_path, dpi=300)
        plt.close()

        plt.figure(figsize=(12, 8))
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        sns.lineplot(x=df['Date'], y=df['Balance'], marker='o', color='b')
        plt.title("Balance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Balance (USD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(line_path, dpi=300)
        plt.close()

        plt.figure(figsize=(12, 8))
        df['Month'] = df['Date'].dt.to_period('M')
        pivot_table = df.pivot_table(values='USD', index='Normalized Category', columns='Month', aggfunc='sum', fill_value=0)
        sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='.0f')
        plt.title("Net USD by Category and Month")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()

        logging.info("Visualizations generated successfully")
    except Exception as e:
        logging.error(f"Visualization error: {str(e)}")
        return None, None, None, None

    return pie_path, bar_path, line_path, heatmap_path

# Global store for transactions
processed_transactions = None

# Dashboard logic
def finance_dashboard(transactions_df, file):
    global processed_transactions

    input_csv = "temp_statement_report.csv"
    output_csv = "standardized_classified_transactions.csv"

    if file is not None:
        df, error = load_data(file)
        if error:
            return error, None, None, None, None, None, None
        df.to_csv(input_csv, index=False)
    elif transactions_df is not None and not transactions_df.empty:
        # Preprocess NaN values for numeric columns
        for col in ['Money In', 'Money Out', 'Balance']:
            if col in transactions_df.columns:
                transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce').fillna(0)
        # Validate manual input DataFrame
        try:
            for _, row in transactions_df.iterrows():
                row_dict = {k.replace(' ', '_'): v for k, v in row.to_dict().items()}
                Transaction(**row_dict)
        except ValidationError as e:
            error_msg = (
                f"Error: Invalid data format in manual input.\n"
                f"Validation errors: {str(e)}\n"
                f"Required columns: {', '.join(REQUIRED_COLUMNS)}\n"
                f"Optional columns: {', '.join(OPTIONAL_COLUMNS)}\n"
                f"Example format:\n"
                f"{','.join(REQUIRED_COLUMNS)}\n"
                f"2025-05-01,Supermarket Purchase,0,50,950"
            )
            logging.error(f"Pydantic validation error: {str(e)}")
            return error_msg, None, None, None, None, None, None
        # Add optional columns
        for col in OPTIONAL_COLUMNS:
            if col not in transactions_df.columns:
                transactions_df[col] = 'USD'
                logging.info(f"Added missing column {col} with default 'USD'")
        input_df = transactions_df[ALL_COLUMNS]
        input_df.to_csv(input_csv, index=False)
        df = input_df
    else:
        logging.error("No transaction data provided")
        return "Please enter transaction data or upload a CSV file.", None, None, None, None, None, None

    success, message = process_transactions(input_csv, output_csv)
    if not success:
        return message, None, None, None, None, None, None

    df, error = load_data(output_csv)
    if error:
        return error, None, None, None, None, None, None

    processed_transactions = df
    pie_chart, bar_chart, line_chart, heatmap_chart = create_visualizations(df)

    table_columns = ['Date', 'Transaction Details', 'Expense/Income', 'USD', 'Notes', 'Balance']
    table_df = df[table_columns]

    anomalies_text = ""
    if os.path.exists("anomalies.txt"):
        with open("anomalies.txt", "r") as f:
            anomalies_text = f"⚠️ Anomalies detected:\n{f.read()}"

    logging.info("Dashboard loaded successfully")
    return (
        f"✅ Finance Dashboard loaded successfully!\n{anomalies_text}",
        table_df,
        pie_chart,
        bar_chart,
        line_chart,
        heatmap_chart,
        output_csv
    )

# Chatbot logic
llm = Ollama(model="Financial-Assistant:latest")

def is_small_talk(user_input):
    greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
    return user_input.strip().lower() in greetings

def handle_chat(user_input, history):
    global processed_transactions
    history = history or []
    history.append({"role": "user", "content": user_input})

    if is_small_talk(user_input):
        response = "Hello! I'm your finance assistant, ready to help with your transactions. Try asking about your spending or trends!"
    elif processed_transactions is None or processed_transactions.empty:
        response = "Please upload or enter transaction data first so I can assist you."
    else:
        try:
            transactions_summary = processed_transactions[[
                'Date', 'Transaction Details', 'Normalized Category', 'USD', 'Expense/Income', 'Notes', 'Balance'
            ]]
            transactions_summary_str = transactions_summary.to_csv(index=False)

            recent_history = history[-4:]
            history_str = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in recent_history])

            prompt = f"""
You are an advanced financial assistant. Using the user's transaction data below (in CSV format), answer their question accurately and concisely. Provide insights, summaries, or specific details as needed. For example:
- Summarize spending by category or time period.
- Identify trends or anomalies.
- Offer simple financial advice based on the data.
- Answer specific queries about transactions.

Transaction Data (CSV):
{transactions_summary_str}

Recent Chat History:
{history_str}

User's Question: {user_input}

Respond in a clear, professional, and conversational tone. If the question is vague, suggest relevant questions or ask for clarification. Format lists or numbers clearly. If relevant, provide proactive insights (e.g., budgeting tips).
"""
            response = llm.invoke(prompt).strip()
            if not response:
                response = "Sorry, I couldn't generate a response. Please try rephrasing your question."
            if "spending" in user_input.lower() and "Dining" in processed_transactions['Normalized Category'].values:
                dining_spend = processed_transactions[processed_transactions['Normalized Category'] == 'Dining']['USD'].sum()
                if dining_spend < 0 and abs(dining_spend) > 100:
                    response += "\n\n💡 Insight: You're spending a lot on dining. Consider setting a monthly dining budget!"
        except Exception as e:
            logging.error(f"Chat error: {str(e)}")
            response = f"Error processing your request: {str(e)}. Please try again."

    history.append({"role": "assistant", "content": response})
    return history, history, ""

# Gradio UI
with gr.Blocks(title="Personal Finance Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💰 Personal Finance Dashboard")
    gr.Markdown("""
    Upload a CSV or enter transactions manually to analyze your finances. 
    **CSV Format**: Must include: Date, Transaction Details, Money In, Money Out, Balance.
    **Optional Columns**: Money In Ccy, Money Out Ccy, Balance Ccy (defaults to USD if missing).
    **Example CSV**:
    ```
    Date,Transaction Details,Money In,Money Out,Balance
    2025-05-01,Supermarket Purchase,0,50,950
    2025-05-02,Salary Deposit,1000,0,1950
    ```
    Chat with the bot to get insights, summaries, or answers about your transactions.
    Example questions: 
    - How much did I spend on groceries?
    - What are my top spending categories?
    - Are there any unusual transactions?
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            transaction_input = gr.Dataframe(
                label="Enter Transactions Manually",
                headers=ALL_COLUMNS,
                datatype=['str', 'str', 'number', 'number', 'number', 'str', 'str', 'str'],
                row_count=5,
                col_count=(8, 'fixed')
            )
            analyze_button = gr.Button("Analyze Transactions", variant="primary")

    status = gr.Textbox(label="Status", interactive=False)
    download_button = gr.File(label="Download Processed Transactions", visible=False)

    with gr.Tabs():
        with gr.Tab("Transaction Table"):
            table = gr.Dataframe(label="Processed Transactions")
        with gr.Tab("Spending by Category"):
            pie_chart = gr.Image(label="Transaction Distribution by Category")
        with gr.Tab("Net USD by Type"):
            bar_chart = gr.Image(label="Net USD by Expense/Income")
        with gr.Tab("Balance Over Time"):
            line_chart = gr.Image(label="Balance Over Time")
        with gr.Tab("Category Heatmap"):
            heatmap_chart = gr.Image(label="Net USD by Category and Month")

    analyze_button.click(
        fn=finance_dashboard,
        inputs=[transaction_input, file_input],
        outputs=[status, table, pie_chart, bar_chart, line_chart, heatmap_chart, download_button]
    ).then(
        fn=lambda x: gr.File(value=x, visible=True) if x else gr.File(visible=False),
        inputs=[download_button],
        outputs=[download_button]
    )

    chatbot_state = gr.State([])

    with gr.Row():
        chatbot = gr.Chatbot(label="Finance Assistant", height=400, type='messages')
        with gr.Column(scale=1):
            chat_input = gr.Textbox(label="Ask about your transactions", placeholder="E.g., 'How much did I spend on dining?'")
            send_button = gr.Button("Send", variant="secondary")

    send_button.click(handle_chat, inputs=[chat_input, chatbot_state], outputs=[chatbot, chatbot_state, chat_input])
    chat_input.submit(handle_chat, inputs=[chat_input, chatbot_state], outputs=[chatbot, chatbot_state, chat_input])

if __name__ == "__main__":
    demo.launch()