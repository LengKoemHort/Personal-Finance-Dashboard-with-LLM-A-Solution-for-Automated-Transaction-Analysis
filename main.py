import pandas as pd
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import json
import logging
from functools import lru_cache
import matplotlib
import seaborn
from pydantic import BaseModel, NonNegativeFloat, constr, ValidationError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check dependency versions
logging.info(f"Matplotlib version: {matplotlib.__version__}")
logging.info(f"Seaborn version: {seaborn.__version__}")

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

# Constants
CATEGORIES = [
    "Groceries", "Utilities", "Entertainment", "Dining", "Rent", "Internet",
    "Transport", "Healthcare", "Shopping", "Investment", "Income", "Other"
]

CATEGORY_PROMPT = PromptTemplate.from_template("""
You are a financial assistant. Classify the transaction below into one of these categories: {categories}.
Provide a brief explanation for your choice.

Transaction: {transaction_details}
Money Out: {money_out}
Money In: {money_in}
Date: {date}

Respond in JSON format:
{
  "category": "CategoryName",
  "explanation": "Reason for classification"
}
""")

# LLM Wrapper
class OllamaLLM:
    def __init__(self, model="Financial-Assistant:latest"):
        self.model = model
        self.llm = Ollama(model=self.model)

    @lru_cache(maxsize=1000)
    def predict(self, prompt):
        try:
            response = self.llm.invoke(prompt).strip()
            logging.info(f"LLM response: {response}")
            return response
        except Exception as e:
            logging.error(f"LLM error: {str(e)}")
            return json.dumps({"error": f"LLM error: {str(e)}"})

# Fallback Rules
def classify_transaction(description):
    description = description.lower()
    rules = {
        "Groceries": ["7-eleven", "supermarket", "grocery", "mart"],
        "Utilities": ["electric", "water", "utility", "gas"],
        "Entertainment": ["movie", "cinema", "netflix", "spotify", "theater"],
        "Dining": ["restaurant", "dine", "coffee", "tea", "kfc", "pizza"],
        "Rent": ["rent", "apartment", "landlord"],
        "Transport": ["uber", "grab", "taxi", "bus", "train", "transport", "car", "vireak buntham"],
        "Healthcare": ["hospital", "clinic", "pharmacy", "doctor"],
        "Shopping": ["mall", "fashion", "shop", "shoes", "clothes"],
        "Investment": ["stock", "crypto", "investment", "nft", "gold"],
        "Income": ["salary", "income", "transfer in", "deposit", "bonus"],
        "Internet": ["metfone", "smart", "cellcard", "internet", "fiber", "wifi"]
    }
    for category, keywords in rules.items():
        if any(keyword in description for keyword in keywords):
            return category, f"Matched keyword: {keywords[0]}"
    return "Other", "No matching keywords found"

# Category Normalization
def normalize_category(transaction_details, llm_response):
    try:
        cleaned_response = llm_response.strip()
        logging.debug(f"Raw LLM response: {cleaned_response}")
        if not cleaned_response:
            logging.warning("Empty LLM response")
            return classify_transaction(transaction_details)
        response = json.loads(cleaned_response)
        if "error" in response:
            logging.warning(f"LLM error in response: {response['error']}")
            return classify_transaction(transaction_details)
        category = response.get("category", "").strip().title()
        explanation = response.get("explanation", "No explanation provided")
        if not category:
            logging.warning(f"Empty category in LLM response: {cleaned_response}")
            return classify_transaction(transaction_details)
        if category in CATEGORIES:
            return category, explanation
        logging.warning(f"Invalid category from LLM: {category}")
        return classify_transaction(transaction_details)
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Invalid JSON from LLM: {cleaned_response}, Error: {str(e)}")
        return classify_transaction(transaction_details)

# Anomaly Detection
def detect_anomalies(df):
    anomalies = []
    usd_values = df['USD'].abs()
    mean_usd = usd_values.mean()
    std_usd = usd_values.std()
    threshold = mean_usd + 2.5 * std_usd
    for idx, row in df.iterrows():
        if abs(row['USD']) > threshold:
            anomalies.append(f"Anomaly: {row['Transaction Details']} (USD {row['USD']:.2f}) on {row['Date']}")
    return anomalies

# Notes Generator
def generate_notes_llm(text, llm):
    prompt = f"""
Summarize the following bank transaction into a simple human-readable explanation (1 short sentence):
"{text}"
Example output: "Payment received from John Doe" or "Purchase at AEON Mall".
Respond with only the sentence.
"""
    try:
        return llm.predict(prompt)
    except Exception as e:
        logging.error(f"Notes generation error: {str(e)}")
        return f"Error: {str(e)}"

# Main Processing
def process_transactions(input_csv, output_csv):
    try:
        df = pd.read_csv(input_csv)
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
            logging.error(f"Pydantic validation error: {str(e)}")
            return False, (
                f"❌ Error: Invalid data format.\n"
                f"Validation errors: {str(e)}\n"
                f"Required columns: {', '.join(['Date', 'Transaction Details', 'Money In', 'Money Out', 'Balance'])}\n"
                f"Optional columns: {', '.join(['Money In Ccy', 'Money Out Ccy', 'Balance Ccy'])}"
            )

        # Add optional columns with default 'USD' if missing
        for col in ['Money In Ccy', 'Money Out Ccy', 'Balance Ccy']:
            if col not in df.columns:
                df[col] = 'USD'
                logging.info(f"Added missing column {col} with default 'USD'")

        logging.info(f"Input CSV columns: {list(df.columns)}")

        df['Expense/Income'] = df.apply(lambda r: 'INCOME' if r['Money In'] > 0 else 'EXPENSE', axis=1)
        df['USD'] = df.apply(lambda r: r['Money In'] if r['Money In'] > 0 else -r['Money Out'], axis=1)

        output_df = pd.DataFrame({
            'Date': df['Date'],
            'Transaction Details': df['Transaction Details'],
            'Expense/Income': df['Expense/Income'],
            'USD': df['USD'],
            'Notes': '',
            'Balance': df['Balance'],
            'Normalized Category': '',
            'Category Explanation': '',
            'Money In': df['Money In'],
            'Money In Ccy': df['Money In Ccy'],
            'Money Out': df['Money Out'],
            'Money Out Ccy': df['Money Out Ccy'],
            'Balance Ccy': df['Balance Ccy']
        })

        llm = OllamaLLM()
        tqdm.pandas(desc="Generating Notes")
        output_df['Notes'] = df['Transaction Details'].progress_apply(lambda x: generate_notes_llm(x, llm))

        tqdm.pandas(desc="Classifying Categories")
        category_results = df.progress_apply(
            lambda row: hop(row['Transaction Details'], row['Money In'], row['Money Out'], row['Date'], llm), axis=1
        )
        output_df['Normalized Category'] = [result[0] for result in category_results]
        output_df['Category Explanation'] = [result[1] for result in category_results]

        # Detect anomalies
        anomalies = detect_anomalies(output_df)
        if anomalies:
            with open("anomalies.txt", "w") as f:
                f.write("\n".join(anomalies))

        output_df.to_csv(output_csv, index=False)
        logging.info(f"Output saved to {output_csv}")
        logging.info(f"Output CSV columns: {list(output_df.columns)}")
        return True, f"✅ Final output saved to {output_csv}. Anomalies saved to anomalies.txt"
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return False, f"❌ Error processing transactions: {str(e)}"

# Classification Hop
def hop(details, money_in, money_out, date, llm):
    try:
        prompt = CATEGORY_PROMPT.format(
            categories=", ".join(CATEGORIES),
            transaction_details=details,
            money_out=money_out,
            money_in=money_in,
            date=date,
        )
        return normalize_category(details, llm.predict(prompt))
    except Exception as e:
        logging.error(f"Category classification error: {str(e)}")
        return classify_transaction(details)

if __name__ == "__main__":
    INPUT_CSV = "statement_report.csv"
    OUTPUT_CSV = "standardized_classified_transactions.csv"
    success, msg = process_transactions(INPUT_CSV, OUTPUT_CSV)
    print(msg)