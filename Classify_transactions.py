import pandas as pd
from langchain.llms import Ollama
import re

# LLM setup
llm = Ollama(model="blueprint-financial-assistant")

# Load CSV with error handling
try:
    df = pd.read_csv("statement_report.csv")
    df = df.dropna(subset=["Transaction Details"])
except FileNotFoundError:
    raise FileNotFoundError("Error: 'statement_report.csv' not found.")
except pd.errors.EmptyDataError:
    raise ValueError("Error: 'statement_report.csv' is empty.")

# Prompt for transaction classification
system_instruction = """
You are a financial assistant. Classify each bank transaction into EXPENSE or INCOME and provide a short note.
Rules:
- 'PURCHASE AT' or 'PAYMENT TO' → EXPENSE, note: "Shop purchase".
- 'FUNDS TRANSFERRED TO' → EXPENSE, note: "Outgoing transfer".
- 'PAYMENT FROM' or 'FUNDS RECEIVED FROM' → INCOME, note: "Incoming transfer".
- Otherwise → UNKNOWN, note: "Other".
- Output format: [Category] | [Note]
"""

# Classify transaction with fallback
def classify_transaction(details):
    # First, try the LLM
    prompt = f"{system_instruction}\nTransaction: {details}"
    try:
        response = llm.invoke(prompt).strip()
        if re.match(r"^(EXPENSE|INCOME|UNKNOWN)\s\|\s.+", response):
            category, note = response.split(" | ", 1)
            return category, note
    except Exception:
        pass

    # Fallback: Manually classify based on keywords if LLM fails
    details_upper = details.upper()
    if "PURCHASE AT" in details_upper or "PAYMENT TO" in details_upper:
        return "EXPENSE", "Shop purchase"
    elif "FUNDS TRANSFERRED TO" in details_upper:
        return "EXPENSE", "Outgoing transfer"
    elif "PAYMENT FROM" in details_upper or "FUNDS RECEIVED FROM" in details_upper:
        return "INCOME", "Incoming transfer"
    return "UNKNOWN", "Other"

# Batch processing for unique transactions
def hop(start, stop, step):
    for i in range(start, stop, step):
        yield i
    yield stop

def categorize_transactions(transaction_names, llm):
    prompt = (
        "Add a category to each expense. Example: Spotify AB by Adyen - Entertainment, "
        "Beta Boulders Ams Amsterdam Nld - Sport. Categories < 4 words: " + transaction_names
    )
    try:
        response = llm.invoke(prompt).strip().split('\n')
        categories_df = pd.DataFrame({'Transaction vs category': response})
        categories_df[['Transaction', 'Category']] = categories_df['Transaction vs category'].str.split(' - ', expand=True)
        # Clean numbering (e.g., "1. ") from Transaction
        categories_df['Transaction'] = categories_df['Transaction'].str.replace(r'^\d+\.\s+', '', regex=True)
        return categories_df[['Transaction', 'Category']].dropna()
    except Exception:
        return pd.DataFrame({'Transaction': [], 'Category': []})

# Get unique transactions and process in batches
unique_transactions = df["Transaction Details"].unique()
index_list = list(hop(0, len(unique_transactions), 30))
categories_df_all = pd.DataFrame()

for i in range(len(index_list) - 1):
    batch = unique_transactions[index_list[i]:index_list[i + 1]]
    transaction_names = ','.join(batch)
    categories_df = categorize_transactions(transaction_names, llm)
    categories_df_all = pd.concat([categories_df_all, categories_df], ignore_index=True)

# Create category mapping
category_map = dict(zip(categories_df_all['Transaction'], categories_df_all['Category']))

# Process transactions
results = []
for _, row in df.iterrows():
    date = row['Date']
    details = row['Transaction Details']
    money_in = row['Money In']
    money_out = row['Money Out']
    balance = row['Balance']

    category, note = classify_transaction(details)
    amount = (
        f"{money_in}" if category == "INCOME" and pd.notna(money_in) else
        f"{money_out}" if category == "EXPENSE" and pd.notna(money_out) else
        ""
    )

    results.append({
        "Date": date,
        "Transaction Details": details,
        "Expense/Income": category,
        "USD": amount,
        "Notes": note,
        "Balance (USD)": balance
    })

# Create DataFrame
final_df = pd.DataFrame(results)
if final_df.empty:
    raise ValueError("Error: No transactions processed.")

# Categorize based on notes, details, and LLM categories
def normalize_category(note, details, llm_category):
    valid_categories = [
        "Food and Drinks", "Clothing", "Health and Wellness", "Travel",
        "Entertainment", "Services", "Sport and Fitness", "Shopping", "Transfers"
    ]
    if llm_category in valid_categories:
        return llm_category
    if note == "Shop purchase":
        if re.search(r"Food|Restaurant|Café|Bistro|Bar", details, re.IGNORECASE):
            return "Food and Drinks"
        elif re.search(r"Clothing|Fashion", details, re.IGNORECASE):
            return "Clothing"
        elif re.search(r"Pharmacy|Etos|Health", details, re.IGNORECASE):
            return "Health and Wellness"
        elif re.search(r"Taxi|Bus|Transport|Travel", details, re.IGNORECASE):
            return "Travel"
        elif re.search(r"Spotify|Entertainment", details, re.IGNORECASE):
            return "Entertainment"
        return "Shopping"
    elif note in ["Incoming transfer", "Outgoing transfer"]:
        return "Transfers"
    elif note == "Other":
        if re.search(r"Service|Bill|Electricity|Justice", details, re.IGNORECASE):
            return "Services"
        elif re.search(r"Sport|Fitness|Boulder", details, re.IGNORECASE):
            return "Sport and Fitness"
        elif re.search(r"Health|Wellness", details, re.IGNORECASE):
            return "Health and Wellness"
        return "Other"
    return "Other"

# Apply categorization
final_df["Normalized Category"] = final_df.apply(
    lambda row: normalize_category(
        row["Notes"], row["Transaction Details"],
        category_map.get(row["Transaction Details"], "")
    ), axis=1
)

# Export
final_df.to_csv("standardized_classified_transactions.csv", index=False)
print(final_df.head())