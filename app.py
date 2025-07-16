import openai
import json
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import os
from flask import send_file
from flask_cors import CORS
app = Flask(__name__)
CORS(app)




openai.api_key = "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"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json_file(file_name, fallback=None):
    try:
        with open(os.path.join(BASE_DIR, file_name), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading {file_name}: {e}")
        return fallback if fallback is not None else {}

mappings = load_json_file("mappings.json")
GL_ACCOUNT_MAPPING = mappings.get("gl_account_mapping", {})
CASH_ORIGIN_MAPPING = mappings.get("cash_origin_mapping", {})
VENDOR_CATEGORIES = mappings.get("vendor_categories", {})

CHART_OF_ACCOUNTS = load_json_file("chart_of_accounts.json")
vendor_data = load_json_file("vendor_data.json")
file_map = load_json_file("file_map.json")
# Define mapping of download keys to actual filenames (adjust filenames as per your files)
file_map = {
    'cashflow': 'cash_flow_statement.txt',
    'originmap': 'gl_cash_origin_mapping_output_fixed.xlsx',
    'transactions': 'transactions_data_updated.csv',
    'unmatched_bank': 'unmatched_bank_missing_in_ap_ar.xlsx',
    'unmatched_ap_ar': 'unmatched_ap_ar_pending_in_bank.xlsx',
    'matched_exact': 'matched_exact_transactions.xlsx',
    'matched_fuzzy': 'matched_fuzzy_transactions.xlsx',
    'final_ap_ar': 'final_ap_ar_transactions.xlsx'
}

manual_keywords = {
    "loan repayment": "Financing",
    "interest": "Interest",
    "tax": "Tax",
    "grant": "Government Grant",
    "salary": "Payroll",
    "wages": "Payroll",
    "scrap": "Scrap",
    "material sale": "MATERIAL_SALES",
    "consultancy": "Consultancy",
    "transport": "Logistics",
    "maintenance": "Maintenance",
    "electricity": "Utilities",
    "utility": "Utilities",
    "utilities": "Utilities",
}

GPT_LOG_FILE = os.path.join(BASE_DIR, "gpt_classification_log.jsonl")

def log_gpt_result(description, predicted_category):
    try:
        with open(GPT_LOG_FILE, 'a', encoding='utf-8') as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "gpt_category": predicted_category
            }
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"⚠️ Error logging GPT result: {e}")

def categorize_transaction(description):
    description_lower = description.lower().strip()

    # Manual match first
    for keyword, category in manual_keywords.items():
        if keyword in description_lower:
            print(f"✅ Manual match: '{keyword}' → '{category}'")
            return category

    # Call GPT fallback
    try:
        prompt = (
            "Classify the following transaction description into one of these categories:\n"
            "1. Payroll, 2. Utilities, 3. Capex, 4. Maintenance, 5. Sales, 6. Logistics, 7. Consultancy, 8. Scrap, "
            "9. Material Purchase, 10. MATERIAL_SALES, 11. Tax, 12. Government Grant, 13. Financing, 14. Interest, 15. Other.\n"
            "If unsure, return 'Other'.\n\n"
            f"Description: {description}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies transaction descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        gpt_category = response['choices'][0]['message']['content'].strip()
        print(f"🔍 GPT predicted category: '{gpt_category}'")

        log_gpt_result(description, gpt_category)

        if gpt_category.lower() != "other":
            return gpt_category
    except Exception as e:
        print(f"❌ GPT classification error: {e}")

    # Fallback mappings
    for keyword, gl_category in GL_ACCOUNT_MAPPING.items():
        if keyword.lower() in description_lower:
            return gl_category

    for keyword, cash_cat in CASH_ORIGIN_MAPPING.items():
        if keyword.lower() in description_lower:
            return cash_cat

    for vendor, info in vendor_data.items():
        if vendor.lower() in description_lower:
            return info.get("category", "Other")

    return "Other"

@app.route("/add-transaction", methods=["POST"])
def add_transaction():
    data = request.get_json()
    description = data.get("description", "")
    amount = float(data.get("amount", 0))
    transaction_type = data.get("type", "")
    date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    reference = data.get("reference", "")

    predicted_category = categorize_transaction(description)

    transactions_file = os.path.join(BASE_DIR, "steel_plant_transactions.xlsx")
    try:
        df = pd.read_excel(transactions_file)
    except FileNotFoundError:
        # Define columns including 'Balance (INR)'
        df = pd.DataFrame(columns=["Description", "Amount (INR)", "Type", "Category", "Date", "Reference", "Balance (INR)"])

    new_transaction = {
        "Description": description,
        "Amount (INR)": amount,
        "Type": transaction_type,
        "Category": predicted_category,
        "Date": date,
        "Reference": reference,
        # Balance will be calculated below
    }

    # Append the new transaction row
    df = pd.concat([df, pd.DataFrame([new_transaction])], ignore_index=True)

    # Calculate running balance
    # First create a column for amount impact: positive for INWARD, negative for OUTWARD
    def amount_signed(row):
        if str(row['Type']).upper() == "INWARD":
            return row["Amount (INR)"]
        elif str(row['Type']).upper() == "OUTWARD":
            return -row["Amount (INR)"]
        else:
            return 0  # for any other types, consider zero impact

    df['Amount_Signed'] = df.apply(amount_signed, axis=1)

    # Running balance: cumulative sum of Amount_Signed
    df['Balance (INR)'] = df['Amount_Signed'].cumsum()

    # Drop helper column
    df.drop(columns=['Amount_Signed'], inplace=True)

    # Save updated dataframe back
    df.to_excel(transactions_file, index=False)

    return jsonify({
        "message": "✅ Transaction added successfully",
        "category": predicted_category,
        "data": new_transaction
    })


@app.route("/reload-mappings", methods=["POST"])
def reload_all_configs():
    global GL_ACCOUNT_MAPPING, CASH_ORIGIN_MAPPING, VENDOR_CATEGORIES
    global CHART_OF_ACCOUNTS, vendor_data, file_map

    mappings = load_json_file("mappings.json")
    GL_ACCOUNT_MAPPING = mappings.get("gl_account_mapping", {})
    CASH_ORIGIN_MAPPING = mappings.get("cash_origin_mapping", {})
    VENDOR_CATEGORIES = mappings.get("vendor_categories", {})
    CHART_OF_ACCOUNTS = load_json_file("chart_of_accounts.json")
    vendor_data = load_json_file("vendor_data.json")
    file_map = load_json_file("file_map.json")

    return jsonify({"message": "✅ All configurations reloaded successfully"})

@app.route("/download/<file_key>", methods=["GET"])
def download_file(file_key):
    if file_key not in file_map:
        return jsonify({"error": "Invalid file key"}), 400

    filename = file_map[file_key]

    # Make sure filename is a valid string
    if not filename or not isinstance(filename, str):
        return jsonify({"error": "Invalid filename"}), 400

    # Construct full file path inside data folder
    print(f"Request to download file_key: {file_key}, filename: {filename}")

    filepath = os.path.join(BASE_DIR, "data", filename)

    # Check if file exists
    if not os.path.isfile(filepath):
        return jsonify({"error": f"File '{filename}' not found on server"}), 404

    # Send the file as a download
    return send_file(filepath, as_attachment=True)

# Add these imports at the top of your app.py (after existing imports)
import difflib

# Add these helper functions before your routes
def calculate_similarity(bank_row, apar_row):
    """Calculate similarity between transactions"""
    try:
        score = 0.0
        
        # Amount similarity
        bank_amount = abs(float(bank_row.get('Amount (INR)', 0)))
        apar_amount = abs(float(apar_row.get('Amount (INR)', 0)))
        
        if bank_amount > 0 and apar_amount > 0:
            amount_diff = abs(bank_amount - apar_amount) / max(bank_amount, apar_amount)
            score += max(0, 1 - amount_diff * 2) * 0.5
        
        # Description similarity
        bank_desc = str(bank_row.get('Description', '')).lower()
        apar_desc = str(apar_row.get('Description', '')).lower()
        score += difflib.SequenceMatcher(None, bank_desc, apar_desc).ratio() * 0.5
        
        return score
    except:
        return 0.0

# ALL MISSING ENDPOINTS - Add these before your @app.route("/")

@app.route("/add-ap-ar", methods=["POST"])
def add_ap_ar():
    data = request.get_json()
    description = data.get("description", "")
    amount = float(data.get("amount", 0))
    entry_type = data.get("type", "")
    status = data.get("status", "Pending")
    date = data.get("date", datetime.now().strftime("%Y-%m-%d"))

    ap_ar_file = os.path.join(BASE_DIR, "ap_ar_entries.xlsx")
    try:
        df = pd.read_excel(ap_ar_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Description", "Amount (INR)", "Type", "Status", "Date", "Category"])

    predicted_category = categorize_transaction(description)

    new_entry = {
        "Description": description,
        "Amount (INR)": amount,
        "Type": entry_type,
        "Status": status,
        "Date": date,
        "Category": predicted_category
    }

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_excel(ap_ar_file, index=False)

    return jsonify({
        "message": "✅ AP/AR entry added successfully",
        "category": predicted_category,
        "data": new_entry
    })

@app.route("/add-bank-entry", methods=["POST"])
def add_bank_entry():
    data = request.get_json()
    description = data.get("description", "")
    amount = float(data.get("amount", 0))
    entry_type = data.get("type", "Debit")
    date = data.get("date", datetime.now().strftime("%Y-%m-%d"))

    bank_file = os.path.join(BASE_DIR, "bank_entries.xlsx")
    try:
        df = pd.read_excel(bank_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Description", "Amount (INR)", "Type", "Date", "Category", "Balance (INR)"])

    predicted_category = categorize_transaction(description)

    new_entry = {
        "Description": description,
        "Amount (INR)": amount,
        "Type": entry_type,
        "Date": date,
        "Category": predicted_category
    }

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    def amount_signed(row):
        if str(row['Type']).upper() == "CREDIT":
            return row["Amount (INR)"]
        elif str(row['Type']).upper() == "DEBIT":
            return -row["Amount (INR)"]
        else:
            return 0

    df['Amount_Signed'] = df.apply(amount_signed, axis=1)
    df['Balance (INR)'] = df['Amount_Signed'].cumsum()
    df.drop(columns=['Amount_Signed'], inplace=True)
    df.to_excel(bank_file, index=False)

    return jsonify({
        "message": "✅ Bank entry added successfully",
        "category": predicted_category,
        "data": new_entry
    })

# Replace your /process-all endpoint with this version that handles NaN values

@app.route("/process-all", methods=["POST"])
def process_all():
    try:
        # Load all transaction data
        bank_file = os.path.join(BASE_DIR, "bank_entries.xlsx")
        apar_file = os.path.join(BASE_DIR, "ap_ar_entries.xlsx")
        trans_file = os.path.join(BASE_DIR, "steel_plant_transactions.xlsx")
        
        # Load dataframes
        try:
            bank_df = pd.read_excel(bank_file)
            bank_count = len(bank_df)
        except FileNotFoundError:
            bank_df = pd.DataFrame()
            bank_count = 0
        
        try:
            apar_df = pd.read_excel(apar_file)
            apar_count = len(apar_df)
        except FileNotFoundError:
            apar_df = pd.DataFrame()
            apar_count = 0
        
        try:
            trans_df = pd.read_excel(trans_file)
            trans_count = len(trans_df)
        except FileNotFoundError:
            trans_df = pd.DataFrame()
            trans_count = 0
        
        # Create data directory
        data_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Simple reconciliation logic
        matched_exact = []
        matched_fuzzy = []
        unmatched_bank = []
        unmatched_apar = []
        matched_apar_indices = set()
        
        # Process reconciliation
        for bank_idx, bank_row in bank_df.iterrows():
            best_match = None
            best_score = 0
            best_apar_idx = None
            
            for apar_idx, apar_row in apar_df.iterrows():
                if apar_idx in matched_apar_indices:
                    continue
                
                similarity = calculate_similarity(bank_row, apar_row)
                
                if similarity > best_score and similarity > 0.65:
                    best_match = apar_row
                    best_score = similarity
                    best_apar_idx = apar_idx
            
            if best_match is not None:
                matched_apar_indices.add(best_apar_idx)
                match_record = {
                    'Bank_Description': str(bank_row.get('Description', '')),
                    'Bank_Amount': float(bank_row.get('Amount (INR)', 0)),
                    'APAR_Description': str(best_match.get('Description', '')),
                    'APAR_Amount': float(best_match.get('Amount (INR)', 0)),
                    'Match_Score': round(best_score, 3)
                }
                
                if best_score > 0.9:
                    matched_exact.append(match_record)
                else:
                    matched_fuzzy.append(match_record)
            else:
                unmatched_bank.append({
                    'Description': str(bank_row.get('Description', '')),
                    'Amount (INR)': float(bank_row.get('Amount (INR)', 0)),
                    'Reason': 'No corresponding AP/AR entry found'
                })
        
        # Find unmatched AP/AR entries
        for apar_idx, apar_row in apar_df.iterrows():
            if apar_idx not in matched_apar_indices:
                unmatched_apar.append({
                    'Description': str(apar_row.get('Description', '')),
                    'Amount (INR)': float(apar_row.get('Amount (INR)', 0)),
                    'Status': str(apar_row.get('Status', '')),
                    'Reason': 'Payment not found in bank records'
                })
        
        # Save reconciliation files
        pd.DataFrame(matched_exact).to_excel(os.path.join(data_dir, "matched_exact_transactions.xlsx"), index=False)
        pd.DataFrame(matched_fuzzy).to_excel(os.path.join(data_dir, "matched_fuzzy_transactions.xlsx"), index=False)
        pd.DataFrame(unmatched_bank).to_excel(os.path.join(data_dir, "unmatched_bank_missing_in_ap_ar.xlsx"), index=False)
        pd.DataFrame(unmatched_apar).to_excel(os.path.join(data_dir, "unmatched_ap_ar_pending_in_bank.xlsx"), index=False)
        
        # Save final AP/AR file
        if not apar_df.empty:
            final_apar = apar_df.copy()
            final_apar['Reconciliation_Status'] = final_apar.index.map(
                lambda x: 'Matched' if x in matched_apar_indices else 'Unmatched'
            )
            final_apar.to_excel(os.path.join(data_dir, "final_ap_ar_transactions.xlsx"), index=False)
        
        # UPDATE CASH FLOW WITH REAL DATA
        all_transactions = pd.concat([bank_df, apar_df, trans_df], ignore_index=True)
        
        # Initialize cash flow calculations
        cash_receipts_customers = 0.0
        cash_received_advances = 0.0
        other_operating_receipts = 0.0
        cash_paid_employees = 0.0
        cash_paid_utilities = 0.0
        cash_paid_statutory = 0.0
        other_operating_payments = 0.0
        
        # Vendor category totals
        vendor_totals = {
            "Electricity Bills": 0.0, "Raw Materials": 0.0, "Spares": 0.0, "Chemicals": 0.0,
            "Maintenance": 0.0, "IT Services": 0.0, "Fuel": 0.0, "Packaging": 0.0,
            "Consultancy": 0.0, "Insurance": 0.0, "Logistics": 0.0, "Transport": 0.0,
            "Tools": 0.0, "Security": 0.0, "Safety": 0.0, "Training": 0.0,
            "Scrap Disposal": 0.0, "Cleaning Services": 0.0, "Spare Parts": 0.0,
            "Canteen": 0.0, "Office Supplies": 0.0, "Lubricants": 0.0, "Calibration": 0.0,
            "Uniforms": 0.0, "Printing": 0.0, "Courier": 0.0, "Pest Control": 0.0
        }
        
        # Process each transaction
        if not all_transactions.empty:
            for _, transaction in all_transactions.iterrows():
                try:
                    amount = float(transaction.get('Amount (INR)', 0))
                    if pd.isna(amount):
                        amount = 0.0
                        
                    category = str(transaction.get('Category', '')).lower()
                    trans_type = str(transaction.get('Type', '')).upper()
                    description = str(transaction.get('Description', '')).lower()
                    
                    # Determine if it's income or expense
                    is_income = (trans_type in ['INWARD', 'CREDIT'] or 
                                'customer' in description or 
                                'sales' in description or 
                                'revenue' in description)
                    
                    # Categorize income
                    if is_income:
                        if 'customer' in description or 'sales' in description:
                            cash_receipts_customers += amount
                        elif 'advance' in description:
                            cash_received_advances += amount
                        else:
                            other_operating_receipts += amount
                    
                    # Categorize expenses
                    else:
                        if 'payroll' in category or 'salary' in category or 'employee' in description:
                            cash_paid_employees += amount
                        elif 'utilities' in category or 'electricity' in description:
                            cash_paid_utilities += amount
                        elif 'tax' in category or 'statutory' in description:
                            cash_paid_statutory += amount
                        else:
                            # Check vendor categories
                            categorized = False
                            for vendor_cat in vendor_totals.keys():
                                if vendor_cat.lower() in category or vendor_cat.lower() in description:
                                    vendor_totals[vendor_cat] += amount
                                    categorized = True
                                    break
                            
                            if not categorized:
                                other_operating_payments += amount
                except (ValueError, TypeError):
                    # Skip invalid transactions
                    continue
        
        # Calculate totals and ensure no NaN values
        total_inflows = cash_receipts_customers + cash_received_advances + other_operating_receipts
        total_vendor_payments = sum(vendor_totals.values())
        total_outflows = -(cash_paid_employees + cash_paid_utilities + cash_paid_statutory + other_operating_payments + total_vendor_payments)
        net_operating_cash_flow = total_inflows + total_outflows
        
        # Fixed values for investing and financing
        fixed_assets = 0.0
        debt_service = -2255000.0
        net_investing = -fixed_assets
        net_financing = debt_service
        net_change_cash = net_operating_cash_flow + net_investing + net_financing
        
        # Helper function to clean values for JSON
        def clean_value(val):
            if pd.isna(val) or val == float('inf') or val == float('-inf'):
                return 0.0
            return float(val)
        
        # Clean all values
        cash_receipts_customers = clean_value(cash_receipts_customers)
        cash_received_advances = clean_value(cash_received_advances)
        other_operating_receipts = clean_value(other_operating_receipts)
        total_inflows = clean_value(total_inflows)
        cash_paid_employees = clean_value(cash_paid_employees)
        cash_paid_utilities = clean_value(cash_paid_utilities)
        cash_paid_statutory = clean_value(cash_paid_statutory)
        other_operating_payments = clean_value(other_operating_payments)
        total_outflows = clean_value(total_outflows)
        net_operating_cash_flow = clean_value(net_operating_cash_flow)
        net_investing = clean_value(net_investing)
        net_financing = clean_value(net_financing)
        net_change_cash = clean_value(net_change_cash)
        
        # Clean vendor totals
        for key in vendor_totals:
            vendor_totals[key] = clean_value(vendor_totals[key])
        
        # Generate updated cash flow statement
        cash_flow_content = f"""OPERATING ACTIVITIES
             Section                                     Description     Amount
Operating Activities                    Cash Receipts from Customers  {cash_receipts_customers}
Operating Activities                       Cash Received as Advances        {cash_received_advances}
Operating Activities                        Other Operating Receipts   {other_operating_receipts}
Operating Activities                         Total Operating Inflows  {total_inflows}
Operating Activities                          Cash Paid to Employees  {-cash_paid_employees}
Operating Activities                         Cash Paid for Utilities  {-cash_paid_utilities}
Operating Activities                         Cash Paid for Statutory       {-cash_paid_statutory}
Operating Activities                        Other Operating Payments       {-other_operating_payments}
Operating Activities Cash Paid to Vendor Category: Electricity Bills {-vendor_totals["Electricity Bills"]}
Operating Activities     Cash Paid to Vendor Category: Raw Materials  {-vendor_totals["Raw Materials"]}
Operating Activities            Cash Paid to Vendor Category: Spares  {-vendor_totals["Spares"]}
Operating Activities         Cash Paid to Vendor Category: Chemicals  {-vendor_totals["Chemicals"]}
Operating Activities       Cash Paid to Vendor Category: Maintenance   {-vendor_totals["Maintenance"]}
Operating Activities       Cash Paid to Vendor Category: IT Services   {-vendor_totals["IT Services"]}
Operating Activities              Cash Paid to Vendor Category: Fuel   {-vendor_totals["Fuel"]}
Operating Activities         Cash Paid to Vendor Category: Packaging   {-vendor_totals["Packaging"]}
Operating Activities       Cash Paid to Vendor Category: Consultancy   {-vendor_totals["Consultancy"]}
Operating Activities         Cash Paid to Vendor Category: Insurance   {-vendor_totals["Insurance"]}
Operating Activities         Cash Paid to Vendor Category: Logistics   {-vendor_totals["Logistics"]}
Operating Activities         Cash Paid to Vendor Category: Transport   {-vendor_totals["Transport"]}
Operating Activities             Cash Paid to Vendor Category: Tools   {-vendor_totals["Tools"]}
Operating Activities          Cash Paid to Vendor Category: Security   {-vendor_totals["Security"]}
Operating Activities            Cash Paid to Vendor Category: Safety   {-vendor_totals["Safety"]}
Operating Activities          Cash Paid to Vendor Category: Training   {-vendor_totals["Training"]}
Operating Activities    Cash Paid to Vendor Category: Scrap Disposal   {-vendor_totals["Scrap Disposal"]}
Operating Activities Cash Paid to Vendor Category: Cleaning Services   {-vendor_totals["Cleaning Services"]}
Operating Activities       Cash Paid to Vendor Category: Spare Parts   {-vendor_totals["Spare Parts"]}
Operating Activities           Cash Paid to Vendor Category: Canteen   {-vendor_totals["Canteen"]}
Operating Activities   Cash Paid to Vendor Category: Office Supplies   {-vendor_totals["Office Supplies"]}
Operating Activities        Cash Paid to Vendor Category: Lubricants   {-vendor_totals["Lubricants"]}
Operating Activities       Cash Paid to Vendor Category: Calibration   {-vendor_totals["Calibration"]}
Operating Activities          Cash Paid to Vendor Category: Uniforms    {-vendor_totals["Uniforms"]}
Operating Activities          Cash Paid to Vendor Category: Printing    {-vendor_totals["Printing"]}
Operating Activities           Cash Paid to Vendor Category: Courier    {-vendor_totals["Courier"]}
Operating Activities      Cash Paid to Vendor Category: Pest Control    {-vendor_totals["Pest Control"]}
Operating Activities                        Total Operating Outflows {total_outflows}
Operating Activities                         Net Operating Cash Flow  {net_operating_cash_flow}
INVESTING ACTIVITIES
             Description  Amount
Purchase of Fixed Assets    {-fixed_assets}
 Net Investing Cash Flow    {net_investing}
FINANCING ACTIVITIES
            Description     Amount
  Debt Service Payments {debt_service}
Net Financing Cash Flow {net_financing}
CASH FLOW SUMMARY
                     Metric      Value
         Net Change in Cash {net_change_cash}
Cash at Beginning of Period        0.0
      Cash at End of Period        {net_change_cash}
  Calculated Ending Balance        {net_change_cash}"""
        
        # Save updated cash flow
        cash_flow_file = os.path.join(data_dir, "cash_flow_statement.txt")
        with open(cash_flow_file, 'w', encoding='utf-8') as f:
            f.write(cash_flow_content)
        
        # Save other required files
        if not trans_df.empty:
            trans_df.to_csv(os.path.join(data_dir, "transactions_data_updated.csv"), index=False)
        
        # Create GL mapping file
        gl_data = []
        if not all_transactions.empty:
            for _, row in all_transactions.iterrows():
                gl_data.append({
                    'Description': str(row.get('Description', '')),
                    'Amount': clean_value(row.get('Amount (INR)', 0)),
                    'Category': str(row.get('Category', 'Other'))
                })
        
        if gl_data:
            pd.DataFrame(gl_data).to_excel(os.path.join(data_dir, "gl_cash_origin_mapping_output_fixed.xlsx"), index=False)
        
        return jsonify({
            "message": "✅ Full reconciliation processing completed successfully!",
            "summary": {
                "input_data": {
                    "bank_entries": bank_count,
                    "apar_entries": apar_count,
                    "main_transactions": trans_count
                },
                "reconciliation_results": {
                    "exact_matches": len(matched_exact),
                    "fuzzy_matches": len(matched_fuzzy),
                    "unmatched_bank": len(unmatched_bank),
                    "unmatched_apar": len(unmatched_apar)
                },
                "cash_flow_updated": {
                    "total_inflows": total_inflows,
                    "total_outflows": total_outflows,
                    "net_operating_cash_flow": net_operating_cash_flow,
                    "net_change_in_cash": net_change_cash
                }
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/categorize", methods=["GET"])
def categorize_endpoint():
    description = request.args.get('description', '')
    if not description:
        return jsonify({"error": "Description parameter required"}), 400
    
    category = categorize_transaction(description)
    return jsonify({
        "cash_flow_category": category,
        "gl_account": "TBD",
        "account_type": "TBD", 
        "account_name": "TBD",
        "cash_origin": category,
        "vendor_category": category
    })

@app.route("/test-add", methods=["POST"])
def test_add():
    data = request.get_json()
    description = data.get("description", "Test Transaction")
    amount = data.get("amount", 1000)
    trans_type = data.get("type", "OUTWARD")
    
    category = categorize_transaction(description)
    
    return jsonify({
        "message": "Test successful",
        "description": description,
        "amount": amount,
        "type": trans_type,
        "predicted_category": category
    })

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "System operational",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": ["add-transaction", "add-ap-ar", "add-bank-entry", "process-all"]
    })

@app.route("/cleanup-transactions", methods=["POST"])
def cleanup_transactions():
    return jsonify({
        "message": "Cleanup functionality not implemented yet",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/load-file", methods=["GET"])
def load_file():
    return jsonify({
        "message": "Load file functionality not implemented yet",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/data/<data_type>", methods=["GET"])
def get_data(data_type):
    data_dir = os.path.join(BASE_DIR, "data")
    
    # Helper function to clean NaN values
    def clean_dataframe(df):
        return df.fillna(0.0)
    
    if data_type == "cashflow":
        try:
            with open(os.path.join(data_dir, "cash_flow_statement.txt"), 'r') as f:
                return jsonify({"text": f.read()})
        except FileNotFoundError:
            return jsonify({"error": "Cash flow file not found"}), 404
    
    elif data_type == "transactions":
        try:
            df = pd.read_csv(os.path.join(data_dir, "transactions_data_updated.csv"))
            # Clean NaN values before converting to JSON
            df = clean_dataframe(df)
            return jsonify(df.to_dict('records'))
        except FileNotFoundError:
            # Try the original transactions file
            try:
                df = pd.read_excel(os.path.join(BASE_DIR, "steel_plant_transactions.xlsx"))
                df = clean_dataframe(df)
                return jsonify(df.to_dict('records'))
            except FileNotFoundError:
                return jsonify({"error": "Transactions file not found"}), 404
    
    elif data_type == "unmatched_bank":
        try:
            df = pd.read_excel(os.path.join(data_dir, "unmatched_bank_missing_in_ap_ar.xlsx"))
            df = clean_dataframe(df)
            return jsonify(df.to_dict('records'))
        except FileNotFoundError:
            return jsonify({"error": "Unmatched bank file not found"}), 404
    
    elif data_type == "unmatched_ap_ar":
        try:
            df = pd.read_excel(os.path.join(data_dir, "unmatched_ap_ar_pending_in_bank.xlsx"))
            df = clean_dataframe(df)
            return jsonify(df.to_dict('records'))
        except FileNotFoundError:
            return jsonify({"error": "Unmatched AP/AR file not found"}), 404
    
    else:
        return jsonify({"error": "Invalid data type"}), 400

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)