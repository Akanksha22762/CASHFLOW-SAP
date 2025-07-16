# sap_bank_mapping_ai.py

import re
import openai
import difflib
import pandas as pd




# Set OpenAI API key
openai.api_key = "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"

def smart_match_description(description):
    """
    Match transaction description with predefined categories using regex patterns.
    """
    patterns = {
        r"vendor\s+.*\spayment": "Vendor Payment",
        r"invoice\s+\d+": "Invoice Payment",
        r"gst\s+payment": "Tax",
        r"employee\s+(bonus|salary)": "Payroll",
        r"power\s+bill|electricity": "Utilities",
        r"raw\s+material\s+purchase": "Material Purchase",
        r"freight|logistics": "Logistics",
        r"maintenance\s+service": "Maintenance",
        r"sales\s+revenue": "Sales/Revenue"
    }

    for pattern, category in patterns.items():
        if re.search(pattern, description.lower()):
            return category

    return "Unknown"  # If no match is found, return "Unknown"


def classify_unknown_description(description):
    """
    Use OpenAI API to categorize an unknown transaction description.
    """
    prompt = f"Categorize the following financial transaction description: '{description}'"
    response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=20,
    temperature=0
    )

    return response.choices[0].message.content.strip()



def load_master_data():
    """
    Load master data from a single Excel file containing multiple sheets.
    """
    master_file = "steel_plant_master_data.xlsx"  # Path to your combined Excel file
    xls = pd.ExcelFile(master_file)

    gl_df = xls.parse("Chart of Accounts")
    cust_df = xls.parse("Customers")
    vendor_df = xls.parse("Vendors")
    rev_df = xls.parse("Revenue Centres")  # Optional

    return gl_df, cust_df, vendor_df, rev_df


def pre_match_by_description(sap_df, bank_df):
    """
    Try to match SAP descriptions with Bank descriptions before reconciliation.
    Flag unmatched descriptions as "Unknown" and apply regex or OpenAI API for categorization.
    """
    # Preprocess SAP and Bank descriptions
    sap_df['Matched_Bank_Description'] = ''
    sap_df['Category'] = sap_df['Description'].apply(smart_match_description)  # Apply regex-based categorization
    
    unmatched_sap = []

    # Try to match descriptions using fuzzy logic or direct matching
    for i, sap_row in sap_df.iterrows():
        sap_desc = sap_row['Description']
        best_match = None
        highest_score = 0
        for j, bank_row in bank_df.iterrows():
            bank_desc = bank_row['Description']
            score = difflib.SequenceMatcher(None, sap_desc.lower(), bank_desc.lower()).ratio()
            if score > highest_score:
                best_match = bank_desc
                highest_score = score

        if highest_score > 0.6:  # If similarity is above a threshold
            sap_df.at[i, 'Matched_Bank_Description'] = best_match
        else:
            # Flag unmatched SAP rows for further investigation
            unmatched_sap.append(sap_row)
            sap_df.at[i, 'Category'] = 'Unknown'  # Mark as unknown if no match is found
    
    # Handle "Unknown" categories via OpenAI (optional)
    for i, row in sap_df[sap_df['Category'] == 'Unknown'].iterrows():
        description = row['Description']
        category = classify_unknown_description(description)
        sap_df.at[i, 'Category'] = category
    
    return sap_df, unmatched_sap  # Return the updated SAP dataframe and the unmatched SAP entries


def match_vendor(description, vendor_df):
    """
    Match vendor from the description based on the vendor list from the master data.
    This function checks if the vendor name appears in the description.
    """
    vendor_name = 'Unknown'
    
    # Check if any vendor name is in the description
    for _, vendor_row in vendor_df.iterrows():
        vendor = vendor_row['Vendor Name'].lower()
        if vendor in description.lower():
            vendor_name = vendor_row['Vendor Name']
            break
    
    return vendor_name


def match_customer(description, cust_df):
    """
    Match customer from the description based on the customer list from the master data.
    This function checks if the customer name appears in the description.
    """
    customer_name = 'Unknown'
    
    # Check if any customer name is in the description
    for _, customer_row in cust_df.iterrows():
        customer = customer_row['Customer Name'].lower()
        if customer in description.lower():
            customer_name = customer_row['Customer Name']
            break
    
    return customer_name


def categorize_with_gl(description, gl_account, gl_df):
    """
    Categorize SAP transaction based on G/L Account.
    This function looks up the G/L Account in the Chart of Accounts (gl_df)
    and returns the corresponding category.

    Parameters:
    - description (str): Description of the SAP transaction.
    - gl_account (str): The G/L account number from SAP.
    - gl_df (DataFrame): Dataframe containing the Chart of Accounts.

    Returns:
    - str: The category for the SAP transaction.
    """
    category = 'Unknown'  # Default category if no match is found

    # Ensure the G/L account is a string to match properly
    gl_account = str(gl_account).strip()

    # Try to find the category by matching G/L Account
    if gl_account in gl_df['Account Code'].astype(str).values:
        matched_row = gl_df[gl_df['Account Code'].astype(str) == gl_account]
        category = matched_row['Type'].values[0]  # Assuming 'Type' column represents the category

    return category
