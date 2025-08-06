import pandas as pd
import os

def extract_real_vendors(descriptions):
    """Extract only REAL business vendor names, not generic transaction words"""
    import re
    
    # Common business name patterns and real company indicators
    business_indicators = [
        r'\b(?:Ltd|LLC|Inc|Corp|Company|Co|Limited|Corporation)\b',
        r'\b(?:Steel|Manufacturing|Industries|Group|Enterprises|Solutions|Systems|Technologies)\b',
        r'\b(?:Bank|Financial|Insurance|Investment|Capital|Trust|Fund)\b',
        r'\b(?:Oil|Gas|Petroleum|Energy|Power|Utilities)\b',
        r'\b(?:Automotive|Engineering|Construction|Infrastructure|Development)\b'
    ]
    
    # Filter out transaction-related words (not business names)
    exclude_patterns = [
        r'\b(?:ATM|WITHDRAWAL|DEPOSIT|TRANSFER|PAYMENT|PURCHASE|SALE|BUY|SELL)\b',
        r'\b(?:FEE|INTEREST|BALANCE|CASH|CHECK|DEBIT|CREDIT|ONLINE|BANK)\b',
        r'\b(?:Q1|Q2|Q3|Q4|FINAL|ADVANCE|MILESTONE|RETENTION|BONUS)\b',
        r'\b(?:CAPEX|IMPORT|EXPORT|PROCUREMENT|TRAINING|SALARY)\b',
        r'\b(?:EQUIPMENT|MACHINERY|SOFTWARE|TECHNOLOGY|INFRASTRUCTURE)\b',
        r'\b(?:RAW|MATERIAL|SUPPLIER|LOGISTICS|PROVIDER|SERVICE)\b',
        r'\b(?:CUSTOMER|VIP|NEW|BULK|ORDER|PROPERTY|ASSET|SCRAP)\b'
    ]
    
    vendors = []
    for desc in descriptions:
        desc_str = str(desc).strip()
        if len(desc_str) < 5:  # Skip very short descriptions
            continue
            
        # Skip if contains too many transaction words
        transaction_words = sum(1 for pattern in exclude_patterns if re.search(pattern, desc_str, re.IGNORECASE))
        if transaction_words >= 2:  # Skip if 2+ transaction words
            continue
        
        # Look for real business name patterns
        business_name = None
        
        # Pattern 1: "Company Name - Description" format
        if ' - ' in desc_str:
            company_part = desc_str.split(' - ')[0]
            if len(company_part) > 3 and not any(re.search(pattern, company_part, re.IGNORECASE) for pattern in exclude_patterns):
                business_name = company_part.strip()
        
        # Pattern 2: Look for business indicators
        elif any(re.search(pattern, desc_str, re.IGNORECASE) for pattern in business_indicators):
            words = desc_str.split()
            business_words = []
            for word in words:
                if len(word) > 2 and word[0].isupper() and not word.isupper():
                    if not any(re.search(pattern, word, re.IGNORECASE) for pattern in exclude_patterns):
                        business_words.append(word)
            
            if business_words:
                business_name = ' '.join(business_words[:2])  # Take first 2 business words
        
        # Pattern 3: Look for "Payment to Company" format
        elif 'PAYMENT TO ' in desc_str.upper():
            company_part = desc_str.upper().replace('PAYMENT TO ', '').strip()
            if len(company_part) > 3:
                business_name = company_part.title()
        
        if business_name and len(business_name) > 3:
            vendors.append(business_name)
    
    # Remove duplicates and sort
    unique_vendors = list(set(vendors))
    unique_vendors.sort()
    
    return unique_vendors

# Test the vendor extraction
if __name__ == "__main__":
    # Load bank data
    bank_path = os.path.join('data', 'bank_data_processed.xlsx')
    if os.path.exists(bank_path):
        bank_df = pd.read_excel(bank_path)
        
        if 'Description' in bank_df.columns:
            print("🔍 Testing vendor extraction...")
            print(f"📊 Total transactions: {len(bank_df)}")
            
            # Get sample descriptions
            sample_descriptions = bank_df['Description'].head(20).tolist()
            print("\n📝 Sample descriptions:")
            for i, desc in enumerate(sample_descriptions, 1):
                print(f"{i}. {desc}")
            
            # Extract vendors
            vendors = extract_real_vendors(bank_df['Description'])
            
            print(f"\n🏢 Extracted vendors ({len(vendors)}):")
            for i, vendor in enumerate(vendors, 1):
                print(f"{i}. {vendor}")
            
            if not vendors:
                print("❌ No real vendors found. The data might not contain business names.")
        else:
            print("❌ No 'Description' column found in bank data")
    else:
        print("❌ Bank data file not found. Please upload and process a bank statement first.") 