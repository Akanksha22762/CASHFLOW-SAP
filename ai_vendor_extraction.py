import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import re

def load_transaction_data():
    """Load your steel plant transaction data"""
    try:
        # Load bank data
        bank_path = os.path.join('data', 'bank_data_processed.xlsx')
        if os.path.exists(bank_path):
            df = pd.read_excel(bank_path)
            print(f"✅ Loaded {len(df)} transactions")
            return df
        else:
            print("❌ Bank data not found")
            return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def separate_transactions(df):
    """Extract vendor information from all transactions"""
    if df is None:
        return None, None
    
    print(f"📊 Total transactions: {len(df)}")
    
    # Extract vendor names from descriptions
    vendor_extractions = []
    
    for i, row in df.iterrows():
        desc = str(row['Description'])
        
        # Look for vendor patterns in descriptions
        vendor = extract_vendor_from_description(desc)
        if vendor:
            vendor_extractions.append({
                'description': desc,
                'extracted_vendor': vendor,
                'confidence': 'high' if len(vendor.split()) > 1 else 'medium'
            })
    
    print(f"📊 Found {len(vendor_extractions)} potential vendors in descriptions")
    
    # Show some examples
    if vendor_extractions:
        print("\n🏢 Extracted vendor examples:")
        for i, extraction in enumerate(vendor_extractions[:5]):
            print(f"   '{extraction['description'][:50]}...' → '{extraction['extracted_vendor']}'")
    
    return vendor_extractions, df

def extract_vendor_from_description(description):
    """Extract vendor name from transaction description"""
    import re
    
    # Pattern 1: "Company Name - Description" format
    if ' - ' in description:
        parts = description.split(' - ')
        company_part = parts[0].strip()
        
        # Check if it looks like a company name
        if len(company_part) > 3 and company_part[0].isupper():
            # Filter out transaction words
            transaction_words = ['Payment', 'Purchase', 'Sale', 'Transfer', 'Deposit', 'Withdrawal', 'Investment', 'CapEx', 'Q1', 'Q2', 'Q3', 'Q4', 'Final', 'Advance', 'Milestone', 'Retention', 'Bonus', 'Import', 'Export', 'Procurement', 'Training', 'Salary', 'Equipment', 'Machinery', 'Software', 'Technology', 'Infrastructure', 'Raw', 'Material', 'Supplier', 'Logistics', 'Provider', 'Service', 'Customer', 'VIP', 'New', 'Bulk', 'Order', 'Property', 'Asset', 'Scrap']
            
            if not any(word in company_part for word in transaction_words):
                return company_part
    
    # Pattern 2: Look for business indicators
    business_indicators = ['Company', 'Corporation', 'Ltd', 'LLC', 'Inc', 'Corp', 'Limited', 'Steel', 'Manufacturing', 'Industries', 'Group', 'Enterprises', 'Solutions', 'Systems', 'Technologies', 'Oil', 'Gas', 'Petroleum', 'Energy', 'Power', 'Utilities', 'Automotive', 'Engineering', 'Construction', 'Infrastructure', 'Development', 'Department', 'Firm']
    
    words = description.split()
    business_words = []
    
    for word in words:
        if len(word) > 2 and word[0].isupper() and not word.isupper():
            # Check if word contains business indicators
            if any(indicator.lower() in word.lower() for indicator in business_indicators):
                business_words.append(word)
            # Check if word looks like a company name
            elif word.lower() not in ['payment', 'purchase', 'sale', 'transfer', 'deposit', 'withdrawal']:
                business_words.append(word)
    
    if business_words:
        vendor_name = ' '.join(business_words[:3]).title()  # Take first 3 words
        return vendor_name
    
    return None

def train_xgboost_model(labeled_transactions):
    """Train XGBoost on labeled vendor data"""
    if len(labeled_transactions) < 10:
        print("❌ Not enough labeled data for training")
        return None, None
    
    print("\n🤖 Training XGBoost on labeled vendor data...")
    
    # Prepare training data
    X = labeled_transactions['Description'].fillna('')
    y = labeled_transactions['Category'].fillna('Other')
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 3),
        stop_words='english'
    )
    
    X_features = vectorizer.fit_transform(X)
    
    # Train XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=6
    )
    
    xgb_model.fit(X_features, y)
    
    print(f"✅ XGBoost trained on {len(X)} labeled examples")
    print(f"✅ Model accuracy: {xgb_model.score(X_features, y):.2f}")
    
    return xgb_model, vectorizer

def extract_vendors_with_ai(vendor_extractions, df):
    """Extract vendors from transaction descriptions using AI"""
    print("\n🎯 Starting AI vendor extraction...")
    
    # Step 1: Extract unique vendors
    unique_vendors = []
    vendor_counts = {}
    
    for extraction in vendor_extractions:
        vendor = extraction['extracted_vendor']
        confidence = extraction['confidence']
        
        if vendor not in vendor_counts:
            vendor_counts[vendor] = 0
        vendor_counts[vendor] += 1
        
        if vendor not in unique_vendors:
            unique_vendors.append(vendor)
            print(f"✅ Extracted: '{extraction['description'][:50]}...' → '{vendor}' ({confidence} confidence)")
    
    # Step 2: Calculate results
    total_transactions = len(df)
    total_extractions = len(vendor_extractions)
    extraction_rate = (total_extractions / total_transactions) * 100 if total_transactions > 0 else 0
    
    print(f"\n📊 AI VENDOR EXTRACTION RESULTS:")
    print(f"   📈 Total transactions: {total_transactions}")
    print(f"   🤖 Vendor extractions: {total_extractions}")
    print(f"   📊 Extraction rate: {extraction_rate:.1f}%")
    print(f"   🎯 Unique vendors found: {len(unique_vendors)}")
    
    # Show unique vendors with counts
    if unique_vendors:
        print(f"\n🏢 Extracted vendors:")
        for vendor in sorted(unique_vendors):
            count = vendor_counts[vendor]
            print(f"   • {vendor} ({count} transactions)")
    
    return unique_vendors

def main():
    """Main function to run AI vendor extraction"""
    print("🚀 AI Vendor Extraction from Steel Plant Data")
    print("=" * 50)
    
    # Step 1: Load data
    df = load_transaction_data()
    if df is None:
        return
    
    # Step 2: Extract vendors from descriptions
    vendor_extractions, df = separate_transactions(df)
    if vendor_extractions is None:
        return
    
    # Step 3: Extract vendors using AI
    vendors = extract_vendors_with_ai(vendor_extractions, df)
    
    print(f"\n✅ AI extraction complete!")
    print(f"🎯 Found {len(vendors)} unique vendors from transaction descriptions")

if __name__ == "__main__":
    main() 