import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import re
import os

def analyze_real_vendors():
    """Analyze what real vendors exist in your data"""
    print("🔍 ANALYZING REAL VENDORS IN YOUR DATA")
    print("=" * 50)
    
    # Load your data
    try:
        df = pd.read_excel('data/bank_data_processed.xlsx')
        print(f"✅ Loaded {len(df)} transactions")
        
        # Look for actual vendor patterns
        print("\n📊 ANALYZING TRANSACTION DESCRIPTIONS:")
        
        # Pattern 1: Look for "Payment to [Vendor Name]"
        payment_to_pattern = r'Payment to ([A-Za-z\s&]+?)(?:\s+\d+|\s*-\s*|\s*$)'
        
        # Pattern 2: Look for "Vendor Name - Description"
        vendor_dash_pattern = r'^([A-Za-z\s&]+?)\s*-\s*'
        
        # Pattern 3: Look for business indicators
        business_indicators = ['Ltd', 'LLC', 'Inc', 'Corp', 'Company', 'Corporation', 'Limited', 'Department', 'Firm', 'Provider', 'Supplier']
        
        real_vendors = []
        
        for i, row in df.iterrows():
            desc = str(row['Description'])
            
            # Method 1: Extract from "Payment to Vendor"
            payment_matches = re.findall(payment_to_pattern, desc)
            for match in payment_matches:
                if len(match.strip()) > 3:
                    real_vendors.append(match.strip())
                    print(f"✅ Found vendor: '{match.strip()}' from '{desc[:60]}...'")
            
            # Method 2: Extract from "Vendor - Description"
            dash_matches = re.findall(vendor_dash_pattern, desc)
            for match in dash_matches:
                if len(match.strip()) > 3:
                    real_vendors.append(match.strip())
                    print(f"✅ Found vendor: '{match.strip()}' from '{desc[:60]}...'")
            
            # Method 3: Look for business indicators
            words = desc.split()
            for word in words:
                if any(indicator in word for indicator in business_indicators):
                    if len(word) > 3:
                        real_vendors.append(word)
                        print(f"✅ Found business: '{word}' from '{desc[:60]}...'")
        
        # Show results
        unique_vendors = list(set(real_vendors))
        print(f"\n📊 REAL VENDOR ANALYSIS RESULTS:")
        print(f"   🎯 Total unique vendors found: {len(unique_vendors)}")
        
        if unique_vendors:
            print(f"\n🏢 REAL VENDORS FOUND:")
            for vendor in sorted(unique_vendors):
                count = real_vendors.count(vendor)
                print(f"   • {vendor} ({count} transactions)")
        
        return unique_vendors
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def create_vendor_training_data():
    """Create training data with real vendor names"""
    print("\n🤖 CREATING VENDOR TRAINING DATA")
    print("=" * 50)
    
    # Load data
    df = pd.read_excel('data/bank_data_processed.xlsx')
    
    # Create training examples
    training_data = []
    
    # Example 1: If description contains "Railway Department" → vendor is "Railway Department"
    # Example 2: If description contains "Construction Company" → vendor is "Construction Company"
    
    vendor_keywords = {
        'Railway Department': ['Railway Department', 'Railway'],
        'Construction Company': ['Construction Company', 'Construction'],
        'Engineering Firm': ['Engineering Firm', 'Engineering'],
        'Oil & Gas Company': ['Oil & Gas Company', 'Oil Gas'],
        'Automotive Manufacturer': ['Automotive Manufacturer', 'Automotive'],
        'Defense Contractor': ['Defense Contractor', 'Defense'],
        'Shipbuilding Yard': ['Shipbuilding Yard', 'Shipbuilding'],
        'Real Estate Developer': ['Real Estate Developer', 'Real Estate'],
        'Infrastructure Project': ['Infrastructure Project', 'Infrastructure'],
        'Technology Provider': ['Technology Provider', 'Technology'],
        'Equipment Supplier': ['Equipment Supplier', 'Equipment'],
        'Raw Material Supplier': ['Raw Material Supplier', 'Raw Material'],
        'Maintenance Contractor': ['Maintenance Contractor', 'Maintenance'],
        'Logistics Provider': ['Logistics Provider', 'Logistics'],
        'Service Provider': ['Service Provider', 'Service']
    }
    
    for vendor_name, keywords in vendor_keywords.items():
        for keyword in keywords:
            # Find transactions containing this keyword
            matching_transactions = df[df['Description'].str.contains(keyword, case=False, na=False)]
            
            for _, row in matching_transactions.iterrows():
                training_data.append({
                    'description': row['Description'],
                    'vendor': vendor_name,
                    'confidence': 'high'
                })
                print(f"✅ Training: '{row['Description'][:50]}...' → '{vendor_name}'")
    
    print(f"\n📊 TRAINING DATA CREATED:")
    print(f"   🎯 Total training examples: {len(training_data)}")
    print(f"   🏢 Unique vendors: {len(set([item['vendor'] for item in training_data]))}")
    
    return training_data

def train_vendor_classifier(training_data):
    """Train XGBoost to classify vendors"""
    print("\n🤖 TRAINING VENDOR CLASSIFIER")
    print("=" * 50)
    
    if not training_data:
        print("❌ No training data available")
        return None, None
    
    # Prepare training data
    descriptions = [item['description'] for item in training_data]
    vendors = [item['vendor'] for item in training_data]
    
    # Create label encoder for vendor names
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    vendor_labels = label_encoder.fit_transform(vendors)
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1, 3))
    X_features = vectorizer.fit_transform(descriptions)
    
    # Train XGBoost classifier
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_features, vendor_labels)
    
    print(f"✅ XGBoost trained on {len(training_data)} examples")
    print(f"✅ Model accuracy: {xgb_model.score(X_features, vendor_labels):.2f}")
    print(f"✅ Unique vendors: {len(label_encoder.classes_)}")
    
    return xgb_model, vectorizer, label_encoder

def extract_real_vendors_with_ai():
    """Extract real vendors using trained AI"""
    print("\n🎯 EXTRACTING REAL VENDORS WITH AI")
    print("=" * 50)
    
    # Step 1: Analyze existing vendors
    existing_vendors = analyze_real_vendors()
    
    # Step 2: Create training data
    training_data = create_vendor_training_data()
    
    # Step 3: Train classifier
    xgb_model, vectorizer, label_encoder = train_vendor_classifier(training_data)
    
    if xgb_model is None:
        return []
    
    # Step 4: Predict vendors for all transactions
    df = pd.read_excel('data/bank_data_processed.xlsx')
    all_descriptions = df['Description'].fillna('')
    all_features = vectorizer.transform(all_descriptions)
    
    # Get predictions
    predictions = xgb_model.predict(all_features)
    confidence_scores = xgb_model.predict_proba(all_features).max(axis=1)
    
    # Convert numeric predictions back to vendor names
    predicted_vendors = label_encoder.inverse_transform(predictions)
    
    # Extract high-confidence predictions
    extracted_vendors = []
    for i, (desc, pred, conf) in enumerate(zip(all_descriptions, predicted_vendors, confidence_scores)):
        if conf > 0.6:  # High confidence threshold
            extracted_vendors.append(pred)
            print(f"✅ AI extracted: '{desc[:50]}...' → '{pred}' (confidence: {conf:.2f})")
    
    # Show results
    unique_vendors = list(set(extracted_vendors))
    print(f"\n📊 REAL VENDOR EXTRACTION RESULTS:")
    print(f"   🎯 Total transactions: {len(df)}")
    print(f"   🤖 High-confidence extractions: {len(extracted_vendors)}")
    print(f"   🏢 Unique real vendors: {len(unique_vendors)}")
    
    if unique_vendors:
        print(f"\n🏢 REAL VENDORS EXTRACTED:")
        for vendor in sorted(unique_vendors):
            count = extracted_vendors.count(vendor)
            print(f"   • {vendor} ({count} transactions)")
    
    return unique_vendors

def main():
    """Main function"""
    print("🚀 REAL VENDOR EXTRACTION SYSTEM")
    print("=" * 50)
    
    vendors = extract_real_vendors_with_ai()
    
    print(f"\n✅ Real vendor extraction complete!")
    print(f"🎯 Found {len(vendors)} real vendors")

if __name__ == "__main__":
    main() 