#!/usr/bin/env python3
"""
Debug the real dataset to understand the customer counting
"""

import pandas as pd
import numpy as np
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def debug_real_dataset():
    """Debug the real dataset customer counting"""
    try:
        # Initialize the AI system
        ai_system = AdvancedRevenueAISystem()
        print("✅ AI system initialized")
        
        # Load your actual data (you'll need to provide the path)
        # For now, let's simulate what might be happening
        print("🔍 Let's check what your real data looks like...")
        
        # Try to load the actual data
        try:
            # Try to load from the uploads folder
            import os
            uploads_dir = "uploads"
            if os.path.exists(uploads_dir):
                files = os.listdir(uploads_dir)
                print(f"📁 Files in uploads: {files}")
                
                # Look for bank data files
                bank_files = [f for f in files if 'bank' in f.lower()]
                if bank_files:
                    print(f"🏦 Bank files found: {bank_files}")
                    
                    # Try to load the first bank file
                    bank_file = os.path.join(uploads_dir, bank_files[0])
                    print(f"📊 Loading: {bank_file}")
                    
                    if bank_file.endswith('.xlsx'):
                        df = pd.read_excel(bank_file)
                    elif bank_file.endswith('.csv'):
                        df = pd.read_csv(bank_file)
                    else:
                        print("❌ Unsupported file format")
                        return
                    
                    print(f"✅ Loaded data shape: {df.shape}")
                    print(f"✅ Columns: {list(df.columns)}")
                    print(f"✅ Sample data:")
                    print(df.head(3).to_string())
                    
                    # Check for customer-related columns
                    customer_columns = [col for col in df.columns if 'customer' in col.lower() or 'client' in col.lower()]
                    print(f"🔍 Customer-related columns: {customer_columns}")
                    
                    # Check descriptions for customer patterns
                    if 'Description' in df.columns:
                        descriptions = df['Description'].head(10).tolist()
                        print(f"🔍 Sample descriptions: {descriptions}")
                        
                        # Test the regex extraction
                        customer_extractions = df['Description'].str.extract(r'(Customer\s+[A-Z]|Client\s+[A-Z]|Customer\s+\w+|Client\s+\w+)')
                        unique_customers = customer_extractions.iloc[:, 0].dropna().unique()
                        print(f"🔍 Extracted customers: {unique_customers}")
                        print(f"🔍 Total unique customers: {len(unique_customers)}")
                    
                else:
                    print("❌ No bank files found in uploads directory")
            else:
                print("❌ Uploads directory not found")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_real_dataset() 