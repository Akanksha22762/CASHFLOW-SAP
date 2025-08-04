"""
Test script for the Universal Data Adapter
=========================================
This script tests the Universal Data Adapter with various datasets to ensure it can handle any format.
"""

import pandas as pd
import numpy as np
import os
import logging
from universal_data_adapter import UniversalDataAdapter
from data_adapter_integration import preprocess_for_analysis, load_and_preprocess_file, get_adaptation_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_datasets():
    """Create various test datasets with different formats."""
    test_datasets = {}
    
    # Dataset 1: Standard format
    standard_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=20),
        'Description': [f"Transaction {i}" for i in range(20)],
        'Amount': [1000 * (i - 10) for i in range(20)],
        'Type': ['INWARD' if i % 2 == 0 else 'OUTWARD' for i in range(20)]
    })
    test_datasets['standard'] = standard_data
    
    # Dataset 2: Different column names
    different_columns = pd.DataFrame({
        'TransactionDate': pd.date_range(start='2023-01-01', periods=20),
        'Narration': [f"Payment {i}" for i in range(20)],
        'Value': [1000 * (i - 10) for i in range(20)],
        'EntryType': ['Credit' if i % 2 == 0 else 'Debit' for i in range(20)]
    })
    test_datasets['different_columns'] = different_columns
    
    # Dataset 3: Missing columns
    missing_columns = pd.DataFrame({
        'TransactionDate': pd.date_range(start='2023-01-01', periods=20),
        'Value': [1000 * (i - 10) for i in range(20)]
    })
    test_datasets['missing_columns'] = missing_columns
    
    # Dataset 4: Extra columns
    extra_columns = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=20),
        'Description': [f"Transaction {i}" for i in range(20)],
        'Amount': [1000 * (i - 10) for i in range(20)],
        'Type': ['INWARD' if i % 2 == 0 else 'OUTWARD' for i in range(20)],
        'Category': [f"Category {i % 5}" for i in range(20)],
        'Reference': [f"REF-{i:04d}" for i in range(20)],
        'Balance': [10000 + sum(1000 * (j - 10) for j in range(i+1)) for i in range(20)]
    })
    test_datasets['extra_columns'] = extra_columns
    
    # Dataset 5: Mixed data types
    mixed_types = pd.DataFrame({
        'Date': [f"2023-{i+1:02d}-01" for i in range(20)],
        'Description': [f"Transaction {i}" for i in range(20)],
        'Amount': [f"${1000 * (i - 10):,.2f}" for i in range(20)],
        'Type': [i % 2 for i in range(20)]
    })
    test_datasets['mixed_types'] = mixed_types
    
    # Dataset 6: Banking format
    banking_format = pd.DataFrame({
        'ValueDate': pd.date_range(start='2023-01-01', periods=20),
        'Particulars': [f"Banking Transaction {i}" for i in range(20)],
        'Withdrawal': [1000 * i if i % 2 == 1 else np.nan for i in range(20)],
        'Deposit': [1000 * i if i % 2 == 0 else np.nan for i in range(20)],
        'Balance': [10000 + sum(1000 * (j if j % 2 == 0 else -j) for j in range(i+1)) for i in range(20)]
    })
    test_datasets['banking_format'] = banking_format
    
    # Dataset 7: SAP format
    sap_format = pd.DataFrame({
        'PostingDate': pd.date_range(start='2023-01-01', periods=20),
        'DocumentNo': [f"DOC-{i:04d}" for i in range(20)],
        'AccountName': [f"Account {i % 5}" for i in range(20)],
        'DebitAmount': [1000 * i if i % 2 == 1 else 0 for i in range(20)],
        'CreditAmount': [1000 * i if i % 2 == 0 else 0 for i in range(20)],
        'Text': [f"SAP Transaction {i}" for i in range(20)]
    })
    test_datasets['sap_format'] = sap_format
    
    return test_datasets

def test_adapter_with_datasets():
    """Test the Universal Data Adapter with various datasets."""
    print("=" * 80)
    print("TESTING UNIVERSAL DATA ADAPTER")
    print("=" * 80)
    
    # Create test datasets
    test_datasets = create_test_datasets()
    
    # Initialize adapter
    adapter = UniversalDataAdapter()
    
    # Test each dataset
    for name, dataset in test_datasets.items():
        print(f"\nTesting dataset: {name}")
        print("-" * 50)
        
        print(f"Original columns: {dataset.columns.tolist()}")
        print(f"Original shape: {dataset.shape}")
        
        # Adapt the dataset
        try:
            adapted_data = adapter.adapt(dataset)
            
            print(f"Adapted columns: {adapted_data.columns.tolist()}")
            print(f"Adapted shape: {adapted_data.shape}")
            print(f"Column mapping: {adapter.column_mapping}")
            
            # Check if required columns are present
            required_columns = ['Date', 'Amount']
            missing = [col for col in required_columns if col not in adapted_data.columns]
            
            if missing:
                print(f"❌ Missing required columns: {missing}")
            else:
                print("✅ All required columns present")
            
            # Check data types
            if 'Date' in adapted_data.columns:
                if pd.api.types.is_datetime64_dtype(adapted_data['Date']):
                    print("✅ Date column has correct datetime type")
                else:
                    print(f"❌ Date column has incorrect type: {adapted_data['Date'].dtype}")
            
            if 'Amount' in adapted_data.columns:
                if pd.api.types.is_numeric_dtype(adapted_data['Amount']):
                    print("✅ Amount column has correct numeric type")
                else:
                    print(f"❌ Amount column has incorrect type: {adapted_data['Amount'].dtype}")
            
            # Print sample data
            print("\nSample of adapted data:")
            print(adapted_data.head(3).to_string())
            
        except Exception as e:
            print(f"❌ Adaptation failed: {str(e)}")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

def test_with_real_files():
    """Test the adapter with real files if available."""
    test_files = [
        "Bank_Statement_Combined.csv",
        "SAP_Data_Combined.csv",
        "steel_plant_bank_data.xlsx",
        "steel_plant_sap_data.xlsx"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING WITH REAL FILES")
    print("=" * 80)
    
    for file in test_files:
        if os.path.exists(file):
            print(f"\nTesting with file: {file}")
            print("-" * 50)
            
            try:
                # Use the adapter integration
                adapted_data = load_and_preprocess_file(file)
                
                print(f"File successfully adapted")
                print(f"Adapted shape: {adapted_data.shape}")
                print(f"Adapted columns: {adapted_data.columns.tolist()}")
                print(f"Column mapping: {get_adaptation_report().get('column_mapping', {})}")
                
                # Print sample data
                print("\nSample of adapted data:")
                print(adapted_data.head(3).to_string())
                
            except Exception as e:
                print(f"❌ Adaptation failed: {str(e)}")
        else:
            print(f"Skipping file {file} (not found)")
    
    print("\n" + "=" * 80)
    print("FILE TESTING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    # Test with synthetic datasets
    test_adapter_with_datasets()
    
    # Test with real files if available
    test_with_real_files()