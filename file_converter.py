import pandas as pd
import os
from datetime import datetime
import traceback

def convert_files_to_2file_system():
    """
    Convert your existing 4 files into 2 standardized files for the reconciliation system
    """
    
    print("🔄 Converting files for 2-file reconciliation system...")
    
    # Input files (adjust paths as needed)
    input_files = {
        'transactions': 'steel_plant_transactions.xlsx',
        'bank': 'steel_plant_bank_statement.xlsx', 
        'ap_ar': 'steel_plant_ap_ar_data.xlsx',
        'master': 'steel_plant_master_data.xlsx'
    }
    
    # Check if files exist and inspect them
    print("\n🔍 Checking files...")
    available_files = {}
    for file_type, filename in input_files.items():
        if os.path.exists(filename):
            print(f"   ✅ Found: {filename}")
            available_files[file_type] = filename
        else:
            print(f"   ❌ Missing: {filename}")
    
    if len(available_files) < 2:
        print("❌ Need at least 2 files (transactions and bank) to proceed")
        return False
    
    try:
        # ====== STEP 1: CREATE SAP DATA FILE ======
        print("\n📊 Step 1: Creating SAP Data File...")
        
        sap_data_parts = []
        
        # Read transactions file (main SAP data)
        if 'transactions' in available_files:
            print("   - Reading transactions data...")
            try:
                transactions_df = read_excel_safely(available_files['transactions'])
                if transactions_df is not None and not transactions_df.empty:
                    print(f"     📈 Loaded {len(transactions_df)} transaction records")
                    print(f"     📋 Columns: {list(transactions_df.columns)}")
                    
                    # Standardize transactions data
                    sap_standardized = standardize_sap_data(transactions_df, 'Transactions')
                    if not sap_standardized.empty:
                        sap_data_parts.append(sap_standardized)
                else:
                    print("     ⚠️ Transactions file is empty")
            except Exception as e:
                print(f"     ❌ Error reading transactions: {e}")
        
        # Read AP/AR file if available
        if 'ap_ar' in available_files:
            print("   - Reading AP/AR data...")
            try:
                apar_df = read_excel_safely(available_files['ap_ar'])
                if apar_df is not None and not apar_df.empty:
                    print(f"     📈 Loaded {len(apar_df)} AP/AR records")
                    print(f"     📋 Columns: {list(apar_df.columns)}")
                    
                    # Standardize AP/AR data
                    apar_standardized = standardize_sap_data(apar_df, 'AP/AR')
                    if not apar_standardized.empty:
                        sap_data_parts.append(apar_standardized)
                else:
                    print("     ⚠️ AP/AR file is empty")
            except Exception as e:
                print(f"     ❌ Error reading AP/AR: {e}")
        
        # Combine SAP data parts
        if sap_data_parts:
            print("   - Combining SAP data...")
            sap_data = pd.concat(sap_data_parts, ignore_index=True, sort=False)
        else:
            print("   ❌ No valid SAP data found")
            return False
        
        # Read master data for lookups (optional)
        if 'master' in available_files:
            print("   - Reading master data...")
            try:
                master_df = read_excel_safely(available_files['master'])
                if master_df is not None and not master_df.empty:
                    print(f"     📈 Loaded {len(master_df)} master records")
                    print(f"     📋 Columns: {list(master_df.columns)}")
                    sap_data = add_master_data_lookups(sap_data, master_df)
                else:
                    print("     ⚠️ Master data file is empty")
            except Exception as e:
                print(f"     ❌ Error reading master data: {e}")
        
        # Save SAP file
        sap_filename = 'SAP_Data_Combined.xlsx'
        sap_data.to_excel(sap_filename, index=False)
        print(f"   ✅ SAP data saved to: {sap_filename}")
        print(f"   📈 Total SAP records: {len(sap_data)}")
        
        # ====== STEP 2: CREATE BANK DATA FILE ======
        print("\n🏦 Step 2: Creating Bank Statement File...")
        
        if 'bank' not in available_files:
            print("   ❌ Bank file not found")
            return False
        
        # Read bank statement
        print("   - Reading bank statement data...")
        try:
            bank_df = read_excel_safely(available_files['bank'])
            if bank_df is not None and not bank_df.empty:
                print(f"     📈 Loaded {len(bank_df)} bank records")
                print(f"     📋 Columns: {list(bank_df.columns)}")
                
                # Standardize bank data
                bank_data = standardize_bank_data(bank_df)
                
                # Save bank file
                bank_filename = 'Bank_Statement_Combined.xlsx'
                bank_data.to_excel(bank_filename, index=False)
                print(f"   ✅ Bank data saved to: {bank_filename}")
                print(f"   📈 Total bank records: {len(bank_data)}")
            else:
                print("   ❌ Bank file is empty")
                return False
        except Exception as e:
            print(f"   ❌ Error reading bank data: {e}")
            return False
        
        # ====== STEP 3: GENERATE SUMMARY ======
        print("\n📋 Conversion Summary:")
        print("="*50)
        print(f"SAP File: {sap_filename}")
        print(f"  - Records: {len(sap_data)}")
        print(f"  - Columns: {', '.join(sap_data.columns)}")
        
        if 'Date' in sap_data.columns:
            try:
                sap_data['Date'] = pd.to_datetime(sap_data['Date'], errors='coerce')
                valid_dates = sap_data['Date'].dropna()
                if not valid_dates.empty:
                    print(f"  - Date range: {valid_dates.min()} to {valid_dates.max()}")
            except:
                print("  - Date range: Could not determine")
        
        print(f"\nBank File: {bank_filename}")
        print(f"  - Records: {len(bank_data)}")
        print(f"  - Columns: {', '.join(bank_data.columns)}")
        
        if 'Date' in bank_data.columns:
            try:
                bank_data['Date'] = pd.to_datetime(bank_data['Date'], errors='coerce')
                valid_dates = bank_data['Date'].dropna()
                if not valid_dates.empty:
                    print(f"  - Date range: {valid_dates.min()} to {valid_dates.max()}")
            except:
                print("  - Date range: Could not determine")
        
        print(f"\n🎯 Next Steps:")
        print(f"1. Upload {sap_filename} as 'SAP File'")
        print(f"2. Upload {bank_filename} as 'Bank File'")
        print(f"3. Run reconciliation in the web interface")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        print("\n🔍 Detailed error:")
        traceback.print_exc()
        return False

def read_excel_safely(filename):
    """Safely read Excel file with error handling"""
    try:
        # Try reading the first sheet
        df = pd.read_excel(filename)
        
        # Check if dataframe is empty or has no valid data
        if df.empty:
            print(f"     ⚠️ File {filename} is empty")
            return None
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        if df.empty:
            print(f"     ⚠️ File {filename} has no valid data after cleaning")
            return None
        
        return df
        
    except Exception as e:
        print(f"     ❌ Error reading {filename}: {e}")
        
        # Try reading as CSV if Excel fails
        try:
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                print(f"     🔄 Trying to read as CSV...")
                csv_filename = filename.replace('.xlsx', '.csv').replace('.xls', '.csv')
                if os.path.exists(csv_filename):
                    df = pd.read_csv(csv_filename)
                    return df.dropna(how='all').dropna(axis=1, how='all')
        except:
            pass
        
        return None

def standardize_sap_data(df, source_name):
    """Standardize SAP transactions data with better error handling"""
    try:
        standardized = df.copy()
        
        print(f"     🔄 Standardizing {source_name} data...")
        print(f"     📊 Original columns: {list(standardized.columns)}")
        
        # Column mapping for common SAP field names (case insensitive)
        column_mapping = {
            # Amount fields
            'amount (inr)': 'Amount',
            'amount': 'Amount',
            'amt': 'Amount',
            'value': 'Amount',
            'transaction_amount': 'Amount',
            'debit': 'Amount',
            'credit': 'Amount',
            
            # Description fields  
            'description': 'Description',
            'particulars': 'Description',
            'narration': 'Description',
            'item_text': 'Description',
            'details': 'Description',
            'remarks': 'Description',
            
            # Date fields
            'date': 'Date',
            'transaction_date': 'Date',
            'posting_date': 'Date',
            'entry_date': 'Date',
            
            # Type fields
            'type': 'Type',
            'transaction_type': 'Type',
            'entry_type': 'Type',
            'dr/cr': 'Type',
            
            # Reference fields
            'reference': 'Reference',
            'document_no': 'Reference',
            'invoice_no': 'Reference',
            'ref_no': 'Reference'
        }
        
        # Rename columns (case insensitive)
        columns_renamed = {}
        for old_col in standardized.columns:
            old_col_clean = str(old_col).lower().strip()
            if old_col_clean in column_mapping:
                new_col = column_mapping[old_col_clean]
                standardized = standardized.rename(columns={old_col: new_col})
                columns_renamed[old_col] = new_col
        
        if columns_renamed:
            print(f"     📝 Renamed columns: {columns_renamed}")
        
        # Ensure Amount column exists and is numeric
        if 'Amount' not in standardized.columns:
            print("     🔍 Looking for amount column...")
            # Find first numeric column
            numeric_cols = standardized.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                amount_col = numeric_cols[0]
                standardized = standardized.rename(columns={amount_col: 'Amount'})
                print(f"     📊 Using '{amount_col}' as Amount column")
            else:
                print("     ⚠️ No numeric column found, setting Amount to 0")
                standardized['Amount'] = 0
        
        # Clean Amount column
        try:
            standardized['Amount'] = pd.to_numeric(standardized['Amount'], errors='coerce')
            standardized['Amount'] = standardized['Amount'].fillna(0)
            print(f"     💰 Amount range: {standardized['Amount'].min()} to {standardized['Amount'].max()}")
        except Exception as e:
            print(f"     ⚠️ Error processing Amount column: {e}")
            standardized['Amount'] = 0
        
        # Ensure Description column exists
        if 'Description' not in standardized.columns:
            print("     🔍 Looking for description column...")
            text_cols = standardized.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                desc_col = text_cols[0]
                standardized = standardized.rename(columns={desc_col: 'Description'})
                print(f"     📝 Using '{desc_col}' as Description column")
            else:
                print("     ⚠️ No text column found, using default description")
                standardized['Description'] = f'{source_name} Transaction'
        
        # Clean Description column
        standardized['Description'] = standardized['Description'].astype(str).str.strip()
        standardized['Description'] = standardized['Description'].replace(['nan', 'NaN', ''], f'{source_name} Transaction')
        
        # Ensure Date column exists
        if 'Date' not in standardized.columns:
            print("     📅 No date column found, using current date")
            standardized['Date'] = datetime.now().strftime('%Y-%m-%d')
        else:
            # Try to standardize dates
            try:
                standardized['Date'] = pd.to_datetime(standardized['Date'], errors='coerce')
                standardized['Date'] = standardized['Date'].dt.strftime('%Y-%m-%d')
                standardized['Date'] = standardized['Date'].fillna(datetime.now().strftime('%Y-%m-%d'))
            except Exception as e:
                print(f"     ⚠️ Error processing dates: {e}")
                standardized['Date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Ensure Type column exists
        if 'Type' not in standardized.columns:
            print("     🏷️ Creating Type column based on Amount")
            standardized['Type'] = standardized['Amount'].apply(
                lambda x: 'Credit' if float(x) > 0 else 'Debit' if float(x) < 0 else 'Unknown'
            )
        
        # Add category and source
        print("     🎯 Adding categories...")
        standardized['Category'] = standardized['Description'].apply(categorize_transaction)
        standardized['Source'] = source_name
        
        # Select final columns
        final_columns = ['Amount', 'Description', 'Date', 'Type', 'Category', 'Source']
        extra_columns = [col for col in standardized.columns if col not in final_columns]
        final_columns.extend(extra_columns)
        
        result = standardized[final_columns]
        
        print(f"     ✅ Standardized {len(result)} records")
        print(f"     📋 Final columns: {list(result.columns)}")
        
        return result
        
    except Exception as e:
        print(f"     ❌ Error standardizing {source_name} data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def standardize_bank_data(df):
    """Standardize bank statement data with better error handling"""
    try:
        standardized = df.copy()
        
        print("     🔄 Standardizing bank data...")
        print(f"     📊 Original columns: {list(standardized.columns)}")
        
        # Column mapping for bank fields
        column_mapping = {
            'amount (inr)': 'Amount',
            'amount': 'Amount',
            'amt': 'Amount',
            'debit': 'Debit_Amount',
            'credit': 'Credit_Amount',
            'withdrawal': 'Debit_Amount',
            'deposit': 'Credit_Amount',
            'dr': 'Debit_Amount',
            'cr': 'Credit_Amount',
            
            'description': 'Description',
            'narration': 'Description',
            'particulars': 'Description',
            'details': 'Description',
            'remarks': 'Description',
            
            'date': 'Date',
            'transaction_date': 'Date',
            'value_date': 'Date',
            'trans_date': 'Date',
            
            'type': 'Type',
            'transaction_type': 'Type',
            'dr/cr': 'Type',
            
            'balance (inr)': 'Balance',
            'balance': 'Balance',
            'running_balance': 'Balance',
            'closing_balance': 'Balance'
        }
        
        # Rename columns (case insensitive)
        columns_renamed = {}
        for old_col in standardized.columns:
            old_col_clean = str(old_col).lower().strip()
            if old_col_clean in column_mapping:
                new_col = column_mapping[old_col_clean]
                standardized = standardized.rename(columns={old_col: new_col})
                columns_renamed[old_col] = new_col
        
        if columns_renamed:
            print(f"     📝 Renamed columns: {columns_renamed}")
        
        # Handle Amount column - check for separate debit/credit columns
        if 'Amount' not in standardized.columns:
            if 'Debit_Amount' in standardized.columns and 'Credit_Amount' in standardized.columns:
                print("     💰 Combining separate Debit/Credit columns...")
                debit = pd.to_numeric(standardized['Debit_Amount'], errors='coerce').fillna(0)
                credit = pd.to_numeric(standardized['Credit_Amount'], errors='coerce').fillna(0)
                standardized['Amount'] = credit - debit  # Positive for credits, negative for debits
                
                # Create Type column
                standardized['Type'] = standardized['Amount'].apply(
                    lambda x: 'Credit' if x > 0 else 'Debit' if x < 0 else 'Unknown'
                )
            else:
                print("     🔍 Looking for amount column...")
                numeric_cols = standardized.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    amount_col = numeric_cols[0]
                    standardized = standardized.rename(columns={amount_col: 'Amount'})
                    print(f"     📊 Using '{amount_col}' as Amount column")
                else:
                    print("     ⚠️ No numeric column found, setting Amount to 0")
                    standardized['Amount'] = 0
        
        # Clean Amount column
        try:
            standardized['Amount'] = pd.to_numeric(standardized['Amount'], errors='coerce')
            standardized['Amount'] = standardized['Amount'].fillna(0)
            print(f"     💰 Amount range: {standardized['Amount'].min()} to {standardized['Amount'].max()}")
        except Exception as e:
            print(f"     ⚠️ Error processing Amount column: {e}")
            standardized['Amount'] = 0
        
        # Ensure other required columns exist
        if 'Description' not in standardized.columns:
            text_cols = standardized.select_dtypes(include=['object']).columns
            if len(text_cols) > 0:
                desc_col = text_cols[0]
                standardized = standardized.rename(columns={desc_col: 'Description'})
                print(f"     📝 Using '{desc_col}' as Description column")
            else:
                standardized['Description'] = 'Bank Transaction'
        
        # Clean Description
        standardized['Description'] = standardized['Description'].astype(str).str.strip()
        standardized['Description'] = standardized['Description'].replace(['nan', 'NaN', ''], 'Bank Transaction')
        
        # Handle Date
        if 'Date' not in standardized.columns:
            standardized['Date'] = datetime.now().strftime('%Y-%m-%d')
        else:
            try:
                standardized['Date'] = pd.to_datetime(standardized['Date'], errors='coerce')
                standardized['Date'] = standardized['Date'].dt.strftime('%Y-%m-%d')
                standardized['Date'] = standardized['Date'].fillna(datetime.now().strftime('%Y-%m-%d'))
            except:
                standardized['Date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Handle Type
        if 'Type' not in standardized.columns:
            standardized['Type'] = standardized['Amount'].apply(
                lambda x: 'Credit' if float(x) > 0 else 'Debit' if float(x) < 0 else 'Unknown'
            )
        
        # Add source
        standardized['Source'] = 'Bank'
        
        # Select final columns
        final_columns = ['Amount', 'Description', 'Date', 'Type', 'Source']
        extra_columns = [col for col in standardized.columns if col not in final_columns]
        final_columns.extend(extra_columns)
        
        result = standardized[final_columns]
        
        print(f"     ✅ Standardized {len(result)} records")
        print(f"     📋 Final columns: {list(result.columns)}")
        
        return result
        
    except Exception as e:
        print(f"     ❌ Error standardizing bank data: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def add_master_data_lookups(sap_data, master_df):
    """Add master data lookups to SAP data"""
    try:
        print("     🔗 Adding master data lookups...")
        
        # Convert column names to lowercase for matching
        master_cols_lower = [col.lower() for col in master_df.columns]
        
        # Look for vendor and category columns
        vendor_col = None
        category_col = None
        
        for i, col in enumerate(master_cols_lower):
            if 'vendor' in col or 'supplier' in col or 'party' in col:
                vendor_col = master_df.columns[i]
            if 'category' in col or 'type' in col or 'class' in col:
                category_col = master_df.columns[i]
        
        if vendor_col and category_col:
            print(f"     📊 Using vendor column: {vendor_col}")
            print(f"     📊 Using category column: {category_col}")
            
            # Create vendor lookup dictionary
            vendor_lookup = {}
            for _, row in master_df.iterrows():
                vendor_name = str(row[vendor_col]).lower().strip()
                category = str(row[category_col]).strip()
                if vendor_name and vendor_name != 'nan':
                    vendor_lookup[vendor_name] = category
            
            print(f"     📈 Created lookup for {len(vendor_lookup)} vendors")
            
            # Apply lookups
            def lookup_vendor_category(description):
                desc_lower = str(description).lower()
                for vendor, category in vendor_lookup.items():
                    if vendor in desc_lower:
                        return category
                return None
            
            sap_data['Vendor_Category'] = sap_data['Description'].apply(lookup_vendor_category)
            
            # Count successful lookups
            successful_lookups = sap_data['Vendor_Category'].notna().sum()
            print(f"     ✅ Applied {successful_lookups} vendor category lookups")
        else:
            print("     ⚠️ Could not find vendor and category columns in master data")
        
        return sap_data
        
    except Exception as e:
        print(f"     ❌ Error adding master data lookups: {e}")
        return sap_data

def categorize_transaction(description):
    """Categorize transaction based on description keywords"""
    try:
        desc_lower = str(description).lower()
        
        categories = {
            'utilities': ['electricity', 'utility', 'power', 'water', 'gas', 'electric'],
            'payroll': ['salary', 'wages', 'payroll', 'employee', 'staff'],
            'maintenance': ['maintenance', 'repair', 'service', 'fix'],
            'logistics': ['transport', 'logistics', 'freight', 'delivery', 'shipping'],
            'materials': ['material', 'raw material', 'steel', 'iron', 'coal', 'ore'],
            'sales': ['sales', 'revenue', 'customer', 'receipt', 'income'],
            'tax': ['tax', 'gst', 'vat', 'duty', 'cess'],
            'scrap': ['scrap', 'waste', 'disposal'],
            'consultancy': ['consultancy', 'consultant', 'advisory', 'professional'],
            'insurance': ['insurance', 'policy', 'premium'],
            'interest': ['interest', 'bank charges', 'charges', 'fees']
        }
        
        for category, keywords in categories.items():
            if any(keyword in desc_lower for keyword in keywords):
                return category.title()
        
        return 'Other'
        
    except:
        return 'Other'

if __name__ == "__main__":
    print("🏭 Steel Plant File Converter")
    print("Converting 4 files → 2 files for reconciliation system")
    print("="*60)
    
    success = convert_files_to_2file_system()
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("You can now upload the generated files to the reconciliation system.")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")
    
    input("\nPress Enter to exit...")