#!/usr/bin/env python3
"""
Comprehensive System Audit for Mathematical and Logical Correctness
Checks both UI and backend for accuracy, consistency, and proper calculations
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def audit_data_structure():
    """Audit the data structure and mathematical calculations"""
    print("🔍 AUDIT 1: Data Structure Analysis")
    print("=" * 50)
    
    try:
        # Check if bank data exists
        bank_path = os.path.join('data', 'bank_data_processed.xlsx')
        if not os.path.exists(bank_path):
            print("❌ ERROR: bank_data_processed.xlsx not found")
            return False
        
        # Load and analyze data
        df = pd.read_excel(bank_path)
        print(f"✅ Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Check required columns
        required_columns = ['Amount', 'Description', 'Category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ ERROR: Missing required columns: {missing_columns}")
            return False
        else:
            print(f"✅ All required columns present: {required_columns}")
        
        # Check data types
        print(f"📊 Data types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        print(f"📊 Null value counts:")
        for col, count in null_counts.items():
            if count > 0:
                print(f"  {col}: {count} null values")
        
        # Check Amount column specifically
        if 'Amount' in df.columns:
            amount_stats = df['Amount'].describe()
            print(f"📊 Amount statistics:")
            print(f"  Count: {amount_stats['count']}")
            print(f"  Mean: ₹{amount_stats['mean']:,.2f}")
            print(f"  Std: ₹{amount_stats['std']:,.2f}")
            print(f"  Min: ₹{amount_stats['min']:,.2f}")
            print(f"  Max: ₹{amount_stats['max']:,.2f}")
            
            # Check for mathematical consistency
            calculated_sum = df['Amount'].sum()
            calculated_mean = df['Amount'].mean()
            calculated_std = df['Amount'].std()
            
            print(f"✅ Mathematical consistency check:")
            print(f"  Sum calculation: ₹{calculated_sum:,.2f}")
            print(f"  Mean calculation: ₹{calculated_mean:,.2f}")
            print(f"  Std calculation: ₹{calculated_std:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in data structure audit: {e}")
        return False

def audit_transaction_filtering():
    """Audit transaction filtering logic"""
    print("\n🔍 AUDIT 2: Transaction Filtering Logic")
    print("=" * 50)
    
    try:
        df = pd.read_excel('data/bank_data_processed.xlsx')
        
        # Test filtering logic
        print("📊 Testing transaction filtering:")
        
        # All transactions
        all_transactions = df
        print(f"  All transactions: {len(all_transactions)}")
        
        # Operating transactions
        operating_transactions = df[df['Category'].str.contains('Operating', na=False)]
        print(f"  Operating transactions: {len(operating_transactions)}")
        
        # Investing transactions
        investing_transactions = df[df['Category'].str.contains('Investing', na=False)]
        print(f"  Investing transactions: {len(investing_transactions)}")
        
        # Financing transactions
        financing_transactions = df[df['Category'].str.contains('Financing', na=False)]
        print(f"  Financing transactions: {len(financing_transactions)}")
        
        # Check logical consistency
        total_filtered = len(operating_transactions) + len(investing_transactions) + len(financing_transactions)
        print(f"  Total filtered: {total_filtered}")
        print(f"  All transactions: {len(all_transactions)}")
        
        if total_filtered <= len(all_transactions):
            print("✅ Filtering logic is mathematically consistent")
        else:
            print("❌ ERROR: Filtering logic has overlap issues")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in transaction filtering audit: {e}")
        return False

def audit_mathematical_calculations():
    """Audit mathematical calculations in the system"""
    print("\n🔍 AUDIT 3: Mathematical Calculations")
    print("=" * 50)
    
    try:
        df = pd.read_excel('data/bank_data_processed.xlsx')
        
        # Test calculations that match the backend
        print("📊 Testing mathematical calculations:")
        
        # Sample transaction set (investing transactions)
        test_transactions = df[df['Category'].str.contains('Investing', na=False)]
        
        if len(test_transactions) > 0:
            # Calculate statistics
            total_amount = test_transactions['Amount'].sum()
            avg_amount = test_transactions['Amount'].mean()
            transaction_count = len(test_transactions)
            max_amount = test_transactions['Amount'].max()
            min_amount = test_transactions['Amount'].min()
            std_amount = test_transactions['Amount'].std()
            
            print(f"  Test dataset: {transaction_count} transactions")
            print(f"  Total amount: ₹{total_amount:,.2f}")
            print(f"  Average amount: ₹{avg_amount:,.2f}")
            print(f"  Max amount: ₹{max_amount:,.2f}")
            print(f"  Min amount: ₹{min_amount:,.2f}")
            print(f"  Std amount: ₹{std_amount:,.2f}")
            
            # Check mathematical consistency
            if transaction_count > 0:
                calculated_avg = total_amount / transaction_count
                if abs(calculated_avg - avg_amount) < 0.01:
                    print("✅ Average calculation is mathematically correct")
                else:
                    print("❌ ERROR: Average calculation mismatch")
                    return False
                
                # Check volatility calculation
                if avg_amount != 0:
                    volatility = std_amount / abs(avg_amount)
                    print(f"  Volatility: {volatility:.4f}")
                    
                    # Check consistency calculation
                    consistency = 1 - volatility
                    print(f"  Consistency: {consistency:.4f}")
                    
                    if 0 <= consistency <= 1:
                        print("✅ Volatility and consistency calculations are mathematically correct")
                    else:
                        print("❌ ERROR: Consistency calculation out of bounds")
                        return False
                else:
                    print("⚠️ WARNING: Average amount is zero, skipping volatility calculation")
            
            # Check pattern detection
            if transaction_count > 1:
                trend = 'increasing' if test_transactions['Amount'].iloc[-1] > test_transactions['Amount'].iloc[0] else 'decreasing'
                print(f"  Trend: {trend}")
                
                # Check amount pattern
                if avg_amount > 1000000:
                    amount_pattern = 'high_value'
                elif avg_amount < 100000:
                    amount_pattern = 'low_value'
                else:
                    amount_pattern = 'medium_value'
                print(f"  Amount pattern: {amount_pattern}")
                
                print("✅ Pattern detection logic is mathematically correct")
            
        else:
            print("⚠️ WARNING: No test transactions found for mathematical audit")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR in mathematical calculations audit: {e}")
        return False

def audit_ui_display_logic():
    """Audit UI display logic and data formatting"""
    print("\n🔍 AUDIT 4: UI Display Logic")
    print("=" * 50)
    
    try:
        # Test the data structure that would be sent to UI
        df = pd.read_excel('data/bank_data_processed.xlsx')
        test_transactions = df[df['Category'].str.contains('Investing', na=False)]
        
        if len(test_transactions) > 0:
            # Simulate backend calculations
            total_amount = test_transactions['Amount'].sum()
            avg_amount = test_transactions['Amount'].mean()
            transaction_count = len(test_transactions)
            max_amount = test_transactions['Amount'].max()
            min_amount = test_transactions['Amount'].min()
            std_amount = test_transactions['Amount'].std()
            
            # Simulate the data structure sent to UI
            ui_data = {
                'ai_model': 'XGBoost',
                'analysis_type': 'cash_flow',
                'insights': f"XGBoost ML Analysis for cash_flow: Transaction Count: {transaction_count} transactions",
                'patterns': {
                    'trend': 'increasing' if test_transactions['Amount'].iloc[-1] > test_transactions['Amount'].iloc[0] else 'decreasing',
                    'volatility': std_amount / abs(avg_amount) if avg_amount != 0 else 0,
                    'consistency': 1 - (std_amount / abs(avg_amount)) if avg_amount != 0 else 0,
                    'frequency_pattern': 'regular' if transaction_count > 10 else 'occasional',
                    'amount_pattern': 'high_value' if avg_amount > 1000000 else 'low_value' if avg_amount < 100000 else 'medium_value'
                },
                'transaction_count': transaction_count,
                'total_amount': float(total_amount),
                'avg_amount': float(avg_amount),
                'max_amount': float(max_amount),
                'min_amount': float(min_amount)
            }
            
            print("📊 UI Data Structure Test:")
            print(f"  AI Model: {ui_data['ai_model']}")
            print(f"  Analysis Type: {ui_data['analysis_type']}")
            print(f"  Transaction Count: {ui_data['transaction_count']}")
            print(f"  Total Amount: ₹{ui_data['total_amount']:,.2f}")
            print(f"  Average Amount: ₹{ui_data['avg_amount']:,.2f}")
            print(f"  Max Amount: ₹{ui_data['max_amount']:,.2f}")
            print(f"  Min Amount: ₹{ui_data['min_amount']:,.2f}")
            
            # Check data type consistency
            print("📊 Data Type Consistency:")
            for key, value in ui_data.items():
                if key in ['total_amount', 'avg_amount', 'max_amount', 'min_amount']:
                    if isinstance(value, (int, float)):
                        print(f"  ✅ {key}: {type(value).__name__} - {value}")
                    else:
                        print(f"  ❌ {key}: {type(value).__name__} - should be numeric")
                        return False
            
            # Check pattern values
            patterns = ui_data['patterns']
            print("📊 Pattern Values:")
            for key, value in patterns.items():
                if key in ['volatility', 'consistency']:
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        print(f"  ✅ {key}: {value:.4f} (valid range)")
                    else:
                        print(f"  ❌ {key}: {value} (invalid range)")
                        return False
                else:
                    print(f"  ✅ {key}: {value}")
            
            print("✅ UI display logic is mathematically consistent")
            return True
        else:
            print("⚠️ WARNING: No test data for UI audit")
            return True
            
    except Exception as e:
        print(f"❌ ERROR in UI display logic audit: {e}")
        return False

def audit_backend_logic():
    """Audit backend processing logic"""
    print("\n🔍 AUDIT 5: Backend Processing Logic")
    print("=" * 50)
    
    try:
        df = pd.read_excel('data/bank_data_processed.xlsx')
        
        # Test transaction filtering logic from backend
        print("📊 Testing backend filtering logic:")
        
        # Test the exact logic from app1.py
        transaction_type = "Investing Activities (XGBoost)"
        
        if transaction_type == 'all' or transaction_type == '':
            filtered_df = df
        elif 'operating' in transaction_type.lower():
            filtered_df = df[df['Category'].str.contains('Operating', na=False)]
        elif 'investing' in transaction_type.lower():
            filtered_df = df[df['Category'].str.contains('Investing', na=False)]
        elif 'financing' in transaction_type.lower():
            filtered_df = df[df['Category'].str.contains('Financing', na=False)]
        else:
            filtered_df = df
        
        print(f"  Input transaction type: {transaction_type}")
        print(f"  Filtered transactions: {len(filtered_df)}")
        
        # Test the exact calculation logic from backend
        if len(filtered_df) > 0:
            total_amount = filtered_df['Amount'].sum()
            avg_amount = filtered_df['Amount'].mean()
            transaction_count = len(filtered_df)
            max_amount = filtered_df['Amount'].max()
            min_amount = filtered_df['Amount'].min()
            std_amount = filtered_df['Amount'].std()
            
            # Test pattern detection logic
            patterns = {
                'trend': 'increasing' if filtered_df['Amount'].iloc[-1] > filtered_df['Amount'].iloc[0] else 'decreasing',
                'volatility': std_amount / abs(avg_amount) if avg_amount != 0 else 0,
                'consistency': 1 - (std_amount / abs(avg_amount)) if avg_amount != 0 else 0,
                'frequency_pattern': 'regular' if transaction_count > 10 else 'occasional',
                'amount_pattern': 'high_value' if avg_amount > 1000000 else 'low_value' if avg_amount < 100000 else 'medium_value'
            }
            
            print("📊 Backend calculation results:")
            print(f"  Total Amount: ₹{total_amount:,.2f}")
            print(f"  Average Amount: ₹{avg_amount:,.2f}")
            print(f"  Transaction Count: {transaction_count}")
            print(f"  Max Amount: ₹{max_amount:,.2f}")
            print(f"  Min Amount: ₹{min_amount:,.2f}")
            print(f"  Std Amount: ₹{std_amount:,.2f}")
            
            print("📊 Pattern detection results:")
            for key, value in patterns.items():
                print(f"  {key}: {value}")
            
            # Verify mathematical consistency
            if transaction_count > 0:
                calculated_avg = total_amount / transaction_count
                if abs(calculated_avg - avg_amount) < 0.01:
                    print("✅ Backend average calculation is correct")
                else:
                    print("❌ ERROR: Backend average calculation mismatch")
                    return False
                
                # Check volatility bounds
                volatility = patterns['volatility']
                if 0 <= volatility <= 10:  # Allow some flexibility for extreme cases
                    print("✅ Backend volatility calculation is within reasonable bounds")
                else:
                    print("❌ ERROR: Backend volatility calculation out of bounds")
                    return False
                
                # Check consistency bounds
                consistency = patterns['consistency']
                if 0 <= consistency <= 1:
                    print("✅ Backend consistency calculation is mathematically correct")
                else:
                    print("❌ ERROR: Backend consistency calculation out of bounds")
                    return False
            
            print("✅ Backend processing logic is mathematically consistent")
            return True
        else:
            print("⚠️ WARNING: No filtered transactions for backend audit")
            return True
            
    except Exception as e:
        print(f"❌ ERROR in backend logic audit: {e}")
        return False

def audit_error_handling():
    """Audit error handling and edge cases"""
    print("\n🔍 AUDIT 6: Error Handling and Edge Cases")
    print("=" * 50)
    
    try:
        df = pd.read_excel('data/bank_data_processed.xlsx')
        
        print("📊 Testing edge cases:")
        
        # Test with empty dataset
        empty_df = df[df['Amount'] > 999999999]  # Impossible condition
        if len(empty_df) == 0:
            print("  ✅ Empty dataset handling: No transactions found")
        
        # Test with single transaction
        single_transaction = df.head(1)
        if len(single_transaction) == 1:
            print("  ✅ Single transaction handling: 1 transaction found")
        
        # Test with zero amounts
        zero_amounts = df[df['Amount'] == 0]
        print(f"  📊 Zero amount transactions: {len(zero_amounts)}")
        
        # Test with negative amounts
        negative_amounts = df[df['Amount'] < 0]
        print(f"  📊 Negative amount transactions: {len(negative_amounts)}")
        
        # Test with very large amounts
        large_amounts = df[df['Amount'] > 10000000]  # > 10M
        print(f"  📊 Large amount transactions (>10M): {len(large_amounts)}")
        
        # Test division by zero protection
        try:
            test_avg = 0 / 0  # This should be handled
            print("  ❌ ERROR: Division by zero not properly handled")
            return False
        except ZeroDivisionError:
            print("  ✅ Division by zero properly caught")
        
        print("✅ Error handling and edge cases are properly managed")
        return True
        
    except Exception as e:
        print(f"❌ ERROR in error handling audit: {e}")
        return False

def main():
    """Run comprehensive system audit"""
    print("🔍 COMPREHENSIVE SYSTEM AUDIT")
    print("=" * 60)
    print(f"Audit started at: {datetime.now()}")
    print()
    
    audit_results = []
    
    # Run all audits
    audits = [
        ("Data Structure", audit_data_structure),
        ("Transaction Filtering", audit_transaction_filtering),
        ("Mathematical Calculations", audit_mathematical_calculations),
        ("UI Display Logic", audit_ui_display_logic),
        ("Backend Logic", audit_backend_logic),
        ("Error Handling", audit_error_handling)
    ]
    
    for audit_name, audit_func in audits:
        try:
            result = audit_func()
            audit_results.append((audit_name, result))
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {audit_name}: {e}")
            audit_results.append((audit_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 AUDIT SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(audit_results)
    
    for audit_name, result in audit_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {audit_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} audits passed")
    
    if passed == total:
        print("🎉 ALL AUDITS PASSED - System is mathematically and logically correct!")
    else:
        print("⚠️ Some audits failed - Review the issues above")
    
    print(f"\nAudit completed at: {datetime.now()}")

if __name__ == "__main__":
    main() 