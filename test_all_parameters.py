#!/usr/bin/env python3
"""
Test All Parameters - Comprehensive Check
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def create_comprehensive_test_data():
    """Create comprehensive test data with all types of transactions"""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    test_data = []
    
    # Revenue transactions (positive amounts)
    for i in range(30):
        test_data.append({
            'Date': dates[i],
            'Description': f'Revenue Transaction {i+1}',
            'Amount': np.random.uniform(10000, 100000),
            'Type': 'Credit'
        })
    
    # Expense transactions (negative amounts)
    for i in range(20):
        test_data.append({
            'Date': dates[i+30],
            'Description': f'Expense Transaction {i+1}',
            'Amount': -np.random.uniform(5000, 50000),
            'Type': 'Debit'
        })
    
    # Loan repayment transactions
    for i in range(10):
        test_data.append({
            'Date': dates[i+50],
            'Description': f'Loan Repayment {i+1}',
            'Amount': -np.random.uniform(20000, 100000),
            'Type': 'Debit'
        })
    
    # Tax transactions
    for i in range(8):
        test_data.append({
            'Date': dates[i+60],
            'Description': f'Tax Payment {i+1}',
            'Amount': -np.random.uniform(10000, 30000),
            'Type': 'Debit'
        })
    
    # CapEx transactions
    for i in range(5):
        test_data.append({
            'Date': dates[i+68],
            'Description': f'Capital Expenditure {i+1}',
            'Amount': -np.random.uniform(50000, 200000),
            'Type': 'Debit'
        })
    
    # Equity/Debt inflows
    for i in range(7):
        test_data.append({
            'Date': dates[i+73],
            'Description': f'Funding Inflow {i+1}',
            'Amount': np.random.uniform(50000, 300000),
            'Type': 'Credit'
        })
    
    # Other income/expenses
    for i in range(10):
        if i % 2 == 0:
            test_data.append({
                'Date': dates[i+80],
                'Description': f'Other Income {i+1}',
                'Amount': np.random.uniform(5000, 25000),
                'Type': 'Credit'
            })
        else:
            test_data.append({
                'Date': dates[i+80],
                'Description': f'Other Expense {i+1}',
                'Amount': -np.random.uniform(3000, 15000),
                'Type': 'Debit'
            })
    
    return pd.DataFrame(test_data)

def test_all_parameters():
    """Test all 14 parameters"""
    ai_system = AdvancedRevenueAISystem()
    test_data = create_comprehensive_test_data()
    
    print("=" * 80)
    print("COMPREHENSIVE PARAMETER TESTING")
    print("=" * 80)
    
    # Test all 14 parameters
    parameters = [
        ('A1', 'Historical Revenue Trends', ai_system.analyze_historical_revenue_trends),
        ('A2', 'Sales Forecast', ai_system.xgboost_sales_forecasting),
        ('A3', 'Customer Contracts', ai_system.analyze_customer_contracts),
        ('A4', 'Pricing Models', ai_system.detect_pricing_models),
        ('A5', 'AR Aging', ai_system.calculate_dso_and_collection_probability),
        ('A6', 'Operating Expenses', ai_system.analyze_operating_expenses),
        ('A7', 'Accounts Payable', ai_system.analyze_accounts_payable_terms),
        ('A8', 'Inventory Turnover', ai_system.analyze_inventory_turnover),
        ('A9', 'Loan Repayments', ai_system.analyze_loan_repayments),
        ('A10', 'Tax Obligations', ai_system.analyze_tax_obligations),
        ('A11', 'Capital Expenditure', ai_system.analyze_capital_expenditure),
        ('A12', 'Equity & Debt Inflows', ai_system.analyze_equity_debt_inflows),
        ('A13', 'Other Income/Expenses', ai_system.analyze_other_income_expenses),
        ('A14', 'Cash Flow Types', ai_system.analyze_cash_flow_types)
    ]
    
    results = {}
    
    for param_id, param_name, function in parameters:
        print(f"\n{param_id}: {param_name}")
        print("-" * 50)
        
        try:
            result = function(test_data)
            results[param_id] = result
            
            if 'error' in result:
                print(f"❌ ERROR: {result['error']}")
            else:
                print(f"✅ SUCCESS: {len(result)} metrics calculated")
                # Show key metrics
                for key, value in result.items():
                    if isinstance(value, (str, int, float)) and key not in ['error', 'analysis_type']:
                        print(f"   {key}: {value}")
                    elif key in ['error', 'analysis_type']:
                        print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            results[param_id] = {'error': str(e)}
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    working_count = 0
    error_count = 0
    
    for param_id, result in results.items():
        if 'error' in result:
            print(f"❌ {param_id}: FAILED - {result['error']}")
            error_count += 1
        else:
            print(f"✅ {param_id}: WORKING - {len(result)} metrics")
            working_count += 1
    
    print(f"\n📊 RESULTS: {working_count} working, {error_count} failed")
    print(f"🎯 SUCCESS RATE: {(working_count/14)*100:.1f}%")

if __name__ == "__main__":
    test_all_parameters() 