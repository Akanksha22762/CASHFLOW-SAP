#!/usr/bin/env python3
"""
Test Real Output - See what's actually being displayed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def create_test_data():
    """Create realistic test data"""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(90)]
    test_data = []
    
    # Revenue transactions
    for i in range(40):
        test_data.append({
            'Date': dates[i],
            'Description': f'Steel Sales Revenue {i+1}',
            'Amount': np.random.uniform(50000, 200000),
            'Type': 'Credit'
        })
    
    # Expense transactions
    for i in range(25):
        test_data.append({
            'Date': dates[i+40],
            'Description': f'Operating Expense {i+1}',
            'Amount': -np.random.uniform(10000, 50000),
            'Type': 'Debit'
        })
    
    # Loan repayments
    for i in range(8):
        test_data.append({
            'Date': dates[i+65],
            'Description': f'Loan Repayment {i+1}',
            'Amount': -np.random.uniform(30000, 80000),
            'Type': 'Debit'
        })
    
    # Tax payments
    for i in range(6):
        test_data.append({
            'Date': dates[i+73],
            'Description': f'Tax Payment {i+1}',
            'Amount': -np.random.uniform(15000, 40000),
            'Type': 'Debit'
        })
    
    # CapEx
    for i in range(4):
        test_data.append({
            'Date': dates[i+79],
            'Description': f'Capital Expenditure {i+1}',
            'Amount': -np.random.uniform(100000, 300000),
            'Type': 'Debit'
        })
    
    # Other transactions
    for i in range(7):
        if i % 2 == 0:
            test_data.append({
                'Date': dates[i+83],
                'Description': f'Other Income {i+1}',
                'Amount': np.random.uniform(5000, 25000),
                'Type': 'Credit'
            })
        else:
            test_data.append({
                'Date': dates[i+83],
                'Description': f'Other Expense {i+1}',
                'Amount': -np.random.uniform(3000, 15000),
                'Type': 'Debit'
            })
    
    return pd.DataFrame(test_data)

def test_parameter_outputs():
    """Test all parameter outputs to see what's wrong"""
    ai_system = AdvancedRevenueAISystem()
    test_data = create_test_data()
    
    print("=" * 80)
    print("TESTING PARAMETER OUTPUTS")
    print("=" * 80)
    
    # Test all 14 parameters
    parameters = [
        ('A1', 'Historical Revenue Trends', ai_system.enhanced_analyze_historical_revenue_trends),
        ('A2', 'Sales Forecast', ai_system.xgboost_sales_forecasting),
        ('A3', 'Customer Contracts', ai_system.analyze_customer_contracts),
        ('A4', 'Pricing Models', ai_system.detect_pricing_models),
        ('A5', 'AR Aging', ai_system.calculate_dso_and_collection_probability),
        ('A6', 'Operating Expenses', ai_system.enhanced_analyze_operating_expenses),
        ('A7', 'Accounts Payable', ai_system.enhanced_analyze_accounts_payable_terms),
        ('A8', 'Inventory Turnover', ai_system.enhanced_analyze_inventory_turnover),
        ('A9', 'Loan Repayments', ai_system.enhanced_analyze_loan_repayments),
        ('A10', 'Tax Obligations', ai_system.enhanced_analyze_tax_obligations),
        ('A11', 'Capital Expenditure', ai_system.enhanced_analyze_capital_expenditure),
        ('A12', 'Equity & Debt Inflows', ai_system.enhanced_analyze_equity_debt_inflows),
        ('A13', 'Other Income/Expenses', ai_system.enhanced_analyze_other_income_expenses),
        ('A14', 'Cash Flow Types', ai_system.enhanced_analyze_cash_flow_types)
    ]
    
    for param_id, param_name, function in parameters:
        print(f"\n{param_id}: {param_name}")
        print("-" * 50)
        
        try:
            result = function(test_data)
            
            if 'error' in result:
                print(f"❌ ERROR: {result['error']}")
            else:
                print(f"✅ SUCCESS: {len(result)} metrics calculated")
                
                # Show key metrics that should be displayed
                key_metrics = [
                    'total_revenue', 'total_expenses', 'total_payables', 'inventory_value',
                    'total_repayments', 'total_taxes', 'total_capex', 'total_inflows',
                    'total_other', 'avg_revenue', 'avg_expense', 'avg_payable'
                ]
                
                print("📊 Key Metrics:")
                for metric in key_metrics:
                    if metric in result:
                        print(f"   {metric}: {result[metric]}")
                
                # Show advanced AI features
                if 'advanced_ai_features' in result:
                    ai_features = result['advanced_ai_features']
                    print(f"🤖 Advanced AI Features: {len(ai_features)}")
                    for feature in ai_features:
                        print(f"   - {feature}")
                else:
                    print("❌ No advanced AI features found")
                
                # Show analysis type
                if 'analysis_type' in result:
                    print(f"📋 Analysis Type: {result['analysis_type']}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
    
    print("\n" + "=" * 80)
    print("SUMMARY OF ISSUES")
    print("=" * 80)
    print("🔍 Check what's wrong with the output cards!")

if __name__ == "__main__":
    test_parameter_outputs() 