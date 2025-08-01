#!/usr/bin/env python3
"""
Simple Test for Advanced AI Features
Tests core advanced AI capabilities without external dependencies
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_revenue_ai_system import AdvancedRevenueAISystem

def create_simple_test_data():
    """Create simple test data"""
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Create simple test data
    test_data = []
    
    # Revenue transactions
    for i, date in enumerate(dates[:50]):
        test_data.append({
            'Date': date,
            'Description': f'Steel Sales Revenue {i+1}',
            'Amount': np.random.uniform(50000, 200000),
            'Type': 'Credit'
        })
    
    # Expense transactions
    for i, date in enumerate(dates[50:]):
        test_data.append({
            'Date': date,
            'Description': f'Raw Material Purchase {i+1}',
            'Amount': -np.random.uniform(10000, 50000),
            'Type': 'Debit'
        })
    
    return pd.DataFrame(test_data)

def test_basic_advanced_features():
    """Test basic advanced AI features"""
    print("🚀 Testing Basic Advanced AI Features...")
    
    # Initialize AI system
    ai_system = AdvancedRevenueAISystem()
    
    # Create test data
    test_data = create_simple_test_data()
    print(f"✅ Created test data with {len(test_data)} transactions")
    
    # Test enhanced functions
    enhanced_functions = [
        ('A6 - Operating Expenses', ai_system.enhanced_analyze_operating_expenses),
        ('A7 - Accounts Payable', ai_system.enhanced_analyze_accounts_payable_terms),
        ('A8 - Inventory Turnover', ai_system.enhanced_analyze_inventory_turnover),
        ('A9 - Loan Repayments', ai_system.enhanced_analyze_loan_repayments),
        ('A10 - Tax Obligations', ai_system.enhanced_analyze_tax_obligations),
        ('A11 - Capital Expenditure', ai_system.enhanced_analyze_capital_expenditure),
        ('A12 - Equity & Debt Inflows', ai_system.enhanced_analyze_equity_debt_inflows),
        ('A13 - Other Income/Expenses', ai_system.enhanced_analyze_other_income_expenses),
        ('A14 - Cash Flow Types', ai_system.enhanced_analyze_cash_flow_types)
    ]
    
    results = {}
    success_count = 0
    
    for name, func in enhanced_functions:
        try:
            print(f"\n📊 Testing {name}...")
            result = func(test_data)
            
            if 'error' not in result:
                # Check for advanced AI features
                if 'advanced_ai_features' in result:
                    ai_features = result['advanced_ai_features']
                    print(f"  ✅ {name} - Advanced AI features found:")
                    
                    for feature_name, feature_data in ai_features.items():
                        if isinstance(feature_data, dict):
                            if 'recommendations' in feature_data:
                                print(f"    🤖 {feature_name}: {len(feature_data['recommendations'])} recommendations")
                            elif 'forecast_total' in feature_data:
                                print(f"    📈 {feature_name}: ₹{feature_data['forecast_total']:,.2f} forecast")
                            elif 'count' in feature_data:
                                print(f"    🔍 {feature_name}: {feature_data['count']} detected")
                            else:
                                print(f"    📊 {feature_name}: {len(feature_data)} data points")
                        else:
                            print(f"    📊 {feature_name}: {feature_data}")
                else:
                    print(f"  ⚠️ {name} - No advanced AI features found")
                
                results[name] = result
                success_count += 1
            else:
                print(f"  ❌ {name} - Error: {result['error']}")
                
        except Exception as e:
            print(f"  ❌ {name} - Exception: {str(e)}")
    
    # Print summary
    print(f"\n🎯 Test Summary:")
    print(f"  ✅ Successful: {success_count}/{len(enhanced_functions)}")
    print(f"  ❌ Failed: {len(enhanced_functions) - success_count}/{len(enhanced_functions)}")
    
    # Check specific AI features
    ai_features_found = []
    for name, result in results.items():
        if 'advanced_ai_features' in result:
            ai_features = result['advanced_ai_features']
            for feature_name in ai_features.keys():
                if feature_name not in ai_features_found:
                    ai_features_found.append(feature_name)
    
    print(f"\n🤖 Advanced AI Features Implemented:")
    for feature in ai_features_found:
        print(f"  ✅ {feature}")
    
    return success_count == len(enhanced_functions)

def test_individual_ai_models():
    """Test individual AI models"""
    print("\n🧠 Testing Individual AI Models...")
    
    ai_system = AdvancedRevenueAISystem()
    
    # Test data
    test_data = np.random.normal(1000, 200, 30)
    
    # Test Anomaly Detection
    try:
        anomalies = ai_system._detect_anomalies(test_data, 'statistical')
        anomaly_count = np.sum(anomalies)
        print(f"  ✅ Anomaly Detection: {anomaly_count} anomalies found")
    except Exception as e:
        print(f"  ❌ Anomaly Detection: {str(e)}")
    
    # Test Confidence Intervals
    try:
        forecast = np.random.normal(1000, 100, 6)
        confidence_intervals = ai_system._calculate_confidence_intervals(forecast)
        print(f"  ✅ Confidence Intervals: {confidence_intervals['confidence_level']} level")
    except Exception as e:
        print(f"  ❌ Confidence Intervals: {str(e)}")
    
    # Test Scenario Planning
    try:
        scenarios = ai_system._generate_scenarios(forecast)
        print(f"  ✅ Scenario Planning: {len(scenarios)} scenarios generated")
    except Exception as e:
        print(f"  ❌ Scenario Planning: {str(e)}")
    
    # Test Clustering
    try:
        customer_data = [
            {'avg_payment_time': 30, 'payment_reliability': 0.8, 'avg_amount': 10000, 'payment_frequency': 1, 'credit_score': 700},
            {'avg_payment_time': 45, 'payment_reliability': 0.9, 'avg_amount': 15000, 'payment_frequency': 2, 'credit_score': 750},
            {'avg_payment_time': 60, 'payment_reliability': 0.7, 'avg_amount': 8000, 'payment_frequency': 1, 'credit_score': 650}
        ]
        cluster_analysis = ai_system._cluster_customer_behavior(customer_data)
        print(f"  ✅ Customer Clustering: {len(cluster_analysis)} clusters found")
    except Exception as e:
        print(f"  ❌ Customer Clustering: {str(e)}")

if __name__ == "__main__":
    print("🚀 Simple Advanced AI Features Test Suite")
    print("=" * 50)
    
    # Test individual models
    test_individual_ai_models()
    
    # Test comprehensive features
    success = test_basic_advanced_features()
    
    if success:
        print("\n🎉 All Advanced AI Features Tested Successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some Advanced AI Features Failed!")
        sys.exit(1) 