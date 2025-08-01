#!/usr/bin/env python3
"""
Comprehensive Test for Advanced AI Features
Tests all enhanced analysis functions with advanced AI capabilities
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_revenue_ai_system import AdvancedRevenueAISystem

def create_test_data():
    """Create comprehensive test data for all parameters"""
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365)]
    
    # Create comprehensive test data
    test_data = []
    
    # Revenue transactions
    for i, date in enumerate(dates[:100]):
        test_data.append({
            'Date': date,
            'Description': f'Steel Sales Revenue {i+1}',
            'Amount': np.random.uniform(50000, 200000),
            'Type': 'Credit'
        })
    
    # Expense transactions
    for i, date in enumerate(dates[100:200]):
        test_data.append({
            'Date': date,
            'Description': f'Raw Material Purchase {i+1}',
            'Amount': -np.random.uniform(10000, 50000),
            'Type': 'Debit'
        })
    
    # Payable transactions
    for i, date in enumerate(dates[200:250]):
        test_data.append({
            'Date': date,
            'Description': f'Vendor Payment {i+1}',
            'Amount': -np.random.uniform(5000, 25000),
            'Type': 'Debit'
        })
    
    # Inventory transactions
    for i, date in enumerate(dates[250:300]):
        test_data.append({
            'Date': date,
            'Description': f'Inventory Purchase {i+1}',
            'Amount': -np.random.uniform(15000, 75000),
            'Type': 'Debit'
        })
    
    # Loan transactions
    for i, date in enumerate(dates[300:350]):
        test_data.append({
            'Date': date,
            'Description': f'Loan EMI Payment {i+1}',
            'Amount': -np.random.uniform(10000, 30000),
            'Type': 'Debit'
        })
    
    # Tax transactions
    for i, date in enumerate(dates[350:380]):
        test_data.append({
            'Date': date,
            'Description': f'GST Tax Payment {i+1}',
            'Amount': -np.random.uniform(5000, 15000),
            'Type': 'Debit'
        })
    
    # CapEx transactions
    for i, date in enumerate(dates[380:400]):
        test_data.append({
            'Date': date,
            'Description': f'Equipment Purchase {i+1}',
            'Amount': -np.random.uniform(50000, 200000),
            'Type': 'Debit'
        })
    
    # Funding transactions
    for i, date in enumerate(dates[400:420]):
        test_data.append({
            'Date': date,
            'Description': f'Investment Received {i+1}',
            'Amount': np.random.uniform(100000, 500000),
            'Type': 'Credit'
        })
    
    # Other transactions
    for i, date in enumerate(dates[420:450]):
        test_data.append({
            'Date': date,
            'Description': f'Asset Sale {i+1}',
            'Amount': np.random.uniform(25000, 100000),
            'Type': 'Credit'
        })
    
    # Cash flow transactions
    for i, date in enumerate(dates[450:500]):
        test_data.append({
            'Date': date,
            'Description': f'Cash Flow Transaction {i+1}',
            'Amount': np.random.choice([-1, 1]) * np.random.uniform(1000, 50000),
            'Type': 'Mixed'
        })
    
    return pd.DataFrame(test_data)

def test_advanced_ai_features():
    """Test all advanced AI features"""
    print("🚀 Testing Advanced AI Features...")
    
    # Initialize AI system
    ai_system = AdvancedRevenueAISystem()
    
    # Create test data
    test_data = create_test_data()
    print(f"✅ Created test data with {len(test_data)} transactions")
    
    # Test all enhanced functions
    enhanced_functions = [
        ('A1 - Historical Revenue Trends', ai_system.enhanced_analyze_historical_revenue_trends),
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
    
    # Test comprehensive summary
    print(f"\n📋 Testing Advanced AI Summary...")
    try:
        summary = ai_system.get_advanced_ai_summary(test_data)
        if 'error' not in summary:
            print(f"  ✅ Summary generated successfully:")
            print(f"    🤖 AI Models Used: {summary.get('ai_models_used', [])}")
            print(f"    📈 Predictions Generated: {len(summary.get('predictions_generated', []))}")
            print(f"    💡 Optimization Recommendations: {len(summary.get('optimization_recommendations', []))}")
            print(f"    ⚠️ Risk Assessments: {len(summary.get('risk_assessments', []))}")
        else:
            print(f"  ❌ Summary Error: {summary['error']}")
    except Exception as e:
        print(f"  ❌ Summary Exception: {str(e)}")
    
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
    test_data = np.random.normal(1000, 200, 50)
    
    # Test LSTM
    try:
        lstm_forecast = ai_system._forecast_with_lstm(test_data, 6)
        if lstm_forecast is not None:
            print(f"  ✅ LSTM Forecasting: {len(lstm_forecast)} predictions")
        else:
            print(f"  ❌ LSTM Forecasting: Failed")
    except Exception as e:
        print(f"  ❌ LSTM Forecasting: {str(e)}")
    
    # Test ARIMA
    try:
        arima_model = ai_system._fit_arima_model(test_data)
        if arima_model is not None:
            arima_forecast = arima_model.forecast(steps=6)
            print(f"  ✅ ARIMA Forecasting: {len(arima_forecast)} predictions")
        else:
            print(f"  ❌ ARIMA Forecasting: Failed")
    except Exception as e:
        print(f"  ❌ ARIMA Forecasting: {str(e)}")
    
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

if __name__ == "__main__":
    print("🚀 Advanced AI Features Test Suite")
    print("=" * 50)
    
    # Test individual models
    test_individual_ai_models()
    
    # Test comprehensive features
    success = test_advanced_ai_features()
    
    if success:
        print("\n🎉 All Advanced AI Features Tested Successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some Advanced AI Features Failed!")
        sys.exit(1) 