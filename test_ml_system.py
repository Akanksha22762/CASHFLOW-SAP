#!/usr/bin/env python3
"""
Test script for the new lightweight AI/ML system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the ML system
try:
    from app1 import lightweight_ai, ml_based_categorize
    print("✅ Successfully imported ML system from app1.py")
except ImportError as e:
    print(f"❌ Error importing ML system: {e}")
    exit(1)

def test_ml_system():
    """Test the lightweight AI/ML system"""
    print("\n🧪 Testing Lightweight AI/ML System")
    print("=" * 50)
    
    # Create test data
    test_data = pd.DataFrame({
        'Description': [
            'Salary payment to employees',
            'Electricity bill payment',
            'Raw material purchase from vendor',
            'Customer payment for steel coils',
            'Bank loan repayment',
            'Equipment maintenance service',
            'GST tax payment',
            'Freight charges for delivery',
            'Office supplies purchase',
            'Interest payment on loan'
        ],
        'Amount': [
            -500000,  # Negative for expenses
            -150000,
            -2000000,
            3000000,  # Positive for income
            -1000000,
            -75000,
            -250000,
            -50000,
            -25000,
            -100000
        ],
        'Date': [
            datetime.now() - timedelta(days=i) for i in range(10)
        ],
        'Type': [
            'Debit', 'Debit', 'Debit', 'Credit', 'Debit', 
            'Debit', 'Debit', 'Debit', 'Debit', 'Debit'
        ]
    })
    
    print(f"📊 Test data created: {len(test_data)} transactions")
    
    # Test 1: Basic categorization without training
    print("\n🔍 Test 1: Basic ML categorization (without training)")
    for _, row in test_data.iterrows():
        category = ml_based_categorize(row['Description'], row['Amount'], row['Type'])
        print(f"   {row['Description'][:30]:<30} -> {category}")
    
    # Test 2: Train ML models
    print("\n🎯 Test 2: Training ML models")
    
    # Add categories for training
    training_data = test_data.copy()
    training_data['Category'] = [
        'Operating Activities',  # Salary
        'Operating Activities',  # Electricity
        'Operating Activities',  # Raw materials
        'Operating Activities',  # Customer payment
        'Financing Activities',  # Loan repayment
        'Operating Activities',  # Maintenance
        'Operating Activities',  # Tax
        'Operating Activities',  # Freight
        'Operating Activities',  # Office supplies
        'Financing Activities'   # Interest
    ]
    
    success = lightweight_ai.train_transaction_classifier(training_data)
    print(f"   Training result: {'✅ Success' if success else '❌ Failed'}")
    
    # Test 3: Categorization after training
    print("\n🤖 Test 3: ML categorization after training")
    for _, row in test_data.iterrows():
        category = ml_based_categorize(row['Description'], row['Amount'], row['Type'])
        print(f"   {row['Description'][:30]:<30} -> {category}")
    
    # Test 4: Anomaly detection
    print("\n🔍 Test 4: Anomaly detection")
    anomalies = lightweight_ai.detect_anomalies_ml(test_data)
    print(f"   Anomalies detected: {len(anomalies)}")
    if anomalies:
        print(f"   Anomaly indices: {anomalies}")
    
    # Test 5: Cash flow forecasting
    print("\n📈 Test 5: Cash flow forecasting")
    forecast = lightweight_ai.forecast_cash_flow_ml(test_data, days_ahead=3)
    if forecast:
        print(f"   Forecast model: {forecast.get('model', 'Unknown')}")
        print(f"   Predictions: {forecast.get('predictions', [])}")
    else:
        print("   Forecasting not available")
    
    print("\n✅ ML system test completed!")

def test_performance():
    """Test performance of ML system"""
    print("\n⚡ Performance Test")
    print("=" * 30)
    
    # Create larger dataset
    large_data = pd.DataFrame({
        'Description': [f'Transaction {i}' for i in range(100)],
        'Amount': np.random.uniform(-100000, 100000, 100),
        'Date': [datetime.now() - timedelta(days=i) for i in range(100)],
        'Type': ['Debit' if i % 2 == 0 else 'Credit' for i in range(100)]
    })
    
    import time
    start_time = time.time()
    
    # Test categorization speed
    categories = []
    for _, row in large_data.iterrows():
        category = ml_based_categorize(row['Description'], row['Amount'], row['Type'])
        categories.append(category)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"📊 Processed {len(large_data)} transactions in {processing_time:.2f} seconds")
    print(f"⚡ Speed: {len(large_data)/processing_time:.1f} transactions/second")
    
    # Show category distribution
    category_counts = pd.Series(categories).value_counts()
    print(f"📈 Category distribution:")
    for cat, count in category_counts.items():
        print(f"   {cat}: {count} transactions")

if __name__ == "__main__":
    print("🚀 Starting ML System Tests")
    print("=" * 50)
    
    try:
        test_ml_system()
        test_performance()
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 