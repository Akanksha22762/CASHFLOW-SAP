#!/usr/bin/env python3
"""
Test script for Revenue Analysis System
Verifies that the revenue analysis is working properly with current data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_revenue_ai_system import AdvancedRevenueAISystem
    print("✅ Advanced Revenue AI System imported successfully!")
except ImportError as e:
    print(f"❌ Error importing Advanced Revenue AI System: {e}")
    sys.exit(1)

def create_test_data():
    """Create test data for revenue analysis"""
    print("📊 Creating test revenue data...")
    
    # Create sample bank data with revenue transactions
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate realistic revenue data
    np.random.seed(42)
    revenue_amounts = np.random.normal(50000, 15000, len(dates))  # Average ₹50k per day
    revenue_amounts = np.abs(revenue_amounts)  # Ensure positive amounts
    
    # Add some seasonal patterns
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    revenue_amounts = revenue_amounts * seasonal_factor
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'Date': dates,
        'Description': [f'Revenue Transaction {i+1}' for i in range(len(dates))],
        'Amount': revenue_amounts,
        'Type': ['Credit'] * len(dates)
    })
    
    print(f"✅ Created test data with {len(test_data)} transactions")
    print(f"📈 Total revenue: ₹{test_data['Amount'].sum():,.2f}")
    print(f"📊 Average daily revenue: ₹{test_data['Amount'].mean():,.2f}")
    
    return test_data

def test_revenue_analysis():
    """Test the revenue analysis system"""
    print("\n🧠 Testing Revenue Analysis System...")
    print("=" * 50)
    
    try:
        # Initialize the system
        revenue_ai = AdvancedRevenueAISystem()
        print("✅ Revenue AI System initialized")
        
        # Create test data
        test_data = create_test_data()
        
        # Run complete revenue analysis
        print("\n🔍 Running SMART OLLAMA Revenue Analysis...")
        results = revenue_ai.complete_revenue_analysis_system_smart_ollama(test_data)
        
        # Display results
        print("\n📊 REVENUE ANALYSIS RESULTS:")
        print("=" * 50)
        
        for key, result in results.items():
            print(f"\n🔹 {key.upper()}:")
            if isinstance(result, dict):
                for sub_key, value in result.items():
                    if sub_key not in ['method', 'accuracy', 'speed', 'grade']:
                        print(f"   {sub_key}: {value}")
            else:
                print(f"   Result: {result}")
        
        # Test individual components
        print("\n🔍 Testing Individual Components...")
        
        # Test historical trends
        enhanced_features = revenue_ai.extract_hybrid_features(test_data['Description'].tolist())
        historical_trends = revenue_ai.analyze_historical_revenue_trends_professional(test_data, enhanced_features)
        print(f"📈 Historical Trends - Total Revenue: ₹{historical_trends.get('total_revenue', 0):,.2f}")
        
        # Test pricing models
        pricing_models = revenue_ai.detect_pricing_models_professional(test_data, enhanced_features)
        print(f"💰 Pricing Strategy: {pricing_models.get('pricing_strategy', 'N/A')}")
        print(f"📊 Price Elasticity: {pricing_models.get('price_elasticity', 0)}")
        print(f"🏢 Revenue Model: {pricing_models.get('revenue_model', 'N/A')}")
        
        print("\n✅ Revenue Analysis Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in revenue analysis test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data():
    """Test with real uploaded data if available"""
    print("\n📁 Testing with Real Data...")
    
    # Check for uploaded data files
    data_files = [
        'Bank_Statement_Combined.xlsx',
        'SAP_Data_Combined.xlsx',
        'steel_plant_bank_data.xlsx'
    ]
    
    for file_name in data_files:
        if os.path.exists(file_name):
            print(f"📂 Found data file: {file_name}")
            try:
                data = pd.read_excel(file_name)
                print(f"✅ Loaded {len(data)} records from {file_name}")
                
                # Check for required columns
                required_cols = ['Date', 'Amount', 'Description']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    print(f"⚠️ Missing columns: {missing_cols}")
                    continue
                
                # Test with real data
                revenue_ai = AdvancedRevenueAISystem()
                enhanced_features = revenue_ai.extract_hybrid_features(data['Description'].tolist())
                
                # Test historical trends
                historical_trends = revenue_ai.analyze_historical_revenue_trends_professional(data, enhanced_features)
                print(f"📈 Real Data - Total Revenue: ₹{historical_trends.get('total_revenue', 0):,.2f}")
                
                # Test pricing models
                pricing_models = revenue_ai.detect_pricing_models_professional(data, enhanced_features)
                print(f"💰 Real Data - Pricing Strategy: {pricing_models.get('pricing_strategy', 'N/A')}")
                print(f"📊 Real Data - Price Elasticity: {pricing_models.get('price_elasticity', 0)}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error processing {file_name}: {e}")
                continue
    
    print("⚠️ No suitable real data files found")
    return False

if __name__ == "__main__":
    print("🚀 Revenue Analysis System Test")
    print("=" * 50)
    
    # Test with synthetic data
    success1 = test_revenue_analysis()
    
    # Test with real data
    success2 = test_with_real_data()
    
    if success1 or success2:
        print("\n✅ Revenue Analysis System is working properly!")
    else:
        print("\n❌ Revenue Analysis System has issues that need to be addressed.") 