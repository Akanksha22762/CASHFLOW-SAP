#!/usr/bin/env python3
"""
Quick Test for All 14 Parameters
================================

Quick comparison of models for all 14 AI nurturing parameters.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    """Load and prepare enhanced data"""
    try:
        enhanced_bank = pd.read_excel('data/bank_data_processed.xlsx')
        print(f"✅ Loaded enhanced bank data: {enhanced_bank.shape}")
        
        # Convert categorical columns to numeric
        for col in enhanced_bank.columns:
            if enhanced_bank[col].dtype == 'object':
                enhanced_bank[col] = pd.Categorical(enhanced_bank[col]).codes
        
        return enhanced_bank
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None

def test_models_for_parameter(param_name, features, target):
    """Test models for a specific parameter"""
    print(f"\n📊 {param_name}")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test models
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'LinearRegression': LinearRegression()
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            accuracy_20 = np.mean(np.abs((y_pred - y_test) / y_test) <= 0.2) * 100
            
            results[name] = {
                'RMSE': rmse,
                'R2': r2,
                'Accuracy_20%': accuracy_20
            }
            
            print(f"  {name}:")
            print(f"    Accuracy: {accuracy_20:.1f}%")
            print(f"    R²: {r2:.3f}")
            
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    return results

def main():
    """Main test function"""
    print("🚀 Quick Test for All 14 Parameters...")
    print("=" * 60)
    
    # Load data
    data = load_and_prepare_data()
    if data is None:
        return
    
    # Create simple features
    features = data.select_dtypes(include=[np.number])
    features = features.fillna(0)
    
    print(f"📊 Using {len(features.columns)} numeric features")
    
    # Define parameters to test
    parameters = {
        '1. Revenue Forecasts': 'Amount',
        '2. Customer Payment Terms': 'Payment_Terms',
        '3. Accounts Receivable Aging': 'Balance',
        '4. Sales Pipeline': 'Transaction_ID',
        '5. Seasonality Factors': 'Date',
        '6. Operating Expenses': 'Bank_Charges',
        '7. Accounts Payable Terms': 'Reference_Number',
        '8. Inventory Turnover': 'Account_Number',
        '9. Loan Repayments': 'Balance',
        '10. Tax Obligations': 'Bank_Charges',
        '11. Capital Expenditure': 'Amount',
        '12. Equity & Debt Inflows': 'Balance',
        '13. Other Income/Expenses': 'Amount',
        '14. Cash Flow Types': 'Amount'
    }
    
    all_results = {}
    
    # Test each parameter
    for param_name, target_col in parameters.items():
        if target_col in features.columns:
            target = features[target_col].values
            
            # Clean target data
            mask = np.isfinite(target) & (target != 0)
            if mask.sum() > 10:
                clean_features = features[mask]
                clean_target = target[mask]
                
                results = test_models_for_parameter(param_name, clean_features, clean_target)
                all_results[param_name] = results
            else:
                print(f"❌ Insufficient data for {param_name}")
        else:
            print(f"❌ Target column {target_col} not found for {param_name}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE RESULTS")
    print("=" * 60)
    
    # Summary by parameter
    print("\n🏆 BEST MODEL FOR EACH PARAMETER:")
    print("-" * 40)
    
    for param_name, results in all_results.items():
        best_model = None
        best_accuracy = 0
        
        for model_name, result in results.items():
            if 'error' not in result and result['Accuracy_20%'] > best_accuracy:
                best_accuracy = result['Accuracy_20%']
                best_model = model_name
        
        if best_model:
            print(f"{param_name}: {best_model} ({best_accuracy:.1f}%)")
    
    # Model performance comparison
    print("\n🤖 MODEL PERFORMANCE COMPARISON:")
    print("-" * 40)
    
    model_stats = {}
    for param_name, results in all_results.items():
        for model_name, result in results.items():
            if 'error' not in result:
                if model_name not in model_stats:
                    model_stats[model_name] = []
                model_stats[model_name].append(result['Accuracy_20%'])
    
    for model_name, accuracies in model_stats.items():
        avg_accuracy = np.mean(accuracies)
        print(f"{model_name}: {avg_accuracy:.1f}% average accuracy")
    
    # Find best overall model
    best_overall_model = None
    best_avg_accuracy = 0
    
    for model_name, accuracies in model_stats.items():
        avg_accuracy = np.mean(accuracies)
        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_overall_model = model_name
    
    print(f"\n🏆 BEST OVERALL MODEL: {best_overall_model}")
    print(f"📈 Average Accuracy: {best_avg_accuracy:.1f}%")
    print(f"🎯 Recommendation: Use {best_overall_model} for all 14 parameters")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 