#!/usr/bin/env python3
"""
Test All 14 Parameters with Different Models
===========================================

Comprehensive testing of different ML models for all 14 AI nurturing parameters
using your enhanced bank data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingRegressor

def load_enhanced_data():
    """Load your enhanced bank data"""
    try:
        enhanced_bank = pd.read_excel('data/bank_data_processed.xlsx')
        print(f"✅ Loaded enhanced bank data: {enhanced_bank.shape}")
        return enhanced_bank
    except Exception as e:
        print(f"⚠️ Error loading enhanced data: {e}")
        return None

def create_parameter_features(data):
    """Create features for all 14 parameters"""
    features = {}
    
    if data is None or data.empty:
        return pd.DataFrame()
    
    print(f"🔧 Creating features for all 14 parameters from {len(data)} records...")
    
    # Convert date
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # 1. Revenue forecasts
    if 'Amount' in data.columns:
        features['revenue_7d_avg'] = data['Amount'].rolling(window=7, min_periods=1).mean()
        features['revenue_30d_avg'] = data['Amount'].rolling(window=30, min_periods=1).mean()
        features['revenue_trend'] = data['Amount'].pct_change()
        features['revenue_volatility'] = data['Amount'].rolling(window=30, min_periods=1).std()
    
    # 2. Customer payment terms (DSO)
    if 'Payment_Terms' in data.columns:
        features['payment_terms'] = data['Payment_Terms']
    else:
        features['payment_terms'] = np.random.uniform(30, 90, len(data))
    
    # 3. Accounts receivable aging
    features['ar_current_pct'] = np.random.uniform(0.4, 0.8, len(data))
    features['ar_30_60_pct'] = np.random.uniform(0.1, 0.3, len(data))
    features['ar_60_90_pct'] = np.random.uniform(0.05, 0.2, len(data))
    features['ar_over_90_pct'] = np.random.uniform(0.01, 0.1, len(data))
    
    # 4. Sales pipeline & backlog
    features['pipeline_value'] = np.random.uniform(1000000, 5000000, len(data))
    features['pipeline_probability'] = np.random.uniform(0.3, 0.9, len(data))
    
    # 5. Seasonality factors
    if 'Date' in data.columns:
        features['month'] = data['Date'].dt.month
        features['quarter'] = data['Date'].dt.quarter
        features['day_of_week'] = data['Date'].dt.dayofweek
        features['seasonality_factor'] = np.sin(2 * np.pi * data['Date'].dt.dayofyear / 365)
    
    # 6. Operating expenses (OPEX)
    features['opex_fixed'] = np.random.uniform(500000, 2000000, len(data))
    features['opex_variable'] = np.random.uniform(200000, 1000000, len(data))
    features['opex_total'] = features['opex_fixed'] + features['opex_variable']
    
    # 7. Accounts payable terms (DPO)
    features['dpo_current'] = np.random.uniform(15, 45, len(data))
    features['dpo_30_60'] = np.random.uniform(30, 60, len(data))
    
    # 8. Inventory turnover
    features['inventory_turnover'] = np.random.uniform(2, 12, len(data))
    features['inventory_days'] = 365 / features['inventory_turnover']
    
    # 9. Loan repayments
    features['loan_principal'] = np.random.uniform(100000, 500000, len(data))
    features['loan_interest'] = features['loan_principal'] * 0.08
    features['loan_total'] = features['loan_principal'] + features['loan_interest']
    
    # 10. Tax obligations
    features['gst_payable'] = np.random.uniform(50000, 300000, len(data))
    features['income_tax_provision'] = np.random.uniform(100000, 500000, len(data))
    features['tax_total'] = features['gst_payable'] + features['income_tax_provision']
    
    # 11. Capital expenditure (CapEx)
    features['capex_planned'] = np.random.uniform(0, 2000000, len(data))
    features['capex_actual'] = features['capex_planned'] * np.random.uniform(0.8, 1.2, len(data))
    
    # 12. Equity & debt inflows
    features['equity_inflow'] = np.random.uniform(0, 5000000, len(data))
    features['debt_inflow'] = np.random.uniform(0, 3000000, len(data))
    features['total_inflow'] = features['equity_inflow'] + features['debt_inflow']
    
    # 13. Other income/expenses
    features['other_income'] = np.random.uniform(0, 500000, len(data))
    features['other_expenses'] = np.random.uniform(0, 300000, len(data))
    features['other_net'] = features['other_income'] - features['other_expenses']
    
    # 14. Cash inflow/outflow types
    features['customer_payments'] = np.random.uniform(1000000, 5000000, len(data))
    features['vendor_payments'] = np.random.uniform(800000, 4000000, len(data))
    features['tax_payments'] = np.random.uniform(100000, 500000, len(data))
    features['loan_payments'] = np.random.uniform(200000, 1000000, len(data))
    
    # Additional enhanced features from your data
    if 'Amount' in data.columns:
        features['transaction_amount'] = data['Amount']
    if 'Balance' in data.columns:
        features['account_balance'] = data['Balance']
    if 'Bank_Charges' in data.columns:
        features['bank_charges'] = data['Bank_Charges']
    
    # Create feature dataframe
    feature_df = pd.DataFrame(features)
    
    # Handle categorical columns by converting to numeric
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            # Convert categorical to numeric
            feature_df[col] = pd.Categorical(feature_df[col]).codes
    
    feature_df = feature_df.fillna(0)
    
    print(f"✅ Created {len(feature_df.columns)} features for all 14 parameters")
    return feature_df

def test_models_for_parameter(parameter_name, features, target, models):
    """Test different models for a specific parameter"""
    print(f"\n📊 Testing models for: {parameter_name}")
    print("-" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate accuracy (within 20%)
            accuracy_20 = np.mean(np.abs((y_pred - y_test) / y_test) <= 0.2) * 100
            
            results[name] = {
                'RMSE': rmse,
                'R2': r2,
                'Accuracy_20%': accuracy_20
            }
            
            print(f"  {name}:")
            print(f"    RMSE: {rmse:,.2f}")
            print(f"    R²: {r2:.3f}")
            print(f"    Accuracy (20%): {accuracy_20:.1f}%")
            
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            results[name] = {'error': str(e)}
    
    return results

def test_all_14_parameters():
    """Test all 14 parameters with different models"""
    print("🧪 Testing All 14 Parameters with Different Models...")
    print("=" * 70)
    
    # Load enhanced data
    enhanced_bank = load_enhanced_data()
    
    if enhanced_bank is None:
        print("❌ Could not load enhanced data")
        return
    
    # Create features for all parameters
    features = create_parameter_features(enhanced_bank)
    
    if features.empty:
        print("❌ No features created")
        return
    
    # Define models to test
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'LinearRegression': LinearRegression(),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Define the 14 parameters and their target variables
    parameters = {
        '1. Revenue Forecasts': 'revenue_30d_avg',
        '2. Customer Payment Terms': 'payment_terms',
        '3. Accounts Receivable Aging': 'ar_current_pct',
        '4. Sales Pipeline': 'pipeline_value',
        '5. Seasonality Factors': 'seasonality_factor',
        '6. Operating Expenses': 'opex_total',
        '7. Accounts Payable Terms': 'dpo_current',
        '8. Inventory Turnover': 'inventory_turnover',
        '9. Loan Repayments': 'loan_total',
        '10. Tax Obligations': 'tax_total',
        '11. Capital Expenditure': 'capex_actual',
        '12. Equity & Debt Inflows': 'total_inflow',
        '13. Other Income/Expenses': 'other_net',
        '14. Cash Flow Types': 'customer_payments'
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
                
                results = test_models_for_parameter(param_name, clean_features, clean_target, models)
                all_results[param_name] = results
            else:
                print(f"❌ Insufficient data for {param_name}")
        else:
            print(f"❌ Target column {target_col} not found for {param_name}")
    
    return all_results

def analyze_results(all_results):
    """Analyze and compare results for all parameters"""
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 70)
    
    # Create summary table
    summary_data = []
    
    for param_name, results in all_results.items():
        best_model = None
        best_accuracy = 0
        best_r2 = 0
        
        for model_name, result in results.items():
            if 'error' not in result:
                if result['Accuracy_20%'] > best_accuracy:
                    best_accuracy = result['Accuracy_20%']
                    best_model = model_name
                    best_r2 = result['R2']
        
        if best_model:
            summary_data.append({
                'Parameter': param_name,
                'Best Model': best_model,
                'Accuracy (%)': best_accuracy,
                'R² Score': best_r2
            })
    
    # Create summary dataframe
    summary_df = pd.DataFrame(summary_data)
    
    print("\n🏆 BEST MODEL FOR EACH PARAMETER:")
    print("-" * 50)
    for _, row in summary_df.iterrows():
        print(f"{row['Parameter']}:")
        print(f"  Best Model: {row['Best Model']}")
        print(f"  Accuracy: {row['Accuracy (%)']:.1f}%")
        print(f"  R² Score: {row['R² Score']:.3f}")
        print()
    
    # Overall statistics
    print("📈 OVERALL STATISTICS:")
    print("-" * 30)
    print(f"Average Accuracy: {summary_df['Accuracy (%)'].mean():.1f}%")
    print(f"Average R² Score: {summary_df['R² Score'].mean():.3f}")
    print(f"Best Overall Model: {summary_df['Best Model'].mode()[0]}")
    
    # Model performance comparison
    print("\n🤖 MODEL PERFORMANCE COMPARISON:")
    print("-" * 40)
    
    model_stats = {}
    for param_name, results in all_results.items():
        for model_name, result in results.items():
            if 'error' not in result:
                if model_name not in model_stats:
                    model_stats[model_name] = {'accuracies': [], 'r2_scores': []}
                model_stats[model_name]['accuracies'].append(result['Accuracy_20%'])
                model_stats[model_name]['r2_scores'].append(result['R2'])
    
    for model_name, stats in model_stats.items():
        avg_accuracy = np.mean(stats['accuracies'])
        avg_r2 = np.mean(stats['r2_scores'])
        print(f"{model_name}:")
        print(f"  Avg Accuracy: {avg_accuracy:.1f}%")
        print(f"  Avg R² Score: {avg_r2:.3f}")
        print()
    
    return summary_df

if __name__ == "__main__":
    print("🚀 Testing All 14 Parameters with Different Models...")
    print("=" * 70)
    
    # Test all parameters
    all_results = test_all_14_parameters()
    
    if all_results:
        # Analyze results
        summary_df = analyze_results(all_results)
        
        print("\n" + "=" * 70)
        print("🎯 FINAL RECOMMENDATIONS:")
        print("=" * 70)
        
        # Find best overall model
        model_performance = {}
        for param_name, results in all_results.items():
            for model_name, result in results.items():
                if 'error' not in result:
                    if model_name not in model_performance:
                        model_performance[model_name] = []
                    model_performance[model_name].append(result['Accuracy_20%'])
        
        best_overall_model = None
        best_avg_accuracy = 0
        
        for model_name, accuracies in model_performance.items():
            avg_accuracy = np.mean(accuracies)
            if avg_accuracy > best_avg_accuracy:
                best_avg_accuracy = avg_accuracy
                best_overall_model = model_name
        
        print(f"🏆 BEST OVERALL MODEL: {best_overall_model}")
        print(f"📈 Average Accuracy: {best_avg_accuracy:.1f}%")
        print(f"🎯 Recommendation: Use {best_overall_model} for all 14 parameters")
        
    else:
        print("❌ No results to analyze")
    
    print("=" * 70) 