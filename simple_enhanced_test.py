#!/usr/bin/env python3
"""
Simple Test with Enhanced Bank Data
===================================

Testing core models with your enhanced data for better accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_enhanced_data():
    """Load your enhanced bank data"""
    try:
        # Load enhanced bank data
        enhanced_bank = pd.read_excel('data/bank_data_processed.xlsx')
        print(f"✅ Loaded enhanced bank data: {enhanced_bank.shape}")
        print(f"📊 Columns: {enhanced_bank.columns.tolist()}")
        
        return enhanced_bank
        
    except Exception as e:
        print(f"⚠️ Error loading enhanced data: {e}")
        return None

def create_simple_features(data):
    """Create simple features from enhanced data"""
    features = {}
    
    if data is None or data.empty:
        print("❌ No data available")
        return pd.DataFrame()
    
    print(f"🔧 Creating features from {len(data)} records...")
    
    # Basic features
    if 'Amount' in data.columns:
        features['amount'] = data['Amount']
        features['amount_7d_avg'] = data['Amount'].rolling(window=7, min_periods=1).mean()
        features['amount_30d_avg'] = data['Amount'].rolling(window=30, min_periods=1).mean()
    
    # Date features
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        features['month'] = data['Date'].dt.month
        features['day_of_week'] = data['Date'].dt.dayofweek
    
    # Any other numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['Amount', 'Date']:
            features[col] = data[col]
    
    feature_df = pd.DataFrame(features)
    feature_df = feature_df.fillna(0)
    
    print(f"✅ Created {len(feature_df.columns)} features")
    return feature_df

def test_simple_models():
    """Test simple models with enhanced data"""
    print("🧪 Testing with Enhanced Data...")
    print("=" * 50)
    
    # Load enhanced data
    enhanced_bank = load_enhanced_data()
    
    if enhanced_bank is None:
        print("❌ Could not load enhanced data")
        return
    
    # Create features
    features = create_simple_features(enhanced_bank)
    
    if features.empty:
        print("❌ No features created")
        return
    
    # Prepare target
    if 'Amount' in enhanced_bank.columns:
        target = enhanced_bank['Amount'].values
    else:
        print("❌ No Amount column found")
        return
    
    # Clean data
    mask = np.isfinite(target) & (target != 0)
    features = features[mask]
    target = target[mask]
    
    if len(features) < 10:
        print("❌ Insufficient data")
        return
    
    print(f"📊 Final dataset: {len(features)} samples, {len(features.columns)} features")
    
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
    
    print("\n📊 Model Performance:")
    print("-" * 30)
    
    for name, model in models.items():
        try:
            # Train and predict
            model.fit(X_train_scaled, y_train)
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
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:,.2f}")
            print(f"  R²: {r2:.3f}")
            print(f"  Accuracy (20%): {accuracy_20:.1f}%")
            
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
    
    # Find best model
    best_model = None
    best_accuracy = 0
    
    for name, result in results.items():
        if result['Accuracy_20%'] > best_accuracy:
            best_accuracy = result['Accuracy_20%']
            best_model = name
    
    print("\n" + "=" * 50)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"📈 Best Accuracy: {best_accuracy:.1f}%")
    
    if best_accuracy > 26.3:
        improvement = best_accuracy - 26.3
        print(f"🎉 IMPROVEMENT: +{improvement:.1f}% over previous 26.3%")
    else:
        print(f"⚠️ Still needs improvement over 26.3%")
    
    print("=" * 50)
    
    return results, best_model, best_accuracy

if __name__ == "__main__":
    print("🚀 Testing Enhanced Data...")
    results, best_model, accuracy = test_simple_models()
    
    if best_model:
        print(f"\n✅ Recommendation: Use {best_model} with your enhanced data")
        print(f"📈 Expected accuracy: {accuracy:.1f}% (vs 26.3% before)")
    else:
        print("\n❌ No models performed well") 