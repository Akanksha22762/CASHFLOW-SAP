#!/usr/bin/env python3
"""
Detailed debug script to identify the exact location of the .mean() error
"""

import pandas as pd
import numpy as np
import traceback
from advanced_revenue_ai_system import AdvancedRevenueAISystem

def debug_opex_detailed():
    """Debug the Operating Expenses function with detailed error tracking"""
    try:
        # Load the real data
        df = pd.read_excel("uploads/bank_Bank_Statement_Combined.xlsx")
        print(f"✅ Loaded data shape: {df.shape}")
        
        # Test the enhanced function with detailed error tracking
        ai_system = AdvancedRevenueAISystem()
        print("\n🎯 Testing enhanced Operating Expenses function with detailed tracking...")
        
        # Add debug prints to track where the error occurs
        try:
            # Step 1: Enhance data with advanced AI features
            print("Step 1: Enhancing data with advanced AI features...")
            enhanced_transactions = ai_system._enhance_with_advanced_ai_features(df.copy())
            print("✅ Data enhancement completed")
            
            # Step 2: Get basic analysis
            print("Step 2: Getting basic analysis...")
            basic_analysis = ai_system.analyze_operating_expenses(enhanced_transactions)
            print("✅ Basic analysis completed")
            
            if 'error' in basic_analysis:
                print(f"❌ Basic analysis error: {basic_analysis['error']}")
                return
            
            # Step 3: Add advanced AI features
            print("Step 3: Adding advanced AI features...")
            advanced_features = {}
            
            # Step 3.1: Cost Optimization
            print("Step 3.1: Cost optimization...")
            if 'total_expenses' in basic_analysis:
                total_expenses = float(basic_analysis['total_expenses'].replace('₹', '').replace(',', ''))
                if total_expenses > 0:
                    amount_column = ai_system._get_amount_column(enhanced_transactions)
                    if amount_column:
                        expense_data = enhanced_transactions[enhanced_transactions[amount_column] < 0]
                        if len(expense_data) > 0:
                            print("Step 3.1.1: Calculating monthly expenses...")
                            monthly_expenses = expense_data.groupby(pd.Grouper(key='Date', freq='M'))[amount_column].sum()
                            print(f"✅ Monthly expenses shape: {monthly_expenses.shape}")
                            
                            if len(monthly_expenses) > 0 and monthly_expenses.sum() > 0:
                                print("Step 3.1.2: Calculating volatility...")
                                volatility = np.std(monthly_expenses) / np.mean(monthly_expenses)
                                print(f"✅ Volatility calculated: {volatility}")
                            else:
                                volatility = 0.0
                                print("✅ Using default volatility: 0.0")
                            
                            advanced_features['cost_optimization'] = {
                                'expense_volatility': float(volatility),
                                'optimization_potential': float(volatility * 0.1 * total_expenses),
                                'recommendations': [
                                    'Implement expense tracking automation',
                                    'Negotiate better vendor terms',
                                    'Optimize inventory levels',
                                    'Review subscription services'
                                ]
                            }
                            print("✅ Cost optimization completed")
            
            # Step 3.2: Anomaly Detection
            print("Step 3.2: Anomaly detection...")
            amount_column = ai_system._get_amount_column(enhanced_transactions)
            if amount_column:
                expense_data = enhanced_transactions[enhanced_transactions[amount_column] < 0]
                if len(expense_data) > 10:
                    try:
                        print("Step 3.2.1: Detecting anomalies...")
                        anomalies = ai_system._detect_anomalies(expense_data[amount_column].values, 'statistical')
                        anomaly_count = np.sum(anomalies)
                        if anomaly_count > 0:
                            advanced_features['expense_anomalies'] = {
                                'count': int(anomaly_count),
                                'percentage': float((anomaly_count / len(expense_data)) * 100),
                                'anomaly_amounts': expense_data.iloc[np.where(anomalies)[0]][amount_column].tolist()
                            }
                        print("✅ Anomaly detection completed")
                    except Exception as e:
                        print(f"⚠️ Anomaly detection failed: {e}")
            
            # Step 3.3: Predictive Cost Modeling
            print("Step 3.3: Predictive cost modeling...")
            if 'Date' in enhanced_transactions.columns and len(enhanced_transactions) > 12:
                try:
                    enhanced_transactions['Date'] = pd.to_datetime(enhanced_transactions['Date'])
                    monthly_expenses = enhanced_transactions[enhanced_transactions[amount_column] < 0].groupby(pd.Grouper(key='Date', freq='M'))[amount_column].sum()
                    if len(monthly_expenses) > 6:
                        print("Step 3.3.1: Forecasting expenses...")
                        expense_forecast = ai_system._forecast_with_lstm(monthly_expenses.values, 3)
                        if expense_forecast is not None:
                            print("Step 3.3.2: Applying external adjustments...")
                            inflation_adjustment = 0.0
                            if 'inflation_impact' in enhanced_transactions.columns:
                                inflation_series = enhanced_transactions['inflation_impact']
                                if len(inflation_series) > 0 and not inflation_series.isna().all():
                                    inflation_adjustment = float(inflation_series.mean())
                                expense_forecast = expense_forecast * (1 + inflation_adjustment)
                            
                            external_adjustments = {}
                            for col in ['inflation_impact', 'interest_rate_impact', 'tax_rate_impact']:
                                if col in enhanced_transactions.columns:
                                    series = enhanced_transactions[col]
                                    if len(series) > 0 and not series.isna().all():
                                        external_adjustments[col] = float(series.mean())
                                    else:
                                        external_adjustments[col] = 0.0
                                else:
                                    external_adjustments[col] = 0.0
                            
                            advanced_features['expense_forecast'] = {
                                'next_3_months': expense_forecast.tolist(),
                                'forecast_total': float(np.sum(expense_forecast)),
                                'external_adjustments': external_adjustments
                            }
                            print("✅ Predictive modeling completed")
                except Exception as e:
                    print(f"⚠️ Predictive modeling failed: {e}")
            
            # Step 3.4: Operational Drivers Impact
            print("Step 3.4: Operational drivers impact...")
            if 'headcount_cost' in enhanced_transactions.columns:
                headcount_cost = enhanced_transactions['headcount_cost']
                headcount_sum = float(headcount_cost.sum()) if len(headcount_cost) > 0 else 0.0
                
                expansion_investment = enhanced_transactions.get('expansion_investment', pd.Series([0]))
                expansion_sum = float(expansion_investment.sum()) if len(expansion_investment) > 0 else 0.0
                
                marketing_roi = enhanced_transactions.get('marketing_roi', pd.Series([0]))
                marketing_mean = float(marketing_roi.mean()) if len(marketing_roi) > 0 and not marketing_roi.isna().all() else 0.0
                
                advanced_features['operational_impact'] = {
                    'headcount_cost': headcount_sum,
                    'expansion_investment': expansion_sum,
                    'marketing_roi': marketing_mean
                }
                print("✅ Operational drivers completed")
            
            # Step 4: Merge with basic analysis
            print("Step 4: Merging results...")
            basic_analysis['advanced_ai_features'] = advanced_features
            basic_analysis['analysis_type'] = 'Enhanced AI Analysis with External Variables & Modeling Considerations'
            
            print("✅ Enhanced analysis completed successfully!")
            print(f"✅ Total expenses: {basic_analysis.get('total_expenses', 'N/A')}")
            print(f"✅ Advanced features: {len(basic_analysis.get('advanced_ai_features', {}))}")
            
        except Exception as e:
            print(f"❌ Error in enhanced analysis: {e}")
            print("Full traceback:")
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_opex_detailed() 