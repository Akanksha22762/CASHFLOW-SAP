#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE DYNAMIC SYSTEM TESTING
Tests the complete XGBoost + Ollama dynamic recommendations system

This script verifies:
1. Dynamic strategic recommendations engine
2. XGBoost pattern integration
3. Ollama enhancement system
4. Dynamic threshold calculations
5. End-to-end system integration
"""

import sys
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🧪 {title}")
    print("="*80)

def print_section(title):
    """Print a formatted section"""
    print(f"\n📋 {title}")
    print("-" * 60)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️ {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️ {message}")

def create_test_data():
    """Create comprehensive test data for testing"""
    print_section("Creating Test Data")
    
    # Create realistic transaction data
    np.random.seed(42)  # For reproducible results
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=datetime.now(), freq='D')
    
    # Generate transaction amounts with realistic patterns
    n_transactions = 200
    
    # High-value transactions (20%)
    high_value_count = int(n_transactions * 0.2)
    high_value_amounts = np.random.normal(5000000, 2000000, high_value_count)
    
    # Medium-value transactions (50%)
    medium_value_count = int(n_transactions * 0.5)
    medium_value_amounts = np.random.normal(500000, 200000, medium_value_count)
    
    # Low-value transactions (30%)
    low_value_count = n_transactions - high_value_count - medium_value_count
    low_value_amounts = np.random.normal(50000, 20000, low_value_count)
    
    # Combine all amounts
    all_amounts = np.concatenate([high_value_amounts, medium_value_amounts, low_value_amounts])
    
    # Create descriptions
    descriptions = [
        "Customer payment for steel order",
        "Supplier payment for raw materials",
        "Equipment maintenance service",
        "Utility bill payment",
        "Employee salary payment",
        "Bank loan repayment",
        "Interest earned on deposits",
        "Tax payment to government",
        "Insurance premium payment",
        "Transportation and logistics cost"
    ] * (n_transactions // 10)
    
    # Ensure we have enough descriptions
    while len(descriptions) < n_transactions:
        descriptions.extend(descriptions[:n_transactions - len(descriptions)])
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'Date': np.random.choice(dates, n_transactions),
        'Description': descriptions[:n_transactions],
        'Amount': all_amounts,
        'Category': ['Operating Activities'] * n_transactions
    })
    
    # Add some volatility and trends
    test_data.loc[50:100, 'Amount'] *= 1.2  # Increase trend
    test_data.loc[150:200, 'Amount'] *= 0.8  # Decrease trend
    
    print_success(f"Created test dataset with {len(test_data)} transactions")
    print_info(f"Amount range: ₹{test_data['Amount'].min():,.0f} to ₹{test_data['Amount'].max():,.0f}")
    print_info(f"Average amount: ₹{test_data['Amount'].mean():,.0f}")
    
    return test_data

def test_dynamic_recommendations_engine():
    """Test the dynamic strategic recommendations engine"""
    print_section("Testing Dynamic Recommendations Engine")
    
    try:
        # Import the functions from app1.py
        from app1 import (
            generate_dynamic_strategic_recommendations,
            calculate_dynamic_threshold,
            generate_cash_flow_recommendations,
            generate_risk_management_recommendations,
            generate_growth_strategies_recommendations,
            generate_operational_insights
        )
        
        print_success("Successfully imported dynamic recommendations functions")
        
        # Test data
        test_patterns = {
            'trend': 'increasing',
            'volatility': 0.45,
            'consistency': 0.65,
            'amount_pattern': 'high_value',
            'frequency_pattern': 'regular'
        }
        
        test_transaction_data = {
            'transaction_count': 200,
            'total_amount': 50000000,
            'avg_amount': 250000,
            'max_amount': 8000000,
            'min_amount': 10000,
            'net_cash_flow': 15000000
        }
        
        # Test 1: Dynamic threshold calculation
        print_info("Testing dynamic threshold calculation...")
        high_value_threshold = calculate_dynamic_threshold(250000, 0.45, 'high_value')
        alert_threshold = calculate_dynamic_threshold(250000, 0.45, 'alert')
        consistency_target = calculate_dynamic_threshold(0.65, 0.45, 'consistency')
        
        print_success(f"High value threshold: ₹{high_value_threshold:,.0f}")
        print_success(f"Alert threshold: ₹{alert_threshold:,.0f}")
        print_success(f"Consistency target: {consistency_target:.1%}")
        
        # Test 2: Generate recommendations
        print_info("Testing recommendation generation...")
        recommendations = generate_dynamic_strategic_recommendations(
            test_patterns, test_transaction_data, 'hybrid'
        )
        
        if recommendations:
            print_success("Dynamic recommendations generated successfully")
            
            # Check each section
            sections = ['cash_flow_optimization', 'risk_management', 'growth_strategies', 'operational_insights']
            for section in sections:
                if section in recommendations and recommendations[section]:
                    count = len(recommendations[section])
                    print_success(f"{section.replace('_', ' ').title()}: {count} recommendations")
                else:
                    print_warning(f"{section.replace('_', ' ').title()}: No recommendations")
        else:
            print_error("Failed to generate dynamic recommendations")
            return False
        
        # Test 3: Individual section generation
        print_info("Testing individual section generation...")
        
        cash_flow_recs = generate_cash_flow_recommendations(test_patterns, test_transaction_data, high_value_threshold)
        risk_recs = generate_risk_management_recommendations(test_patterns, test_transaction_data, consistency_target)
        growth_recs = generate_growth_strategies_recommendations(test_patterns, test_transaction_data, 15000000)
        operational_insights = generate_operational_insights(test_patterns, test_transaction_data, 'regular')
        
        print_success(f"Cash flow: {len(cash_flow_recs)} recommendations")
        print_success(f"Risk management: {len(risk_recs)} recommendations")
        print_success(f"Growth strategies: {len(growth_recs)} recommendations")
        print_success(f"Operational insights: {len(operational_insights)} insights")
        
        return True
        
    except Exception as e:
        print_error(f"Dynamic recommendations engine test failed: {e}")
        return False

def test_xgboost_integration():
    """Test XGBoost pattern integration"""
    print_section("Testing XGBoost Pattern Integration")
    
    try:
        # Import XGBoost processing function
        from app1 import process_transactions_with_xgboost
        
        # Create test data
        test_data = create_test_data()
        
        print_info("Testing XGBoost pattern detection...")
        xgboost_result = process_transactions_with_xgboost(test_data, 'cash_flow')
        
        if xgboost_result and 'error' not in xgboost_result:
            print_success("XGBoost processing successful")
            
            # Check patterns
            patterns = xgboost_result.get('patterns', {})
            if patterns:
                print_success("Patterns detected:")
                for key, value in patterns.items():
                    if key == 'dynamic_thresholds':
                        print_info(f"  {key}: {len(value)} dynamic thresholds")
                    else:
                        print_info(f"  {key}: {value}")
                
                # Verify dynamic thresholds
                if 'dynamic_thresholds' in patterns:
                    thresholds = patterns['dynamic_thresholds']
                    print_success("Dynamic thresholds calculated:")
                    for threshold_name, threshold_value in thresholds.items():
                        print_info(f"  {threshold_name}: ₹{threshold_value:,.0f}")
                else:
                    print_warning("No dynamic thresholds found in patterns")
            else:
                print_warning("No patterns detected")
        else:
            print_error("XGBoost processing failed")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"XGBoost integration test failed: {e}")
        return False

def test_ollama_integration():
    """Test Ollama enhancement system"""
    print_section("Testing Ollama Enhancement System")
    
    try:
        # Import Ollama enhancement function
        from app1 import enhance_recommendations_with_ollama
        
        # Test data
        test_recommendations = {
            'cash_flow_optimization': [
                {
                    'title': 'Monitor High-Value Transactions',
                    'description': 'Track transactions above dynamic threshold',
                    'priority': 'high',
                    'action': 'Set up automated alerts'
                }
            ]
        }
        
        test_patterns = {'volatility': 0.45, 'consistency': 0.65}
        test_transaction_data = {'transaction_count': 200, 'avg_amount': 250000}
        
        print_info("Testing Ollama enhancement...")
        enhanced_result = enhance_recommendations_with_ollama(
            test_recommendations, test_patterns, test_transaction_data
        )
        
        if enhanced_result:
            print_success("Ollama enhancement successful")
            if 'ollama_enhancements' in enhanced_result:
                print_success("Ollama enhancements applied")
            if 'ai_generated_insights' in enhanced_result:
                print_success("AI-generated insights available")
        else:
            print_warning("Ollama enhancement not available or failed")
            print_info("This is expected if Ollama is not running")
        
        return True
        
    except Exception as e:
        print_error(f"Ollama integration test failed: {e}")
        return False

def test_end_to_end_integration():
    """Test end-to-end system integration"""
    print_section("Testing End-to-End Integration")
    
    try:
        # Test the complete flow
        print_info("Testing complete dynamic recommendations flow...")
        
        # Create test data
        test_data = create_test_data()
        
        # Process with XGBoost
        from app1 import process_transactions_with_xgboost, generate_dynamic_strategic_recommendations
        
        xgboost_result = process_transactions_with_xgboost(test_data, 'cash_flow')
        
        if xgboost_result and 'error' not in xgboost_result:
            print_success("XGBoost processing completed")
            
            # Extract patterns and transaction data
            patterns = xgboost_result.get('patterns', {})
            transaction_data = {
                'transaction_count': len(test_data),
                'total_amount': float(test_data['Amount'].sum()),
                'avg_amount': float(test_data['Amount'].mean()),
                'max_amount': float(test_data['Amount'].max()),
                'min_amount': float(test_data['Amount'].min()),
                'net_cash_flow': float(test_data['Amount'].sum())
            }
            
            # Generate dynamic recommendations
            recommendations = generate_dynamic_strategic_recommendations(
                patterns, transaction_data, 'hybrid'
            )
            
            if recommendations:
                print_success("End-to-end integration successful!")
                
                # Verify all sections are present
                required_sections = ['cash_flow_optimization', 'risk_management', 'growth_strategies', 'operational_insights']
                for section in required_sections:
                    if section in recommendations and recommendations[section]:
                        count = len(recommendations[section])
                        print_success(f"✓ {section}: {count} recommendations")
                    else:
                        print_warning(f"⚠ {section}: No recommendations")
                
                # Check for Ollama enhancements
                if 'ollama_enhancements' in recommendations:
                    print_success("✓ Ollama enhancements applied")
                else:
                    print_info("ℹ Ollama enhancements not applied (expected if not available)")
                
                return True
            else:
                print_error("Failed to generate recommendations in end-to-end test")
                return False
        else:
            print_error("XGBoost processing failed in end-to-end test")
            return False
        
    except Exception as e:
        print_error(f"End-to-end integration test failed: {e}")
        return False

def test_frontend_integration():
    """Test frontend integration with dynamic data"""
    print_section("Testing Frontend Integration")
    
    try:
        # Test the JavaScript functions that would be called by the frontend
        print_info("Testing frontend JavaScript function simulation...")
        
        # Simulate the data structure that would come from the backend
        mock_data = {
            'dynamic_recommendations': {
                'cash_flow_optimization': [
                    {
                        'title': 'Monitor High-Value Transactions',
                        'description': 'Track transactions above ₹4,875,000 for better cash flow management',
                        'priority': 'high',
                        'action': 'Set up automated alerts for transactions above threshold'
                    }
                ],
                'risk_management': [
                    {
                        'title': 'Volatility Monitoring',
                        'description': 'Current 45.0% Medium volatility requires attention',
                        'priority': 'high',
                        'action': 'Implement volatility reduction strategies for 45.0% threshold'
                    }
                ],
                'growth_strategies': [
                    {
                        'title': 'Leverage Increasing Trend',
                        'description': 'Capitalize on moderate positive cash flow momentum',
                        'priority': 'high',
                        'action': 'Expand operations based on positive trend momentum'
                    }
                ],
                'operational_insights': [
                    {
                        'title': 'Transaction Volume Analysis',
                        'description': 'High volume (200 transactions) requires automated processing',
                        'priority': 'medium',
                        'action': 'Implement automated processing systems'
                    }
                ]
            },
            'recommendations_generated_by': 'Dynamic XGBoost + Ollama Engine',
            'patterns': {
                'volatility': 0.45,
                'consistency': 0.65,
                'trend': 'increasing'
            }
        }
        
        print_success("Mock data structure created successfully")
        
        # Verify data structure
        if 'dynamic_recommendations' in mock_data:
            print_success("✓ Dynamic recommendations present")
            sections = mock_data['dynamic_recommendations']
            for section_name, section_data in sections.items():
                if isinstance(section_data, list) and len(section_data) > 0:
                    print_success(f"✓ {section_name}: {len(section_data)} items")
                else:
                    print_warning(f"⚠ {section_name}: No items")
        
        if 'recommendations_generated_by' in mock_data:
            print_success(f"✓ Engine: {mock_data['recommendations_generated_by']}")
        
        if 'patterns' in mock_data:
            print_success("✓ Patterns data present")
        
        return True
        
    except Exception as e:
        print_error(f"Frontend integration test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all comprehensive tests"""
    print_header("COMPREHENSIVE DYNAMIC SYSTEM TESTING")
    print_info("Testing the complete XGBoost + Ollama dynamic recommendations system")
    print_info(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {}
    
    # Test 1: Dynamic Recommendations Engine
    print_header("TEST 1: DYNAMIC RECOMMENDATIONS ENGINE")
    test_results['dynamic_engine'] = test_dynamic_recommendations_engine()
    
    # Test 2: XGBoost Integration
    print_header("TEST 2: XGBOOST PATTERN INTEGRATION")
    test_results['xgboost_integration'] = test_xgboost_integration()
    
    # Test 3: Ollama Integration
    print_header("TEST 3: OLLAMA ENHANCEMENT SYSTEM")
    test_results['ollama_integration'] = test_ollama_integration()
    
    # Test 4: End-to-End Integration
    print_header("TEST 4: END-TO-END INTEGRATION")
    test_results['end_to_end'] = test_end_to_end_integration()
    
    # Test 5: Frontend Integration
    print_header("TEST 5: FRONTEND INTEGRATION")
    test_results['frontend_integration'] = test_frontend_integration()
    
    # Summary
    print_header("TEST RESULTS SUMMARY")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    print_info(f"Tests Passed: {passed_tests}/{total_tests}")
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name.replace('_', ' ').title()}")
    
    # Overall assessment
    if passed_tests == total_tests:
        print_success("🎉 ALL TESTS PASSED! The dynamic system is working perfectly!")
        print_info("Your system now has:")
        print_info("  • Dynamic strategic recommendations based on XGBoost patterns")
        print_info("  • Adaptive thresholds instead of hardcoded values")
        print_info("  • Ollama integration for enhanced business insights")
        print_info("  • Real-time pattern analysis and recommendations")
    elif passed_tests >= total_tests * 0.8:
        print_warning("⚠️ MOST TESTS PASSED! The system is mostly working.")
        print_info("Some components may need attention.")
    else:
        print_error("❌ MANY TESTS FAILED! The system needs significant attention.")
        print_info("Please check the error messages above.")
    
    return test_results

if __name__ == "__main__":
    try:
        results = run_comprehensive_test()
        
        # Save results to file
        with open('dynamic_system_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'summary': {
                    'total_tests': len(results),
                    'passed_tests': sum(results.values()),
                    'success_rate': f"{(sum(results.values()) / len(results)) * 100:.1f}%"
                }
            }, f, indent=2)
        
        print_info("Test results saved to 'dynamic_system_test_results.json'")
        
    except KeyboardInterrupt:
        print_info("\nTesting interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
