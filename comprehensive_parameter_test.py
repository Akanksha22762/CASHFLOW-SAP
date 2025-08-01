#!/usr/bin/env python3
"""
Comprehensive test to identify all issues with parameter analysis system
"""

import requests
import json
import time

def test_parameter_analysis():
    """Test all parameter analysis endpoints"""
    
    base_url = "http://127.0.0.1:5000"
    
    # Test data - simulate uploaded files
    test_data = {
        "bank_df": {
            "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Description": ["Test Transaction 1", "Test Transaction 2", "Test Transaction 3"],
            "Amount": [1000, 2000, 3000],
            "Type": ["Credit", "Debit", "Credit"]
        }
    }
    
    print("🧪 Comprehensive Parameter Analysis Test")
    print("=" * 60)
    
    # Test all 5 parameters
    parameters = [
        'A1_historical_trends',
        'A2_sales_forecast', 
        'A3_customer_contracts',
        'A4_pricing_models',
        'A5_ar_aging'
    ]
    
    results = {}
    
    for param in parameters:
        print(f"\n🔍 Testing {param}...")
        
        try:
            # Test parameter analysis
            response = requests.post(
                f"{base_url}/run-parameter-analysis",
                json={"parameter_type": param},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    print(f"✅ {param} - Analysis completed successfully")
                    results[param] = {
                        'status': 'success',
                        'data': data.get('results', {}),
                        'processing_time': data.get('processing_time', 'N/A')
                    }
                else:
                    print(f"❌ {param} - Analysis failed: {data.get('error', 'Unknown error')}")
                    results[param] = {
                        'status': 'error',
                        'error': data.get('error', 'Unknown error')
                    }
            else:
                print(f"❌ {param} - HTTP {response.status_code}: {response.text}")
                results[param] = {
                    'status': 'http_error',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {param} - Network error: {e}")
            results[param] = {
                'status': 'network_error',
                'error': str(e)
            }
        except Exception as e:
            print(f"❌ {param} - Unexpected error: {e}")
            results[param] = {
                'status': 'unexpected_error',
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    for param, result in results.items():
        if result['status'] == 'success':
            success_count += 1
            print(f"✅ {param}: SUCCESS")
            if 'data' in result and result['data']:
                print(f"   📊 Data keys: {list(result['data'].keys())}")
        else:
            error_count += 1
            print(f"❌ {param}: FAILED - {result.get('error', 'Unknown error')}")
    
    print(f"\n🎯 Overall Results:")
    print(f"   ✅ Successful: {success_count}/5")
    print(f"   ❌ Failed: {error_count}/5")
    
    return results

if __name__ == "__main__":
    test_parameter_analysis() 