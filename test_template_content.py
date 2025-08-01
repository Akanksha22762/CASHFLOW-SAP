#!/usr/bin/env python3
"""
Test script to check template content
"""

import requests
import time

def test_template_content():
    print("🔍 Testing template content...")
    
    # Wait for Flask app to start
    time.sleep(3)
    
    try:
        # Get the page content
        response = requests.get('http://localhost:5000/advanced-revenue-analysis')
        
        if response.status_code == 200:
            content = response.text
            print("✅ Successfully got page content")
            
            # Check for OPEX content
            if "Operating Expenses (OPEX)" in content:
                print("✅ OPEX card found in template!")
            else:
                print("❌ OPEX card NOT found in template!")
                
            # Check for other cards
            if "Historical Revenue Trends" in content:
                print("✅ Historical Revenue Trends card found")
            else:
                print("❌ Historical Revenue Trends card NOT found")
                
            # Show first 500 characters
            print("\n📄 First 500 characters of template:")
            print(content[:500])
            
        else:
            print(f"❌ Failed to get page. Status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_template_content() 