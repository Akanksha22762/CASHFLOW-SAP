#!/usr/bin/env python3
"""
Test script to verify console output is working properly
"""

import sys
import os
import time

# Force immediate output flushing
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Set environment variable to force immediate output
os.environ['PYTHONUNBUFFERED'] = '1'

def test_console_output():
    """Test various console output methods"""
    print("🧪 Testing Console Output...")
    print("=" * 50)
    
    # Test 1: Basic print statements
    print("✅ Test 1: Basic print statements")
    for i in range(5):
        print(f"   Processing step {i+1}/5...")
        time.sleep(0.5)
    
    # Test 2: Print with flush
    print("✅ Test 2: Print with flush")
    for i in range(3):
        print(f"   Flushed output {i+1}/3...", flush=True)
        time.sleep(0.3)
    
    # Test 3: Error output
    print("✅ Test 3: Error output")
    print("   This is a test error message", file=sys.stderr)
    
    # Test 4: Progress indicators
    print("✅ Test 4: Progress indicators")
    for i in range(10):
        print(f"\r   Progress: {i*10}%", end="", flush=True)
        time.sleep(0.2)
    print("\n   Progress: 100% Complete!")
    
    print("=" * 50)
    print("🎉 Console output test completed!")
    print("💡 If you can see this output, console logging is working properly")

if __name__ == '__main__':
    test_console_output() 