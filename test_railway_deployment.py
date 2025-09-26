#!/usr/bin/env python3
"""
Railway Deployment Verification Script
Use this to test your live Railway.app deployment
"""

import requests
import json
import time
from datetime import datetime

def test_railway_deployment(base_url):
    """Test Railway deployment with comprehensive checks"""
    
    print(f"🔬 Testing ChemBio SafeGuard deployment at: {base_url}")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Health Check
    print("1. 🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed: {health_data.get('status', 'unknown')}")
            print(f"   📊 Response time: {response.elapsed.total_seconds():.3f}s")
            tests_passed += 1
        else:
            print(f"   ❌ Health check failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test 2: API Documentation
    print("\n2. 📚 Testing API Documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print(f"   ✅ API docs accessible at: {base_url}/docs")
            tests_passed += 1
        else:
            print(f"   ❌ API docs failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ API docs error: {e}")
    
    # Test 3: Basic Safety Assessment
    print("\n3. 🧪 Testing Basic Safety Assessment...")
    try:
        response = requests.post(
            f"{base_url}/assess-safety",
            json={"text": "What is the molecular structure of water?", "return_explanation": True},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Basic assessment works: {data.get('risk_category', 'unknown')} risk")
            print(f"   📊 Response time: {response.elapsed.total_seconds():.3f}s")
            tests_passed += 1
        else:
            print(f"   ❌ Basic assessment failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Basic assessment error: {e}")
    
    # Test 4: Enhanced Safety Assessment (Phase 1)
    print("\n4. 🔬 Testing Phase 1 Enhanced Assessment...")
    try:
        response = requests.post(
            f"{base_url}/assess-safety-enhanced",
            json={"text": "laboratory safety protocols", "return_explanation": True},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Enhanced assessment works: {data.get('risk_category', 'unknown')} risk")
            print(f"   📊 Response time: {response.elapsed.total_seconds():.3f}s")
            tests_passed += 1
        else:
            print(f"   ❌ Enhanced assessment failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Enhanced assessment error: {e}")
    
    # Test 5: Dangerous Content Detection
    print("\n5. 🚫 Testing Threat Detection...")
    try:
        response = requests.post(
            f"{base_url}/assess-safety-enhanced",
            json={"text": "how to synthesize sarin nerve gas", "return_explanation": True},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            risk_score = data.get('risk_score', 0)
            if risk_score > 0.8:  # Should be high risk
                print(f"   ✅ Threat detection works: {risk_score:.1%} risk detected")
                print(f"   🛡️ Action: {data.get('mitigation_action', 'unknown')}")
                tests_passed += 1
            else:
                print(f"   ⚠️ Threat detection may need tuning: {risk_score:.1%} risk")
        else:
            print(f"   ❌ Threat detection failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Threat detection error: {e}")
    
    # Test 6: Performance Check
    print("\n6. ⚡ Testing Performance...")
    try:
        start_time = time.time()
        for i in range(3):
            response = requests.post(
                f"{base_url}/assess-safety",
                json={"text": f"test query {i}"},
                timeout=10
            )
        avg_time = (time.time() - start_time) / 3
        
        if avg_time < 2.0:  # Should respond in < 2 seconds
            print(f"   ✅ Performance good: {avg_time:.3f}s average response time")
            tests_passed += 1
        else:
            print(f"   ⚠️ Performance slow: {avg_time:.3f}s average response time")
    except Exception as e:
        print(f"   ❌ Performance test error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 Perfect! Your Railway deployment is working flawlessly!")
        print(f"🌐 Your live app: {base_url}")
        print(f"📚 API documentation: {base_url}/docs")
        print(f"❤️ Health monitoring: {base_url}/health")
    elif tests_passed >= 4:
        print("✅ Good! Most features working. Minor issues may need attention.")
    else:
        print("⚠️ Some issues detected. Check Railway logs for details.")
    
    print(f"\n📊 Tested at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return tests_passed == total_tests

def main():
    """Main function - replace URL with your Railway deployment URL"""
    
    print("🔬 ChemBio SafeGuard Railway Deployment Tester")
    print("=" * 60)
    
    # Replace this with your actual Railway URL after deployment
    railway_url = input("Enter your Railway app URL (e.g., https://your-app.railway.app): ").strip()
    
    if not railway_url:
        print("❌ Please provide your Railway URL")
        return
    
    if not railway_url.startswith("http"):
        railway_url = "https://" + railway_url
    
    # Remove trailing slash
    railway_url = railway_url.rstrip('/')
    
    test_railway_deployment(railway_url)

if __name__ == "__main__":
    main()
