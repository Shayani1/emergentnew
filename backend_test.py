#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Forex Trading Signal Bot
Tests all API endpoints and core functionality
"""

import asyncio
import aiohttp
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Test configuration
BACKEND_URL = "https://c91bfcbf-2498-412c-b042-68fb52ac2a88.preview.emergentagent.com/api"
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

class ForexBotTester:
    def __init__(self):
        self.session = None
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    async def setup(self):
        """Initialize test session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"Content-Type": "application/json"}
        )
        
    async def cleanup(self):
        """Clean up test session"""
        if self.session:
            await self.session.close()
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"
            
        print(f"{status} - {test_name}")
        if details:
            print(f"    Details: {details}")
            
        self.test_results[test_name] = {
            "passed": passed,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    async def test_health_check(self):
        """Test GET /api/ (health check)"""
        try:
            async with self.session.get(f"{BACKEND_URL}/") as response:
                if response.status == 200:
                    data = await response.json()
                    if "message" in data and "status" in data:
                        self.log_test("Health Check", True, f"Status: {data.get('status')}")
                        return True
                    else:
                        self.log_test("Health Check", False, "Missing required fields in response")
                        return False
                else:
                    self.log_test("Health Check", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Health Check", False, f"Exception: {str(e)}")
            return False
    
    async def test_performance_endpoint(self):
        """Test GET /api/performance (system statistics)"""
        try:
            async with self.session.get(f"{BACKEND_URL}/performance") as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ["total_signals", "active_signals", "average_confidence", 
                                     "supported_pairs", "timeframes"]
                    
                    missing_fields = [field for field in required_fields if field not in data]
                    if not missing_fields:
                        self.log_test("Performance Endpoint", True, 
                                    f"Total signals: {data.get('total_signals')}, "
                                    f"Avg confidence: {data.get('average_confidence')}%")
                        return True
                    else:
                        self.log_test("Performance Endpoint", False, 
                                    f"Missing fields: {missing_fields}")
                        return False
                else:
                    self.log_test("Performance Endpoint", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Performance Endpoint", False, f"Exception: {str(e)}")
            return False
    
    async def test_signals_endpoint(self):
        """Test GET /api/signals (fetch trading signals)"""
        try:
            async with self.session.get(f"{BACKEND_URL}/signals?limit=5") as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list):
                        self.log_test("Signals Endpoint", True, f"Retrieved {len(data)} signals")
                        return True
                    else:
                        self.log_test("Signals Endpoint", False, "Response is not a list")
                        return False
                else:
                    self.log_test("Signals Endpoint", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Signals Endpoint", False, f"Exception: {str(e)}")
            return False
    
    async def test_generate_signal_endpoint(self):
        """Test POST /api/generate-signal for all supported pairs"""
        results = []
        
        for pair in FOREX_PAIRS:
            try:
                payload = {"pair": pair, "timeframe": "1h"}
                async with self.session.post(f"{BACKEND_URL}/generate-signal", 
                                           params=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "status" in data:
                            if data["status"] == "success" and "signal" in data:
                                # Validate signal structure
                                signal = data["signal"]
                                required_fields = ["direction", "entry_price", "stop_loss", 
                                                 "take_profit_1", "take_profit_2", "take_profit_3",
                                                 "confidence", "reasoning"]
                                
                                missing_fields = [field for field in required_fields if field not in signal]
                                if not missing_fields:
                                    self.log_test(f"Generate Signal - {pair}", True, 
                                                f"Direction: {signal.get('direction')}, "
                                                f"Confidence: {signal.get('confidence')}%")
                                    results.append(True)
                                else:
                                    self.log_test(f"Generate Signal - {pair}", False, 
                                                f"Missing signal fields: {missing_fields}")
                                    results.append(False)
                            elif data["status"] == "no_signal":
                                self.log_test(f"Generate Signal - {pair}", True, 
                                            "No signal generated (insufficient confluence)")
                                results.append(True)
                            else:
                                self.log_test(f"Generate Signal - {pair}", False, 
                                            f"Unexpected status: {data['status']}")
                                results.append(False)
                        else:
                            self.log_test(f"Generate Signal - {pair}", False, "Missing status field")
                            results.append(False)
                    else:
                        self.log_test(f"Generate Signal - {pair}", False, f"HTTP {response.status}")
                        results.append(False)
            except Exception as e:
                self.log_test(f"Generate Signal - {pair}", False, f"Exception: {str(e)}")
                results.append(False)
        
        return all(results)
    
    async def test_market_data_endpoint(self):
        """Test GET /api/market-data/{pair} for all supported pairs"""
        results = []
        
        for pair in FOREX_PAIRS:
            try:
                async with self.session.get(f"{BACKEND_URL}/market-data/{pair}?timeframe=1h") as response:
                    if response.status == 200:
                        data = await response.json()
                        if "status" in data and data["status"] == "success" and "data" in data:
                            market_data = data["data"]
                            if market_data and "symbol" in market_data:
                                required_fields = ["open", "high", "low", "close", "volume"]
                                missing_fields = [field for field in required_fields if field not in market_data]
                                
                                if not missing_fields:
                                    self.log_test(f"Market Data - {pair}", True, 
                                                f"Price: {market_data.get('close')}")
                                    results.append(True)
                                else:
                                    self.log_test(f"Market Data - {pair}", False, 
                                                f"Missing fields: {missing_fields}")
                                    results.append(False)
                            else:
                                self.log_test(f"Market Data - {pair}", False, "Invalid market data structure")
                                results.append(False)
                        else:
                            self.log_test(f"Market Data - {pair}", False, "Invalid response structure")
                            results.append(False)
                    else:
                        self.log_test(f"Market Data - {pair}", False, f"HTTP {response.status}")
                        results.append(False)
            except Exception as e:
                self.log_test(f"Market Data - {pair}", False, f"Exception: {str(e)}")
                results.append(False)
        
        return all(results)
    
    async def test_scan_all_pairs_endpoint(self):
        """Test POST /api/scan-all-pairs (comprehensive signal generation)"""
        try:
            async with self.session.post(f"{BACKEND_URL}/scan-all-pairs") as response:
                if response.status == 200:
                    data = await response.json()
                    if "status" in data and data["status"] == "success":
                        signals_count = data.get("signals_generated", 0)
                        self.log_test("Scan All Pairs", True, 
                                    f"Generated {signals_count} signals")
                        return True
                    else:
                        self.log_test("Scan All Pairs", False, 
                                    f"Unexpected status: {data.get('status')}")
                        return False
                else:
                    self.log_test("Scan All Pairs", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Scan All Pairs", False, f"Exception: {str(e)}")
            return False
    
    async def test_session_cookie_endpoint(self):
        """Test POST /api/session-cookie (TradingView cookie management)"""
        try:
            payload = {
                "cookie_value": "test_cookie_value_12345",
                "user_agent": "Mozilla/5.0 (Test Browser)"
            }
            
            async with self.session.post(f"{BACKEND_URL}/session-cookie", 
                                       params=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    required_fields = ["id", "cookie_value", "user_agent", "created_at", "is_active"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if not missing_fields:
                        self.log_test("Session Cookie", True, 
                                    f"Cookie stored with ID: {data.get('id')}")
                        return True
                    else:
                        self.log_test("Session Cookie", False, 
                                    f"Missing fields: {missing_fields}")
                        return False
                else:
                    self.log_test("Session Cookie", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Session Cookie", False, f"Exception: {str(e)}")
            return False
    
    async def test_signal_structure_validation(self):
        """Test that generated signals have proper ICT/SMC structure"""
        try:
            # Generate a signal to validate structure
            payload = {"pair": "EURUSD", "timeframe": "1h"}
            async with self.session.post(f"{BACKEND_URL}/generate-signal", 
                                       params=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success" and "signal" in data:
                        signal = data["signal"]
                        
                        # Validate ICT/SMC specific requirements
                        validations = []
                        
                        # Check direction is BUY or SELL
                        if signal.get("direction") in ["BUY", "SELL"]:
                            validations.append(True)
                        else:
                            validations.append(False)
                        
                        # Check confidence is between 60-100% (confluence requirement)
                        confidence = signal.get("confidence", 0)
                        if 60 <= confidence <= 100:
                            validations.append(True)
                        else:
                            validations.append(False)
                        
                        # Check multiple take profit levels
                        tp_levels = [signal.get("take_profit_1"), signal.get("take_profit_2"), 
                                   signal.get("take_profit_3")]
                        if all(tp is not None for tp in tp_levels):
                            validations.append(True)
                        else:
                            validations.append(False)
                        
                        # Check reasoning contains ICT/SMC analysis
                        reasoning = signal.get("reasoning", "").lower()
                        if any(term in reasoning for term in ["ict", "smc", "confluence", "structure"]):
                            validations.append(True)
                        else:
                            validations.append(False)
                        
                        if all(validations):
                            self.log_test("Signal Structure Validation", True, 
                                        f"All ICT/SMC requirements met")
                            return True
                        else:
                            self.log_test("Signal Structure Validation", False, 
                                        f"Failed validations: {validations}")
                            return False
                    else:
                        self.log_test("Signal Structure Validation", True, 
                                    "No signal generated (acceptable)")
                        return True
                else:
                    self.log_test("Signal Structure Validation", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.log_test("Signal Structure Validation", False, f"Exception: {str(e)}")
            return False
    
    async def test_confluence_scoring_system(self):
        """Test that confluence scoring system works properly"""
        try:
            # Test multiple pairs to check confluence scoring
            confluence_results = []
            
            for pair in FOREX_PAIRS[:2]:  # Test first 2 pairs
                payload = {"pair": pair, "timeframe": "1h"}
                async with self.session.post(f"{BACKEND_URL}/generate-signal", 
                                           params=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success" and "signal" in data:
                            confidence = data["signal"].get("confidence", 0)
                            # Signal generated means confluence >= 60%
                            if confidence >= 60:
                                confluence_results.append(True)
                            else:
                                confluence_results.append(False)
                        elif data.get("status") == "no_signal":
                            # No signal means confluence < 60% (acceptable)
                            confluence_results.append(True)
                        else:
                            confluence_results.append(False)
                    else:
                        confluence_results.append(False)
            
            if all(confluence_results):
                self.log_test("Confluence Scoring System", True, 
                            "Confluence requirements properly enforced")
                return True
            else:
                self.log_test("Confluence Scoring System", False, 
                            "Confluence scoring issues detected")
                return False
                
        except Exception as e:
            self.log_test("Confluence Scoring System", False, f"Exception: {str(e)}")
            return False
    
    async def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Forex Trading Signal Bot Backend Tests")
        print("=" * 60)
        
        await self.setup()
        
        try:
            # Core API endpoint tests
            await self.test_health_check()
            await self.test_performance_endpoint()
            await self.test_signals_endpoint()
            await self.test_generate_signal_endpoint()
            await self.test_market_data_endpoint()
            await self.test_scan_all_pairs_endpoint()
            await self.test_session_cookie_endpoint()
            
            # Advanced functionality tests
            await self.test_signal_structure_validation()
            await self.test_confluence_scoring_system()
            
        finally:
            await self.cleanup()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests} ✅")
        print(f"Failed: {self.failed_tests} ❌")
        print(f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Backend is working correctly.")
        else:
            print(f"\n⚠️  {self.failed_tests} tests failed. Check details above.")
        
        return self.failed_tests == 0

async def main():
    """Main test runner"""
    tester = ForexBotTester()
    success = await tester.run_all_tests()
    
    # Save test results to file
    with open("/app/backend_test_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": tester.total_tests,
            "passed_tests": tester.passed_tests,
            "failed_tests": tester.failed_tests,
            "success_rate": tester.passed_tests/tester.total_tests*100 if tester.total_tests > 0 else 0,
            "test_results": tester.test_results
        }, f, indent=2)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))