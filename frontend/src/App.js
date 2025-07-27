import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"];

const ForexTradingBot = () => {
  const [signals, setSignals] = useState([]);
  const [marketData, setMarketData] = useState({});
  const [performance, setPerformance] = useState({});
  const [loading, setLoading] = useState(false);
  const [selectedPair, setSelectedPair] = useState("EURUSD");
  const [cookieValue, setCookieValue] = useState("");
  const [userAgent, setUserAgent] = useState("");
  const [activeTab, setActiveTab] = useState("dashboard");

  useEffect(() => {
    fetchSignals();
    fetchPerformance();
    const interval = setInterval(() => {
      fetchSignals();
      fetchMarketData();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchSignals = async () => {
    try {
      const response = await axios.get(`${API}/signals?limit=20`);
      setSignals(response.data);
    } catch (error) {
      console.error("Error fetching signals:", error);
    }
  };

  const fetchMarketData = async () => {
    try {
      const data = {};
      for (const pair of FOREX_PAIRS) {
        const response = await axios.get(`${API}/market-data/${pair}`);
        if (response.data.status === "success") {
          data[pair] = response.data.data;
        }
      }
      setMarketData(data);
    } catch (error) {
      console.error("Error fetching market data:", error);
    }
  };

  const fetchPerformance = async () => {
    try {
      const response = await axios.get(`${API}/performance`);
      setPerformance(response.data);
    } catch (error) {
      console.error("Error fetching performance:", error);
    }
  };

  const generateSignal = async (pair) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/generate-signal?pair=${pair}&timeframe=1h`);
      if (response.data.status === "success") {
        alert(`Signal generated for ${pair}!`);
        fetchSignals();
      } else {
        alert("No signal generated - insufficient confluence");
      }
    } catch (error) {
      console.error("Error generating signal:", error);
      alert("Error generating signal");
    } finally {
      setLoading(false);
    }
  };

  const scanAllPairs = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/scan-all-pairs`);
      alert(`Scan complete! ${response.data.signals_generated} signals generated.`);
      fetchSignals();
    } catch (error) {
      console.error("Error scanning pairs:", error);
      alert("Error scanning pairs");
    } finally {
      setLoading(false);
    }
  };

  const updateSessionCookie = async () => {
    try {
      const response = await axios.post(`${API}/session-cookie?cookie_value=${encodeURIComponent(cookieValue)}&user_agent=${encodeURIComponent(userAgent)}`);
      alert("Session cookie updated successfully!");
      setCookieValue("");
      setUserAgent("");
    } catch (error) {
      console.error("Error updating cookie:", error);
      alert("Error updating session cookie");
    }
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  const getDirectionColor = (direction) => {
    return direction === "BUY" ? "text-green-600" : "text-red-600";
  };

  const getDirectionBg = (direction) => {
    return direction === "BUY" ? "bg-green-100" : "bg-red-100";
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return "text-green-600";
    if (confidence >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-900 to-purple-900 text-white shadow-lg">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-lg flex items-center justify-center">
                <span className="text-2xl font-bold">💱</span>
              </div>
              <div>
                <h1 className="text-3xl font-bold">ICT/SMC Trading Bot</h1>
                <p className="text-blue-200">Advanced Forex Signal Generation</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-sm text-blue-200">Total Signals</div>
                <div className="text-2xl font-bold">{performance.total_signals || 0}</div>
              </div>
              <div className="text-right">
                <div className="text-sm text-blue-200">Avg. Confidence</div>
                <div className="text-2xl font-bold">{performance.average_confidence || 0}%</div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white shadow-sm border-b">
        <div className="container mx-auto px-4">
          <div className="flex space-x-8">
            {["dashboard", "signals", "market-data", "settings"].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-2 border-b-2 font-medium text-sm capitalize transition-colors duration-200 ${
                  activeTab === tab
                    ? "border-blue-500 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                {tab.replace("-", " ")}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <div className="container mx-auto px-4 py-8">
        {/* Dashboard Tab */}
        {activeTab === "dashboard" && (
          <div className="space-y-8">
            {/* Quick Actions */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Quick Actions</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                  onClick={scanAllPairs}
                  disabled={loading}
                  className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-6 py-4 rounded-lg font-semibold hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 transition-all duration-200 transform hover:scale-105"
                >
                  {loading ? "Scanning..." : "🔍 Scan All Pairs"}
                </button>
                
                <select
                  value={selectedPair}
                  onChange={(e) => setSelectedPair(e.target.value)}
                  className="px-4 py-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {FOREX_PAIRS.map((pair) => (
                    <option key={pair} value={pair}>{pair}</option>
                  ))}
                </select>
                
                <button
                  onClick={() => generateSignal(selectedPair)}
                  disabled={loading}
                  className="bg-gradient-to-r from-green-500 to-green-600 text-white px-6 py-4 rounded-lg font-semibold hover:from-green-600 hover:to-green-700 disabled:opacity-50 transition-all duration-200 transform hover:scale-105"
                >
                  {loading ? "Generating..." : "⚡ Generate Signal"}
                </button>
              </div>
            </div>

            {/* Market Overview */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Market Overview</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {FOREX_PAIRS.map((pair) => {
                  const data = marketData[pair];
                  return (
                    <div key={pair} className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-bold text-lg text-gray-800">{pair}</h3>
                        <span className="text-sm text-gray-500">LIVE</span>
                      </div>
                      {data ? (
                        <div className="space-y-2">
                          <div className="text-2xl font-bold text-blue-600">
                            {data.close}
                          </div>
                          <div className="flex justify-between text-xs text-gray-600">
                            <span>H: {data.high}</span>
                            <span>L: {data.low}</span>
                          </div>
                          <div className="text-xs text-gray-500">
                            Vol: {data.volume?.toLocaleString()}
                          </div>
                        </div>
                      ) : (
                        <div className="text-gray-400">Loading...</div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Recent Signals Preview */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-800">Recent Signals</h2>
                <button
                  onClick={() => setActiveTab("signals")}
                  className="text-blue-600 hover:text-blue-800 font-medium"
                >
                  View All →
                </button>
              </div>
              <div className="space-y-4">
                {signals.slice(0, 3).map((signal) => (
                  <div key={signal.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div className={`px-3 py-1 rounded-full text-sm font-bold ${getDirectionBg(signal.direction)} ${getDirectionColor(signal.direction)}`}>
                          {signal.direction}
                        </div>
                        <div>
                          <div className="font-bold">{signal.pair}</div>
                          <div className="text-sm text-gray-500">{signal.timeframe}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold">Entry: {signal.entry_price}</div>
                        <div className={`text-sm ${getConfidenceColor(signal.confidence)}`}>
                          {signal.confidence.toFixed(1)}% confidence
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Signals Tab */}
        {activeTab === "signals" && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Trading Signals</h2>
            <div className="space-y-4">
              {signals.map((signal) => (
                <div key={signal.id} className="border rounded-lg p-6 hover:shadow-lg transition-all duration-200">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-4">
                      <div className={`px-4 py-2 rounded-full text-lg font-bold ${getDirectionBg(signal.direction)} ${getDirectionColor(signal.direction)}`}>
                        {signal.direction}
                      </div>
                      <div>
                        <h3 className="text-xl font-bold">{signal.pair}</h3>
                        <p className="text-gray-600">{signal.timeframe} • {signal.session} Session</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-lg font-bold ${getConfidenceColor(signal.confidence)}`}>
                        {signal.confidence.toFixed(1)}% Confidence
                      </div>
                      <div className="text-sm text-gray-500">
                        {formatTime(signal.timestamp)}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                    <div className="bg-blue-50 rounded-lg p-3">
                      <div className="text-sm text-blue-600 font-medium">Entry Price</div>
                      <div className="text-lg font-bold text-blue-800">{signal.entry_price}</div>
                    </div>
                    <div className="bg-red-50 rounded-lg p-3">
                      <div className="text-sm text-red-600 font-medium">Stop Loss</div>
                      <div className="text-lg font-bold text-red-800">{signal.stop_loss}</div>
                    </div>
                    <div className="bg-green-50 rounded-lg p-3">
                      <div className="text-sm text-green-600 font-medium">Take Profit 1</div>
                      <div className="text-lg font-bold text-green-800">{signal.take_profit_1}</div>
                    </div>
                    <div className="bg-purple-50 rounded-lg p-3">
                      <div className="text-sm text-purple-600 font-medium">Risk/Reward</div>
                      <div className="text-lg font-bold text-purple-800">{signal.risk_reward}:1</div>
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-lg p-4">
                    <h4 className="font-bold text-gray-800 mb-2">Analysis Reasoning:</h4>
                    <p className="text-gray-700 text-sm leading-relaxed">{signal.reasoning}</p>
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                      TP2: {signal.take_profit_2}
                    </span>
                    <span className="px-3 py-1 bg-blue-100 text-blue-800 text-xs font-medium rounded-full">
                      TP3: {signal.take_profit_3}
                    </span>
                    <span className="px-3 py-1 bg-gray-100 text-gray-800 text-xs font-medium rounded-full">
                      Status: {signal.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Market Data Tab */}
        {activeTab === "market-data" && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Live Market Data</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {FOREX_PAIRS.map((pair) => {
                const data = marketData[pair];
                return (
                  <div key={pair} className="border rounded-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-2xl font-bold">{pair}</h3>
                      <span className="px-2 py-1 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                        LIVE
                      </span>
                    </div>
                    
                    {data ? (
                      <div className="space-y-4">
                        <div className="text-center">
                          <div className="text-4xl font-bold text-blue-600 mb-2">
                            {data.close}
                          </div>
                          <div className="text-sm text-gray-500">
                            Last updated: {formatTime(data.timestamp)}
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                          <div className="bg-gray-50 rounded-lg p-3">
                            <div className="text-sm text-gray-600">Open</div>
                            <div className="text-lg font-bold">{data.open}</div>
                          </div>
                          <div className="bg-red-50 rounded-lg p-3">
                            <div className="text-sm text-red-600">High</div>
                            <div className="text-lg font-bold text-red-800">{data.high}</div>
                          </div>
                          <div className="bg-blue-50 rounded-lg p-3">
                            <div className="text-sm text-blue-600">Low</div>
                            <div className="text-lg font-bold text-blue-800">{data.low}</div>
                          </div>
                          <div className="bg-green-50 rounded-lg p-3">
                            <div className="text-sm text-green-600">Volume</div>
                            <div className="text-lg font-bold text-green-800">
                              {data.volume?.toLocaleString()}
                            </div>
                          </div>
                        </div>
                        
                        <button
                          onClick={() => generateSignal(pair)}
                          className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white py-2 rounded-lg font-semibold hover:from-blue-600 hover:to-blue-700 transition-all duration-200"
                        >
                          Generate Signal for {pair}
                        </button>
                      </div>
                    ) : (
                      <div className="text-center text-gray-400 py-8">
                        <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
                        Loading market data...
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Settings Tab */}
        {activeTab === "settings" && (
          <div className="space-y-8">
            {/* TradingView Session Cookie */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">TradingView Session Settings</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Session Cookie Value
                  </label>
                  <textarea
                    value={cookieValue}
                    onChange={(e) => setCookieValue(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    rows="3"
                    placeholder="Paste TradingView session cookie here..."
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    User Agent (Optional)
                  </label>
                  <input
                    type="text"
                    value={userAgent}
                    onChange={(e) => setUserAgent(e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="Custom user agent or leave empty for automatic rotation"
                  />
                </div>
                <button
                  onClick={updateSessionCookie}
                  className="bg-gradient-to-r from-green-500 to-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:from-green-600 hover:to-green-700 transition-all duration-200"
                >
                  Update Session Cookie
                </button>
              </div>
              
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <h3 className="font-bold text-blue-800 mb-2">How to Get TradingView Session Cookie:</h3>
                <ol className="text-sm text-blue-700 space-y-1">
                  <li>1. Open TradingView.com in your browser</li>
                  <li>2. Log in to your account</li>
                  <li>3. Open Developer Tools (F12)</li>
                  <li>4. Go to Application/Storage → Cookies → tradingview.com</li>
                  <li>5. Find the session cookie and copy its value</li>
                  <li>6. Paste it in the field above and click Update</li>
                </ol>
              </div>
            </div>

            {/* System Status */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">System Status</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl mb-2">🟢</div>
                  <div className="font-bold text-green-800">API Status</div>
                  <div className="text-sm text-green-600">Operational</div>
                </div>
                <div className="text-center p-4 bg-yellow-50 rounded-lg">
                  <div className="text-2xl mb-2">🟡</div>
                  <div className="font-bold text-yellow-800">TradingView</div>
                  <div className="text-sm text-yellow-600">Using Fallback</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl mb-2">🟢</div>
                  <div className="font-bold text-green-800">Telegram Bot</div>
                  <div className="text-sm text-green-600">Connected</div>
                </div>
              </div>
            </div>

            {/* Performance Statistics */}
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">Performance Statistics</h2>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-3xl font-bold text-blue-600">
                    {performance.total_signals || 0}
                  </div>
                  <div className="text-sm text-blue-600">Total Signals</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-3xl font-bold text-green-600">
                    {performance.active_signals || 0}
                  </div>
                  <div className="text-sm text-green-600">Active Signals</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-3xl font-bold text-purple-600">
                    {performance.average_confidence || 0}%
                  </div>
                  <div className="text-sm text-purple-600">Avg. Confidence</div>
                </div>
                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <div className="text-3xl font-bold text-orange-600">
                    {performance.supported_pairs?.length || 0}
                  </div>
                  <div className="text-sm text-orange-600">Supported Pairs</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-16">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <span className="text-2xl">💱</span>
            <span className="text-xl font-bold">ICT/SMC Trading Bot</span>
          </div>
          <p className="text-gray-400">
            Advanced forex signal generation using Inner Circle Trader methodology
          </p>
          <div className="mt-4 flex justify-center space-x-6 text-sm text-gray-500">
            <span>Real-time Market Analysis</span>
            <span>•</span>
            <span>Institutional-Grade Signals</span>
            <span>•</span>
            <span>Multi-Timeframe Confluence</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <ForexTradingBot />
    </div>
  );
}

export default App;