from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timedelta
import asyncio
import json
import requests
import aiohttp
import random
import time
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from emergentintegrations.llm.chat import LlmChat, UserMessage
import telegram
from telegram import Bot

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configuration
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# TradingView User Agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
]

# Pydantic Models
class MarketData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pair: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class TradingSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pair: str
    direction: str  # BUY/SELL
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    confidence: float
    reasoning: str
    timeframe: str
    session: str
    risk_reward: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "ACTIVE"

class SessionCookie(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cookie_value: str
    user_agent: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class ICTAnalysis(BaseModel):
    liquidity_levels: List[float]
    order_blocks: List[Dict[str, Any]]
    fair_value_gaps: List[Dict[str, Any]]
    market_structure: str
    trend_direction: str
    confluence_score: float

# TradingView Scraper with Anti-Detection
class TradingViewScraper:
    def __init__(self):
        self.session = aiohttp.ClientSession()
        self.current_cookie = None
        self.current_user_agent = None
        
    async def get_active_session(self):
        """Get active session cookie from database"""
        cookie_doc = await db.session_cookies.find_one({"is_active": True})
        if cookie_doc:
            self.current_cookie = cookie_doc["cookie_value"]
            self.current_user_agent = cookie_doc["user_agent"]
            return True
        return False
    
    async def scrape_tradingview_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Scrape OHLCV data from TradingView with anti-detection"""
        try:
            # Random delay for anti-detection
            await asyncio.sleep(random.uniform(2, 5))
            
            # Get session or use fallback
            if not await self.get_active_session():
                return await self.fallback_data_source(symbol, timeframe)
            
            headers = {
                "User-Agent": self.current_user_agent or random.choice(USER_AGENTS),
                "Cookie": self.current_cookie,
                "Referer": "https://www.tradingview.com/",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest"
            }
            
            # TradingView API endpoint (simplified for demo)
            url = f"https://scanner.tradingview.com/forex/scan"
            
            payload = {
                "filter": [{"left": "name", "operation": "match", "right": symbol}],
                "columns": ["name", "close", "high", "low", "open", "volume"],
                "sort": {"sortBy": "name", "sortOrder": "asc"},
                "range": [0, 50]
            }
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self.process_tradingview_data(data, symbol)
                else:
                    logger.warning(f"TradingView scraping failed with status {response.status}")
                    return await self.fallback_data_source(symbol, timeframe)
                    
        except Exception as e:
            logger.error(f"TradingView scraping error: {e}")
            return await self.fallback_data_source(symbol, timeframe)
    
    def process_tradingview_data(self, data: Dict, symbol: str) -> Dict:
        """Process TradingView response data"""
        try:
            for item in data.get("data", []):
                if symbol.upper() in item["d"][0].upper():
                    return {
                        "symbol": symbol,
                        "open": item["d"][4],
                        "high": item["d"][2],
                        "low": item["d"][3],
                        "close": item["d"][1],
                        "volume": item["d"][5] if len(item["d"]) > 5 else 1000000,
                        "timestamp": datetime.utcnow()
                    }
            return None
        except Exception as e:
            logger.error(f"Error processing TradingView data: {e}")
            return None
    
    async def fallback_data_source(self, symbol: str, timeframe: str) -> Dict:
        """Fallback to demo data when TradingView fails"""
        logger.info(f"Using fallback data for {symbol}")
        
        # Generate realistic demo data based on symbol
        base_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 149.50,
            "XAUUSD": 2020.00
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        volatility = random.uniform(0.001, 0.005)
        
        # Generate OHLC
        open_price = base_price + random.uniform(-volatility, volatility)
        close_price = open_price + random.uniform(-volatility, volatility)
        high_price = max(open_price, close_price) + random.uniform(0, volatility/2)
        low_price = min(open_price, close_price) - random.uniform(0, volatility/2)
        
        return {
            "symbol": symbol,
            "open": round(open_price, 5),
            "high": round(high_price, 5),
            "low": round(low_price, 5),
            "close": round(close_price, 5),
            "volume": random.randint(500000, 2000000),
            "timestamp": datetime.utcnow()
        }

# ICT/SMC Analysis Engine
class ICTAnalyzer:
    def __init__(self):
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        
    async def analyze_market_structure(self, data: List[Dict], timeframe: str) -> ICTAnalysis:
        """Perform ICT/SMC analysis on market data"""
        try:
            # Extract OHLC data
            highs = [d["high"] for d in data]
            lows = [d["low"] for d in data]
            closes = [d["close"] for d in data]
            
            # Identify key levels
            liquidity_levels = self.find_liquidity_levels(highs, lows)
            order_blocks = self.identify_order_blocks(data)
            fair_value_gaps = self.find_fair_value_gaps(data)
            market_structure = self.determine_market_structure(highs, lows, closes)
            trend_direction = self.analyze_trend(closes)
            
            # Calculate confluence score
            confluence_score = self.calculate_confluence_score(
                liquidity_levels, order_blocks, fair_value_gaps, trend_direction
            )
            
            return ICTAnalysis(
                liquidity_levels=liquidity_levels,
                order_blocks=order_blocks,
                fair_value_gaps=fair_value_gaps,
                market_structure=market_structure,
                trend_direction=trend_direction,
                confluence_score=confluence_score
            )
            
        except Exception as e:
            logger.error(f"ICT Analysis error: {e}")
            return ICTAnalysis(
                liquidity_levels=[],
                order_blocks=[],
                fair_value_gaps=[],
                market_structure="NEUTRAL",
                trend_direction="SIDEWAYS",
                confluence_score=0.0
            )
    
    def find_liquidity_levels(self, highs: List[float], lows: List[float]) -> List[float]:
        """Identify institutional liquidity levels"""
        levels = []
        
        # Find swing highs and lows
        for i in range(2, len(highs) - 2):
            # Swing high
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                levels.append(highs[i])
            
            # Swing low
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                levels.append(lows[i])
        
        # Remove duplicates and sort
        levels = sorted(list(set(levels)))
        return levels[-10:]  # Return last 10 levels
    
    def identify_order_blocks(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify institutional order blocks"""
        order_blocks = []
        
        for i in range(1, len(data) - 1):
            current = data[i]
            prev = data[i-1]
            next_candle = data[i+1]
            
            # Bullish order block
            if (current["close"] > current["open"] and 
                prev["close"] < prev["open"] and 
                next_candle["close"] > current["high"]):
                
                order_blocks.append({
                    "type": "BULLISH",
                    "high": current["high"],
                    "low": current["low"],
                    "timestamp": current["timestamp"],
                    "strength": "HIGH"
                })
            
            # Bearish order block
            elif (current["close"] < current["open"] and 
                  prev["close"] > prev["open"] and 
                  next_candle["close"] < current["low"]):
                
                order_blocks.append({
                    "type": "BEARISH",
                    "high": current["high"],
                    "low": current["low"],
                    "timestamp": current["timestamp"],
                    "strength": "HIGH"
                })
        
        return order_blocks[-5:]  # Return last 5 order blocks
    
    def find_fair_value_gaps(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify Fair Value Gaps (FVG)"""
        fvgs = []
        
        for i in range(1, len(data) - 1):
            prev_candle = data[i-1]
            current = data[i]
            next_candle = data[i+1]
            
            # Bullish FVG
            if (prev_candle["high"] < next_candle["low"] and 
                current["close"] > current["open"]):
                
                fvgs.append({
                    "type": "BULLISH",
                    "high": next_candle["low"],
                    "low": prev_candle["high"],
                    "timestamp": current["timestamp"],
                    "filled": False
                })
            
            # Bearish FVG
            elif (prev_candle["low"] > next_candle["high"] and 
                  current["close"] < current["open"]):
                
                fvgs.append({
                    "type": "BEARISH",
                    "high": prev_candle["low"],
                    "low": next_candle["high"],
                    "timestamp": current["timestamp"],
                    "filled": False
                })
        
        return fvgs[-3:]  # Return last 3 FVGs
    
    def determine_market_structure(self, highs: List[float], lows: List[float], closes: List[float]) -> str:
        """Determine overall market structure"""
        if len(closes) < 10:
            return "NEUTRAL"
        
        recent_highs = highs[-5:]
        recent_lows = lows[-5:]
        
        # Higher highs and higher lows = Bullish structure
        if (max(recent_highs) > max(highs[-10:-5]) and 
            min(recent_lows) > min(lows[-10:-5])):
            return "BULLISH"
        
        # Lower highs and lower lows = Bearish structure
        elif (max(recent_highs) < max(highs[-10:-5]) and 
              min(recent_lows) < min(lows[-10:-5])):
            return "BEARISH"
        
        return "NEUTRAL"
    
    def analyze_trend(self, closes: List[float]) -> str:
        """Simple trend analysis"""
        if len(closes) < 20:
            return "SIDEWAYS"
        
        recent_avg = sum(closes[-10:]) / 10
        older_avg = sum(closes[-20:-10]) / 10
        
        if recent_avg > older_avg * 1.001:
            return "BULLISH"
        elif recent_avg < older_avg * 0.999:
            return "BEARISH"
        
        return "SIDEWAYS"
    
    def calculate_confluence_score(self, liquidity_levels: List[float], 
                                 order_blocks: List[Dict], 
                                 fair_value_gaps: List[Dict], 
                                 trend_direction: str) -> float:
        """Calculate confluence score for signal strength"""
        score = 0.0
        
        # Trend direction weight
        if trend_direction in ["BULLISH", "BEARISH"]:
            score += 0.3
        
        # Liquidity levels
        score += min(len(liquidity_levels) * 0.1, 0.3)
        
        # Order blocks
        score += min(len(order_blocks) * 0.15, 0.2)
        
        # Fair value gaps
        score += min(len(fair_value_gaps) * 0.1, 0.2)
        
        return min(score, 1.0)

# Signal Generator
class SignalGenerator:
    def __init__(self):
        self.ict_analyzer = ICTAnalyzer()
        self.scraper = TradingViewScraper()
        self.telegram_bot = None
        if os.environ.get('TELEGRAM_BOT_TOKEN'):
            self.telegram_bot = Bot(token=os.environ.get('TELEGRAM_BOT_TOKEN'))
    
    async def generate_signal(self, pair: str, timeframe: str) -> Optional[TradingSignal]:
        """Generate trading signal based on ICT/SMC analysis"""
        try:
            # Get market data
            data = await self.scraper.scrape_tradingview_data(pair, timeframe)
            if not data:
                return None
            
            # Simulate historical data for analysis
            historical_data = await self.get_historical_data(pair, timeframe)
            
            # Perform ICT analysis
            ict_analysis = await self.ict_analyzer.analyze_market_structure(historical_data, timeframe)
            
            # Check if confluence requirements are met
            if ict_analysis.confluence_score < 0.6:  # Minimum 60% confluence
                return None
            
            # Generate signal using Gemini AI
            signal = await self.generate_ai_signal(pair, data, ict_analysis, timeframe)
            
            return signal
            
        except Exception as e:
            logger.error(f"Signal generation error for {pair}: {e}")
            return None
    
    async def get_historical_data(self, pair: str, timeframe: str) -> List[Dict]:
        """Get historical data for analysis (simulated for demo)"""
        historical_data = []
        base_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 149.50,
            "XAUUSD": 2020.00
        }
        
        base_price = base_prices.get(pair, 1.0000)
        
        for i in range(50):  # Generate 50 historical candles
            timestamp = datetime.utcnow() - timedelta(hours=50-i)
            volatility = random.uniform(0.001, 0.003)
            
            open_price = base_price + random.uniform(-volatility, volatility)
            close_price = open_price + random.uniform(-volatility, volatility)
            high_price = max(open_price, close_price) + random.uniform(0, volatility/2)
            low_price = min(open_price, close_price) - random.uniform(0, volatility/2)
            
            historical_data.append({
                "timestamp": timestamp,
                "open": round(open_price, 5),
                "high": round(high_price, 5),
                "low": round(low_price, 5),
                "close": round(close_price, 5),
                "volume": random.randint(500000, 1500000)
            })
            
            base_price = close_price  # Update base for next candle
        
        return historical_data
    
    async def generate_ai_signal(self, pair: str, current_data: Dict, 
                               ict_analysis: ICTAnalysis, timeframe: str) -> Optional[TradingSignal]:
        """Use Gemini AI to generate sophisticated trading signal"""
        try:
            # Initialize Gemini chat
            chat = LlmChat(
                api_key=os.environ.get('GEMINI_API_KEY'),
                session_id=f"trading-{pair}-{timeframe}",
                system_message="""You are an expert ICT/SMC forex trader with institutional-grade analysis capabilities. 
                Analyze the provided market data and generate precise trading signals with clear reasoning."""
            ).with_model("gemini", "gemini-2.5-pro-preview-05-06").with_max_tokens(2048)
            
            # Prepare analysis prompt
            prompt = f"""
            Analyze {pair} on {timeframe} timeframe:
            
            Current Market Data:
            - Price: {current_data['close']}
            - High: {current_data['high']}
            - Low: {current_data['low']}
            - Volume: {current_data['volume']}
            
            ICT/SMC Analysis:
            - Market Structure: {ict_analysis.market_structure}
            - Trend Direction: {ict_analysis.trend_direction}
            - Confluence Score: {ict_analysis.confluence_score}
            - Liquidity Levels: {ict_analysis.liquidity_levels}
            - Order Blocks: {len(ict_analysis.order_blocks)} identified
            - Fair Value Gaps: {len(ict_analysis.fair_value_gaps)} identified
            
            Generate a trading signal with:
            1. Direction (BUY/SELL)
            2. Entry price with 3-pip tolerance
            3. Stop loss (ATR-based)
            4. 3 Take profit levels
            5. Risk-reward ratio (minimum 1:2.5)
            6. Confidence percentage
            7. Detailed reasoning
            
            Only generate signal if confluence score > 60% and conditions are favorable.
            Format response as JSON with these exact keys: direction, entry_price, stop_loss, take_profit_1, take_profit_2, take_profit_3, confidence, reasoning, risk_reward
            """
            
            user_message = UserMessage(text=prompt)
            response = await chat.send_message(user_message)
            
            # Parse AI response
            signal_data = self.parse_ai_response(response, pair, current_data, ict_analysis, timeframe)
            
            if signal_data:
                signal = TradingSignal(**signal_data)
                
                # Send to Telegram if configured
                if self.telegram_bot:
                    await self.send_telegram_signal(signal)
                
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"AI signal generation error: {e}")
            return None
    
    def parse_ai_response(self, response: str, pair: str, current_data: Dict, 
                         ict_analysis: ICTAnalysis, timeframe: str) -> Optional[Dict]:
        """Parse AI response and create signal data"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                ai_data = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = ['direction', 'entry_price', 'stop_loss', 'take_profit_1', 
                                 'take_profit_2', 'take_profit_3', 'confidence', 'reasoning']
                
                if all(field in ai_data for field in required_fields):
                    return {
                        "pair": pair,
                        "direction": ai_data['direction'],
                        "entry_price": float(ai_data['entry_price']),
                        "stop_loss": float(ai_data['stop_loss']),
                        "take_profit_1": float(ai_data['take_profit_1']),
                        "take_profit_2": float(ai_data['take_profit_2']),
                        "take_profit_3": float(ai_data['take_profit_3']),
                        "confidence": float(ai_data['confidence']),
                        "reasoning": ai_data['reasoning'],
                        "timeframe": timeframe,
                        "session": self.get_trading_session(),
                        "risk_reward": ai_data.get('risk_reward', 2.5)
                    }
            
            # Fallback: Generate signal based on analysis
            return self.generate_fallback_signal(pair, current_data, ict_analysis, timeframe)
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self.generate_fallback_signal(pair, current_data, ict_analysis, timeframe)
    
    def generate_fallback_signal(self, pair: str, current_data: Dict, 
                               ict_analysis: ICTAnalysis, timeframe: str) -> Dict:
        """Generate fallback signal when AI parsing fails"""
        current_price = current_data['close']
        
        # Determine direction based on trend
        direction = "BUY" if ict_analysis.trend_direction == "BULLISH" else "SELL"
        
        # Calculate levels based on ATR approximation
        atr = abs(current_data['high'] - current_data['low'])
        
        if direction == "BUY":
            entry_price = current_price + (atr * 0.1)
            stop_loss = entry_price - (atr * 1.5)
            take_profit_1 = entry_price + (atr * 2.5)
            take_profit_2 = entry_price + (atr * 4.0)
            take_profit_3 = entry_price + (atr * 6.0)
        else:
            entry_price = current_price - (atr * 0.1)
            stop_loss = entry_price + (atr * 1.5)
            take_profit_1 = entry_price - (atr * 2.5)
            take_profit_2 = entry_price - (atr * 4.0)
            take_profit_3 = entry_price - (atr * 6.0)
        
        return {
            "pair": pair,
            "direction": direction,
            "entry_price": round(entry_price, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit_1": round(take_profit_1, 5),
            "take_profit_2": round(take_profit_2, 5),
            "take_profit_3": round(take_profit_3, 5),
            "confidence": ict_analysis.confluence_score * 100,
            "reasoning": f"ICT/SMC Analysis: {ict_analysis.market_structure} market structure with {ict_analysis.trend_direction} trend. Confluence score: {ict_analysis.confluence_score:.2f}",
            "timeframe": timeframe,
            "session": self.get_trading_session(),
            "risk_reward": 2.5
        }
    
    def get_trading_session(self) -> str:
        """Determine current trading session"""
        from datetime import datetime
        import pytz
        
        utc_now = datetime.now(pytz.UTC)
        london_time = utc_now.astimezone(pytz.timezone('Europe/London'))
        ny_time = utc_now.astimezone(pytz.timezone('America/New_York'))
        
        hour = utc_now.hour
        
        if 8 <= hour < 17:
            return "LONDON"
        elif 13 <= hour < 22:
            return "NEW_YORK"
        elif 22 <= hour or hour < 8:
            return "ASIAN"
        else:
            return "OVERLAP"
    
    async def send_telegram_signal(self, signal: TradingSignal):
        """Send signal to Telegram"""
        try:
            message = f"""
🎯 **FOREX SIGNAL** 🎯

**Pair:** {signal.pair}
**Direction:** {signal.direction}
**Timeframe:** {signal.timeframe}
**Session:** {signal.session}

📈 **Entry:** {signal.entry_price}
🛑 **Stop Loss:** {signal.stop_loss}

🎯 **Take Profits:**
TP1: {signal.take_profit_1}
TP2: {signal.take_profit_2}
TP3: {signal.take_profit_3}

📊 **Analysis:**
Risk/Reward: {signal.risk_reward}:1
Confidence: {signal.confidence:.1f}%

💡 **Reasoning:**
{signal.reasoning}

⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC
            """
            
            chat_id = os.environ.get('TELEGRAM_CHAT_ID')
            if chat_id and self.telegram_bot:
                await self.telegram_bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
            
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

# Global signal generator
signal_generator = SignalGenerator()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Forex Trading Signal Bot API v1.0", "status": "active"}

@api_router.post("/session-cookie", response_model=SessionCookie)
async def update_session_cookie(request: dict):
    """Update TradingView session cookie for scraping"""
    try:
        cookie_value = request.get("cookie_value")
        user_agent = request.get("user_agent") or random.choice(USER_AGENTS)
        
        if not cookie_value:
            raise HTTPException(status_code=400, detail="cookie_value is required")
        
        # Test the cookie immediately
        test_result = await test_cookie_validity(cookie_value, user_agent)
        
        # Deactivate old cookies
        await db.session_cookies.update_many({}, {"$set": {"is_active": False}})
        
        # Add new cookie
        cookie_data = {
            "cookie_value": cookie_value,
            "user_agent": user_agent,
            "created_at": datetime.utcnow(),
            "is_active": True,
            "test_result": test_result
        }
        
        cookie_obj = SessionCookie(**cookie_data)
        await db.session_cookies.insert_one(cookie_obj.dict())
        
        logger.info(f"Cookie updated successfully. Test result: {test_result}")
        
        return cookie_obj
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cookie update failed: {str(e)}")

async def test_cookie_validity(cookie_value: str, user_agent: str) -> str:
    """Test if the provided cookie works with TradingView"""
    try:
        headers = {
            "User-Agent": user_agent,
            "Cookie": cookie_value,
            "Referer": "https://www.tradingview.com/",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest"
        }
        
        # Test with a simple TradingView request
        url = "https://scanner.tradingview.com/forex/scan"
        payload = {
            "filter": [{"left": "name", "operation": "match", "right": "EURUSD"}],
            "columns": ["name", "close", "high", "low", "open", "volume"],
            "sort": {"sortBy": "name", "sortOrder": "asc"},
            "range": [0, 1]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        return "VALID - Real data accessible"
                    else:
                        return "VALID - Response received but no data"
                else:
                    return f"INVALID - HTTP {response.status}"
                    
    except Exception as e:
        return f"ERROR - {str(e)}"

@api_router.get("/signals", response_model=List[TradingSignal])
async def get_signals(limit: int = 10):
    """Get recent trading signals"""
    try:
        signals = await db.trading_signals.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return [TradingSignal(**signal) for signal in signals]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching signals: {str(e)}")

@api_router.post("/generate-signal")
async def generate_signal_endpoint(pair: str, timeframe: str = "1h"):
    """Generate new trading signal for specified pair"""
    try:
        if pair not in FOREX_PAIRS:
            raise HTTPException(status_code=400, detail=f"Unsupported pair: {pair}")
        
        signal = await signal_generator.generate_signal(pair, timeframe)
        
        if signal:
            # Save to database
            await db.trading_signals.insert_one(signal.dict())
            return {"status": "success", "signal": signal}
        else:
            return {"status": "no_signal", "message": "No signal generated - insufficient confluence"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")

@api_router.post("/scan-all-pairs")
async def scan_all_pairs():
    """Scan all pairs for signals"""
    try:
        results = []
        
        for pair in FOREX_PAIRS:
            signal = await signal_generator.generate_signal(pair, "1h")
            if signal:
                await db.trading_signals.insert_one(signal.dict())
                results.append(signal)
        
        return {
            "status": "success",
            "signals_generated": len(results),
            "signals": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

@api_router.get("/market-data/{pair}")
async def get_market_data(pair: str, timeframe: str = "1h"):
    """Get current market data for pair"""
    try:
        scraper = TradingViewScraper()
        data = await scraper.scrape_tradingview_data(pair, timeframe)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)}")

@api_router.get("/performance")
async def get_performance():
    """Get trading performance statistics"""
    try:
        total_signals = await db.trading_signals.count_documents({})
        active_signals = await db.trading_signals.count_documents({"status": "ACTIVE"})
        
        # Calculate average confidence
        pipeline = [
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence"}}}
        ]
        avg_result = await db.trading_signals.aggregate(pipeline).to_list(1)
        avg_confidence = avg_result[0]["avg_confidence"] if avg_result else 0
        
        return {
            "total_signals": total_signals,
            "active_signals": active_signals,
            "average_confidence": round(avg_confidence, 2),
            "supported_pairs": FOREX_PAIRS,
            "timeframes": TIMEFRAMES
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance data error: {str(e)}")

# Background task for continuous signal generation
async def continuous_signal_generation():
    """Background task to continuously generate signals"""
    while True:
        try:
            logger.info("Starting signal generation cycle...")
            
            for pair in FOREX_PAIRS:
                signal = await signal_generator.generate_signal(pair, "1h")
                if signal:
                    await db.trading_signals.insert_one(signal.dict())
                    logger.info(f"Generated signal for {pair}: {signal.direction} at {signal.entry_price}")
                
                # Delay between pairs
                await asyncio.sleep(30)
            
            # Wait 1 hour before next cycle
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Continuous generation error: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes on error

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("Starting Forex Trading Signal Bot...")
    # Start continuous signal generation in background
    asyncio.create_task(continuous_signal_generation())

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()