# ===================================================
# bot.py — Multi-Market Ensemble Bot (US Alpaca + EU/CRYPTO/FX Paper)
# Strategie: SMA, RSI-MR, MACD, Breakout, Bollinger MR/BO, Donchian, Momentum(ROC), Ichimoku
# Anti-overfitting: range parametri stretti, malus trade, soglia miglioramento, ensemble fallback
# Calibrazione: ogni 12 ore (EU ~11:30 e ~23:30), lookback 300g
# ===================================================

import os, time, math, json, logging, datetime as dt, pathlib, random
from typing import List, Dict, Tuple, Callable
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import pytz
import yfinance as yf

# Alpaca (US) — opzionale per esecuzione reale/paper su USA
try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception:
    TradingClient = None

# ---------------------- ENV ----------------------
load_dotenv()
LOGLEVEL = os.getenv("LOGLEVEL","INFO").upper()
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(levelname)s: %(message)s")

# Keys / flags
ALPACA_KEY = os.getenv("ALPACA_KEY_ID")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER = os.getenv("PAPER","true").lower() in ("1","true","yes")

# Symbols
SYMBOLS_US = [s.strip() for s in os.getenv("SYMBOLS_US","AAPL,MSFT,AMZN,GOOGL,NVDA,TSLA,JPM,XOM,JNJ,SPY,QQQ,DIA").split(",") if s.strip()]
SYMBOLS_EU = [s.strip() for s in os.getenv("SYMBOLS_EU","ENEL.MI,ISP.MI,ENI.MI,LUX.MI,DTE.DE,BMW.DE,SAP.DE,BNP.PA,SAN.PA,SAN.MC").split(",") if s.strip()]
SYMBOLS_CRYPTO = [s.strip() for s in os.getenv("SYMBOLS_CRYPTO","BTC-USD,ETH-USD,SOL-USD").split(",") if s.strip()]
SYMBOLS_FX = [s.strip() for s in os.getenv("SYMBOLS_FX","EURUSD=X,GBPUSD=X,USDJPY=X").split(",") if s.strip()]

# Sessions
SESSION_TZ_EU = os.getenv("SESSION_TZ_EU","Europe/Rome")
SESSION_START_EU = os.getenv("SESSION_START_EU","09:00")
SESSION_END_EU = os.getenv("SESSION_END_EU","17:30")
SESSION_TZ_US = os.getenv("SESSION_TZ_US","America/New_York")
SESSION_START_US = os.getenv("SESSION_START_US","09:30")
SESSION_END_US = os.getenv("SESSION_END_US","15:55")

# Loop cadence
BAR_TIMEFRAME_LIVE = os.getenv("BAR_TIMEFRAME_LIVE","5Min")  # 5Min o 15Min
LOOP_MINUTES = int(os.getenv("LOOP_MINUTES","10"))

# Risk
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE","0.015"))   # 1.5% equity per trade
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT","0.03"))
DAILY_PROFIT_CAP_PCT = float(os.getenv("DAILY_PROFIT_CAP_PCT","0.04"))
COOLDOWN_LOSSES = int(os.getenv("COOLDOWN_LOSSES","3"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD","14"))
ATR_MULT_STOP = float(os.getenv("ATR_MULT_STOP","2.0"))
ATR_REGIME_PCTL = int(os.getenv("ATR_REGIME_PCTL","30"))

# Learning
LEARN_LOOKBACK_DAYS = int(os.getenv("LEARN_LOOKBACK_DAYS","300"))
RAND_SEARCH_ITERS = int(os.getenv("RAND_SEARCH_ITERS","220"))
WALK_FORWARD_FOLDS = int(os.getenv("WALK_FORWARD_FOLDS","4"))
IMPROVE_THRESHOLD = float(os.getenv("IMPROVE_THRESHOLD","0.10"))  # +10% metrica per sostituzione modello

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID")

# Storage
DATA = pathlib.Path("data"); DATA.mkdir(exist_ok=True)
EQ_CSV = DATA/"equity.csv"
TRADES_CSV = DATA/"trades.csv"
SIG_CSV = DATA/"signals.csv"
POS_CSV = DATA/"positions.csv"
LEARN_CSV = DATA/"learning.csv"
MODEL_JSON = DATA/"model.json"
STATE_JSON = DATA/"state.json"

# ---------------------- Utils ----------------------
def save_json(path: pathlib.Path, obj: Dict): path.write_text(json.dumps(obj, indent=2, default=str))
def load_json(path: pathlib.Path, default: Dict=None) -> Dict:
    if path.exists(): return json.loads(path.read_text())
    return {} if default is None else default

def append_csv(path: pathlib.Path, row: Dict):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

def ensure_csv(path: pathlib.Path, headers: List[str]):
    if not path.exists(): pd.DataFrame(columns=headers).to_csv(path, index=False)

def telegram_send(text: str):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT): return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id":TELEGRAM_CHAT, "text":text, "parse_mode":"HTML"}, timeout=10)
    except Exception: logging.exception("Telegram error")

# Time helpers
def now_tz(tz: str) -> dt.datetime: return dt.datetime.now(pytz.timezone(tz))
def parse_hhmm(s: str): h,m=s.split(":"); return int(h),int(m)
def within_session(now_local: dt.datetime, start:str, end:str)->bool:
    sh,sm=parse_hhmm(start); eh,em=parse_hhmm(end)
    b=now_local.replace(hour=sh,minute=sm,second=0,microsecond=0)
    e=now_local.replace(hour=eh,minute=em,second=0,microsecond=0)
    return b <= now_local <= e

# ---------------------- Indicators / Signals ----------------------
def sma_signal(df: pd.DataFrame, fast=10, slow=50):
    fast=max(5, min(fast, 30)); slow=max(30, min(slow, 120)); 
    s = pd.Series(0, index=df.index, dtype=float)
    f = df["close"].rolling(fast).mean(); sl = df["close"].rolling(slow).mean()
    s[f>sl]=1; s[f<sl]=-1; return s

def rsi(series: pd.Series, period=14):
    period=max(10, min(period, 21))
    d=series.diff(); up=d.clip(lower=0).rolling(period).mean(); down=-d.clip(upper=0).rolling(period).mean()
    rs=up/(down+1e-12); return 100-100/(1+rs)

def rsi_mr_signal(df: pd.DataFrame, period=14, low=30, high=70):
    low=max(20, min(low,40)); high=max(low+10, min(high,80))
    s=pd.Series(0,index=df.index,dtype=float); r=rsi(df["close"],period)
    s[r<low]=1; s[r>high]=-1; return s

def macd_signal(df: pd.DataFrame, fast=12, slow=26, sig=9):
    fast=max(8,min(fast,16)); slow=max(fast+4,min(slow,40)); sig=max(7,min(sig,12))
    ef=df["close"].ewm(span=fast,adjust=False).mean()
    es=df["close"].ewm(span=slow,adjust=False).mean()
    m=ef-es; ms=m.ewm(span=sig,adjust=False).mean()
    s=pd.Series(0,index=df.index,dtype=float); s[m>ms]=1; s[m<ms]=-1; return s

def breakout_signal(df: pd.DataFrame, lookback=20):
    lb=max(10, min(lookback, 40))
    hh = df["close"].rolling(lb).max()
    ll = df["close"].rolling(lb).min()
    s = pd.Series(0, index=df.index, dtype=float)
    s[df["close"]>=hh]=1; s[df["close"]<=ll]=-1; return s

def bbands(series: pd.Series, n=20, k=2.0):
    n=max(10, min(n,40)); k=max(1.0, min(k,3.0))
    ma=series.rolling(n).mean(); sd=series.rolling(n).std()
    upper=ma+k*sd; lower=ma-k*sd
    return ma, upper, lower

def boll_mr_signal(df: pd.DataFrame, n=20, k=2.0):
    ma,u,l = bbands(df["close"], n, k)
    s=pd.Series(0,index=df.index,dtype=float)
    s[df["close"]<l]=1; s[df["close"]>u]=-1; return s

def boll_break_signal(df: pd.DataFrame, n=20, k=2.0):
    ma,u,l = bbands(df["close"], n, k)
    s=pd.Series(0,index=df.index,dtype=float)
    s[df["close"]>u]=1; s[df["close"]<l]=-1; return s

def donchian_signal(df: pd.DataFrame, n=20):
    n=max(10, min(n,40))
    hh=df["close"].rolling(n).max(); ll=df["close"].rolling(n).min()
    mid=(hh+ll)/2
    s=pd.Series(0,index=df.index,dtype=float)
    s[df["close"]>=hh]=1; s[df["close"]<=ll]=-1
    # filtro: se in mezzo, usa segno rispetto alla mediana
    s[(df["close"]>mid)&(s==0)] = 1
    s[(df["close"]<mid)&(s==0)] = -1
    return s

def roc(series: pd.Series, n=10):
    n=max(5,min(n,30)); return series.pct_change(n).fillna(0.0)

def momentum_signal(df: pd.DataFrame, n=10):
    r=roc(df["close"], n)
    s=pd.Series(0,index=df.index,dtype=float); s[r>0]=1; s[r<0]=-1; return s

def ichimoku_base(df: pd.DataFrame, conv=9, base=26):
    conv=max(9,min(conv,12)); base=max(26,min(base,34))
    high=df["close"].rolling(conv).max(); low=df["close"].rolling(conv).min()
    conversion=(high+low)/2
    high_b=df["close"].rolling(base).max(); low_b=df["close"].rolling(base).min()
    baseline=(high_b+low_b)/2
    s=pd.Series(0,index=df.index,dtype=float); s[conversion>baseline]=1; s[conversion<baseline]=-1; return s

def atr_like(df: pd.DataFrame, period=14):
    period=max(10,min(period,21))
    ret=df["close"].pct_change().fillna(0.0)
    return (ret.rolling(period).std()*np.sqrt(period)).abs()

# ---------------------- Backtest / Metric (anti-overfit) ----------------------
def backtest(df: pd.DataFrame, sig: pd.Series, cost_bps=5, comm=0.0005):
    d=df.copy()
    d["sig"]=sig.reindex(d.index).fillna(0)
    d["pos"]=d["sig"].shift(1).fillna(0)
    ret=d["close"].pct_change().fillna(0)
    trade_change = d["pos"].diff().abs().fillna(0)
    num_trades = int((trade_change>0).sum())
    cost = trade_change*(cost_bps/10000 + comm)
    strat = d["pos"]*ret - cost
    eq=(1+strat).cumprod()
    tot=eq.iloc[-1]-1; roll=eq.cummax(); mdd=((roll-eq)/roll).max()
    vol = strat.std()
    sharpe=(strat.mean()/max(1e-9,vol))*np.sqrt(252) if vol>1e-9 else 0.0
    # Anti-overfit metric: penalizza troppi trade
    trade_penalty = 0.0005 * num_trades  # malus dolce
    metric=float(sharpe - 3.0*float(mdd) - trade_penalty)
    return {"total_return":float(tot),"max_drawdown":float(mdd),"sharpe":float(sharpe),
            "metric":metric,"trades":num_trades}, eq

def walk_forward_score(df: pd.DataFrame, mk_sig: Callable, folds:int=4) -> float:
    n=len(df); k=max(2,folds); fs=n//k; metrics=[]
    for i in range(k-1):
        test=df.iloc[fs*(i+1):fs*(i+2)]
        if len(test)<60: continue
        sig=mk_sig(test); m,_=backtest(test, sig); metrics.append(m["metric"])
    return float(np.mean(metrics)) if metrics else -1e9

# ---------------------- Search (range stretti) ----------------------
def extended_search(df: pd.DataFrame, iters:int=220, folds:int=4)->Dict:
    random.seed(42)
    best={"strategy":"SMA","metric":-1e9,"params":{}}

    # SMA
    for _ in range(iters//6):
        f=random.choice([5,8,10,12,15,20]); s=random.choice([30,40,50,60,80,100])
        if f>=s: continue
        score=walk_forward_score(df, lambda d,f=f,s=s: sma_signal(d,f,s), folds)
        if score>best["metric"]: best={"strategy":"SMA","metric":score,"params":{"fast":f,"slow":s}}

    # RSI MR
    for _ in range(iters//6):
        p=random.choice([10,14,18,21]); lo=random.choice([25,30,35]); hi=100-random.choice([25,30,35])
        if lo>=hi: continue
        score=walk_forward_score(df, lambda d,p=p,lo=lo,hi=hi: rsi_mr_signal(d,p,lo,hi), folds)
        if score>best["metric"]: best={"strategy":"RSI","metric":score,"params":{"period":p,"low":lo,"high":hi}}

    # MACD
    for _ in range(iters//6):
        f=random.choice([8,12,16]); s=random.choice([20,26,32,40]); sg=random.choice([7,9,12])
        if f>=s: continue
        score=walk_forward_score(df, lambda d,f=f,s=s,sg=sg: macd_signal(d,f,s,sg), folds)
        if score>best["metric"]: best={"strategy":"MACD","metric":score,"params":{"fast":f,"slow":s,"sig":sg}}

    # Breakout
    for _ in range(iters//6):
        lb=random.choice([10,20,40])
        score=walk_forward_score(df, lambda d,l=lb: breakout_signal(d,l), folds)
        if score>best["metric"]: best={"strategy":"BRK","metric":score,"params":{"lookback":lb}}

    # Bollinger MR
    for _ in range(iters//6):
        n=random.choice([15,20,25]); k=random.choice([1.8,2.0,2.2])
        score=walk_forward_score(df, lambda d,n=n,k=k: boll_mr_signal(d,n,k), folds)
        if score>best["metric"]: best={"strategy":"BOLL_MR","metric":score,"params":{"n":n,"k":k}}

    # Donchian / Momentum / Ichimoku
    for _ in range(iters//6):
        choice=random.choice(["DON","MOM","ICHI"])
        if choice=="DON":
            n=random.choice([15,20,25,30])
            score=walk_forward_score(df, lambda d,n=n: donchian_signal(d,n), folds)
            if score>best["metric"]: best={"strategy":"DON","metric":score,"params":{"n":n}}
        elif choice=="MOM":
            n=random.choice([10,14,21,28])
            score=walk_forward_score(df, lambda d,n=n: momentum_signal(d,n), folds)
            if score>best["metric"]: best={"strategy":"MOM","metric":score,"params":{"n":n}}
        else:
            cv=random.choice([9,10,12]); bs=random.choice([26,30,34])
            score=walk_forward_score(df, lambda d,cv=cv,bs=bs: ichimoku_base(d,cv,bs), folds)
            if score>best["metric"]: best={"strategy":"ICHI","metric":score,"params":{"conv":cv,"base":bs}}

    return best

# ---------------------- Ensemble ----------------------
def ensemble_signal(df: pd.DataFrame, params: Dict) -> pd.Series:
    comps = [
        sma_signal(df, params.get("sma_fast",10), params.get("sma_slow",50)),
        rsi_mr_signal(df, params.get("rsi_period",14), params.get("rsi_low",30), params.get("rsi_high",70)),
        macd_signal(df, params.get("macd_fast",12), params.get("macd_slow",26), params.get("macd_sig",9)),
        breakout_signal(df, params.get("brk_lookback",20)),
        boll_mr_signal(df, params.get("boll_n",20), params.get("boll_k",2.0)),
        donchian_signal(df, params.get("don_n",20)),
        momentum_signal(df, params.get("mom_n",14)),
        ichimoku_base(df, params.get("ichi_conv",9), params.get("ichi_base",26)),
    ]
    look = min(len(df), 84)  # ~4 mesi
    def perf(sig):
        m,_ = backtest(df.tail(look), sig.tail(look))
        return max(0.0, m["metric"])
    scores = np.array([perf(s) for s in comps], dtype=float) + 1e-9
    w = scores / scores.sum()
    ens_raw = sum(w[i]*comps[i] for i in range(len(comps)))
    ens = ens_raw.apply(lambda x: 1 if x>0.20 else (-1 if x<-0.20 else 0))
    return ens

# ---------------------- Sizing ----------------------
def position_size_atr(equity: float, price: float, atr_val: float) -> int:
    per_share_risk = max(1e-9, ATR_MULT_STOP * atr_val * price)
    qty = math.floor((equity * RISK_PER_TRADE) / per_share_risk)
    return max(0, int(qty))

# ---------------------- Brokers ----------------------
class AlpacaBroker:
    def __init__(self):
        if TradingClient is None:
            raise RuntimeError("alpaca-py non installato (aggiungi 'alpaca-py' a requirements).")
        if not (ALPACA_KEY and ALPACA_SECRET):
            raise RuntimeError("Mancano ALPACA_KEY_ID/ALPACA_SECRET_KEY")
        self.trading = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)
        self.data = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    def name(self): return "Alpaca(US-paper)" if PAPER else "Alpaca(US-live)"
    def equity(self):
        try: return float(self.trading.get_account().equity)
        except: return 0.0
    def position_qty(self, symbol):
        try: return float(self.trading.get_open_position(symbol).qty)
        except: return 0.0
    def historical(self, symbol, timeframe="1Day", limit=400)->pd.DataFrame:
        tf_map={"1Min":TimeFrame.Minute,"5Min":TimeFrame.FiveMinutes,"15Min":TimeFrame.FifteenMinutes,"1Hour":TimeFrame.Hour,"1Day":TimeFrame.Day}
        tf=tf_map.get(timeframe,TimeFrame.Day)
        bars=self.data.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol,timeframe=tf,limit=limit)).df
        df = bars.xs(symbol, level=0).reset_index() if isinstance(bars.index, pd.MultiIndex) else bars.reset_index()
        df.rename(columns={"timestamp":"time","close":"close"}, inplace=True)
        return df[["time","close"]].dropna().set_index("time")
    def market_order(self, symbol, qty, side):
        try:
            order=MarketOrderRequest(symbol=symbol, qty=abs(int(qty)),
                                     side=OrderSide.BUY if side=="buy" else OrderSide.SELL,
                                     time_in_force=TimeInForce.DAY)
            res=self.trading.submit_order(order); return res.id
        except Exception as e:
            logging.warning(f"Alpaca order fail {symbol} {side}: {e}"); return None

class PaperSimBroker:
    def __init__(self, label:str, starting_cash:float=10000.0):
        self.label=label; self.cash=starting_cash; self.pos: Dict[str,int]={}
    def name(self): return f"{self.label}-PaperSim"
    def equity(self)->float:
        total = self.cash
        for sym,q in self.pos.items():
            price = self._last_price(sym); total += q*price
        return float(total)
    def position_qty(self, symbol)->float: return float(self.pos.get(symbol,0))
    def _last_price(self, symbol)->float:
        try:
            df = self.historical(symbol, "1Day", limit=2)
            return float(df["close"].iloc[-1])
        except: return 0.0
    def close_position(self, symbol):
        q=self.pos.get(symbol,0)
        if q!=0:
            price=self._last_price(symbol); self.cash += q*price; self.pos[symbol]=0
    def close_all(self):
        for s in list(self.pos.keys()): self.close_position(s)
    def historical(self, symbol, timeframe="1Day", limit=400)->pd.DataFrame:
        if timeframe in ("1Min","5Min","15Min"):
            interval="5m" if timeframe!="15Min" else "15m"; period="60d"
        else:
            interval="1d"; period=f"{limit}d"
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        data = data.rename(columns={"Close":"close"})[["close"]].dropna(); data.index.name="time"
        return data.tail(limit)
    def market_order(self, symbol, qty, side):
        price=self._last_price(symbol)
        if side=="buy":
            cost=price*qty
            if cost<=self.cash:
                self.cash-=cost; self.pos[symbol]=self.pos.get(symbol,0)+qty; return f"SIMBUY-{symbol}-{int(time.time())}"
            return None
        else:
            have=self.pos.get(symbol,0); sell=min(have,qty)
            self.pos[symbol]=have-sell; self.cash+=sell*price; return f"SIMSELL-{symbol}-{int(time.time())}"

# ---------------------- Model / State ----------------------
def load_model()->Dict:
    return load_json(MODEL_JSON, {"by_symbol":{}, "updated_utc":None})

def save_model(m:Dict): save_json(MODEL_JSON, m)

def load_state()->Dict:
    st=load_json(STATE_JSON, {})
    for b in ("EU","US","CRYPTO","FX"):
        st.setdefault(f"{b}_baseline_eq", None)
        st.setdefault(f"{b}_losses", 0)
        st.setdefault(f"{b}_last_report","")
        st.setdefault(f"{b}_baseline_day","")
    return st

def save_state(st:Dict): save_json(STATE_JSON, st)

# ---------------------- Risk gates ----------------------
def should_open_new(book:str, equity_now:float)->bool:
    st=load_state()
    tz = {"EU":SESSION_TZ_EU,"US":SESSION_TZ_US,"CRYPTO":"UTC","FX":"UTC"}[book]
    now = now_tz(tz)
    base_key=f"{book}_baseline_eq"
    if st[base_key] is None or st.get(f"{book}_baseline_day","") != str(now.date()):
        st[base_key]=equity_now; st[f"{book}_baseline_day"]=str(now.date()); st[f"{book}_losses"]=0; save_state(st)
    baseline = st[base_key] or equity_now
    pnl_pct = (equity_now/baseline - 1.0)
    if pnl_pct <= -MAX_DAILY_LOSS_PCT: 
        logging.warning(f"{book}: max daily loss hit ({pnl_pct:.2%})"); return False
    if DAILY_PROFIT_CAP_PCT>0 and pnl_pct >= DAILY_PROFIT_CAP_PCT:
        logging.info(f"{book}: profit cap reached ({pnl_pct:.2%})"); return False
    if st.get(f"{book}_losses",0) >= COOLDOWN_LOSSES:
        logging.info(f"{book}: cooldown after losses"); return False
    return True

def update_losses(book:str, pnl_positive: bool):
    st=load_state()
    st[f"{book}_losses"] = max(0, st.get(f"{book}_losses",0)-1) if pnl_positive else st.get(f"{book}_losses",0)+1
    save_state(st)

# ---------------------- Direction + Intraday timing ----------------------
def ensemble_defaults():
    return {"sma_fast":10,"sma_slow":50,"rsi_period":14,"rsi_low":30,"rsi_high":70,
            "macd_fast":12,"macd_slow":26,"macd_sig":9,"brk_lookback":20,
            "boll_n":20,"boll_k":2.0,"don_n":20,"mom_n":14,"ichi_conv":9,"ichi_base":26}

def pick_daily_signal(df_daily: pd.DataFrame, sym: str) -> float:
    model = load_model()
    p = model.get("by_symbol", {}).get(sym, {})
    if p.get("mode","ensemble")=="single":
        strat=p.get("strategy","SMA")
        if strat=="SMA": return float(sma_signal(df_daily, p.get("fast",10), p.get("slow",50)).iloc[-1])
        if strat=="RSI": return float(rsi_mr_signal(df_daily, p.get("period",14), p.get("low",30), p.get("high",70)).iloc[-1])
        if strat=="MACD":return float(macd_signal(df_daily, p.get("fast",12), p.get("slow",26), p.get("sig",9)).iloc[-1])
        if strat=="BRK": return float(breakout_signal(df_daily, p.get("lookback",20)).iloc[-1])
        if strat=="BOLL_MR": return float(boll_mr_signal(df_daily, p.get("n",20), p.get("k",2.0)).iloc[-1])
        if strat=="DON": return float(donchian_signal(df_daily, p.get("n",20)).iloc[-1])
        if strat=="MOM": return float(momentum_signal(df_daily, p.get("n",14)).iloc[-1])
        if strat=="ICHI": return float(ichimoku_base(df_daily, p.get("conv",9), p.get("base",26)).iloc[-1])
    # ensemble di default / fallback
    params = {**ensemble_defaults(), **p.get("ensemble", {})}
    return float(ensemble_signal(df_daily, params).iloc[-1])

def intraday_timing(df_intr: pd.DataFrame) -> Tuple[float,float,bool,bool]:
    sig = sma_signal(df_intr, 10, 50).iloc[-1]  # timing base
    slow = df_intr["close"].rolling(50).mean()
    slope_ok = slow.diff().iloc[-1] > 0
    atrs = atr_like(df_intr, ATR_PERIOD)
    atr_now = float(atrs.iloc[-1]); pctl = float(atrs.rank(pct=True).iloc[-1]*100)
    ret = df_intr["close"].pct_change().fillna(0.0)
    spike_veto = abs(ret.iloc[-1]) > (2.0 * max(1e-9, atrs.iloc[-1]))
    regime_ok = (pctl >= ATR_REGIME_PCTL) and not spike_veto
    return float(sig), float(atr_now), bool(slope_ok), bool(regime_ok)

# ---------------------- Learning (ogni 12 ore) ----------------------
def learn_symbol_params(symbol:str, broker) -> Dict:
    df = broker.historical(symbol, "1Day", LEARN_LOOKBACK_DAYS+60)
    if df.shape[0] < 180: return {}
    best = extended_search(df.tail(LEARN_LOOKBACK_DAYS), iters=RAND_SEARCH_ITERS, folds=WALK_FORWARD_FOLDS)
    return {"mode":"single","strategy":best["strategy"], **best.get("params",{}),
            "metric":best["metric"], "ensemble":ensemble_defaults()}

def nightly_learning(brokers: Dict[str,object], symbols_by_book: Dict[str,List[str]]):
    updated = {}
    model = load_model()
    cur = model.get("by_symbol", {})
    for book, syms in symbols_by_book.items():
        for s in syms:
            try:
                p=learn_symbol_params(s, brokers[book])
                if not p: continue
                prev = cur.get(s, {})
                prev_metric = prev.get("metric", -1e9)
                new_metric = p.get("metric", -1e9)
                if prev_metric <= 0:  # nessun modello precedente
                    cur[s]=p; updated[s]=("init", new_metric)
                else:
                    # sostituisci solo se miglioramento >= +10% (threshold)
                    if new_metric >= prev_metric*(1.0+IMPROVE_THRESHOLD):
                        cur[s]=p; updated[s]=("improved", new_metric)
                    else:
                        # fallback a ensemble robusto
                        cur[s]={"mode":"ensemble", **prev.get("ensemble", ensemble_defaults()), "metric":prev_metric}
                        updated[s]=("fallback_ens", prev_metric)
                append_csv(LEARN_CSV, {"ts_utc": dt.datetime.utcnow().isoformat(),
                                       "symbol": s, "book": book,
                                       "strategy": p.get("strategy","ensemble"),
                                       "metric": p.get("metric",0.0),
                                       "params": json.dumps(p)})
            except Exception:
                logging.exception(f"Learning error {s}")
    if updated:
        model["by_symbol"]=cur; model["updated_utc"]=dt.datetime.utcnow().isoformat()
        save_model(model)
        telegram_send("🧠 Calibrazione aggiornata: " + ", ".join([f"{k}:{v[0]}" for k,v in updated.items()]))

# ---------------------- Trading core ----------------------
def trade_book(book:str, symbols: List[str], broker, session_open: bool):
    append_csv(EQ_CSV, {"ts_utc":dt.datetime.utcnow().isoformat(),"book":book,"equity":broker.equity()})
    if not session_open: return
    can_open = should_open_new(book, broker.equity())

    for sym in symbols:
        try:
            df_d = broker.historical(sym, "1Day", 360)
            tf = "5Min" if BAR_TIMEFRAME_LIVE.lower().startswith("5") else "15Min"
            df_i = broker.historical(sym, tf, 400)
            if df_d.shape[0] < 120 or df_i.shape[0] < 80: continue

            dir_daily = pick_daily_signal(df_d, sym)
            sig_intra, atr_now, slope_ok, ok_filters = intraday_timing(df_i)
            price = float(df_i["close"].iloc[-1])
            pos = broker.position_qty(sym)

            append_csv(SIG_CSV, {"ts_utc":dt.datetime.utcnow().isoformat(),"symbol":sym,
                                 "sig_daily":dir_daily,"sig_intra":sig_intra,"price":price,"book":book})

            open_long = (dir_daily>0 and sig_intra>0 and slope_ok and ok_filters and can_open)
            close_long = (pos>0 and (sig_intra<0 or not slope_ok or not ok_filters))

            if open_long and pos==0:
                qty = position_size_atr(broker.equity(), price, atr_now)
                if qty>0:
                    oid = broker.market_order(sym, qty, "buy")
                    if oid:
                        append_csv(TRADES_CSV, {"ts_utc":dt.datetime.utcnow().isoformat(),"symbol":sym,
                                                "side":"BUY","qty":qty,"price":price,"order_id":oid,"book":book})
                        telegram_send(f"{'🇪🇺' if book=='EU' else '🇺🇸' if book=='US' else '₿' if book=='CRYPTO' else '💱'} BUY {sym} x{qty} @ {price:.2f}")
            elif close_long and pos>0:
                eq_before = broker.equity()
                oid = broker.market_order(sym, pos, "sell")
                if oid:
                    append_csv(TRADES_CSV, {"ts_utc":dt.datetime.utcnow().isoformat(),"symbol":sym,
                                            "side":"CLOSE","qty":pos,"price":price,"order_id":oid,"book":book})
                    pnl_positive = broker.equity() >= eq_before
                    update_losses(book, pnl_positive)
                    telegram_send(f"{'🇪🇺' if book=='EU' else '🇺🇸' if book=='US' else '₿' if book=='CRYPTO' else '💱'} CLOSE {sym} x{int(pos)} @ {price:.2f}")

            append_csv(POS_CSV, {"ts_utc":dt.datetime.utcnow().isoformat(),"symbol":sym,"qty":broker.position_qty(sym),"book":book})
        except Exception:
            logging.exception(f"{book} cycle error {sym}")

# ---------------------- Sessions ----------------------
def within_eu_session()->bool: 
    return within_session(now_tz(SESSION_TZ_EU), SESSION_START_EU, SESSION_END_EU)
def within_us_session()->bool: 
    return within_session(now_tz(SESSION_TZ_US), SESSION_START_US, SESSION_END_US)
def crypto_open()->bool: return True
def fx_open()->bool: return True

# ---------------------- Reports ----------------------
def send_daily_report():
    try:
        today_eu = now_tz(SESSION_TZ_EU).date().isoformat()
        df_eq = pd.read_csv(EQ_CSV) if EQ_CSV.exists() else pd.DataFrame()
        df_tr = pd.read_csv(TRADES_CSV) if TRADES_CSV.exists() else pd.DataFrame()
        msg = ["📮 <b>Report giornaliero</b>"]
        if not df_eq.empty:
            df_eq["ts_utc"]=pd.to_datetime(df_eq["ts_utc"], utc=True)
            for book in ["EU","US","CRYPTO","FX"]:
                sub=df_eq[df_eq["book"]==book].copy()
                if sub.empty: continue
                sub["date_local"]=sub["ts_utc"].dt.tz_convert(SESSION_TZ_EU).dt.date.astype(str)
                tt=sub[sub["date_local"]==today_eu]["equity"]
                if not tt.empty:
                    open_eq=float(tt.iloc[0]); close_eq=float(tt.iloc[-1])
                    chg=(close_eq/open_eq -1)*100 if open_eq else 0
                    msg.append(f"{book}: {close_eq:,.2f} ({chg:+.2f}%)")
        if not df_tr.empty:
            df_tr["ts_utc"]=pd.to_datetime(df_tr["ts_utc"], utc=True)
            df_tr["date_local"]=df_tr["ts_utc"].dt.tz_convert(SESSION_TZ_EU).dt.date.astype(str)
            tday=df_tr[df_tr["date_local"]==today_eu]
            if not tday.empty:
                lines=tday.apply(lambda r: f"{r['book']} {r['side']} {r['symbol']} x{int(float(r['qty']))} @ {float(r['price']):.2f}", axis=1).tolist()
                msg.append("Operazioni di oggi:\n- " + "\n- ".join(lines))
        model=load_model()
        if model.get("updated_utc"): msg.append(f"Ultimo learning: {model['updated_utc']}")
        telegram_send("\n".join(msg))
    except Exception:
        logging.exception("Report error")

# ---------------------- Main ----------------------
def main():
    ensure_csv(EQ_CSV, ["ts_utc","book","equity"])
    ensure_csv(TRADES_CSV, ["ts_utc","symbol","side","qty","price","order_id","book"])
    ensure_csv(SIG_CSV, ["ts_utc","symbol","sig_daily","sig_intra","price","book"])
    ensure_csv(POS_CSV, ["ts_utc","symbol","qty","book"])
    ensure_csv(LEARN_CSV, ["ts_utc","symbol","book","strategy","metric","params"])

    # brokers
    us = AlpacaBroker() if (ALPACA_KEY and ALPACA_SECRET and TradingClient) else PaperSimBroker("US", starting_cash=10000.0)
    eu = PaperSimBroker("EU", starting_cash=10000.0)
    crypto = PaperSimBroker("CRYPTO", starting_cash=10000.0)
    fx = PaperSimBroker("FX", starting_cash=10000.0)

    brokers = {"US":us, "EU":eu, "CRYPTO":crypto, "FX":fx}
    symbols_by_book = {"US":SYMBOLS_US, "EU":SYMBOLS_EU, "CRYPTO":SYMBOLS_CRYPTO, "FX":SYMBOLS_FX}

    telegram_send(f"🤖 Avvio bot — US:{len(SYMBOLS_US)} EU:{len(SYMBOLS_EU)} CRYPTO:{len(SYMBOLS_CRYPTO)} FX:{len(SYMBOLS_FX)}")

    # learning iniziale
    nightly_learning(brokers, symbols_by_book)

    interval = max(1, LOOP_MINUTES)*60
    last_report_day = ""
    while True:
        try:
            trade_book("EU", SYMBOLS_EU, eu, within_eu_session())
            trade_book("US", SYMBOLS_US, us, within_us_session())
            trade_book("CRYPTO", SYMBOLS_CRYPTO, crypto, crypto_open())
            trade_book("FX", SYMBOLS_FX, fx, fx_open())

            nowEU = now_tz(SESSION_TZ_EU)

            # Report giornaliero alle 18:00 EU
            if nowEU.hour==18 and last_report_day != nowEU.date().isoformat():
                send_daily_report(); last_report_day = nowEU.date().isoformat()

            # Calibrazione ogni 12 ore: ~11:30 e ~23:30 EU
            if (nowEU.hour==11 and nowEU.minute>=30) or (nowEU.hour==23 and nowEU.minute>=30):
                nightly_learning(brokers, symbols_by_book)

            time.sleep(interval)
        except Exception:
            logging.exception("Main loop error")
            time.sleep(interval)

if __name__ == "__main__":
    main()
