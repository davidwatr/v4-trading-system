from production_v18 import fetch_data, get_market_regime, calc_atr, UNIVERSE, SECTOR_ETFS, extract_features, dynamic_triple_barriers, kelly_criterion, select_features, MODEL_FILE
from alpaca_executor import AlpacaExecutor
from datetime import datetime
import pickle
import numpy as np

print("="*60)
print("  PRE-MARKET ANALYSIS")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

# ============================================
# 1. MARKET REGIME
# ============================================
print("\n📊 MARKET REGIME")
print("-"*40)

result = fetch_data('SPY', days=200)
spy_closes = result[0]
spy_volumes = result[1]
spy_highs = result[2]
spy_lows = result[3]

regime, penalty, big_cap_max, small_cap_max, kelly_mult = get_market_regime(spy_closes, spy_highs, spy_lows)
atr = calc_atr(spy_highs, spy_lows, spy_closes)

# Calculate MAs manually for reporting
ma5 = sum(spy_closes[-5:])/5
ma10 = sum(spy_closes[-10:])/10
ma20 = sum(spy_closes[-20:])/20
ma50 = sum(spy_closes[-50:])/50

print(f"  SPY Price:    ${spy_closes[-1]:>8.2f}")
print(f"  MA5:          ${ma5:>8.2f}")
print(f"  MA10:         ${ma10:>8.2f}")
print(f"  MA20:         ${ma20:>8.2f}")
print(f"  MA50:         ${ma50:>8.2f}")
print(f"  ATR(14):      ${atr:>8.2f}")
print(f"  Data Points:  {len(spy_closes)}")
print(f"  Regime:       {regime.upper()}")
print(f"  Max Positions: {max_pos}")
print(f"  Kelly Mult:   {kelly_mult}x")
print(f"  Penalty:      {penalty}")

# ============================================
# 2. ACCOUNT & POSITIONS
# ============================================
print("\n💰 ACCOUNT & POSITIONS")
print("-"*40)

try:
    executor = AlpacaExecutor()
    account = executor.get_account()
    positions = executor.get_positions()
    
    equity = float(account.portfolio_value)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    
    print(f"  Equity:        ${equity:>12,.2f}")
    print(f"  Cash:          ${cash:>12,.2f}")
    print(f"  Buying Power:  ${buying_power:>12,.2f}")
    print(f"  Open Positions: {len(positions)}")
    
    total_pnl = 0
    for p in positions:
        pnl = float(p.unrealized_pl)
        total_pnl += pnl
        pnl_pct = float(p.unrealized_plpc) * 100
        qty = float(p.qty)
        avg_entry = float(p.avg_entry_price)
        current = float(p.current_price)
        emoji = "🟢" if pnl >= 0 else "🔴"
        print(f"  {emoji} {p.symbol:6s} {qty:>4.0f} shares @ ${avg_entry:.2f} → ${current:.2f} | P&L: ${pnl:>+8.2f} ({pnl_pct:>+5.2f}%)")
    
    print(f"\n  Total Unrealized P&L: ${total_pnl:>+12,.2f}")
    
except Exception as e:
    print(f"  ❌ Account error: {e}")
    import traceback
    traceback.print_exc()
    positions = []
    equity = 0

# ============================================
# 3. SIGNAL GENERATION
# ============================================
print("\n🎯 TOP SIGNALS")
print("-"*40)

if MODEL_FILE.exists():
    try:
        with open(MODEL_FILE, 'rb') as f:
            xgb_saved = pickle.load(f)
        xgb_model = xgb_saved['model']
        xgb_features = xgb_saved['feature_names']
        selected_features, _ = select_features(xgb_model, xgb_features, threshold=0.02)
        
        results = []
        for symbol in UNIVERSE[:15]:  # Top 15 for speed
            try:
                closes, volumes, highs, lows = fetch_data(symbol, days=200)
                if not closes or len(closes) < 50:
                    continue
                sector_etf = SECTOR_ETFS.get(symbol, "SPY")
                sector_closes, _, _, _ = fetch_data(sector_etf, days=200)
                if not sector_closes:
                    sector_closes = spy_closes
                features = extract_features(closes, volumes, highs, lows, symbol, spy_closes, sector_closes)
                if features:
                    results.append({'symbol': symbol, 'features': features})
            except Exception as e:
                continue
        
        results.sort(key=lambda x: x['features']['residual_return'], reverse=True)
        
        for i, r in enumerate(results[:5]):
            f = r['features']
            X = [[f[name] for name in selected_features]]
            raw_prob = xgb_model.predict_proba(X)[0][1]
            penalized_prob = raw_prob * penalty
            
            upper_mult, lower_mult, max_hold = dynamic_triple_barriers(f['atr'], closes, f['iv_rank'])
            payout_ratio = upper_mult / lower_mult
            base_kelly = kelly_criterion(penalized_prob, payout_ratio, max_risk=0.1)
            kelly_size = min(base_kelly * kelly_mult, 0.25)
            
            print(f"  {i+1}. {r['symbol']:6s} ${f['current_price']:.2f} | Win: {penalized_prob:.1%} | Kelly: {kelly_size:.1%} | Residual: {f['residual_return']:+.4f}")
    except Exception as e:
        print(f"  ❌ Signal error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  ⚠️ Model file not found")

print("\n" + "="*60)
print("  Analysis complete")
print("="*60)