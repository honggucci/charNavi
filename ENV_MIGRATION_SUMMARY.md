# Environment Variables Migration Summary

## Overview
Migrated all hardcoded configuration parameters from backtest scripts to environment variables management via `.env` file.

## Date
2025-12-31

## Changes Made

### 1. Created `.env` File
New file: `.env`
- Contains all configurable parameters for backtesting
- Default values match previous hardcoded settings
- Organized into logical sections:
  - CCXT API settings
  - Backtest configuration (symbol, timeframe, days)
  - Futures settings (leverage, position size)
  - Risk management (SL/TP multipliers)
  - Scoring & filtering
  - Timing parameters
  - Wyckoff box detection (Theta parameters)
  - Technical indicators
  - Costs (fees, slippage)
  - Probability model settings
  - Navigation Gate settings
  - Multi-timeframe settings
  - Execution settings
  - Spot engine settings

### 2. Updated Scripts

#### run_backtest_3months_1x.py (Futures V8 Engine)
**Changes:**
- Added `import os` and `from dotenv import load_dotenv`
- Added `load_dotenv()` call at startup
- Removed hardcoded function parameter defaults (timeframe, days)
- Replaced all hardcoded V8Config parameters with `os.getenv()` calls
- Parameters now read from environment with fallback defaults

**Key Parameters from ENV:**
- SYMBOL, TIMEFRAME, BACKTEST_DAYS
- All V8Config parameters (leverage, position_pct, sl_atr_mult, etc.)
- Navigation Gate parameters (use_navigation_gate, conf_min, edge_min, etc.)

#### run_spot_15m_3months.py (Spot BacktestEngine)
**Changes:**
- Added `import os` and `from dotenv import load_dotenv`
- Added `load_dotenv()` call at startup
- Removed hardcoded function parameter defaults
- Replaced BacktestCosts parameters with ENV (converted to bps)
- Replaced BacktestConfig parameters with ENV
- Added MTF timeframes parsing from ENV (comma-separated list)
- Added scalping_mode and use_phase_accumulation flags from ENV

**Key Parameters from ENV:**
- SYMBOL, TIMEFRAME, BACKTEST_DAYS
- MTF_TIMEFRAMES (parsed as comma-separated list)
- BacktestCosts: fee_bps, slippage_bps
- BacktestConfig: initial_equity, max_hold_bars, tp1_frac, etc.
- Execution flags: SCALPING_MODE, USE_PHASE_ACCUMULATION

#### run_spot_backtest_3months.py (Spot broker_sim)
**Changes:**
- Added `import os` and `from dotenv import load_dotenv`
- Added `load_dotenv()` call at startup
- **FIXED CRITICAL BUG**: Theta class was using non-existent parameters (tp_pct, sl_pct)
  - Corrected to use proper Theta parameters: pivot_lr, box_L, m_freeze, atr_len, x_atr, m_bw, N_reclaim, N_fill, F_min
- **FIXED CRITICAL BUG**: BacktestCosts was using non-existent parameters (maker_fee, taker_fee, slippage)
  - Corrected to use proper parameters: fee_bps, slippage_bps (converted from decimal to basis points)
- **FIXED CRITICAL BUG**: BacktestConfig was using non-existent parameter (initial_capital)
  - Corrected to use proper parameter: initial_equity
- Removed hardcoded function parameter defaults
- Added MTF timeframes parsing from ENV
- Added execution flags from ENV (scalping_mode, use_phase_accumulation, spot_mode)

**Key Parameters from ENV:**
- SYMBOL, TIMEFRAME, BACKTEST_DAYS
- Theta parameters: BOX_LOOKBACK, M_FREEZE, N_FILL, PIVOT_LR, ATR_LEN, X_ATR, M_BW, N_RECLAIM, F_MIN
- BacktestCosts: MAKER_FEE, SLIPPAGE (converted to bps)
- BacktestConfig: INITIAL_CAPITAL, MAX_HOLD_BARS, TP1_FRAC, etc.
- MTF_TIMEFRAMES, SCALPING_MODE, USE_PHASE_ACCUMULATION, SPOT_MODE

### 3. Updated .env File
Added missing Theta parameters:
- PIVOT_LR=3
- ATR_LEN=14
- X_ATR=2.0
- M_BW=0.02
- N_RECLAIM=8
- F_MIN=0.3

## Critical Bugs Fixed

### run_spot_backtest_3months.py
1. **Theta Parameters**: File was using `tp_pct` and `sl_pct` which don't exist in Theta dataclass
   - Fixed: Now uses correct parameters (pivot_lr, box_L, m_freeze, atr_len, x_atr, m_bw, N_reclaim, N_fill, F_min)

2. **BacktestCosts Parameters**: File was using `maker_fee`, `taker_fee`, `slippage`
   - Fixed: Now uses `fee_bps` and `slippage_bps` (converted from decimal to basis points)

3. **BacktestConfig Parameters**: File was using `initial_capital`, `position_pct`, `cooldown_bars`
   - Fixed: Now uses `initial_equity` and proper BacktestConfig parameters

## Testing

### Verified Working
- `run_backtest_3months_1x.py` - Successfully executed with ENV variables
  - Navigation Gate: 16/8641 bars allowed
  - 543 Fibonacci grid signals
  - 4 trades executed
  - Return: -0.01%, MDD: -0.04%

### To Be Tested
- `run_spot_15m_3months.py` - Uses BacktestEngine
- `run_spot_backtest_3months.py` - Uses broker_sim (bugs fixed, needs testing)

## Environment Variable Benefits

1. **Centralized Configuration**: All parameters in one place (`.env`)
2. **No Hardcoding**: Timeframes, symbols, and all parameters are configurable
3. **Easy Experimentation**: Change parameters without modifying code
4. **Version Control**: `.env` can be gitignored, `.env.example` tracks structure
5. **Deployment Flexibility**: Different environments can have different configs
6. **Type Safety**: Default values in `os.getenv()` provide fallbacks

## Usage

To modify backtest parameters:
1. Edit `.env` file
2. Change desired parameters
3. Run any backtest script - it will automatically use new values

Example:
```bash
# Change to 5m timeframe
TIMEFRAME=5m

# Change to ETH
SYMBOL=ETH-USDT

# Increase leverage
LEVERAGE=3.0
```

## Next Steps

1. Test `run_spot_15m_3months.py` with ENV variables
2. Test `run_spot_backtest_3months.py` with fixed bugs
3. Consider adding parameter validation
4. Consider adding `.env.example` file for documentation
5. Update other legacy backtest scripts in `src/wpcn/_01_crypto/` folders
