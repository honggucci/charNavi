from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_backtest_results(run_dir: Path, df: pd.DataFrame, equity_df: pd.DataFrame, trades_df: pd.DataFrame, nav_df: pd.DataFrame):
    """Generate charts for backtest results."""
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Main price chart with targets and trades
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Price chart
    ax1.plot(df.index, df['close'], label='Close Price', color='black', linewidth=1)
    ax1.set_title('Price Chart with Targets and Trades')
    ax1.set_ylabel('Price')
    
    # Plot support levels
    for i, row in nav_df.iterrows():
        if row['support_levels']:
            for level, prob in zip(row['support_levels'], row['support_probs']):
                if prob > 0.5:  # Only high probability
                    ax1.axhline(y=level, color='green', alpha=0.3, linewidth=0.5)
    
    # Plot resistance levels
    for i, row in nav_df.iterrows():
        if row['resist_levels']:
            for level, prob in zip(row['resist_levels'], row['resist_probs']):
                if prob > 0.5:
                    ax1.axhline(y=level, color='red', alpha=0.3, linewidth=0.5)
    
    # Plot trades
    for _, trade in trades_df.iterrows():
        if trade['side'] == 'long':
            ax1.scatter(trade.name, trade['entry_price'], marker='^', color='green', s=100, label='Buy' if _ == 0 else "")
        elif trade['side'] == 'short':
            ax1.scatter(trade.name, trade['entry_price'], marker='v', color='red', s=100, label='Sell' if _ == 0 else "")
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI chart
    ax2.plot(nav_df.index, nav_df['rsi'], label='RSI', color='blue')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
    ax2.set_title('RSI')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Stoch RSI chart
    ax3.plot(nav_df.index, nav_df['stoch_k'], label='Stoch K', color='orange')
    ax3.plot(nav_df.index, nav_df['stoch_d'], label='Stoch D', color='purple')
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.7)
    ax3.axhline(y=0.2, color='green', linestyle='--', alpha=0.7)
    ax3.set_title('Stochastic RSI')
    ax3.set_ylabel('Stoch')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(charts_dir / "price_rsi_stoch_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Equity curve
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    ax.plot(equity_df.index, equity_df['equity'], label='Equity', color='blue', linewidth=2)
    ax.set_title('Equity Curve')
    ax.set_ylabel('Equity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(charts_dir / "equity_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Regime and edge score
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    # Regime
    regime_colors = {'CHAOS': 'red', 'RANGE': 'orange', 'TREND_UP': 'green', 'TREND_DOWN': 'blue'}
    for regime in nav_df['regime'].unique():
        mask = nav_df['regime'] == regime
        ax1.fill_between(nav_df.index, 0, 1, where=mask, color=regime_colors.get(regime, 'gray'), alpha=0.3, label=regime)
    ax1.set_title('Market Regime')
    ax1.set_yticks([])
    ax1.legend()
    
    # Edge score
    ax2.plot(nav_df.index, nav_df['edge_score'], label='Edge Score', color='purple')
    ax2.set_title('Edge Score')
    ax2.set_ylabel('Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(charts_dir / "regime_edge_chart.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Charts saved to {charts_dir}")