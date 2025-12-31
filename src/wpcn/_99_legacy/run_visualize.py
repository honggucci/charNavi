import os
import sys
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

def plot_targets(run_id: str, save_path: str = None):
    run_dir = Path("runs") / run_id
    
    # Load data
    nav_df = pd.read_parquet(run_dir / "nav.parquet")
    candles_df = pd.read_parquet(run_dir / "candles.parquet")
    
    # Merge on time
    df = candles_df.join(nav_df[['support_levels', 'support_probs', 'resist_levels', 'resist_probs']], how='left')
    
    # Plot last 200 bars for visibility
    df_plot = df.tail(200)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price
    ax.plot(df_plot.index, df_plot['close'], label='Close Price', color='blue', linewidth=1)
    
    # Plot support levels
    for idx in df_plot.index:
        levels = df_plot.loc[idx, 'support_levels']
        probs = df_plot.loc[idx, 'support_probs']
        if isinstance(levels, list) and levels:
            for i, level in enumerate(levels):
                prob = probs[i] if i < len(probs) else 0
                alpha = min(1.0, prob * 2)  # Scale alpha
                ax.axhline(y=level, color='green', linestyle='--', alpha=alpha, linewidth=0.5)
                if prob > 0.5:  # Only label high prob
                    ax.text(idx, level, f'S:{level:.0f}({prob:.2f})', fontsize=6, color='green', ha='left', va='bottom')
    
    # Plot resistance levels
    for idx in df_plot.index:
        levels = df_plot.loc[idx, 'resist_levels']
        probs = df_plot.loc[idx, 'resist_probs']
        if isinstance(levels, list) and levels:
            for i, level in enumerate(levels):
                prob = probs[i] if i < len(probs) else 0
                alpha = min(1.0, prob * 2)
                ax.axhline(y=level, color='red', linestyle='--', alpha=alpha, linewidth=0.5)
                if prob > 0.5:
                    ax.text(idx, level, f'R:{level:.0f}({prob:.2f})', fontsize=6, color='red', ha='left', va='top')
    
    ax.set_title(f'WPCN Targets - {run_id}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize WPCN targets")
    parser.add_argument("--run-id", required=True, help="Run ID (e.g., 2025-12-26_17-44-12_binanceusdm_BTC-USDT_1h)")
    parser.add_argument("--save", help="Save chart to file (e.g., chart.png)")
    
    args = parser.parse_args()
    plot_targets(args.run_id, args.save)

if __name__ == "__main__":
    main()