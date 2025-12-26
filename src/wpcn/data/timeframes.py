from __future__ import annotations

def timeframe_to_pandas_rule(tf: str) -> str:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return f"{int(tf[:-1])}min"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}h"
    if tf.endswith("d"):
        return f"{int(tf[:-1])}D"
    if tf.endswith("w"):
        return f"{int(tf[:-1])}W"
    raise ValueError(f"Unsupported timeframe: {tf}")
