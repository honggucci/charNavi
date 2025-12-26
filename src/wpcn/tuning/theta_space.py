from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any
import numpy as np
import yaml

from wpcn.core.types import Theta

class ThetaSpace:
    def __init__(self, space: Dict[str, Any], seed: int = 42):
        self.space = space
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_yaml(cls, path: str, seed: int = 42) -> "ThetaSpace":
        with open(path, "r", encoding="utf-8") as f:
            space = yaml.safe_load(f)
        return cls(space, seed=seed)

    def sample(self) -> Theta:
        s = self.space
        pivot_lr = int(self.rng.choice(s["pivot_lr"]))
        atr_len = int(self.rng.choice(s["atr_len"]))
        box_L = int(self.rng.choice(s["box_L"]))
        m_freeze = int(self.rng.choice(s["m_freeze"]))
        x_atr = float(self.rng.uniform(s["x_atr"][0], s["x_atr"][1]))
        m_bw = float(self.rng.uniform(s["m_bw"][0], s["m_bw"][1]))
        N_reclaim = int(self.rng.choice(s["N_reclaim"]))
        N_fill = int(self.rng.choice(s["N_fill"]))
        F_min = float(self.rng.uniform(s["F_min"][0], s["F_min"][1]))
        return Theta(
            pivot_lr=pivot_lr, box_L=box_L, m_freeze=m_freeze, atr_len=atr_len,
            x_atr=x_atr, m_bw=m_bw, N_reclaim=N_reclaim, N_fill=N_fill, F_min=F_min
        )
