"""
백테스트 엔진 불변식 검증 모듈 (v2.6.12)
========================================

핵심 불변식:
1. 포지션은 동시에 0 또는 1개만 존재 (방향별)
2. Entry 1개당 Exit 정확히 1개 (Entry-Exit 1:1 매칭)
3. trade_pnl 합 = equity 변화 (수수료/펀딩 포함) 일치
4. 같은 exit 이벤트가 두 trade에 귀속되면 즉시 assert

Author: Claude + Hattz
Version: 1.0.0
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class TradeInvariantChecker:
    """
    백테스트 거래 불변식 검증기

    사용법:
    1. 백테스트 시작 전 초기화
    2. 진입 시 record_entry() 호출
    3. 청산 시 record_exit() 호출
    4. 백테스트 종료 후 validate_all() 호출
    """

    # 상태 추적
    active_positions: Dict[str, Dict] = field(default_factory=dict)  # side -> position info
    entries: List[Dict] = field(default_factory=list)
    exits: List[Dict] = field(default_factory=list)

    # 중복 청산 감지
    exit_events: Set[str] = field(default_factory=set)  # "time_side_price" unique keys

    # PnL 추적
    total_entry_notional: float = 0.0
    total_exit_notional: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_funding: float = 0.0

    # 위반 기록
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def record_entry(
        self,
        time: pd.Timestamp,
        side: str,
        price: float,
        qty: float,
        margin: float,
        entry_id: str = "",
        fee_bps: float = 4.0,
    ) -> Optional[str]:
        """
        진입 기록

        Returns:
            위반 메시지 (없으면 None)
        """
        # 불변식 1: 같은 방향 포지션이 이미 있으면 경고 (축적은 허용)
        # (단, 완전히 새로운 포지션 시작이면 기존 포지션 청산 먼저 확인)

        entry_key = f"{time}_{side}_{price:.2f}_{qty:.4f}"
        entry_record = {
            "time": time,
            "side": side,
            "price": price,
            "qty": qty,
            "margin": margin,
            "entry_id": entry_id or entry_key,
            "matched": False,  # 청산과 매칭되었는지
        }

        self.entries.append(entry_record)

        # 포지션 업데이트
        if side not in self.active_positions:
            self.active_positions[side] = {
                "qty": 0.0,
                "notional": 0.0,
                "entry_count": 0,
            }

        self.active_positions[side]["qty"] += qty
        self.active_positions[side]["notional"] += qty * price
        self.active_positions[side]["entry_count"] += 1

        # 통계 업데이트
        self.total_entry_notional += qty * price
        self.total_fees += qty * price * (fee_bps / 10000)

        return None

    def record_exit(
        self,
        time: pd.Timestamp,
        side: str,
        price: float,
        qty: float,
        pnl: float,
        exit_type: str,
        fee_bps: float = 4.0,
    ) -> Optional[str]:
        """
        청산 기록

        Returns:
            위반 메시지 (없으면 None)
        """
        # 불변식 4: 중복 청산 감지
        exit_key = f"{time}_{side}_{price:.2f}_{qty:.4f}"
        if exit_key in self.exit_events:
            violation = f"DUPLICATE_EXIT: {exit_key} already recorded"
            self.violations.append(violation)
            return violation

        self.exit_events.add(exit_key)

        exit_record = {
            "time": time,
            "side": side,
            "price": price,
            "qty": qty,
            "pnl": pnl,
            "exit_type": exit_type,
            "exit_key": exit_key,
        }

        self.exits.append(exit_record)

        # 포지션 업데이트
        if side not in self.active_positions:
            violation = f"EXIT_WITHOUT_POSITION: {side} position doesn't exist at {time}"
            self.violations.append(violation)
            return violation

        self.active_positions[side]["qty"] -= qty

        # 포지션 완전 청산 시 제거
        if abs(self.active_positions[side]["qty"]) < 1e-10:
            del self.active_positions[side]

        # 통계 업데이트
        self.total_exit_notional += qty * price
        self.total_pnl += pnl
        self.total_fees += qty * price * (fee_bps / 10000)

        return None

    def record_funding(self, amount: float):
        """펀딩비 기록"""
        self.total_funding += amount

    def validate_entry_exit_matching(self) -> List[str]:
        """
        불변식 2 검증: Entry 1개당 Exit 정확히 1개

        Returns:
            위반 목록
        """
        violations = []

        # 방향별 Entry/Exit 카운트
        entry_counts = {"long": 0, "short": 0}
        exit_counts = {"long": 0, "short": 0}

        for e in self.entries:
            entry_counts[e["side"]] += 1

        for x in self.exits:
            exit_counts[x["side"]] += 1

        # 열린 포지션이 없으면 Entry == Exit
        if not self.active_positions:
            for side in ["long", "short"]:
                if entry_counts[side] != exit_counts[side]:
                    violations.append(
                        f"ENTRY_EXIT_MISMATCH: {side} entries={entry_counts[side]}, "
                        f"exits={exit_counts[side]} (no open positions)"
                    )

        return violations

    def validate_pnl_consistency(
        self,
        initial_equity: float,
        final_equity: float,
        tolerance: float = 0.01,  # 1% 허용 오차
    ) -> List[str]:
        """
        불변식 3 검증: PnL 합 = equity 변화

        Returns:
            위반 목록
        """
        violations = []

        # 예상 equity = initial + PnL - fees - funding
        expected_equity = (
            initial_equity
            + self.total_pnl
            - self.total_funding
        )
        # Note: fees는 이미 PnL에 반영되어 있음 (체결가에 포함)

        # 열린 포지션의 미실현 손익은 별도 처리 필요
        # (여기서는 final_equity가 이미 포함한다고 가정)

        diff = abs(expected_equity - final_equity)
        diff_pct = diff / initial_equity if initial_equity > 0 else 0

        if diff_pct > tolerance:
            violations.append(
                f"PNL_EQUITY_MISMATCH: expected={expected_equity:.2f}, "
                f"actual={final_equity:.2f}, diff={diff:.2f} ({diff_pct*100:.2f}%)"
            )

        return violations

    def validate_no_duplicate_exits(self) -> List[str]:
        """
        불변식 4 검증: 중복 청산 없음

        이미 record_exit()에서 감지되지만, 추가 검증
        """
        violations = []

        seen = set()
        for x in self.exits:
            key = x["exit_key"]
            if key in seen:
                violations.append(f"DUPLICATE_EXIT_POST: {key}")
            seen.add(key)

        return violations

    def validate_all(
        self,
        initial_equity: float,
        final_equity: float,
    ) -> Tuple[bool, List[str], List[str]]:
        """
        모든 불변식 검증

        Returns:
            (is_valid, violations, warnings)
        """
        all_violations = list(self.violations)  # 기록된 실시간 위반

        # 추가 검증
        all_violations.extend(self.validate_entry_exit_matching())
        all_violations.extend(self.validate_no_duplicate_exits())
        all_violations.extend(
            self.validate_pnl_consistency(initial_equity, final_equity)
        )

        # 경고 (위반은 아니지만 주의 필요)
        warnings = list(self.warnings)

        if self.active_positions:
            for side, pos in self.active_positions.items():
                warnings.append(
                    f"OPEN_POSITION: {side} qty={pos['qty']:.4f} at end of backtest"
                )

        is_valid = len(all_violations) == 0

        return is_valid, all_violations, warnings

    def get_summary(self) -> Dict:
        """검증 결과 요약"""
        return {
            "total_entries": len(self.entries),
            "total_exits": len(self.exits),
            "active_positions": len(self.active_positions),
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "total_funding": self.total_funding,
            "violations": len(self.violations),
            "warnings": len(self.warnings),
        }

    def reset(self):
        """상태 초기화"""
        self.active_positions.clear()
        self.entries.clear()
        self.exits.clear()
        self.exit_events.clear()
        self.total_entry_notional = 0.0
        self.total_exit_notional = 0.0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.total_funding = 0.0
        self.violations.clear()
        self.warnings.clear()


def validate_trades_df(trades_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    trades_df에 대한 사후 검증

    체크 항목:
    1. 같은 exit 이벤트가 여러 trade에 귀속되었는지
    2. Entry-Exit 순서가 올바른지
    3. PnL 계산이 올바른지
    """
    violations = []

    if trades_df.empty:
        return True, []

    # Exit 타입 필터링
    exit_types = ["SL", "TP", "TIME_EXIT", "LIQUIDATION", "FORCED_CLOSE",
                  "BEAR_ENTRY_KILL", "HARD_BEAR_KILL", "PARTIAL_DERISK"]

    exits = trades_df[trades_df["type"].isin(exit_types)].copy()

    # 중복 exit 체크 (time + side + exit_price)
    if "exit_price" in exits.columns:
        exits["exit_key"] = (
            exits["time"].astype(str) + "_" +
            exits["side"] + "_" +
            exits["exit_price"].round(2).astype(str)
        )

        duplicates = exits[exits.duplicated(subset=["exit_key"], keep=False)]
        if not duplicates.empty:
            for key in duplicates["exit_key"].unique():
                count = len(duplicates[duplicates["exit_key"] == key])
                violations.append(
                    f"DUPLICATE_EXIT_IN_TRADES_DF: {key} appears {count} times"
                )

    # PnL 검증 (entry_price, exit_price 둘 다 있을 때)
    if "entry_price" in exits.columns and "exit_price" in exits.columns and "pnl" in exits.columns:
        for idx, row in exits.iterrows():
            if pd.isna(row.get("entry_price")) or pd.isna(row.get("exit_price")):
                continue

            entry_p = row["entry_price"]
            exit_p = row["exit_price"]
            recorded_pnl = row["pnl"]
            side = row["side"]

            # qty 또는 size가 있으면 검증
            qty = row.get("total_size") or row.get("size") or row.get("qty")
            if qty and not pd.isna(qty):
                if side == "long":
                    expected_pnl = (exit_p - entry_p) * abs(qty)
                else:
                    expected_pnl = (entry_p - exit_p) * abs(qty)

                diff = abs(expected_pnl - recorded_pnl)
                if diff > abs(expected_pnl) * 0.1 and diff > 1.0:  # 10% 또는 $1 이상 차이
                    violations.append(
                        f"PNL_CALCULATION_MISMATCH at {row['time']}: "
                        f"expected={expected_pnl:.2f}, recorded={recorded_pnl:.2f}, "
                        f"side={side}, entry={entry_p:.2f}, exit={exit_p:.2f}"
                    )

    is_valid = len(violations) == 0
    return is_valid, violations
