# WPCN - 와이코프 암호화폐 트레이딩 봇

> **v2.6.14** | 마지막 업데이트: 2026-01-13
>
> **"안 들어가는 날이 많아져야 계좌가 산다"**

## 이게 뭔데?

100년 전 월스트리트 전설 **Richard Wyckoff**의 매집/분산 이론을 암호화폐에 적용한 자동 트레이딩 시스템입니다.

**핵심 아이디어:**
- 세력이 매집하는 구간(박스권)을 찾아내고
- Spring(속임수 하락) 신호가 나오면 진입
- 멀티 타임프레임(15분~1주) 점수로 확신도 체크
- **HMM 기반 리스크 필터**로 잘못된 진입 차단

## 왜 이게 좋은데?

| 일반 봇 | WPCN |
|--------|------|
| 신호 나오면 무조건 진입 | 확신 없으면 안 들어감 |
| 과거 데이터에 과적합 | Walk-Forward로 검증 |
| 단일 타임프레임 | 5개 타임프레임 종합 분석 |
| 시장가 진입 | **지정가 진입** (슬리피지 최소화) |

## 뭘 할 수 있어?

- **현물 백테스트**: BTC 롱 포지션, 스윙 트레이딩
- **선물 백테스트**: 롱/숏 양방향, 레버리지 지원
- **15분봉 지정가 백테스트** (v2.6.14): ATR 기반 SL/TP, HMM 필터
- **파라미터 최적화**: Walk-Forward + A/B Testing

## 최근 결과 (v1.2 Relaxed Cooldown)

```
Total PnL: +$991 (v1 대비 +$32,473 개선)
2023: +$71 (72 trades)
2024: +$920 (143 trades)
SL Hit Rate: 46-50% (v1의 94%에서 대폭 감소)
```

## 시작하기

```bash
pip install -e .

# 15분봉 지정가 백테스트 (권장)
python run_dlite_wfo_v2.py

# 현물 백테스트
python -m wpcn._01_crypto.001_binance.001_spot.002_sub.run_spot_backtest_mtf
```

---

더 자세한 내용은 [SYSTEM_ARCHITECTURE.md](conversations/SYSTEM_ARCHITECTURE.md) 참고
