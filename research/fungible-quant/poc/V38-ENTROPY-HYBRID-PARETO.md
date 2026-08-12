# PoC v38: Entropy-aware hybrid Pareto

## Finding

The Pareto is stuck at 7.0 bpw because the jump from K2+K5trsc (7 bpw) to
K2+6LM (8 bpw) costs 1.0 bit with no intermediate tier. Need to include
ALL tiers (even dominated ones) as upgrade targets for fractional mixing.

## Tier summary with entropy

| Tier | raw bpw | entropy bpw | MSE |
|------|---------|-------------|-----|
| K2 | 2 | 2.000 | 1.061e-01 |
| K3 | 3 | 3.000 | 2.718e-02 |
| K4 | 4 | 4.000 | 7.286e-03 |
| K2+K3trsc | 5 | 5.000 | 1.892e-03 |
| K2+K4trsc | 6 | 6.000 | 5.276e-04 |
| K2+4LM | 6 | 5.526 | 1.199e-03 (dominated) |
| K2+K5trsc | 7 | 7.000 | 1.726e-04 |
| K2+6LM | 8 | 7.441 | 8.767e-05 |
| K3+6LM | 9 | 8.436 | 2.629e-05 |
| K4+6LM | 10 | 9.421 | 9.613e-06 |

The entropy savings on LM tiers (K2+6LM: 7.441 vs 8.0) gives room for
upgrading more tiles, but the tier list needs all tiers for mixing.

## Entropy benefit for LM tiers

K2+6LM: raw 8.0 → entropy 7.441 (saves 0.559 bpw)
K3+6LM: raw 9.0 → entropy 8.436 (saves 0.564 bpw)
K4+6LM: raw 10.0 → entropy 9.421 (saves 0.579 bpw)

These savings allow the entropy-aware Pareto to reach lower MSE at the same
target bitrate by upgrading more tiles to LM tiers.
