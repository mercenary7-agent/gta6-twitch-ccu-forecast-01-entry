"""gta6-twitch-ccu-forecast-01 entry: comparables-regression forecaster (Python 3.13, stdlib only).

Reads each case in /input, forecasts hours 2..24 of the release day as five quantiles
(5/25/50/75/95) of the Twitch category's concurrent viewers, writes /output/<id>.out.

Model (v4): for each comparable launch in the case, the release-day ratio
r_i(h) = viewers(hour h) / viewers(hour 1). For every hour h we fit, across the
comparable launches, a weighted linear regression of ln r_i(h) on ln v1_i (the
launch's own hour-1 print), where each comparable is weighted by a Gaussian kernel
on its log-hour-1 distance from the case's hour-1 print p1 (launches of similar
size decay more alike). The regression is strongly regularized (LAM) and the kernel
is wide (KERNEL), which shrinks the slope toward the pooled median — empirically the
most robust center on the 4 practice cases. Quantiles come from the empirical
distribution of the regression residuals, shrunk by SPREAD, with the spread itself
shrinking with hour (SPREAD_GROW < 0: late-hour ratios are less dispersed than the
pooled residual spread suggests).
"""

import json
import math
import os
import sys
import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Residual quantiles are shrunk to this spread: 38 launches over-disperse vs one launch.
SPREAD = 0.65
# Per-hour spread multiplier: sp(h) = SPREAD * (1 + SPREAD_GROW * (h-1)/23).
SPREAD_GROW = -0.85
# The regression median is blended with the plain empirical median (robustness).
BLEND = 0.0
# Gaussian kernel width (in log hour-1 viewers) for weighting similar-size launches.
KERNEL = 10.0
# Ridge regularization strength (large => slope shrinks toward pooled median).
LAM = 0.5
FLOOR = 0.01


def hour_key(day_start_iso, h):
    """The 'from' timestamp of release-day hour h (h=1 covers [day_start, day_start+1h))."""
    ds = datetime.datetime.fromisoformat(day_start_iso.replace("Z", "+00:00"))
    return (ds + datetime.timedelta(hours=h - 1)).strftime("%Y-%m-%dT%H:00:00Z")


def quantile(sorted_vals, tau):
    if not sorted_vals:
        return 0.0
    pos = tau * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def solve_ridge_w(X, Y, W, lam=LAM):
    """Weighted least squares with intercept for X rows -> Y with per-row weights W."""
    k = len(X[0]) + 1
    A = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for xi, yi, w in zip(X, Y, W):
        xf = [1.0] + xi
        for a in range(k):
            for b in range(k):
                A[a][b] += w * xf[a] * xf[b]
            rhs[a] += w * xf[a] * yi
    for a in range(k):
        A[a][a] += lam
    M = [A[i] + [rhs[i]] for i in range(k)]
    for i in range(k):
        piv = max(range(i, k), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-12:
            continue
        for r in range(k):
            if r != i:
                f = M[r][i] / M[i][i]
                for c in range(i, k + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(k)]


def comparable_ratios(case):
    """For each comparable launch: {h: viewers(h)/viewers(hour1), h=2..24} and its hour-1 print."""
    out = []
    for comp in case.get("comparables") or []:
        by_from = {p.get("from"): p.get("viewers") for p in (comp.get("hourly") or [])}
        ds = comp.get("day_start")
        if not ds:
            continue
        vals = {}
        ok = True
        for h in range(1, 25):
            v = by_from.get(hour_key(ds, h))
            if v is None:
                ok = False
                break
            vals[h] = v
        if not ok or vals[1] <= 0:
            continue
        v1 = max(1.0, vals[1])
        out.append({"v1": v1, "r": {h: vals[h] / v1 for h in range(2, 25)}})
    return out


def forecast_case(case):
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first.get("viewers"))
    lp1 = math.log(p1)
    comps = comparable_ratios(case)
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        used = [c for c in comps if h in c["r"]]
        X = [[math.log(c["v1"])] for c in used]
        Y = [math.log(max(1e-6, c["r"][h])) for c in used]
        if not X:
            med = 0.0
            resid = []
        else:
            W = [math.exp(-((math.log(c["v1"]) - lp1) ** 2) / (2.0 * KERNEL * KERNEL)) for c in used]
            beta = solve_ridge_w(X, Y, W)
            med = (1.0 - BLEND) * (beta[0] + beta[1] * lp1) + BLEND * quantile(sorted(Y), 0.5)
            resid = sorted(yi - (beta[0] + beta[1] * xi[0]) for xi, yi in zip(X, Y))
        sp = SPREAD * (1.0 + SPREAD_GROW * (h - 1) / 23.0)
        qs = [math.exp(med + quantile(resid, tau) * sp) for tau in LEVELS]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        rows.append({"hour": h, "q": [max(FLOOR, p1 * x) for x in qs]})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
