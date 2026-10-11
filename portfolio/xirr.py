"""Actual/365 XIRR with bracketed root finding and explicit unavailable cases."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass
class XirrResult:
    rate: float | None
    reason: str = ""


def xirr(flows):
    grouped = {}
    for day, amount in flows:
        grouped[day] = grouped.get(day, 0.0) + float(amount)
    grouped = sorted((d, v) for d, v in grouped.items() if v != 0)
    if len(grouped) < 2:
        return XirrResult(None, "Need cash flows on at least two distinct dates.")
    values = np.array([v for _, v in grouped])
    if not (np.any(values < 0) and np.any(values > 0)):
        return XirrResult(
            None,
            "Cash flows need both positive and negative amounts; a complete loss has no finite XIRR.",
        )
    years = np.array([(d - grouped[0][0]).days / 365 for d, _ in grouped])

    # Solve in log(1+r) space; rescale exponential terms to prevent overflow.
    def npv(log_rate):
        exponents = -years * log_rate
        exponents -= max(exponents)
        return float(np.dot(values / max(abs(values)), np.exp(exponents)))

    grid = np.unique(np.concatenate((np.linspace(-25, math.log(11), 4000), [0.0])))
    roots = []
    for a, b in zip(grid[:-1], grid[1:]):
        fa, fb = npv(a), npv(b)
        if abs(fa) < 1e-14:
            roots.append(math.expm1(a))
        elif fa * fb < 0:
            roots.append(math.expm1(brentq(npv, a, b, xtol=1e-13)))
    if not roots:
        return XirrResult(None, "No stable root found within -100% to 1,000%.")
    rate = min(roots, key=abs)
    return XirrResult(
        rate,
        "Multiple roots detected; using the root nearest zero." if len(roots) > 1 else "",
    )
