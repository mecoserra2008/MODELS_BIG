"""Positioning <-> yields study + US10Y ARIMA/ARIMAX forecaster.

Turns the asserted "crowded positioning precedes higher yields" claim into a rigorous,
reproducible study (conditional base rates with CIs, sub-period robustness, Granger,
forward-return regressions), and adds a dynamic (walk-forward) US10Y ARIMA/ARIMAX forecaster.

Data is 100% free/open (FRED via the positioning yields adapter; CFTC TFF via the positioning
CFTC adapter). Nothing here needs Bloomberg.
"""
