"""Oil <-> US rates conditional-sensitivity study.

Why don't rates move much with oil? Core answer under test: nominal 10Y = real + breakeven;
oil passes into breakevens (inflation channel) while the real leg prices the growth/risk
consequence - rates look insensitive exactly when the two offset. Low sensitivity is a
REGIME (found endogenously via Markov switching and profiled on observable conditions),
and the unconditional beta is skewed by a handful of identifiable tail dates.

Free FRED data only. House style mirrors rates_study/.
"""
