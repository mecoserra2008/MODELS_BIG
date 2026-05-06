"""
Generate all notebooks and LaTeX reports for the 5-country VAR spillover project.

Usage: python generate_project.py
"""

import json
import os

ALL_COUNTRIES = ["US", "EU", "CA", "BR", "UK"]

COUNTRY_NAMES = {
    "UK": "UK Gilt Curve",
    "US": "US Treasury Curve",
    "EU": "Euro Area AAA Curve",
    "CA": "Canadian Government Curve",
    "BR": "Brazilian DI Curve",
}

COUNTRY_FULL = {
    "UK": "United Kingdom",
    "US": "United States",
    "EU": "Euro Area",
    "CA": "Canada",
    "BR": "Brazil",
}

# Maturities per country for NS fitting
MATURITIES_CODE = {
    "UK": "np.array([1, 2, 3, 5, 10, 20], dtype=float)",
    "US": "np.array([1, 2, 3, 5, 10, 20], dtype=float)",
    "EU": "np.array([1, 2, 3, 5, 10, 20], dtype=float)",
    "CA": "np.array([2, 3, 5, 7, 10, 30], dtype=float)",
    "BR": "np.array([1, 3, 6, 12], dtype=float)",
}


def make_cell(source, cell_type="code"):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else source.split("\n"),
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def make_notebook(cells):
    return {
        "cells": [make_cell(s, t) for s, t in cells],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def save_notebook(nb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Fix source lines to have proper newlines
    for cell in nb["cells"]:
        if isinstance(cell["source"], list):
            cell["source"] = [
                line if line.endswith("\n") else line + "\n"
                for line in cell["source"]
            ]
            # Remove trailing newline from last line
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Created {path}")


# ===================================================================
# common/01_data_retrieval.ipynb
# ===================================================================
def gen_data_retrieval():
    cells = [
        (
            "# 01 — Yield Curve Data Retrieval (5 Countries)\n\n"
            "Download daily yield curve data:\n"
            "- **UK**: Bank of England nominal spot curve\n"
            "- **US**: FRED Treasury constant-maturity rates\n"
            "- **EU**: ECB AAA-rated government bond spot rates\n"
            "- **CA**: Bank of Canada benchmark bond yields\n"
            "- **BR**: ANBIMA/IPEA Brazilian DI pre-fixed curve",
            "markdown",
        ),
        (
            "import sys\n"
            "sys.path.insert(0, '..')\n\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "from src.data_retrieval import (\n"
            "    fetch_uk_yields, fetch_us_yields, fetch_eu_yields,\n"
            "    fetch_ca_yields, fetch_br_yields, align_yield_data,\n"
            ")\n\n"
            "sns.set_style('whitegrid')\n"
            "plt.rcParams['figure.figsize'] = (14, 5)\n\n"
            "START = '2005-01-01'\n"
            "END = '2026-04-30'",
            "code",
        ),
        ("## 1. Fetch All Countries", "markdown"),
        (
            "uk = fetch_uk_yields(start_date=START, end_date=END)\n"
            "print(f'UK: {uk.shape}')\n\n"
            "us = fetch_us_yields(start_date=START, end_date=END)\n"
            "print(f'US: {us.shape}')\n\n"
            "eu = fetch_eu_yields(start_date=START, end_date=END)\n"
            "print(f'EU: {eu.shape}')\n\n"
            "ca = fetch_ca_yields(start_date=START, end_date=END)\n"
            "print(f'CA: {ca.shape}')\n\n"
            "br = fetch_br_yields(start_date=START, end_date=END)\n"
            "print(f'BR: {br.shape}')",
            "code",
        ),
        ("## 2. Align", "markdown"),
        (
            "combined = align_yield_data(uk, us, eu, ca, br)\n"
            "print(f'Combined: {combined.shape}')\n"
            "print(f'Range: {combined.index.min()} to {combined.index.max()}')\n"
            "print(f'NaN: {combined.isna().sum().sum()}')\n"
            "combined.describe()",
            "code",
        ),
        ("## 3. Visualization — 10Y Yields", "markdown"),
        (
            "fig, ax = plt.subplots(figsize=(14, 6))\n"
            "for col, label in [('UK_10Y','UK'), ('US_10Y','US'), ('EU_10Y','EU'),\n"
            "                    ('CA_10Y','CA'), ('BR_12Y','BR (12Y)')]:\n"
            "    if col in combined.columns:\n"
            "        ax.plot(combined.index, combined[col], label=label, alpha=0.8, lw=0.8)\n"
            "ax.set_ylabel('Yield (%)')\n"
            "ax.set_title('Long-Term Government Bond Yields (5 Countries)')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()",
            "code",
        ),
        ("## 4. Save", "markdown"),
        (
            "combined.to_csv('../data/raw/aligned_yields_5c.csv')\n"
            "print('Saved aligned_yields_5c.csv')",
            "code",
        ),
    ]
    nb = make_notebook(cells)
    save_notebook(nb, "common/01_data_retrieval.ipynb")


# ===================================================================
# common/02_nelson_siegel.ipynb
# ===================================================================
def gen_nelson_siegel():
    mats_code = "\n".join(
        f"    '{cc}': {MATURITIES_CODE[cc]}," for cc in ALL_COUNTRIES
    )
    cells = [
        (
            "# 02 — Nelson-Siegel Decomposition (5 Countries)\n\n"
            "Fit Level, Slope, Curvature factors for UK, US, EU, CA, BR.",
            "markdown",
        ),
        (
            "import sys\n"
            "sys.path.insert(0, '..')\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "from src.nelson_siegel import (\n"
            "    ns_factor_loadings, fit_ns_ols, compute_rmse, select_lambda,\n"
            ")\n\n"
            "sns.set_style('whitegrid')\n"
            "plt.rcParams['figure.figsize'] = (14, 5)\n\n"
            f"MATURITIES = {{\n{mats_code}\n}}\n\n"
            "ALL_COUNTRIES = " + repr(ALL_COUNTRIES),
            "code",
        ),
        ("## 1. Load Data", "markdown"),
        (
            "combined = pd.read_csv('../data/raw/aligned_yields_5c.csv', index_col=0, parse_dates=True)\n"
            "print(f'Shape: {combined.shape}')",
            "code",
        ),
        ("## 2. Fit NS Factors Per Country", "markdown"),
        (
            "all_factors = {}\n"
            "lambdas = {}\n\n"
            "for cc in ALL_COUNTRIES:\n"
            "    cols = [c for c in combined.columns if c.startswith(f'{cc}_')]\n"
            "    yields = combined[cols]\n"
            "    mats = MATURITIES[cc]\n\n"
            "    lam_result = select_lambda(yields, mats)\n"
            "    best_lam = lam_result['best_lambda']\n"
            "    lambdas[cc] = best_lam\n\n"
            "    factors = fit_ns_ols(yields, mats, lam=best_lam)\n"
            "    rmse = compute_rmse(yields, factors, mats, lam=best_lam)\n\n"
            "    print(f'{cc}: lambda={best_lam:.4f}, RMSE={rmse.mean():.4f} bps, shape={factors.shape}')\n"
            "    all_factors[cc] = factors",
            "code",
        ),
        ("## 3. Factor Time Series", "markdown"),
        (
            "fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)\n"
            "for i, factor in enumerate(['Level', 'Slope', 'Curvature']):\n"
            "    ax = axes[i]\n"
            "    for cc in ALL_COUNTRIES:\n"
            "        ax.plot(all_factors[cc].index, all_factors[cc][factor],\n"
            "                label=cc, alpha=0.8, lw=0.7)\n"
            "    ax.set_ylabel(factor)\n"
            "    ax.legend(loc='upper right', fontsize=7)\n"
            "plt.suptitle('Nelson-Siegel Factors (5 Countries)', fontsize=12)\n"
            "plt.tight_layout()\n"
            "plt.show()",
            "code",
        ),
        ("## 4. Save", "markdown"),
        (
            "factor_dfs = [all_factors[cc].add_prefix(f'{cc}_') for cc in ALL_COUNTRIES]\n"
            "all_df = pd.concat(factor_dfs, axis=1)\n"
            "all_df.to_csv('../data/factors/ns_factors_5c.csv')\n"
            "print(f'Saved ns_factors_5c.csv: {all_df.shape}')\n"
            "all_df.describe().round(2)",
            "code",
        ),
    ]
    nb = make_notebook(cells)
    save_notebook(nb, "common/02_nelson_siegel.ipynb")


# ===================================================================
# Per-country VAR notebook
# ===================================================================
def gen_var_notebook(target):
    others = [c for c in ALL_COUNTRIES if c != target]
    cholesky_str = " -> ".join(others + [target])

    cells = [
        (
            f"# VAR Spillover Analysis: {COUNTRY_FULL[target]} as Target\n\n"
            f"15-variable VAR on NS factors (5 countries). "
            f"Cholesky ordering: {cholesky_str}\n\n"
            f"Target: **{target}** ({COUNTRY_NAMES[target]})",
            "markdown",
        ),
        (
            "import sys\n"
            "sys.path.insert(0, '..')\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from src.var_analysis import (\n"
            "    build_var_order, prepare_var_data, adf_tests,\n"
            "    select_var_lag_detailed, estimate_var, compute_irfs,\n"
            "    compute_fevd, fevd_summary, granger_causality_tests,\n"
            "    var_diagnostics, FACTOR_NAMES,\n"
            ")\n\n"
            "sns.set_style('whitegrid')\n"
            "plt.rcParams['figure.figsize'] = (14, 5)\n\n"
            f"TARGET = '{target}'\n"
            f"ALL_COUNTRIES = {repr(ALL_COUNTRIES)}\n"
            f"OTHERS = [c for c in ALL_COUNTRIES if c != TARGET]",
            "code",
        ),
        ("## 1. Load and Prepare Data", "markdown"),
        (
            "factors_raw = pd.read_csv('../data/factors/ns_factors_5c.csv', index_col=0, parse_dates=True)\n\n"
            "factors_dict = {}\n"
            "for cc in ALL_COUNTRIES:\n"
            "    cols = [c for c in factors_raw.columns if c.startswith(f'{cc}_')]\n"
            "    factors_dict[cc] = factors_raw[cols].rename(columns=lambda x: x.replace(f'{cc}_', ''))\n\n"
            "var_data = prepare_var_data(factors_dict, target=TARGET, countries=ALL_COUNTRIES)\n"
            "print(f'VAR data: {var_data.shape}')\n"
            "print(f'Date range: {var_data.index.min()} to {var_data.index.max()}')\n"
            "print(f'Columns: {list(var_data.columns)}')",
            "code",
        ),
        ("## 2. Stationarity Tests", "markdown"),
        (
            "adf_levels = adf_tests(var_data)\n"
            "print('ADF Tests (Levels):')\n"
            "print(adf_levels[['ADF_stat', 'p_value', 'stationary_5%']].to_string())",
            "code",
        ),
        ("## 3. Lag Selection & Estimation", "markdown"),
        (
            "lag_result = select_var_lag_detailed(var_data, max_lags=4)\n"
            "print('Optimal lag by criterion:')\n"
            "for criterion, lag in lag_result['recommended'].items():\n"
            "    print(f'  {criterion}: {lag}')\n\n"
            "LAGS = lag_result['recommended']['BIC']\n"
            "print(f'\\nUsing lag: {LAGS}')\n\n"
            "var_results = estimate_var(var_data, lags=LAGS)\n"
            "print(f'VAR({LAGS}) estimated. Obs: {var_results.nobs}')",
            "code",
        ),
        ("## 4. Diagnostics", "markdown"),
        (
            "diag = var_diagnostics(var_results)\n"
            "print('Durbin-Watson:')\n"
            "for var, dw in diag['durbin_watson'].items():\n"
            "    if var.startswith(TARGET):\n"
            "        print(f'  {var}: {dw:.3f}')\n"
            "print(f'\\nPortmanteau: stat={diag[\"portmanteau\"][\"statistic\"]:.1f}, '\n"
            "      f'p={diag[\"portmanteau\"][\"p_value\"]:.4f}')",
            "code",
        ),
        (
            "## 5. Impulse Response Functions\n\n"
            f"Response of {target} factors to all foreign shocks.",
            "markdown",
        ),
        (
            "HORIZON = 40\n"
            "irf = var_results.irf(HORIZON)\n"
            "irf_lower, irf_upper = irf.errband_mc(orth=True, repl=500, seed=42)\n\n"
            "var_names = list(var_results.names)\n"
            "target_responses = [f'{TARGET}_{f}' for f in FACTOR_NAMES]\n"
            "foreign_shocks = [f'{c}_{f}' for c in OTHERS for f in FACTOR_NAMES]\n\n"
            "n_foreign = len(OTHERS)\n"
            "fig, axes = plt.subplots(3, n_foreign * 3, figsize=(5 * n_foreign, 8), sharex=True)\n\n"
            "for i, response in enumerate(target_responses):\n"
            "    for j, impulse in enumerate(foreign_shocks):\n"
            "        ax = axes[i, j]\n"
            "        imp_idx = var_names.index(impulse)\n"
            "        resp_idx = var_names.index(response)\n"
            "        irf_vals = irf.orth_irfs[:, resp_idx, imp_idx]\n"
            "        ax.plot(range(HORIZON + 1), irf_vals, 'b-', lw=1.2)\n"
            "        ax.axhline(0, color='grey', lw=0.5, ls='--')\n"
            "        ax.fill_between(range(HORIZON + 1),\n"
            "                        irf_lower[:, resp_idx, imp_idx],\n"
            "                        irf_upper[:, resp_idx, imp_idx],\n"
            "                        alpha=0.15, color='blue')\n"
            "        if i == 0:\n"
            "            ax.set_title(impulse.replace('_', ' '), fontsize=7)\n"
            "        if j == 0:\n"
            "            ax.set_ylabel(response.replace('_', ' '), fontsize=8)\n"
            "        ax.tick_params(labelsize=5)\n\n"
            f"plt.suptitle('IRFs: {target} Responses to Foreign Shocks (95% CI)', fontsize=11)\n"
            "plt.tight_layout()\n"
            f"plt.savefig('fig_irf.png', dpi=200, bbox_inches='tight')\n"
            "plt.show()",
            "code",
        ),
        ("## 6. Forecast Error Variance Decomposition", "markdown"),
        (
            "fevd = compute_fevd(var_results, periods=52)\n"
            "horizons = [1, 4, 12, 26, 52]\n\n"
            "for response in target_responses:\n"
            "    print(f'\\nFEVD for {response}:')\n"
            "    summary = fevd_summary(fevd, response, horizons=horizons, var_names=var_names)\n"
            "    # Aggregate by country\n"
            "    agg = {}\n"
            "    for cc in ALL_COUNTRIES:\n"
            "        cc_cols = [c for c in summary.columns if c.startswith(f'{cc}_')]\n"
            "        agg[cc] = summary[cc_cols].sum(axis=1)\n"
            "    print(pd.DataFrame(agg).round(1).to_string())",
            "code",
        ),
        (
            "# FEVD bar charts\n"
            "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
            "colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']\n\n"
            "for idx, response in enumerate(target_responses):\n"
            "    ax = axes[idx]\n"
            "    summary = fevd_summary(fevd, response, horizons=horizons, var_names=var_names)\n"
            "    agg = pd.DataFrame()\n"
            "    for i, cc in enumerate(ALL_COUNTRIES):\n"
            "        cc_cols = [c for c in summary.columns if c.startswith(f'{cc}_')]\n"
            "        label = f'{cc} (own)' if cc == TARGET else cc\n"
            "        agg[label] = summary[cc_cols].sum(axis=1)\n"
            "    agg.plot(kind='bar', stacked=True, ax=ax, color=colors)\n"
            "    ax.set_title(response.replace('_', ' '))\n"
            "    ax.set_xlabel('Horizon (weeks)')\n"
            "    ax.set_ylabel('% Variance')\n"
            "    ax.set_ylim(0, 105)\n"
            "    ax.legend(loc='lower right', fontsize=6)\n"
            "    ax.set_xticklabels(horizons, rotation=0)\n\n"
            f"plt.suptitle('FEVD — {target} Factors', fontsize=12)\n"
            "plt.tight_layout()\n"
            f"plt.savefig('fig_fevd.png', dpi=200, bbox_inches='tight')\n"
            "plt.show()",
            "code",
        ),
        ("## 7. Granger Causality", "markdown"),
        (
            "for cc in OTHERS:\n"
            "    causing = [f'{cc}_{f}' for f in FACTOR_NAMES]\n"
            "    gc = granger_causality_tests(var_results, causing=causing, target=TARGET)\n"
            "    print(f'\\n{cc} -> {TARGET}:')\n"
            "    print(gc[['caused', 'F_stat', 'p_value', 'significant']].to_string(index=False))",
            "code",
        ),
        ("## 8. Factor Time Series", "markdown"),
        (
            "fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)\n"
            "for i, factor in enumerate(FACTOR_NAMES):\n"
            "    ax = axes[i]\n"
            "    for cc in ALL_COUNTRIES:\n"
            "        ax.plot(factors_dict[cc].index, factors_dict[cc][factor],\n"
            "                label=cc, alpha=0.8, lw=0.7)\n"
            "    ax.set_ylabel(factor)\n"
            "    ax.legend(loc='upper right', fontsize=7)\n"
            f"plt.suptitle('NS Factors — All Countries (target: {target})', fontsize=11)\n"
            "plt.tight_layout()\n"
            f"plt.savefig('fig_factors.png', dpi=200, bbox_inches='tight')\n"
            "plt.show()",
            "code",
        ),
    ]
    nb = make_notebook(cells)
    save_notebook(nb, f"{target}/03_var_spillover.ipynb")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating common notebooks...")
    gen_data_retrieval()
    gen_nelson_siegel()

    print("\nGenerating per-country VAR notebooks...")
    for cc in ALL_COUNTRIES:
        gen_var_notebook(cc)

    print("\nDone.")
