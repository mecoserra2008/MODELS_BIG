"""
Generate per-country LaTeX reports for the 5-country VAR spillover project.
Run after all notebooks have been executed and figures generated.

Usage: python generate_reports.py
"""

import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.var_analysis import (
    prepare_var_data, estimate_var, compute_fevd, fevd_summary,
    granger_causality_tests, FACTOR_NAMES, var_diagnostics
)

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

CURVE_SHORT = {
    "UK": "gilts",
    "US": "Treasuries",
    "EU": "euro-area AAA bonds",
    "CA": "Government of Canada bonds",
    "BR": "DI pre-fixed curve",
}

DATA_SOURCE = {
    "UK": "Bank of England IADB (nominal spot curve)",
    "US": "FRED (constant-maturity Treasury rates)",
    "EU": "ECB Statistical Data Warehouse (AAA-rated)",
    "CA": "Bank of Canada Valet API (benchmark bonds)",
    "BR": "IPEA/ANBIMA (DI pre-fixed yield curve)",
}

# Country-specific macro context for page 4
MACRO_CONTEXT = {
    "UK": r"""
\subsection*{Macro \& Fiscal Backdrop}
\begin{itemize}[nosep, leftmargin=1.2em]
    \item \textbf{Inflation:} CPI at 3.3\% (March 2026), up from 3.0\% in February. Energy-driven (Iran conflict oil shock, Ofgem cap $\uparrow$20\% from July). BoE projects 3.0--3.5\% through Q3 2026.
    \item \textbf{Growth:} GDP grew 0.5\% in three months to Feb 2026; OBR forecasts full-year 2026 at 1.1\%.
    \item \textbf{BoE MPC (30 Apr):} Held at 3.75\%, 8--1 vote (1 member voted for a hike). Markets price $\sim$3 hikes in 2026.
    \item \textbf{Gilt issuance:} DMO 2026--27 remit at \pounds252.1bn, above consensus. Heavy tilt to short/medium maturities.
    \item \textbf{Debt:} PSND at 94.3\% of GDP, forecast to rise to 96.1\%. OBR flags unsustainable trajectory.
\end{itemize}

\subsection*{Recent Market Action}
\begin{itemize}[nosep, leftmargin=1.2em]
    \item 10Y gilt: 5.06\% (30 Apr), highest since July 2008.
    \item 30Y gilt: 5.74\% (5 May), highest since 1998.
    \item Sell-off accelerated after MPC hawkish dissent and municipal election uncertainty.
\end{itemize}

\subsection*{Key Drivers Ahead}
\begin{enumerate}[nosep, leftmargin=1.4em]
    \item \textbf{UST moves} dominate UK rate variance ($\sim$29\%). Any further UST sell-off mechanically widens gilt yields.
    \item \textbf{Energy/Iran:} Re-escalation pushes CPI above 3.5\%, forcing BoE hikes.
    \item \textbf{Gilt supply:} \pounds252bn remit is front-loaded. Poor bid-to-cover signals absorption stress.
    \item \textbf{BoE rate path:} 8--1 split suggests June hike is live.
    \item \textbf{Fiscal credibility:} NIESR flags gilt market ``derailment'' risk.
\end{enumerate}
""",
    "US": r"""
\subsection*{Macro \& Fiscal Backdrop}
\begin{itemize}[nosep, leftmargin=1.2em]
    \item \textbf{Inflation:} Core PCE at 2.6\% (Mar 2026). Energy pass-through from Iran conflict adding $\sim$0.5pp to headline.
    \item \textbf{Growth:} Q1 2026 GDP at 1.8\% annualised, showing resilience despite tighter financial conditions.
    \item \textbf{Fed:} Held at 4.25--4.50\%. Markets no longer pricing cuts in 2026; $\sim$2 hikes now priced.
    \item \textbf{Issuance:} US Treasury running $\sim$\$2tn annual deficit financing. Refunding mix shifting toward bills.
    \item \textbf{Debt ceiling:} Ongoing uncertainty adds term premium pressure.
\end{itemize}

\subsection*{Key Drivers Ahead}
\begin{enumerate}[nosep, leftmargin=1.4em]
    \item \textbf{US is the global anchor:} As the most exogenous rate in our VAR, US shocks propagate to all other curves. Own-variance share ($\sim$36\%) is relatively low, reflecting deep global integration.
    \item \textbf{Canada co-movement:} CA explains 28\% of US Level variance---reflecting NAFTA integration and BoC/Fed policy synchronisation.
    \item \textbf{Fiscal trajectory:} CBO projections of rising debt/GDP create persistent term premium.
    \item \textbf{Iran ceasefire durability:} Key swing factor for energy pass-through and Fed reaction function.
\end{enumerate}
""",
    "EU": r"""
\subsection*{Macro \& Fiscal Backdrop}
\begin{itemize}[nosep, leftmargin=1.2em]
    \item \textbf{Inflation:} Headline HICP at 2.4\% (Apr 2026). Core remains sticky at 2.7\%.
    \item \textbf{Growth:} Euro area GDP grew 0.3\% q/q in Q1 2026. Germany flat, periphery outperforming.
    \item \textbf{ECB:} Held deposit rate at 2.50\%. Forward guidance cautious on cuts given energy uncertainty.
    \item \textbf{Issuance:} Joint EU debt issuance (NextGenEU) continues. National programmes vary widely.
    \item \textbf{Fragmentation:} BTP-Bund spread stable at $\sim$130bp; no TPI activation concerns.
\end{itemize}

\subsection*{Key Drivers Ahead}
\begin{enumerate}[nosep, leftmargin=1.4em]
    \item \textbf{High domestic determination:} EU curve is the most self-driven (65\% own-variance for Level). ECB policy is the primary driver.
    \item \textbf{US and CA spillovers:} Combined $\sim$26\% of Level variance. Global risk-free rate channel.
    \item \textbf{Defence spending:} EU joint debt for defence could increase supply pressure.
    \item \textbf{ECB rate path:} Next decision 5 June 2026.
\end{enumerate}
""",
    "CA": r"""
\subsection*{Macro \& Fiscal Backdrop}
\begin{itemize}[nosep, leftmargin=1.2em]
    \item \textbf{Inflation:} CPI at 2.8\% (Mar 2026). Shelter costs remain the largest contributor.
    \item \textbf{Growth:} GDP grew 0.4\% in Q4 2025. Labour market softening; unemployment at 6.7\%.
    \item \textbf{BoC:} Policy rate at 3.25\%, on hold. Markets pricing 1--2 cuts later in 2026.
    \item \textbf{Housing:} Canadian housing vulnerability adds sensitivity to rate movements.
    \item \textbf{Trade:} US tariff uncertainty weighing on business investment.
\end{itemize}

\subsection*{Key Drivers Ahead}
\begin{enumerate}[nosep, leftmargin=1.4em]
    \item \textbf{US dominance:} US explains $\sim$29\% of CA Level and $\sim$47--49\% of Slope and Curvature. The BoC--Fed policy link is the single largest external driver.
    \item \textbf{Brazil connection:} Surprisingly, BR explains $\sim$15\% of CA Level---likely via commodity/EM risk channels.
    \item \textbf{BoC vs Fed divergence:} CAD depreciation constrains BoC's ability to cut independently.
    \item \textbf{Housing correction risk:} Could force BoC to cut aggressively, decoupling from Fed.
\end{enumerate}
""",
    "BR": r"""
\subsection*{Macro \& Fiscal Backdrop}
\begin{itemize}[nosep, leftmargin=1.2em]
    \item \textbf{Inflation:} IPCA at 5.5\% (Mar 2026), above target ceiling (4.5\%). Food and energy pressures persistent.
    \item \textbf{Growth:} GDP grew 2.1\% in 2025. Fiscal stimulus supporting consumption.
    \item \textbf{COPOM:} Selic at 14.75\%, hiked 50bp in March. Forward guidance signals further tightening.
    \item \textbf{Fiscal:} Primary deficit at 0.8\% of GDP. New fiscal framework credibility under pressure.
    \item \textbf{FX:} BRL/USD at 5.80, depreciation adds to imported inflation.
\end{itemize}

\subsection*{Key Drivers Ahead}
\begin{enumerate}[nosep, leftmargin=1.4em]
    \item \textbf{Highest own-determination:} BR curve is the most insular (73\% own-variance across all factors). Domestic monetary policy and fiscal credibility dominate.
    \item \textbf{Limited foreign spillover:} No single foreign block explains more than 10\% of BR variance. Brazil's high carry and idiosyncratic risk dominate.
    \item \textbf{EU as largest foreign driver} ($\sim$10\%): Likely via risk sentiment and European investor flows into EM.
    \item \textbf{Fiscal credibility:} The binding constraint. Any spending cap revision would widen DI rates sharply.
    \item \textbf{Selic path:} Market expects terminal rate of 15.25\%. Any overshoot reprices the entire curve.
\end{enumerate}
""",
}


def generate_report(target):
    others = [c for c in ALL_COUNTRIES if c != target]
    cholesky_str = " $\\rightarrow$ ".join(others + [target])
    cholesky_list = ", ".join(
        [f"{c}$_{{\\text{{Level}}}}$, {c}$_{{\\text{{Slope}}}}$, {c}$_{{\\text{{Curv}}}}$"
         for c in others + [target]]
    )

    # Compute FEVD and Granger results
    factors_raw = pd.read_csv("data/factors/ns_factors_5c.csv", index_col=0, parse_dates=True)
    factors_dict = {}
    for cc in ALL_COUNTRIES:
        cols = [c for c in factors_raw.columns if c.startswith(f"{cc}_")]
        factors_dict[cc] = factors_raw[cols].rename(columns=lambda x: x.replace(f"{cc}_", ""))

    var_data = prepare_var_data(factors_dict, target=target, countries=ALL_COUNTRIES)
    var_results = estimate_var(var_data, lags=1)
    fevd = compute_fevd(var_results, periods=52)
    var_names = list(var_results.names)

    # FEVD table data
    fevd_rows = []
    for f in FACTOR_NAMES:
        resp = f"{target}_{f}"
        s = fevd_summary(fevd, resp, horizons=[52], var_names=var_names)
        shares = {}
        for cc in ALL_COUNTRIES:
            cc_cols = [c for c in s.columns if c.startswith(f"{cc}_")]
            shares[cc] = s[cc_cols].sum(axis=1).iloc[0]
        fevd_rows.append(shares)

    # Granger table
    gc_lines = []
    for cc in others:
        causing = [f"{cc}_{f}" for f in FACTOR_NAMES]
        gc = granger_causality_tests(var_results, causing=causing, target=target)
        for _, row in gc.iterrows():
            sig = "\\textbf{Sig.}" if row["significant"] else "Not sig."
            caused_short = row["caused"].replace(f"{target}_", "")
            gc_lines.append(
                f"    {cc} & {caused_short} & {row['F_stat']:.2f} & {row['p_value']:.3f} & {sig} \\\\"
            )

    # FEVD table LaTeX
    fevd_header = " & ".join([f"\\textbf{{{cc}}}" for cc in ALL_COUNTRIES])
    fevd_table_rows = []
    for i, f in enumerate(FACTOR_NAMES):
        vals = " & ".join([f"{fevd_rows[i][cc]:.1f}" for cc in ALL_COUNTRIES])
        fevd_table_rows.append(f"    {target} {f} & {vals} \\\\")

    tex = rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{helvet}}
\renewcommand{{\familydefault}}{{\sfdefault}}
\usepackage[margin=2.2cm, top=2cm, bottom=2cm]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage{{tikz}}
\usepackage{{enumitem}}
\usepackage{{tabularx}}

\definecolor{{bigorange}}{{RGB}}{{255,130,0}}
\definecolor{{biggrey}}{{RGB}}{{60,60,60}}

\hypersetup{{colorlinks=true, linkcolor=bigorange, urlcolor=bigorange}}

\pagestyle{{fancy}}
\fancyhf{{}}
\renewcommand{{\headrulewidth}}{{0.5pt}}
\renewcommand{{\headrule}}{{\hbox to\headwidth{{\color{{bigorange}}\leaders\hrule height \headrulewidth\hfill}}}}
\fancyhead[L]{{\small\color{{biggrey}} Banco BIG $|$ Rates Research}}
\fancyhead[R]{{\small\color{{biggrey}} May 2026}}
\fancyfoot[C]{{\small\color{{biggrey}}\thepage}}

\begin{{document}}

% COVER
\thispagestyle{{empty}}
\begin{{tikzpicture}}[remember picture, overlay]
    \fill[bigorange] (current page.north west) rectangle ([yshift=-9cm]current page.north east);
    \fill[white, opacity=0.15] ([yshift=-3cm]current page.north west) -- ([yshift=-7cm]current page.north west) -- ([yshift=-5cm]current page.north east) -- ([yshift=-1cm]current page.north east) -- cycle;
    \fill[bigorange] ([yshift=1.5cm]current page.south west) rectangle (current page.south east);
\end{{tikzpicture}}

\vspace*{{1cm}}
\begin{{center}}
    \includegraphics[width=4.5cm]{{../Logo-BIG.png}}
\end{{center}}

\vspace{{3.5cm}}

\begin{{center}}
    {{\Huge\bfseries\color{{biggrey}} Interest Rate Spillovers}}\\[0.4cm]
    {{\Huge\bfseries\color{{biggrey}} to the {COUNTRY_NAMES[target]}}}\\[1.0cm]
    {{\Large\color{{biggrey}} 5-Country VAR: US, EU, CA, BR, UK}}\\[0.3cm]
    {{\Large\color{{biggrey}} Nelson-Siegel Level, Slope, Curvature}}\\[2.0cm]
    {{\large\color{{biggrey}} Rates Research $|$ 5 May 2026}}
\end{{center}}

\vfill
\begin{{tikzpicture}}[remember picture, overlay]
    \node[anchor=south, yshift=2.0cm] at (current page.south) {{
        \color{{white}}\small\bfseries Banco BIG $|$ Institutional Research
    }};
\end{{tikzpicture}}

\newpage

% PAGE 2: METHODOLOGY & RESULTS
\section*{{Methodology: 5-Country VAR on Nelson-Siegel Factors}}

We decompose five government yield curves (US, EU, CA, BR, UK) into Level, Slope, and Curvature via Nelson-Siegel ($\lambda=0.10$). A 15-variable VAR(1) is estimated on weekly data (2005--2026, $N={var_results.nobs}$ obs).

\smallskip
\noindent\textbf{{Cholesky ordering:}} {cholesky_str}

\smallskip
\noindent Rationale: {target} is placed last as the ``receiver'' of interest. US is most exogenous (global reserve currency).

\subsection*{{Granger Causality Tests ($\alpha=5\%$)}}

\begin{{table}}[H]
\centering\small
\begin{{tabular}}{{llccc}}
\toprule
\textbf{{Source}} & \textbf{{{target} Factor}} & \textbf{{F-stat}} & \textbf{{p-value}} & \textbf{{Sig.?}} \\
\midrule
{chr(10).join(gc_lines)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{fig_factors.png}}
    \vspace{{-0.3cm}}
    \caption{{\small Nelson-Siegel factors for all 5 countries (2005--2026). {target} highlighted as the target curve.}}
\end{{figure}}

\newpage

% PAGE 3: IRFs AND FEVD
\section*{{Shock Propagation to {COUNTRY_FULL[target]}}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{fig_irf.png}}
    \vspace{{-0.3cm}}
    \caption{{\small Orthogonalised IRFs: {target} factor responses to foreign shocks over 40 weeks. Blue shading = 95\% bootstrap CI.}}
\end{{figure}}

\subsection*{{FEVD at 52-Week Horizon (\% of variance)}}

\begin{{table}}[H]
\centering\small
\begin{{tabular}}{{l{' c' * len(ALL_COUNTRIES)}}}
\toprule
 & {fevd_header} \\
\midrule
{chr(10).join(fevd_table_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
    \centering
    \includegraphics[width=\textwidth]{{fig_fevd.png}}
    \vspace{{-0.3cm}}
    \caption{{\small FEVD: share of {target} factor forecast error variance by source country at horizons 1--52 weeks.}}
\end{{figure}}

\newpage

% PAGE 4: MARKET CONTEXT
\section*{{{COUNTRY_FULL[target]}: Market Context (May 2026)}}

{MACRO_CONTEXT[target]}

\vspace{{0.3cm}}
\noindent\rule{{\textwidth}}{{0.4pt}}
{{\footnotesize\color{{biggrey}} \textbf{{Data:}} {', '.join(f'{cc}: {DATA_SOURCE[cc]}' for cc in ALL_COUNTRIES)}. Model: VAR(1) on weekly NS factors, sample 2005--2026 ({var_results.nobs} obs). This document is for informational purposes only and does not constitute investment advice.}}

\end{{document}}
"""
    path = f"{target}/var_spillover_report.tex"
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"  Created {path}")


if __name__ == "__main__":
    for cc in ALL_COUNTRIES:
        generate_report(cc)
    print("Done.")
