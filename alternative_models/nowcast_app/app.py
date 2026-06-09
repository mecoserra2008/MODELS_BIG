"""
Nowcasting App — 6-tab Streamlit interface.

Tabs:
  1. Data & Target       — select target, fetch data, view series
  2. Feature Lab         — pick regressors, configure lags (Almon / Beta / Direct)
  3. PCA & t-SNE         — dimensionality reduction + visualisation
  4. VIF & Collinearity  — multicollinearity analysis + iterative elimination
  5. Estimation          — multi-method model fit + results
  6. Diagnostics         — full assumption test suite

Launch: streamlit run alternative_models/nowcast_app/app.py
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup — must be done before any local imports
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).parent.resolve()
_ALT_DIR = _APP_DIR.parent.resolve()
_PROJ_DIR = _ALT_DIR.parent.resolve()

for p in [str(_APP_DIR), str(_ALT_DIR), str(_PROJ_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(str(_ALT_DIR))  # required so data_macro.py resolves cache paths correctly

from config import (
    TARGETS, TARGET_GROUPS, REGRESSORS, REGRESSOR_GROUPS,
    TARGET_DEFAULT_REGRESSORS, DEFAULT_LAG_CONFIG, TRAIN_END,
)
from feature_builder import (
    build_full_feature_matrix, config_hash,
    almon_pdl_transform, build_almon_weights_from_delta,
)
from estimators import fit_model, ESTIMATORS
from diagnostics import run_all_diagnostics, make_qq_plot, make_residual_fitted_plot, make_residual_time_plot, make_cusum_plot
from pca_viz import run_pca, make_eigenvalue_table, make_scree_plot, make_loadings_heatmap, make_biplot
from tsne_viz import run_tsne, make_tsne_plot
from vif_utils import compute_vif, iterative_vif_elimination, make_vif_bar_chart, make_correlation_heatmap
from plots import (
    make_target_series_plot, make_feature_matrix_heatmap,
    make_coefficient_plot, make_fitted_vs_actual_plot,
    make_almon_weight_profile, make_feature_importance_bar,
    make_probability_chart,
)

from data_loader import fetch_all_macro_safe

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nowcasting App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Nowcasting Model Builder")
st.caption(
    "Strictly lagged features only. Supports Almon PDL, Beta MIDAS, direct lags, "
    "PCA, t-SNE, VIF analysis, and multiple estimation methods."
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_SS_DEFAULTS = {
    "raw_data":             None,
    "target_key":           "US_CPI_YOY",
    "y":                    None,
    "X_scaled":             None,
    "scaler":               None,
    "build_config_hash":    None,
    "build_meta":           None,
    "vif_df":               None,
    "vif_clean_cols":       None,
    "pca_object":           None,
    "X_pca":                None,
    "pca_loadings":         None,
    "selected_pcs":         [1, 2, 3],
    "tsne_embedding":       None,
    "tsne_index":           None,
    "fit_result":           None,
    "diagnostics_result":   None,
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar — global settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Global Settings")

    # FRED API key — prefer env var, allow override via UI
    _env_key = os.environ.get("FRED_API_KEY", "")
    fred_api_key = st.text_input(
        "FRED API Key",
        value=_env_key,
        type="password",
        help="Required to fetch FRED data. Get a free key at fred.stlouisfed.org/docs/api/api_key.html",
    )
    if fred_api_key:
        os.environ["FRED_API_KEY"] = fred_api_key

    st.markdown("---")

    train_end_global = st.text_input(
        "Train / test split date",
        value=TRAIN_END,
        help="Features are scaled on data up to this date only.",
    )

    as_of_date_input = st.date_input(
        "As-of date (knowledge cutoff)",
        value=date.today(),
        help="Simulate real-time: regressors are cut off to data available by this date.",
    )
    as_of_date = pd.Timestamp(as_of_date_input)

    st.markdown("---")
    st.markdown("**Data range**")
    start_date = st.text_input("Start date", "2003-01-01")
    end_date   = st.text_input("End date",   "2026-06-01")

# ---------------------------------------------------------------------------
# Stale-feature warning helper
# ---------------------------------------------------------------------------
def _stale_warning(current_hash: str):
    stored = st.session_state.get("build_config_hash")
    if stored is not None and stored != current_hash:
        st.warning(
            "Configuration has changed since the last feature build. "
            "Go to **Feature Lab** and click **Build Feature Matrix** to refresh."
        )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
(
    tab_data,
    tab_features,
    tab_pca,
    tab_vif,
    tab_estimation,
    tab_diagnostics,
) = st.tabs([
    "Data & Target",
    "Feature Lab",
    "PCA & t-SNE",
    "VIF & Collinearity",
    "Estimation",
    "Diagnostics",
])

# ===========================================================================
# TAB 1 — Data & Target
# ===========================================================================
with tab_data:
    st.header("Data & Target Setup")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Group target selection
        group = st.selectbox("Country / Region", list(TARGET_GROUPS.keys()), key="sb_group")
        targets_in_group = TARGET_GROUPS[group]
        target_key = st.selectbox("Target variable", targets_in_group, key="sb_target")
        st.session_state.target_key = target_key

        _, _, transform, freq, pub_lag = TARGETS[target_key]
        st.info(
            f"Transform: **{transform}** | Frequency: **{freq}** | "
            f"Publication lag: **~{pub_lag} business days**"
        )

        if st.button("Fetch / Refresh Data", key="btn_fetch"):
            if not fred_api_key:
                st.error("Enter your FRED API Key in the sidebar before fetching data.")
                st.stop()
            with st.spinner("Fetching macro data (uses cached files if available)..."):
                try:
                    data = fetch_all_macro_safe(
                        start=start_date,
                        end=end_date,
                        force=False,
                        api_key=fred_api_key or None,
                    )
                    st.session_state.raw_data = data
                    # Reset downstream state
                    for k in ["X_scaled", "y", "build_config_hash", "build_meta",
                              "vif_df", "vif_clean_cols", "pca_object", "X_pca",
                              "pca_loadings", "tsne_embedding", "fit_result",
                              "diagnostics_result"]:
                        st.session_state[k] = None
                    ns_empty = data["ns_factors"].empty
                    yields_empty = data["yields"].empty
                    if ns_empty or yields_empty:
                        missing = []
                        if ns_empty:
                            missing.append("Nelson-Siegel factors")
                        if yields_empty:
                            missing.append("aligned yields")
                        st.warning(
                            f"Optional files not found: **{', '.join(missing)}**. "
                            "NS yield-curve regressors (US_Level, US_Slope, etc.) will be "
                            "unavailable. Run the main VAR pipeline to generate them."
                        )
                    else:
                        st.success("Data loaded.")
                except Exception as e:
                    st.error(f"Data fetch failed: {e}")

    with col2:
        if st.session_state.raw_data is not None:
            try:
                from feature_builder import get_target_series
                y_preview = get_target_series(
                    st.session_state.raw_data, target_key, as_of_date
                )
                st.plotly_chart(make_target_series_plot(y_preview, target_key),
                                use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Observations", len(y_preview))
                c2.metric("Mean",  f"{y_preview.mean():.3f}")
                c3.metric("Std",   f"{y_preview.std():.3f}")
                c4.metric("Last",  f"{y_preview.iloc[-1]:.3f}")
                st.caption(
                    f"Date range: {y_preview.index.min().date()} → "
                    f"{y_preview.index.max().date()}"
                )
            except Exception as e:
                st.warning(f"Cannot preview target series: {e}")
        else:
            st.info("Click **Fetch / Refresh Data** to load macro data.")

# ===========================================================================
# TAB 2 — Feature Lab
# ===========================================================================
with tab_features:
    st.header("Feature Lab")

    if st.session_state.raw_data is None:
        st.warning("Fetch data first in **Data & Target** tab.")
        st.stop()

    target_key = st.session_state.target_key
    default_regs = TARGET_DEFAULT_REGRESSORS.get(target_key, list(REGRESSORS.keys())[:6])

    # Regressor multiselect with grouping hint
    st.subheader("Select Regressors")
    all_regs = list(REGRESSORS.keys())
    selected_regs = st.multiselect(
        "Regressors (strictly lagged — no contemporaneous values)",
        all_regs,
        default=default_regs,
        key="ms_regs",
        help="All features enter the model as lags only.",
    )

    # Per-regressor lag configuration
    lag_configs: dict = {}
    if selected_regs:
        st.subheader("Lag Configuration")
        st.caption(
            "Configure how each regressor's lags are parameterised. "
            "**Almon PDL**: polynomial constraints on K lag coefficients → linear OLS. "
            "**Beta MIDAS**: beta-polynomial weighted aggregate. "
            "**Direct lags**: k separate lag columns."
        )

        for reg in selected_regs:
            _, obs_freq, default_K, default_pub = REGRESSORS[reg]
            default_cfg = DEFAULT_LAG_CONFIG[obs_freq].copy()

            with st.expander(f"{reg}  ({obs_freq}-frequency, default K={default_K})", expanded=False):
                c1, c2, c3 = st.columns(3)
                weight_type = c1.selectbox(
                    "Weight type",
                    ["Almon PDL", "Beta MIDAS", "Direct lags"],
                    index=["Almon PDL","Beta MIDAS","Direct lags"].index(default_cfg["weight_type"]),
                    key=f"wt_{reg}",
                )
                K = c2.slider(
                    "K (number of lags)",
                    min_value=1, max_value=44,
                    value=default_K, step=1,
                    key=f"K_{reg}",
                )
                pub_lag = c3.number_input(
                    "Publication lag (business days)",
                    min_value=0, max_value=90,
                    value=default_pub,
                    key=f"pub_{reg}",
                )

                cfg = {"weight_type": weight_type, "K": K, "pub_lag": int(pub_lag)}

                if weight_type == "Almon PDL":
                    c4, c5 = st.columns(2)
                    cfg["degree"] = c4.slider(
                        "Polynomial degree", 1, 4, default_cfg["degree"],
                        key=f"deg_{reg}",
                    )
                    cfg["constraint"] = c5.selectbox(
                        "Endpoint constraint",
                        ["none", "far", "near", "both"],
                        index=["none","far","near","both"].index(default_cfg["constraint"]),
                        key=f"con_{reg}",
                    )
                elif weight_type == "Beta MIDAS":
                    c4, c5 = st.columns(2)
                    cfg["theta1"] = c4.slider("θ₁", 1.0, 10.0, default_cfg["theta1"],
                                               0.1, key=f"t1_{reg}")
                    cfg["theta2"] = c5.slider("θ₂", 1.0, 10.0, default_cfg["theta2"],
                                               0.1, key=f"t2_{reg}")

                lag_configs[reg] = cfg

    # Compute current config hash
    cur_hash = config_hash(
        target_key, selected_regs, lag_configs,
        str(as_of_date.date()), train_end_global,
    )

    st.markdown("---")
    if st.button("Build Feature Matrix", key="btn_build", type="primary"):
        if not selected_regs:
            st.error("Select at least one regressor.")
        else:
            with st.spinner("Building lagged feature matrix..."):
                try:
                    X_scaled, y, build_meta, scaler = build_full_feature_matrix(
                        raw_data=st.session_state.raw_data,
                        target_key=target_key,
                        regressor_keys=selected_regs,
                        lag_configs=lag_configs,
                        as_of_date=as_of_date,
                        train_end=train_end_global,
                    )
                    st.session_state.X_scaled = X_scaled
                    st.session_state.y = y
                    st.session_state.build_meta = build_meta
                    st.session_state.scaler = scaler
                    st.session_state.build_config_hash = cur_hash
                    # Reset downstream
                    for k in ["vif_df", "vif_clean_cols", "pca_object", "X_pca",
                              "pca_loadings", "tsne_embedding", "fit_result",
                              "diagnostics_result"]:
                        st.session_state[k] = None
                    st.success(
                        f"Feature matrix built: {X_scaled.shape[0]} observations × "
                        f"{X_scaled.shape[1]} features."
                    )
                except Exception as e:
                    st.error(f"Feature build failed: {e}")

    # Feature matrix preview
    if st.session_state.X_scaled is not None:
        X = st.session_state.X_scaled
        _stale_warning(cur_hash)

        c1, c2, c3 = st.columns(3)
        c1.metric("Observations", X.shape[0])
        c2.metric("Features", X.shape[1])
        c3.metric("Date range",
                  f"{X.index.min().date()} → {X.index.max().date()}")

        # Missingness
        na_pct = (X.isna().mean() * 100).sort_values(ascending=False)
        if na_pct.max() > 0:
            st.warning("Some features have missing values (shown below). "
                       "The model uses the complete-cases intersection.")
            st.dataframe(na_pct[na_pct > 0].rename("Missing %").to_frame(),
                         use_container_width=True)

        with st.expander("Feature matrix preview (first 5 rows)"):
            st.dataframe(X.head(), use_container_width=True)

        st.plotly_chart(make_feature_matrix_heatmap(X), use_container_width=True)

# ===========================================================================
# TAB 3 — PCA & t-SNE
# ===========================================================================
with tab_pca:
    st.header("Dimensionality Reduction")

    if st.session_state.X_scaled is None:
        st.warning("Build the feature matrix first in **Feature Lab**.")
        st.stop()

    X = st.session_state.X_scaled
    _stale_warning(
        config_hash(st.session_state.target_key,
                    list(X.columns), {}, str(as_of_date.date()), train_end_global)
    )

    # ---- PCA ----
    st.subheader("PCA")
    max_pc = min(X.shape[1], max(X.dropna().shape[0] - 1, 2), 20)
    col1, col2 = st.columns([1, 3])

    with col1:
        n_comp = st.slider("Number of PCs", 2, max_pc, min(5, max_pc), key="sl_n_pca")
        if st.button("Run PCA", key="btn_pca"):
            with st.spinner("Running PCA..."):
                try:
                    pca_obj, scores_df, loadings_df = run_pca(X, n_comp, train_end_global)
                    st.session_state.pca_object = pca_obj
                    st.session_state.X_pca = scores_df
                    st.session_state.pca_loadings = loadings_df
                    st.session_state.selected_pcs = list(range(1, min(4, n_comp + 1)))
                    st.success("PCA complete.")
                except Exception as e:
                    st.error(f"PCA failed: {e}")

        if st.session_state.pca_object is not None:
            n_fit = len(st.session_state.pca_object.explained_variance_)
            all_pc_labels = [f"PC{i+1}" for i in range(n_fit)]
            sel_pcs = st.multiselect(
                "Use these PCs in Estimation",
                all_pc_labels,
                default=[f"PC{i}" for i in st.session_state.selected_pcs],
                key="ms_pcs",
            )
            st.session_state.selected_pcs = [int(pc[2:]) for pc in sel_pcs]

    with col2:
        if st.session_state.pca_object is not None:
            pca = st.session_state.pca_object
            st.subheader("Eigenvalue Table")
            st.dataframe(make_eigenvalue_table(pca), use_container_width=True)
            st.plotly_chart(make_scree_plot(pca), use_container_width=True)
            st.subheader("Loadings Heatmap")
            st.plotly_chart(
                make_loadings_heatmap(st.session_state.pca_loadings),
                use_container_width=True,
            )
            st.subheader("Biplot")
            bc1, bc2 = st.columns(2)
            bp_x = bc1.number_input("Biplot X-axis PC", 1, n_fit, 1, key="bp_x")
            bp_y = bc2.number_input("Biplot Y-axis PC", 1, n_fit, 2, key="bp_y")
            color_opt = st.selectbox(
                "Colour by", ["None", "Target value", "Year"], key="sb_bcolor"
            )
            color_series = None
            if color_opt == "Target value" and st.session_state.y is not None:
                color_series = st.session_state.y
            elif color_opt == "Year":
                idx = st.session_state.X_pca.index
                color_series = pd.Series(idx.year, index=idx, name="Year")

            st.plotly_chart(
                make_biplot(
                    st.session_state.X_pca,
                    st.session_state.pca_loadings,
                    bp_x, bp_y, color_series,
                ),
                use_container_width=True,
            )

    st.markdown("---")
    # ---- t-SNE ----
    st.subheader("t-SNE")
    col3, col4 = st.columns([1, 3])
    with col3:
        perp     = st.slider("Perplexity", 5, 50, 30, key="sl_tsne_perp")
        n_iter_t = st.slider("Iterations", 250, 2000, 1000, 250, key="sl_tsne_iter")
        tsne_color = st.selectbox(
            "Colour by",
            ["None", "Target value", "Year", "Prediction error"],
            key="sb_tsne_color",
        )
        if st.button("Run t-SNE", key="btn_tsne"):
            if len(X.dropna()) > 5000:
                st.warning("Large dataset — t-SNE may take a few minutes.")
            with st.spinner("Running t-SNE..."):
                try:
                    X_clean = X.dropna()
                    embedding = run_tsne(X_clean, perplexity=perp, n_iter=n_iter_t)
                    st.session_state.tsne_embedding = embedding
                    st.session_state.tsne_index = X_clean.index
                    st.success("t-SNE complete.")
                except Exception as e:
                    st.error(f"t-SNE failed: {e}")

    with col4:
        if st.session_state.tsne_embedding is not None:
            idx = st.session_state.tsne_index
            color_s = None
            clabel  = ""
            if tsne_color == "Target value" and st.session_state.y is not None:
                color_s = st.session_state.y.reindex(idx)
                clabel = "Target"
            elif tsne_color == "Year":
                color_s = pd.Series(idx.year, index=idx, name="Year")
                clabel = "Year"
            elif tsne_color == "Prediction error" and st.session_state.fit_result is not None:
                r = st.session_state.fit_result
                color_s = pd.Series(r["residuals"], index=r["y_index"], name="Residual").reindex(idx)
                clabel = "Residual"

            st.plotly_chart(
                make_tsne_plot(st.session_state.tsne_embedding, idx, color_s, clabel),
                use_container_width=True,
            )
            st.caption(
                "t-SNE is a non-linear technique for exploratory visualisation only. "
                "Cluster topology is sensitive to perplexity — re-run to check robustness."
            )

# ===========================================================================
# TAB 4 — VIF & Collinearity
# ===========================================================================
with tab_vif:
    st.header("VIF & Multicollinearity")

    if st.session_state.X_scaled is None:
        st.warning("Build the feature matrix first in **Feature Lab**.")
        st.stop()

    X = st.session_state.X_scaled

    col1, col2 = st.columns([1, 3])
    with col1:
        vif_thresh = st.slider("VIF threshold", 5.0, 30.0, 10.0, 0.5, key="sl_vif")

        if st.button("Compute VIF", key="btn_vif"):
            with st.spinner("Computing VIF..."):
                try:
                    vif_df = compute_vif(X.dropna())
                    st.session_state.vif_df = vif_df
                    st.session_state.vif_clean_cols = None
                except Exception as e:
                    st.error(f"VIF failed: {e}")

        if st.session_state.vif_df is not None:
            n_high = (st.session_state.vif_df["vif"] >= vif_thresh).sum()
            st.metric("Features with VIF ≥ threshold", int(n_high))

            if st.button("Run Iterative VIF Elimination", key="btn_vif_elim"):
                with st.spinner("Eliminating..."):
                    kept, dropped_list = iterative_vif_elimination(
                        X.dropna(), threshold=vif_thresh
                    )
                    st.session_state.vif_clean_cols = kept
                    st.success(
                        f"Kept {len(kept)} features, "
                        f"dropped {len(dropped_list)}."
                    )
                    if dropped_list:
                        st.dataframe(
                            pd.DataFrame(dropped_list, columns=["feature", "vif_at_drop"]),
                            use_container_width=True,
                        )

    with col2:
        if st.session_state.vif_df is not None:
            st.plotly_chart(
                make_vif_bar_chart(st.session_state.vif_df, vif_thresh),
                use_container_width=True,
            )
            with st.expander("VIF Table"):
                st.dataframe(st.session_state.vif_df, use_container_width=True)

        st.subheader("Correlation Matrix")
        st.plotly_chart(make_correlation_heatmap(X.dropna()), use_container_width=True)

# ===========================================================================
# TAB 5 — Estimation
# ===========================================================================
with tab_estimation:
    st.header("Model Estimation")

    if st.session_state.X_scaled is None:
        st.warning("Build the feature matrix first in **Feature Lab**.")
        st.stop()

    X_full = st.session_state.X_scaled
    y      = st.session_state.y

    # --- Feature input selector ---
    st.subheader("Feature Input")
    feat_options = ["Original features"]
    if st.session_state.vif_clean_cols:
        feat_options.append("VIF-cleaned features")
    if st.session_state.X_pca is not None:
        feat_options.append("PCA components")

    feat_input = st.radio("Use features from", feat_options, horizontal=True, key="ri_feat")

    if feat_input == "VIF-cleaned features" and st.session_state.vif_clean_cols:
        X_use = X_full[st.session_state.vif_clean_cols].dropna()
    elif feat_input == "PCA components" and st.session_state.X_pca is not None:
        pc_cols = [f"PC{i}" for i in st.session_state.selected_pcs]
        X_use = st.session_state.X_pca[pc_cols].dropna()
    else:
        X_use = X_full.dropna()

    # Align y to X_use
    aligned = X_use.join(y, how="inner").dropna()
    X_use = aligned.drop(columns=[y.name])
    y_use = aligned[y.name]

    st.caption(
        f"Using {X_use.shape[0]} observations × {X_use.shape[1]} features. "
        f"Train period: up to {train_end_global}."
    )

    # --- Method selector ---
    st.subheader("Estimation Method")
    method = st.selectbox("Method", list(ESTIMATORS.keys()), key="sb_method")

    # Method-specific params
    extra_kwargs: dict = {}
    with st.expander("Method hyperparameters", expanded=False):
        if method == "Quantile":
            extra_kwargs["quantile"] = st.slider("Quantile", 0.05, 0.95, 0.5, 0.05,
                                                  key="sl_quant")
        elif method == "OrderedLogit":
            extra_kwargs["sigma_threshold"] = st.slider(
                "Beat/miss threshold (σ)", 0.1, 1.5, 0.5, 0.05, key="sl_sigma"
            )
        elif method == "XGBoost":
            extra_kwargs["train_end"] = train_end_global
            n_est = st.slider("n_estimators", 50, 500, 300, 50, key="sl_xgb_n")
            depth = st.slider("max_depth", 2, 8, 4, 1, key="sl_xgb_d")
            lr    = st.slider("learning_rate", 0.01, 0.3, 0.05, 0.01, key="sl_xgb_lr")
            extra_kwargs["params"] = {
                "n_estimators": n_est, "max_depth": depth,
                "learning_rate": lr, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_alpha": 0.1,
                "reg_lambda": 1.0, "random_state": 42,
            }
        elif method == "Random Forest":
            extra_kwargs["n_estimators"] = st.slider("n_estimators", 50, 500, 300, 50,
                                                       key="sl_rf_n")
            extra_kwargs["max_depth"]    = st.slider("max_depth", 2, 15, 5, 1,
                                                       key="sl_rf_d")
        elif method in ("Ridge",):
            pass  # uses CV alphas automatically

    if st.button("Fit Model", key="btn_fit", type="primary"):
        with st.spinner(f"Fitting {method}..."):
            try:
                result = fit_model(method, X_use, y_use, **extra_kwargs)
                st.session_state.fit_result = result
                st.session_state.diagnostics_result = None  # reset
                st.success(
                    f"{method} fitted. "
                    f"R² = {result['r2']:.4f}, RMSE = {result['rmse']:.4f}"
                    + (f", AIC = {result['aic']:.1f}" if result["aic"] else "")
                )
            except Exception as e:
                st.error(f"Estimation failed: {e}")

    # --- Results ---
    if st.session_state.fit_result is not None:
        r = st.session_state.fit_result
        st.markdown("---")
        st.subheader("Results")

        c1, c2, c3, c4 = st.columns(4)
        label_r2 = "Accuracy" if r["method"] == "OrderedLogit" else "R²"
        c1.metric(label_r2, f"{r['r2']:.4f}")
        c2.metric("RMSE / MAE", f"{r['rmse']:.4f}")
        if r["aic"]:
            c3.metric("AIC", f"{r['aic']:.1f}")
        if r["bic"]:
            c4.metric("BIC", f"{r['bic']:.1f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Coefficients / Importances")
            coef_df_display = r["coef_df"].copy()
            for col in ["coef","importance","std_err","t_stat","p_value","ci_low","ci_high"]:
                if col in coef_df_display.columns:
                    coef_df_display[col] = coef_df_display[col].round(4)
            st.dataframe(coef_df_display, use_container_width=True)
            st.plotly_chart(make_coefficient_plot(r["coef_df"], r["method"]),
                            use_container_width=True)

        with col_b:
            st.subheader("Actual vs Fitted")
            st.plotly_chart(
                make_fitted_vs_actual_plot(y_use, r["fitted"], r["method"]),
                use_container_width=True,
            )

        # Method-specific extras
        if r["method"] in ("XGBoost", "Random Forest"):
            imp = r["extra"].get("feature_importance")
            if imp is not None:
                st.subheader("Feature Importance")
                st.plotly_chart(make_feature_importance_bar(imp), use_container_width=True)

        if r["method"] == "OrderedLogit":
            probs = r["extra"].get("probabilities")
            if probs is not None:
                st.subheader("Class Probabilities Over Time")
                st.plotly_chart(
                    make_probability_chart(probs, r["y_index"],
                                           r["extra"]["class_labels"]),
                    use_container_width=True,
                )

        # Almon lag weight profiles
        if st.session_state.build_meta is not None:
            almon_profiles = []
            for reg_key, meta in st.session_state.build_meta.items():
                if meta.get("type") == "almon":
                    almon_profiles.append((reg_key, meta))

            if almon_profiles and r["method"] in ("OLS", "Quantile"):
                st.subheader("Almon PDL Implied Lag Profiles")
                coef_series = r["coef_df"].set_index("feature")["coef"]
                for reg_key, meta in almon_profiles:
                    col_names = meta["col_names"]
                    delta = np.array([
                        coef_series.get(c, np.nan) for c in col_names
                    ])
                    if np.isnan(delta).any():
                        continue
                    weights = build_almon_weights_from_delta(
                        delta, meta["K"], meta["degree"], meta["constraint"]
                    )
                    st.plotly_chart(
                        make_almon_weight_profile(weights, reg_key, meta["K"]),
                        use_container_width=True,
                    )

# ===========================================================================
# TAB 6 — Diagnostics
# ===========================================================================
with tab_diagnostics:
    st.header("Model Diagnostics")

    if st.session_state.fit_result is None:
        st.warning("Fit a model first in **Estimation**.")
        st.stop()

    r = st.session_state.fit_result

    # Get X aligned to residuals
    if st.session_state.X_scaled is not None:
        X_diag = st.session_state.X_scaled.reindex(r["y_index"])
    else:
        st.error("Feature matrix not available for diagnostics.")
        st.stop()

    if st.button("Run Diagnostic Tests", key="btn_diag", type="primary"):
        with st.spinner("Running diagnostics..."):
            try:
                diag = run_all_diagnostics(r, X_diag)
                st.session_state.diagnostics_result = diag
            except Exception as e:
                st.error(f"Diagnostics error: {e}")

    if st.session_state.diagnostics_result is not None:
        diag = st.session_state.diagnostics_result

        # --- Summary table ---
        st.subheader("Test Summary")

        rows = []

        def _fmt(name, d, stat_key="stat", pval_key="pval"):
            if not d or "error" in d:
                return {"Test": name, "Statistic": "ERROR",
                        "p-value": d.get("error","") if d else "",
                        "Result": "ERROR"}
            stat = d.get(stat_key)
            pval = d.get(pval_key)
            rej  = d.get("reject_h0")
            return {
                "Test": name,
                "Statistic": f"{stat:.4f}" if stat is not None else "—",
                "p-value":   f"{pval:.4f}" if pval is not None else "—",
                "Result": ("Reject H0 (5%)" if rej else "Fail to reject H0")
                          if rej is not None else "—",
            }

        rows.append(_fmt("Jarque-Bera (Normality)",    diag.get("jarque_bera",{})))
        rows.append(_fmt("Shapiro-Wilk (Normality)",   diag.get("shapiro_wilk",{})))
        rows.append(_fmt("Breusch-Pagan (Heterosked.)",diag.get("breusch_pagan",{}),
                         "lm_stat","pval"))
        wt = diag.get("white_test",{})
        row_white = _fmt("White Test (Heterosked.)", wt, "lm_stat","pval")
        if "note" in wt:
            row_white["Result"] += f" [{wt['note']}]"
        rows.append(row_white)

        dw = diag.get("durbin_watson",{})
        rows.append({
            "Test": "Durbin-Watson (Autocorr.)",
            "Statistic": f"{dw.get('stat', '—'):.4f}" if dw.get("stat") else "—",
            "p-value": "—",
            "Result": dw.get("interpretation","—"),
        })

        lb = diag.get("ljung_box",{})
        if "error" not in lb and "lags" in lb:
            for lag, stat, pval in zip(lb["lags"], lb["stats"], lb["pvals"]):
                rows.append({
                    "Test": f"Ljung-Box (lag={lag})",
                    "Statistic": f"{stat:.4f}",
                    "p-value":   f"{pval:.4f}",
                    "Result": "Reject H0 (5%)" if pval < 0.05 else "Fail to reject H0",
                })

        rows.append(_fmt("Breusch-Godfrey (Autocorr.)",diag.get("breusch_godfrey",{}),
                         "lm_stat","pval"))

        cusum = diag.get("cusum",{})
        if "error" not in cusum and cusum:
            rows.append({
                "Test": "CUSUM (Structural Stability)",
                "Statistic": "—", "p-value": "—",
                "Result": "Stable" if cusum.get("is_stable") else "Instability detected",
            })

        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True)

        # --- Diagnostic plots ---
        st.subheader("Diagnostic Plots")
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(make_qq_plot(np.array(r["residuals"])),
                            use_container_width=True)
            st.plotly_chart(
                make_residual_time_plot(r["y_index"], np.array(r["residuals"])),
                use_container_width=True,
            )
        with p2:
            st.plotly_chart(
                make_residual_fitted_plot(np.array(r["fitted"]), np.array(r["residuals"])),
                use_container_width=True,
            )
            if "error" not in cusum and cusum:
                st.plotly_chart(
                    make_cusum_plot(cusum, r["y_index"]),
                    use_container_width=True,
                )

        # --- Normality detail ---
        with st.expander("Normality Details"):
            jb = diag.get("jarque_bera",{})
            if "error" not in jb:
                st.write(f"Skewness: **{jb.get('skewness','-'):.4f}**")
                st.write(f"Kurtosis: **{jb.get('kurtosis','-'):.4f}**")
                st.write(f"JB statistic: **{jb.get('stat','-'):.4f}**, "
                         f"p-value: **{jb.get('pval','-'):.4f}**")

        with st.expander("Heteroskedasticity Notes"):
            st.caption(
                "White test cross-products are limited to the first 6 features "
                "to avoid rank deficiency. For large feature matrices, Breusch-Pagan "
                "is the primary heteroskedasticity test."
            )
