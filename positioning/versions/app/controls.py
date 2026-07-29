"""Widget factory: render one Streamlit control per `params.Param`.

`render_params(version)` reads the ORDERED schema `params.VERSION_PARAMS[version]` and
renders it into two side-by-side expanders ("Data inputs" vs "Method params"), one widget
per Param keyed by its `kind`:

    int / float  -> st.slider
    choice       -> st.selectbox
    multiselect  -> st.multiselect
    bool         -> st.checkbox
    text         -> st.text_input

It returns a plain ``{key: value}`` dict of the selected values, ready to hand to a
version's ``build(inp, params)``. Widget keys are namespaced by version so the same knob
can appear on multiple tabs without a Streamlit key collision.
"""
from __future__ import annotations

import streamlit as st

from positioning.versions import params as P


def _widget(prm: P.Param, key: str):
    """Render a single Param and return its selected value."""
    kind = prm.kind
    help_ = prm.help or None

    if kind == "int":
        if prm.min is None or prm.max is None:
            return st.number_input(prm.label, value=int(prm.default), step=int(prm.step or 1),
                                   key=key, help=help_)
        return st.slider(prm.label, min_value=int(prm.min), max_value=int(prm.max),
                         value=int(prm.default), step=int(prm.step or 1), key=key, help=help_)

    if kind == "float":
        if prm.min is None or prm.max is None:
            return st.number_input(prm.label, value=float(prm.default),
                                   step=float(prm.step or 0.1), key=key, help=help_)
        return st.slider(prm.label, min_value=float(prm.min), max_value=float(prm.max),
                         value=float(prm.default), step=float(prm.step or 0.1),
                         key=key, help=help_)

    if kind == "choice":
        choices = list(prm.choices or [])
        idx = choices.index(prm.default) if prm.default in choices else 0
        return st.selectbox(prm.label, options=choices, index=idx, key=key, help=help_)

    if kind == "multiselect":
        choices = list(prm.choices or [])
        default = [d for d in (prm.default or []) if d in choices]
        return st.multiselect(prm.label, options=choices, default=default, key=key, help=help_)

    if kind == "bool":
        return st.checkbox(prm.label, value=bool(prm.default), key=key, help=help_)

    # text (and any unknown kind) -> text input
    return st.text_input(prm.label, value=str(prm.default), key=key, help=help_)


def render_params(version: str) -> dict:
    """Render every Param for `version` (from params.VERSION_PARAMS) and collect the values.

    Grouped into two columns: 'Data inputs' and 'Method params'. Returns {key: value}.
    """
    schema = P.VERSION_PARAMS[version]
    data_params = [p for p in schema if p.group == P.GROUP_DATA]
    method_params = [p for p in schema if p.group != P.GROUP_DATA]

    values: dict = {}
    c_data, c_method = st.columns(2)

    with c_data:
        with st.expander("Data inputs", expanded=True):
            if data_params:
                for prm in data_params:
                    values[prm.key] = _widget(prm, key=f"{version}::{prm.key}")
            else:
                st.caption("This version reuses the production data pipeline (no data knobs).")

    with c_method:
        with st.expander("Method params", expanded=True):
            if method_params:
                for prm in method_params:
                    values[prm.key] = _widget(prm, key=f"{version}::{prm.key}")
            else:
                st.caption("No tunable method params for this version.")

    return values
