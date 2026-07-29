"""Streamlit app that unites the positioning-composite methodology VERSIONS.

One interactive tool, one tab per methodology (V0..V4) plus a Compare tab. Every tab
renders its editable knobs straight from `positioning.versions.params.VERSION_PARAMS`,
rebuilds the version via its `build(inp, params)`, and shows the composite, its
responsiveness vs production, and an honest conditional predictive read.

Presentation only: this package codes exclusively against the frozen `base` / `params`
contract and never mutates `positioning.core` or the version builds.
"""
