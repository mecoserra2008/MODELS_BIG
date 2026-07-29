"""Alternative positioning-composite methodologies (normalization versions).

Each version is a self-contained module exposing `build(inp) -> base.VersionResult`, coded
against the FIXED interface in `base.py` so they can be developed independently and compared
side by side. The production composite in positioning.core is NOT changed by this package;
`compare.py` joins the versions and `get_composite(version)` selects one.
"""
from . import base  # noqa: F401
