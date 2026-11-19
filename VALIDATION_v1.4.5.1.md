# PV CrackVision v1.4.5.1 — superseded validation note

The dependency-index change described in the original v1.4.5.1 notes was ineffective. The PyTorch install command supplied `--index-url https://download.pytorch.org/whl/cpu` on the command line, which took precedence over the attempted `PIP_INDEX_URL=https://pypi.org/simple` environment override. The isolated build-dependency subprocess therefore continued to look only at the PyTorch index and could not find `flit_core`.

Do not use the v1.4.5.1 release as the fix for the `flit_core<4,>=3.11` failure. The corrected two-phase installation and checks are in v1.4.5.2; see `VALIDATION_v1.4.5.2.md`.
