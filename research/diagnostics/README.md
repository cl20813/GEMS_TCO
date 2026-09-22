# Experimental diagnostics

This directory is for diagnostic methods that are still being developed or
validated.  It is deliberately outside `src/`, so these experiments are not
installed as part of the stable `GEMS_TCO` package.

Code belongs here while its statistical definition, numerical behavior, or
public API may still change.  Promotion into `src/GEMS_TCO/` requires focused
tests, documented assumptions, and a stable interface.  Existing diagnostic
code is not moved here until its current callers have compatibility coverage.
