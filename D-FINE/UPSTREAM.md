# Upstream provenance

Source: https://github.com/Peterande/D-FINE
Revision: 956d1709314c2c6a4df6f34de232054578a7449f

Source is vendored in the AICOMP root repository so local training changes are
reviewable and recoverable in one checkout. The original LICENSE is retained.
The pre-existing upstream Git metadata is preserved locally under
.git/vendor-history/D-FINE.git (not part of project commits).

Local changes: src/solver/det_engine.py supports optional gradient accumulation,
optimizer-boundary callbacks, and scheduler/EMA updates at optimizer boundaries.
Defaults retain the upstream single-step behavior.
