"""The keyword arm: the Python approximation to NCI CDE Match, and CDE Match-Fuzzy.

Two implementations live here, and the manuscript treats them as different
methods:

``clone``
    The Python approximation to NCI CDE Match. It reimplements the service's
    rule cascade and scoring against the frozen production catalog. Table 4
    reports it as its own method, and seven of the final HGBC's 117 features
    are derived from its ranked list. It contributes no candidates to the pool.

``keyword_retriever``
    The fuzzy stage: character-n-gram, word-n-gram, token-overlap and
    permissible-value evidence over the catalog's text fields.

**CDE Match-Fuzzy is not a third program.** It is ``clone`` run with
``--fuzzy-fallback``, which appends ``keyword_retriever`` candidates after the
clone's exact-rule hits in a three-tier rank. ``FuzzyFallback`` is defined in
``clone``, and the shared allowance gate is the clone's ``ExactMatchControl``.
That is why the two modules ship together: separating them would not produce
two runnable methods, it would produce none.

``build_candidates`` is the entry point that runs either arm over a split and
writes the candidate parquet the pool consumes — ``demap cdematch-candidates``.

Redistribution
--------------
Both implementations are cleared for public release. They were previously held
behind a ``cde_match_derivative_unresolved`` gate and reached through a runtime
adapter; that gate is retired and the adapter is gone.

Still excluded, and unaffected by that clearance: the NCI-supplied Oracle
PL/SQL these were developed from, the CDE Match logic PDF, and saved output of
the live NCI CDE Match service. Those carry their own gates in
``manifests/source_migration.yaml`` and have never been in this repository.
"""
from __future__ import annotations

__all__ = ["clone", "keyword_retriever", "build_candidates"]
