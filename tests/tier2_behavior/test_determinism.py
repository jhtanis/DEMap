"""Ranking is deterministic by construction.

An earlier version of this pipeline was not. Two keyword rules built a counter by
iterating a Python set, and CPython randomizes string hashing per process, so
which exactly-tied candidates survived the per-rule truncation varied between
runs. That moved `keyword_rank`, `n_rule_hits`, `best_rule_code` and every `kw_*`
feature downstream of them.

The fix was to give every ordering step a total order with an explicit final
tie-break, and to forbid any ordering input that depends on process state.
`configs/paper/determinism_policy_v1.json` is the contract; these tests hold the
policy to it and check the one rule the manuscript states outright — ties broken by
ascending CDE public identifier.

Note the honest limitation recorded in the policy itself: outputs produced before
the fix are hash-seed samples and cannot be reproduced by any rule. The policy was
chosen on principle, not fitted to a saved output.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier2

POLICY_PATH = (Path(__file__).resolve().parents[2]
               / "configs" / "paper" / "determinism_policy_v1.json")


@pytest.fixture(scope="module")
def policy():
    return json.loads(POLICY_PATH.read_text())


def test_policy_version_is_pinned(policy):
    assert policy["pipeline_determinism_version"] == "demap_reranker_determinism_v1"


def test_no_environment_variable_is_part_of_the_algorithm(policy):
    """PYTHONHASHSEED must not be needed at runtime. Requiring it would mean the
    algorithm still depended on process state; it would just be papered over."""
    assert "PYTHONHASHSEED" in policy["summary"]
    forbidden = policy["explicitly_forbidden_as_ordering_inputs"]
    assert any("environment" in f.lower() for f in forbidden)


def test_final_tie_break_is_ascending_public_id(policy):
    """S5.5 and Figure 6: 'exact ties broken by ascending CDE public identifier'."""
    assert (policy["policies"]["final_score_tie_break_policy_version"]
            == "final_rank_v1_score_desc_then_public_id_asc")
    keys = policy["policies"]["final_score_tie_break_keys"]
    assert any("final_score DESCENDING" in k for k in keys)
    assert any("public id ASCENDING" in k for k in keys)


def test_public_id_ordering_is_numeric_where_possible(policy):
    """CDE public ids are integers. Lexicographic ordering would put '10' before
    '9', so the rule compares numerically and sends non-numeric ids last."""
    keys = " ".join(policy["policies"]["final_score_tie_break_keys"])
    assert "numeric comparison" in keys
    assert "non-numeric ids sort last" in keys


def test_ranking_invariants_match_the_paper(policy):
    invariants = policy["policies"]["final_ranking_invariants"]
    joined = " ".join(invariants)
    assert "rank starts at 1 within each query" in joined
    assert "top-K applied only after scoring and ranking" in joined
    # Exact-match pinning was removed from every manuscript-facing artifact.
    assert "no exact-match pinning" in joined
    assert "no hidden ranking tier" in joined


def test_forbidden_ordering_inputs_cover_the_known_failure_modes(policy):
    forbidden = " ".join(policy["explicitly_forbidden_as_ordering_inputs"]).lower()
    for source in ("set iteration order", "hash values", "filesystem enumeration",
                   "incoming feature-table row order"):
        assert source in forbidden


def test_candidate_pool_policy_matches_the_paper(policy):
    keys = " ".join(policy["policies"]["candidate_union_keys"])
    assert "bi-encoder top-20" in keys
    assert "keyword_v1 fuzzy top-10" in keys
    assert "deduplicated by CDE public id" in keys


def test_keyword_tie_break_uses_a_frozen_catalog_index(policy):
    """The secondary key is the candidate's row in the keyword index, which is
    unique and frozen at build time — not its position in a runtime container."""
    keys = " ".join(policy["policies"]["keyword_tie_break_keys"])
    assert "rule_score DESCENDING" in keys
    assert "catalog row index ASCENDING" in keys
    assert "frozen at index build" in keys


def test_root_cause_is_recorded_with_its_blast_radius(policy):
    """The policy documents which rules were affected and which features moved, so
    a reader can tell whether a historical artifact predates the fix."""
    root = policy["root_cause_fixed"]
    assert set(root["affected_rules"]) == {"token (all 5 text fields)", "pv (PV_SUMMARY)"}
    assert "exact" in root["unaffected_rules"]
    assert "keyword_rank" in root["downstream_features_that_moved"]


def test_pre_fix_outputs_are_marked_superseded(policy):
    hist = policy["historical_outputs"]
    assert "SUPERSEDED" in hist["status"]
    assert hist["named"], "the affected saved outputs must be named, not implied"


# --------------------------------------------------------------------------
# the tie-break rule itself
# --------------------------------------------------------------------------

def _rank(candidates):
    """Reference implementation of the documented final ordering."""
    def key(item):
        public_id, score = item
        try:
            numeric = (0, int(public_id))
        except (TypeError, ValueError):
            numeric = (1, 0)
        return (-score, numeric, str(public_id))
    return [pid for pid, _ in sorted(candidates, key=key)]


def test_ties_resolve_by_ascending_public_id():
    assert _rank([("300", 0.9), ("100", 0.9), ("200", 0.9)]) == ["100", "200", "300"]


def test_score_dominates_the_tie_break():
    assert _rank([("999", 0.95), ("100", 0.90)]) == ["999", "100"]


def test_public_id_compares_numerically_not_lexicographically():
    """Lexicographic ordering would rank '1000' ahead of '999'."""
    assert _rank([("1000", 0.5), ("999", 0.5)]) == ["999", "1000"]


def test_non_numeric_ids_sort_last():
    assert _rank([("abc", 0.5), ("100", 0.5)]) == ["100", "abc"]


def test_ordering_is_independent_of_input_order():
    """The policy forbids incoming row order as an ordering input."""
    forward = [("300", 0.9), ("100", 0.9), ("200", 0.8)]
    assert _rank(forward) == _rank(list(reversed(forward)))
