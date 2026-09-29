"""SentenceTransformer loader utilities.

This module exists to support "virtual" model variants without forking
checkpoints on Hugging Face.

Currently supported variants
---------------------------

We support suffix-based variants appended to a base HF model id:

- "__cls"  : force CLS-token pooling (last_hidden_state[:,0])
- "__mean" : force mean-token pooling

Example
-------

  cambridgeltl/SapBERT-from-PubMedBERT-fulltext__cls

This allows off-the-shelf ablations such as SapBERT mean vs CLS pooling while
keeping the underlying checkpoint identical.

Notes
-----
- We intentionally do *not* use the Transformer "pooler_output".
- We modify the SentenceTransformers Pooling module flags in-place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


MODEL_ID_ALIASES = {
    "NeuML/pubmedbert-base-embeddings": "neuml/pubmedbert-base-embeddings",
}


STVariant = Literal["base", "cls", "mean"]


@dataclass(frozen=True)
class STModelSpec:
    """Parsed specification for a SentenceTransformer load request."""

    original: str
    base_model: str
    variant: STVariant


def parse_st_model_spec(model_name_or_path: str) -> STModelSpec:
    """Parse a model string into (base_model, variant).

    We reserve the *trailing* suffixes "__cls" and "__mean" to indicate pooling
    variants. If no suffix is present, variant="base".

    This is designed to be robust to normal HF ids containing underscores.
    """

    s = str(model_name_or_path).strip()
    if s.endswith("__cls"):
        base = s[: -len("__cls")]
        return STModelSpec(original=s, base_model=MODEL_ID_ALIASES.get(base, base), variant="cls")
    if s.endswith("__mean"):
        base = s[: -len("__mean")]
        return STModelSpec(original=s, base_model=MODEL_ID_ALIASES.get(base, base), variant="mean")
    return STModelSpec(original=s, base_model=MODEL_ID_ALIASES.get(s, s), variant="base")


#: The per-mode booleans of the sentence-transformers <5 Pooling module.
_LEGACY_POOLING_FLAGS = (
    "pooling_mode_cls_token",
    "pooling_mode_mean_tokens",
    "pooling_mode_max_tokens",
    "pooling_mode_mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens",
    "pooling_mode_lasttoken",
)

#: Legacy boolean -> the equivalent sentence-transformers >=5 ``pooling_mode``
#: string, so :func:`effective_pooling` answers in one vocabulary either way.
_LEGACY_FLAG_TO_MODE = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}

#: Variant -> (legacy boolean, sentence-transformers >=5 ``pooling_mode`` string).
_VARIANT_TO_POOLING = {
    "cls": ("pooling_mode_cls_token", "cls"),
    "mean": ("pooling_mode_mean_tokens", "mean"),
}


def _is_pooling_module(module) -> bool:
    """True for a SentenceTransformers Pooling module of either generation.

    sentence-transformers <5 exposes one boolean per mode; 5.x replaced them with
    a single ``pooling_mode`` string. Duck-typing only the booleans meant that on
    a modern install no Pooling module was ever found and ``__cls``/``__mean``
    raised ``ValueError``.
    """
    if hasattr(module, "pooling_mode_cls_token") and hasattr(module, "pooling_mode_mean_tokens"):
        return True
    return isinstance(getattr(module, "pooling_mode", None), str)


def effective_pooling(st_model) -> Optional[str]:
    """The pooling mode actually in force, read back from the model.

    Returns ``'cls'`` / ``'mean'`` / another mode name, or ``None`` if no Pooling
    module is present. Reads whichever API the installed version uses, so a caller
    can verify it got the pooling it asked for rather than assuming.
    """
    pool = _find_pooling_module(st_model)
    if pool is None:
        return None
    mode = getattr(pool, "pooling_mode", None)
    if isinstance(mode, str):
        return mode
    for flag in _LEGACY_POOLING_FLAGS:
        if getattr(pool, flag, False):
            return _LEGACY_FLAG_TO_MODE[flag]
    return None


def _find_pooling_module(st_model):
    """First Pooling-like submodule, by ``._modules`` then by index."""
    modules = getattr(st_model, "_modules", None)
    if isinstance(modules, dict):
        for m in modules.values():
            if _is_pooling_module(m):
                return m
    try:
        for i in range(0, 10):
            m = st_model[i]  # type: ignore[index]
            if _is_pooling_module(m):
                return m
    except Exception:
        pass
    return None


def apply_pooling_variant(st_model, variant: STVariant) -> None:
    """Modify an in-memory SentenceTransformer to use a specific pooling mode.

    Supports both Pooling generations: the per-mode booleans of
    sentence-transformers <5 and the single ``pooling_mode`` string of 5.x. The
    two are never written together, so an inert legacy flag cannot end up
    disagreeing with the mode actually in force.

    This function is intentionally import-light so unit tests can exercise it
    without requiring sentence-transformers to be installed.
    """

    if variant == "base":
        return

    if variant not in _VARIANT_TO_POOLING:
        raise ValueError(f"Unknown pooling variant: {variant}")

    pool = _find_pooling_module(st_model)
    if pool is None:
        raise ValueError(
            "Could not locate a SentenceTransformers Pooling module to set pooling variant. "
            "(Expected a module with pooling_mode_cls_token/pooling_mode_mean_tokens "
            "attributes, or a sentence-transformers >=5 module with a pooling_mode string.)"
        )

    legacy_flag, mode_string = _VARIANT_TO_POOLING[variant]

    if any(hasattr(pool, flag) for flag in _LEGACY_POOLING_FLAGS):
        for flag in _LEGACY_POOLING_FLAGS:
            if hasattr(pool, flag):
                setattr(pool, flag, False)
        setattr(pool, legacy_flag, True)
    else:
        pool.pooling_mode = mode_string


def enforce_min_max_seq_length(st_model, min_max_seq_length: int) -> int:
    """Ensure a SentenceTransformer uses at least `min_max_seq_length` tokens.

    This implements a *floor* (not a cap):

        effective = max(native_max_seq_length, min_max_seq_length)

    We update both `SentenceTransformer.max_seq_length` (if present) and the first
    transformer's tokenizer `model_max_length` when available.

    The function is duck-typed for unit testing and does not require importing
    sentence-transformers at import time.

    Returns the effective max sequence length.
    """

    try:
        floor = int(min_max_seq_length)
    except Exception:
        return int(getattr(st_model, "max_seq_length", 0) or 0)

    if floor <= 0:
        return int(getattr(st_model, "max_seq_length", 0) or 0)

    native = getattr(st_model, "max_seq_length", None)
    try:
        native_i = int(native) if native is not None else 0
    except Exception:
        native_i = 0

    effective = native_i if native_i >= floor else floor

    # Set on the SentenceTransformer object.
    try:
        st_model.max_seq_length = int(effective)
    except Exception:
        pass

    # Also set on the first transformer module and its tokenizer if present.
    first = None
    try:
        # SentenceTransformer supports indexing.
        first = st_model[0]  # type: ignore[index]
    except Exception:
        first = None

    if first is None:
        modules = getattr(st_model, "_modules", None)
        if isinstance(modules, dict) and modules:
            first = list(modules.values())[0]

    if first is not None:
        try:
            if hasattr(first, "max_seq_length"):
                first.max_seq_length = int(effective)
        except Exception:
            pass
        try:
            tok = getattr(first, "tokenizer", None)
            if tok is not None and hasattr(tok, "model_max_length"):
                tok.model_max_length = int(effective)
        except Exception:
            pass

    return int(effective)



def load_sentence_transformer(
    model_name_or_path: str,
    *,
    device: Optional[str] = None,
    revision: Optional[str] = None,
    min_max_seq_length: int = 256,
):
    """Load a SentenceTransformer model, supporting pooling-variant suffixes."""

    spec = parse_st_model_spec(model_name_or_path)

    # Import lazily so CLIs can show --help in minimal environments.
    from sentence_transformers import SentenceTransformer  # type: ignore

    kwargs = {"revision": revision} if revision is not None else {}
    if device is not None:
        kwargs["device"] = device
    model = SentenceTransformer(spec.base_model, **kwargs)

    apply_pooling_variant(model, spec.variant)

    # Enforce a common minimum max sequence length for fair comparisons.
    enforce_min_max_seq_length(model, int(min_max_seq_length))

    return model
