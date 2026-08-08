from __future__ import annotations

from typing import Final, List, Mapping, Sequence

# Canonical paper model set used for the off-the-shelf / finalists comparisons.
PAPER_CORE_MODEL_IDS: Final[List[str]] = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2',
    'pritamdeka/S-PubMedBert-MS-MARCO',
    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
    'kamalkraj/BioSimCSE-BioLinkBERT-BASE',
    'neuml/pubmedbert-base-embeddings',
]

PAPER_SCREENING_BACKBONE_MODEL_ID: Final[str] = 'neuml/pubmedbert-base-embeddings'
PAPER_HEATMAP_MODEL_ID: Final[str] = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
PAPER_OFF_THE_SHELF_SEEDS: Final[List[int]] = [1, 2]

PAPER_PHASE1_LRS: Final[List[float]] = [7e-5, 1e-4, 1.5e-4]
PAPER_PHASE1_BATCH_SIZES: Final[List[int]] = [128]
PAPER_PHASE1_TEMPERATURES: Final[List[float]] = [0.04, 0.07, 0.10]
PAPER_PHASE1_EPOCHS: Final[List[int]] = [1, 2, 3]

PAPER_PHASE2_LR_BY_PHASE1: Final[Mapping[float, List[float]]] = {
    7e-5: [3e-5, 5e-5, 7e-5],
    1e-4: [5e-5, 7e-5, 1e-4],
    1.5e-4: [7e-5, 1e-4, 1.5e-4],
}

PAPER_PHASE2_TEMPERATURE_BY_PHASE1: Final[Mapping[float, List[float]]] = {
    0.04: [0.03, 0.04, 0.05],
    0.07: [0.04, 0.06, 0.08],
    # Corrected 2026-07-26: the approved neighbourhood for a Phase 1 temperature of 0.10
    # is 0.07/0.08/0.10. The previous 0.06/0.08/0.10 row was erroneous. No completed run
    # used it (no Phase 1 winner landed on 0.10).
    0.10: [0.07, 0.08, 0.10],
}

PAPER_PHASE2_BATCH_SIZE: Final[int] = 11

PAPER_PHASE2_MINING_STRATEGIES: Final[List[str]] = [
    'none',
    'hard_top25',
    'semihard_1_50',
]

PAPER_FINALIST_REPRESENTATIONS: Final[List[tuple[str, str]]] = [
    ('Q1', 'v3'),
    ('Q3', 'v1_v6_v2b_v3_v5'),
    ('Q4', 'v1_v6_v2b_v3_v5'),
    ('Q2', 'v1_v6_v2b_v3_v5'),
]

PAPER_SNAPSHOT_TAGS: Final[Mapping[str, str]] = {
    'dataset': 'dataset_ready',
    'inference': 'paper_inference_complete',
    'phase1': 'paper_phase1_complete',
    'phase2': 'paper_phase2_complete',
    'baselines': 'paper_baselines_complete',
}


SNAPSHOT_TAG_DATASET: Final[str] = PAPER_SNAPSHOT_TAGS['dataset']
SNAPSHOT_TAG_INFERENCE: Final[str] = PAPER_SNAPSHOT_TAGS['inference']
SNAPSHOT_TAG_PHASE1: Final[str] = PAPER_SNAPSHOT_TAGS['phase1']
SNAPSHOT_TAG_PHASE2: Final[str] = PAPER_SNAPSHOT_TAGS['phase2']
SNAPSHOT_TAG_BASELINES: Final[str] = PAPER_SNAPSHOT_TAGS['baselines']

__all__ = [
    'PAPER_CORE_MODEL_IDS',
    'PAPER_SCREENING_BACKBONE_MODEL_ID',
    'PAPER_HEATMAP_MODEL_ID',
    'PAPER_OFF_THE_SHELF_SEEDS',
    'PAPER_PHASE1_LRS',
    'PAPER_PHASE1_BATCH_SIZES',
    'PAPER_PHASE1_TEMPERATURES',
    'PAPER_PHASE1_EPOCHS',
    'PAPER_PHASE2_LR_BY_PHASE1',
    'PAPER_PHASE2_TEMPERATURE_BY_PHASE1',
    'PAPER_PHASE2_BATCH_SIZE',
    'PAPER_PHASE2_MINING_STRATEGIES',
    'PAPER_FINALIST_REPRESENTATIONS',
    'PAPER_SNAPSHOT_TAGS',
    'SNAPSHOT_TAG_DATASET',
    'SNAPSHOT_TAG_INFERENCE',
    'SNAPSHOT_TAG_PHASE1',
    'SNAPSHOT_TAG_PHASE2',
    'SNAPSHOT_TAG_BASELINES',
]
