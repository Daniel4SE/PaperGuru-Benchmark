"""Data loaders for the forecasting-forgetting paper (Jin & Ren, ICML 2024).

D_PT  : 36 balanced P3-train tasks, 100 examples each (§4.1, addendum).
D_R   : mispredicted examples on the model-refinement dataset
        (P3-Test for BART0, MMLU validation for FLAN-T5).
"""

from .tasks import (
    P3_TRAIN_TASKS_36,
    BART0_TEST_TASKS_8,
    P3_TEST_ID_TASKS,
    P3_TEST_OOD_TASKS,
)
from .loader import (
    load_p3_train,
    load_p3_test,
    load_mmlu_validation,
    build_d_pt,
)
from .refinement import (
    build_d_r,
    split_60_40,
    RefinementSplits,
)

__all__ = [
    "P3_TRAIN_TASKS_36",
    "BART0_TEST_TASKS_8",
    "P3_TEST_ID_TASKS",
    "P3_TEST_OOD_TASKS",
    "load_p3_train",
    "load_p3_test",
    "load_mmlu_validation",
    "build_d_pt",
    "build_d_r",
    "split_60_40",
    "RefinementSplits",
]
