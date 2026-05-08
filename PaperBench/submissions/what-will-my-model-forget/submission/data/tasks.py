"""Task lists from the addendum.

The 36 P3-train tasks used to construct D_PT (intersection of T0-Train tasks
and the BART0 ReCross repo data) and the 8 BART0 test tasks used to
construct D_R for BART0 experiments.
"""

# 36 P3-train tasks for D_PT (per addendum to the paper).
# Source: intersection of T0-Train (Sanh et al., 2021, Table 5) and
# https://github.com/INK-USC/ReCross/blob/main/data/
P3_TRAIN_TASKS_36 = [
    "glue-mrpc",
    "glue-qqp",
    "paws_x-en",
    "kilt_tasks-hotpotqa",
    "wiki_qa",
    "adversarial_qa-dbert",
    "adversarial_qa-dbidaf",
    "adversarial_qa-droberta",
    "duorc-SelfRC",
    "duorc-ParaphraseRC",
    "ropes",
    "quoref",
    "cos_e-v1.11",
    "cosmos_qa",
    "dream",
    "qasc",
    "quail",
    "quartz",
    "sciq",
    "social_i_qa",
    "wiki_hop-original",
    "wiqa",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "common_gen",
    "wiki_bio",
    "cnn_dailymail-3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "ag_news",
    "dbpedia_14",
]

assert len(P3_TRAIN_TASKS_36) == 36, "addendum says 36 tasks"


# 8 BART0 model-refinement (P3-Test) tasks (per addendum)
BART0_TEST_TASKS_8 = [
    "super_glue-wsc.fixed",
    "winogrande-winogrande_xl",
    "super_glue-cb",
    "super_glue-rte",
    "anli",
    "super_glue-copa",
    "hellaswag",
    "super_glue-wic",
]


# In-domain / out-of-domain split for Table 2 (BART0, full FT).
# The paper does not give the exact split; we follow the addendum's note
# that "all task variants are used" and pick a balanced 4/4 split. This
# is documented as a configurable choice in configs/default.yaml.
P3_TEST_ID_TASKS = [
    "super_glue-cb",
    "super_glue-rte",
    "anli",
    "super_glue-wic",
]
P3_TEST_OOD_TASKS = [
    "super_glue-wsc.fixed",
    "winogrande-winogrande_xl",
    "super_glue-copa",
    "hellaswag",
]


# MMLU is the full set of 57 tasks (validation split per addendum).
# We do not enumerate them here; they are loaded directly from the
# original Hendrycks data tarball.
