# SurveyBench Results — CCM-Backed Long-Form Survey Generation

This directory contains the **best version** of the 20 generated surveys
on the [SurveyBench](https://github.com/...) benchmark, used to substantiate
the long-document generalization claims of CCM in Section 5.2 of the main paper.

The system under test combines a Claude Opus 4.7 backbone with the CCM
lifecycle-aware memory described in Section 3 of the paper; all
prompt templates, system-level scaffolding, and runtime details are
deliberately excluded from this release.

## Layout

```
results/surveybench/
├── md/        20 markdown surveys (post-revision body, references inline)
├── latex/     20 LaTeX projects with figures (one subdir per topic)
└── pdf/       20 compiled PDFs (single-column ICML-style)
```

Each topic appears in all three subfolders under the same slug
(e.g. `3d-gaussian-splatting.md`, `latex/3d-gaussian-splatting/`,
`pdf/3d-gaussian-splatting.pdf`).

## Headline Numbers (Table~5 in the main paper)

Under the official `claude-opus-4.7` judge at the 200K-character truncation
cap, averaged over three independent judge runs:

| Method                            | Content avg | Richness |
|-----------------------------------|-------------|----------|
| AutoSurvey (Wang et al., 2024)    | 3.510       | 0.00     |
| SurveyForge (Yan et al., 2025)    | 3.770       | 0.00     |
| LLM×MR-V2 (Liang et al., 2025)    | 3.970       | 5.09     |
| OpenAI DeepResearch (2024)        | 4.030       | 1.56     |
| **CCM (this paper)**              | **4.733**   | **10.94**|

## What is masked in this directory

- All authorship / affiliation metadata in the LaTeX headers has been
  rewritten to `Anonymous / Anonymous Institution`.
- All references to specific runtime systems, agent frameworks, prompt
  templates, and proprietary infrastructure have been removed.
- Only the final per-topic survey body is shipped; intermediate revision
  passes, per-pass logs, and judge-side traces are omitted.
