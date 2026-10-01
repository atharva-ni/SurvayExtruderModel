# LLM labeling codebook (training labels for the profile combiner)

Label each paper from its title, venue, publication type and abstract.

- **S (survey)**: the main contribution is reviewing or organizing existing work. Surveys, reviews,
  tutorials, overviews, and magazine articles whose body is mainly an overview of a field (enabling
  technologies, applications, challenges, open issues, future directions), even when a short case study is
  included at the end.
- **R (research)**: the main contribution is the authors' own method, system, analysis, theory or
  measurement, evaluated by the authors, even if the paper opens with background. Magazine articles whose
  core is a new framework or scheme ("we propose ..., simulation results show ...") are research.
- **N (not a paper)**: editorials, front matter, books, errata, calls for papers, reports on events.

Confidence: H (clear), M (leaning), L (could go either way).
The indexer type ("Review") and the model score are not used; labels are blind to the model.

## Sample (`data/llm_labels.csv`)

Pool: 9,506 papers from the Semantic Scholar profiles of the five survey authors (`data/proauthor_s2/`) and
the OpenAlex profiles of both cohorts (`data/cohort_openalex/`), deduplicated by title, with an abstract of at
least 30 words, not in the training dataset or the hand-labeled evaluation set, and not a non-paper by
type or title. The pool was scored with the fine-tuned DistilBERT model and split into three strata:

| Stratum | Definition | Pool | Labeled |
|---|---|---|---|
| flagged | DistilBERT survey probability >= 0.5 | 1,119 | 1,119 |
| magazine_arxiv_low | probability < 0.5, magazine or arXiv venue | 636 | 150 |
| random_low | all other papers | 7,751 | 250 |

1,519 papers were drawn (all 1,119 flagged papers, 150 and 250 from the other strata) and shuffled; all were
labeled, blind to the model score, by Claude Opus 5.5 (`claude-opus-5-5`) on 28-29 September 2026 following the
rules above (the first 1,000 on 28 September, the remaining 519 on 29 September). `StratumWeight` = pool size / labeled papers in the stratum. Labels: 0 survey, 1 research,
skip = not a paper (596 / 845 / 78).
