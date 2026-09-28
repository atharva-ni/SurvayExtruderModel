# Survey classifier evaluation (2026-09-28)

Model: `./distilbert_survey_model` — DistilBERT threshold 0.96

Learned hybrid coefficients: `{'bert_logit': 0.493, 'title_terms': 2.242, 'abstract_survey_cues': 1.404, 'abstract_research_cues': -0.298, 'abstract_missing': 0.262, 'log_refs': 0.807, 'refs_missing': 2.985}`

## Table II(a) — held-out test split (n = 1925, 963 surveys)

| Method                                                   | Acc.   | Prec.   | Recall   | F1    |
|----------------------------------------------------------|--------|---------|----------|-------|
| Keyword only (title)                                     | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| TF-IDF + SVM                                             | 94.8%  | 96.5%   | 92.8%    | 94.7% |
| DistilBERT only                                          | 95.3%  | 97.4%   | 93.1%    | 95.2% |
| Hybrid (OR, paper)                                       | 95.3%  | 97.4%   | 93.1%    | 95.2% |
| Learned hybrid (title+abstract)                          | 94.9%  | 98.0%   | 91.7%    | 94.7% |
| Learned hybrid (+ reference count)                       | 95.4%  | 97.8%   | 92.9%    | 95.3% |
| Indexer type ('Review')                                  | 52.2%  | 100.0%  | 4.4%     | 8.4%  |
| Baseline: distilbert_survey_model_synthetic (OR, th=0.8) | 78.3%  | 87.9%   | 65.6%    | 75.1% |

## Table II(a2) — same test split, title only (abstract removed)

| Method                                                   | Acc.   | Prec.   | Recall   | F1    |
|----------------------------------------------------------|--------|---------|----------|-------|
| Keyword only (title)                                     | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| TF-IDF + SVM                                             | 86.4%  | 90.1%   | 81.9%    | 85.8% |
| DistilBERT only                                          | 88.7%  | 94.5%   | 82.2%    | 88.0% |
| Hybrid (OR, paper)                                       | 88.7%  | 94.5%   | 82.2%    | 88.0% |
| Learned hybrid (title+abstract)                          | 88.7%  | 94.8%   | 81.8%    | 87.8% |
| Learned hybrid (+ reference count)                       | 90.1%  | 95.0%   | 84.7%    | 89.6% |
| Indexer type ('Review')                                  | 52.2%  | 100.0%  | 4.4%     | 8.4%  |
| Baseline: distilbert_survey_model_synthetic (OR, th=0.8) | 78.5%  | 92.7%   | 62.0%    | 74.3% |

## Table II(b) — hand-labeled author-profile papers (n = 323, 111 surveys; labels: first author)

| Method                                                   | Acc.   | Prec.   | Recall   | F1    | F1 (H+M labels)   | Prec. (rw)   | Recall (rw)   | Flagged (rw)   | Prec. (rw) 95% CI   | Recall (rw) 95% CI   |
|----------------------------------------------------------|--------|---------|----------|-------|-------------------|--------------|---------------|----------------|---------------------|----------------------|
| Keyword only (title)                                     | 76.2%  | 97.2%   | 31.5%    | 47.6% | 54.3%             | 97.2%        | 23.3%         | 1.9%           | 91–100%             | 17–33%               |
| TF-IDF + SVM                                             | 88.9%  | 76.6%   | 97.3%    | 85.7% | 88.2%             | 57.6%        | 91.5%         | 12.7%          | 44–75%              | 78–100%              |
| DistilBERT only                                          | 87.0%  | 73.2%   | 98.2%    | 83.8% | 87.9%             | 58.4%        | 92.2%         | 12.6%          | 44–74%              | 78–100%              |
| Hybrid (OR, paper)                                       | 87.0%  | 73.2%   | 98.2%    | 83.8% | 87.9%             | 58.4%        | 92.2%         | 12.6%          | 44–74%              | 78–100%              |
| Learned hybrid (title+abstract)                          | 87.3%  | 74.0%   | 97.3%    | 84.0% | 88.3%             | 68.9%        | 85.0%         | 9.9%           | 56–81%              | 69–100%              |
| Learned hybrid (+ reference count)                       | 87.3%  | 74.0%   | 97.3%    | 84.0% | 88.3%             | 67.2%        | 78.5%         | 9.4%           | 55–80%              | 62–100%              |
| DistilBERT + magazine rule                               | 86.1%  | 77.0%   | 84.7%    | 80.7% | 88.4%             | 66.4%        | 62.7%         | 7.6%           | 52–82%              | 47–86%               |
| Learned hybrid + magazine rule                           | 86.4%  | 77.2%   | 85.6%    | 81.2% | 88.5%             | 71.6%        | 63.4%         | 7.1%           | 59–83%              | 47–86%               |
| Indexer type ('Review')                                  | 84.5%  | 76.1%   | 80.2%    | 78.1% | 86.5%             | 60.9%        | 59.4%         | 7.8%           | 47–78%              | 44–80%               |
| Learned hybrid + magazine rule (type hidden)             | 85.4%  | 78.1%   | 80.2%    | 79.1% | 87.2%             | 78.1%        | 59.4%         | 6.1%           | 71–85%              | 44–80%               |
| Baseline: distilbert_survey_model_synthetic (OR, th=0.8) | 76.8%  | 82.1%   | 41.4%    | 55.1% | 61.7%             | 34.4%        | 30.7%         | 7.2%           | 23–58%              | 23–42%               |
| LLM labels (Claude)                                      | 91.3%  | 85.5%   | 90.1%    | 87.7% | 94.7%             | 80.4%        | 73.2%         | 7.3%           | 67–92%              | 56–93%               |

(rw) = reweighted by sampling stratum to the real mix of papers in the profiles; estimated true survey rate: 8.0% (95% CI 6–11%). CIs: 2000 bootstrap resamples within each stratum.

## Sampling design of the hand-labeled set

| Stratum      | Definition                                         |   Pool |   Sampled |   Labeled |   Surveys |   Weight | Share of est. survey rate   |
|--------------|----------------------------------------------------|--------|-----------|-----------|-----------|----------|-----------------------------|
| abstract_cue | survey phrasing in the abstract, none in the title |    120 |       120 |       117 |        63 |    1     | 3.4%                        |
| random       | everything else                                    |   1717 |       160 |       159 |         4 |   10.731 | 2.3%                        |
| title_kw     | survey term in the title                           |     48 |        48 |        47 |        44 |    1     | 2.4%                        |

Pool = profile papers with an abstract of at least 30 words, deduplicated by title, not in the training set. Weight = pool / sampled. Estimated survey rate = weighted surveys / weighted labeled papers.

## Label agreement: first author (blind) vs. LLM

323 papers labeled survey or research by both: agreement 91.3%, Cohen's kappa 0.81; 28 disagreements (the first author's label is used). All papers were labeled blind by the first author.

## Table IV — impact of excluding detected surveys, survey authors (learned hybrid; data/proauthor_s2/*.csv)

| Author     |   Papers |   Surveys |   Magazine overviews |   Non-papers | Papers excluded   | Citation reduction   | h-index   |   h-index drop | i10-index reduction   | Citation reduction (+ magazine)   |   h-index drop (+ magazine) |
|------------|----------|-----------|----------------------|--------------|-------------------|----------------------|-----------|----------------|-----------------------|-----------------------------------|-----------------------------|
| Zhu Han    |     1353 |       114 |                   21 |           17 | 8.43%             | 21.87%               | 128 → 117 |             11 | 9.26%                 | 25.20%                            |                        14   |
| L. Hanzo   |     2710 |       180 |                   23 |           10 | 6.64%             | 31.77%               | 131 → 108 |             23 | 10.27%                | 34.77%                            |                        26   |
| E. Hossain |      733 |       125 |                    9 |           28 | 17.05%            | 37.53%               | 107 → 80  |             27 | 22.51%                | 40.94%                            |                        29   |
| D. Niyato  |     2514 |       350 |                   68 |           44 | 13.92%            | 38.61%               | 143 → 114 |             29 | 18.34%                | 41.91%                            |                        32   |
| F. Yu      |      985 |        86 |                   13 |           33 | 8.73%             | 30.19%               | 101 → 86  |             15 | 11.96%                | 33.97%                            |                        17   |
| Average    |          |           |                      |              | 10.95%            | 31.99%               |           |             21 | 14.47%                | 35.36%                            |                        23.6 |

Books/editorials are not counted as surveys. '(+ magazine)' also excludes magazine articles that the classifier flagged but that do not present themselves as surveys.

## Table IV(b) — survey authors, papers excluded by each method (averages over authors)

| Method                                                  | Papers excluded   | Citations   |   Δh (avg) |   Δh han |   Δh hanzo |   Δh hossain |   Δh niyato |   Δh yu |
|---------------------------------------------------------|-------------------|-------------|------------|----------|------------|--------------|-------------|---------|
| DistilBERT only (no rules)                              | 18.7%             | 38.7%       |       26.2 |       16 |         32 |           32 |          34 |      17 |
| DistilBERT + rules                                      | 11.6%             | 32.5%       |       21.4 |       11 |         25 |           27 |          29 |      15 |
| Learned hybrid (no rules)                               | 17.6%             | 37.5%       |       25.4 |       15 |         30 |           32 |          33 |      17 |
| Learned hybrid + rules (main)                           | 11.0%             | 32.0%       |       21   |       11 |         23 |           27 |          29 |      15 |
| Learned hybrid + rules, magazine overviews also removed | 12.5%             | 35.4%       |       23.6 |       14 |         26 |           29 |          32 |      17 |
| Learned hybrid + rules, type hidden                     | 9.4%              | 27.7%       |       17.8 |       10 |         19 |           22 |          23 |      15 |
| Indexer type ('Review')                                 | 10.0%             | 31.4%       |       21   |       10 |         27 |           25 |          27 |      16 |

'Rules' = title-only, non-paper and magazine rules (Section III-A.4); 'no rules' excludes every paper the classifier flags.
