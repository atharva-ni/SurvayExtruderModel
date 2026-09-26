# Survey classifier evaluation (2026-09-24)

Model: `./distilbert_survey_model` — DistilBERT threshold 0.95

Learned hybrid coefficients: `{'bert_logit': 0.493, 'title_terms': 2.242, 'abstract_survey_cues': 1.404, 'abstract_research_cues': -0.298, 'abstract_missing': 0.262, 'log_refs': 0.807, 'refs_missing': 2.985}`

## Table II(a) — held-out test split (n = 1925, 963 surveys)

| Method                                                   | Acc.   | Prec.   | Recall   | F1    |
|----------------------------------------------------------|--------|---------|----------|-------|
| Keyword only (title)                                     | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| TF-IDF + SVM                                             | 94.8%  | 96.5%   | 92.8%    | 94.7% |
| DistilBERT only                                          | 95.2%  | 96.9%   | 93.5%    | 95.1% |
| Hybrid (OR, paper)                                       | 95.2%  | 96.9%   | 93.5%    | 95.1% |
| Learned hybrid (title+abstract)                          | 94.9%  | 98.0%   | 91.7%    | 94.7% |
| Learned hybrid (+ reference count)                       | 95.4%  | 97.8%   | 92.9%    | 95.3% |
| Baseline: distilbert_survey_model_synthetic (OR, th=0.8) | 78.3%  | 87.9%   | 65.6%    | 75.1% |

## Table II(a2) — same test split, title only (abstract removed)

| Method                                                   | Acc.   | Prec.   | Recall   | F1    |
|----------------------------------------------------------|--------|---------|----------|-------|
| Keyword only (title)                                     | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| TF-IDF + SVM                                             | 86.4%  | 90.1%   | 81.9%    | 85.8% |
| DistilBERT only                                          | 88.6%  | 94.0%   | 82.6%    | 87.9% |
| Hybrid (OR, paper)                                       | 88.6%  | 94.0%   | 82.6%    | 87.9% |
| Learned hybrid (title+abstract)                          | 88.7%  | 94.8%   | 81.8%    | 87.8% |
| Learned hybrid (+ reference count)                       | 90.1%  | 95.0%   | 84.7%    | 89.6% |
| Baseline: distilbert_survey_model_synthetic (OR, th=0.8) | 78.5%  | 92.7%   | 62.0%    | 74.3% |

## Table II(b) — hand-labeled author-profile papers (n = 323, 117 surveys; labels: claude (unverified))

| Method                                                   | Acc.   | Prec.   | Recall   | F1    | F1 (H+M labels)   | Prec. (rw)   | Recall (rw)   | Flagged (rw)   |
|----------------------------------------------------------|--------|---------|----------|-------|-------------------|--------------|---------------|----------------|
| Keyword only (title)                                     | 74.9%  | 100.0%  | 30.8%    | 47.1% | 55.0%             | 100.0%       | 26.4%         | 1.9%           |
| TF-IDF + SVM                                             | 87.6%  | 77.3%   | 93.2%    | 84.5% | 88.3%             | 53.9%        | 94.1%         | 12.7%          |
| DistilBERT only                                          | 89.8%  | 78.0%   | 100.0%   | 87.6% | 90.5%             | 57.4%        | 100.0%        | 12.7%          |
| Hybrid (OR, paper)                                       | 89.8%  | 78.0%   | 100.0%   | 87.6% | 90.5%             | 57.4%        | 100.0%        | 12.7%          |
| Learned hybrid (title+abstract)                          | 90.4%  | 79.5%   | 99.1%    | 88.2% | 90.4%             | 73.3%        | 99.3%         | 9.9%           |
| Learned hybrid (+ reference count)                       | 90.4%  | 79.5%   | 99.1%    | 88.2% | 90.4%             | 71.8%        | 92.1%         | 9.4%           |
| Learned hybrid + magazine rule                           | 90.4%  | 87.7%   | 85.5%    | 86.6% | 92.6%             | 87.7%        | 73.3%         | 6.1%           |
| Baseline: distilbert_survey_model_synthetic (OR, th=0.8) | 74.9%  | 82.1%   | 39.3%    | 53.2% | 60.9%             | 34.4%        | 33.7%         | 7.2%           |

(rw) = reweighted by sampling stratum to the real mix of papers in the profiles; estimated true survey rate: 7.3%

## Table IV — impact of excluding detected surveys (learned hybrid; data/proauthor_s2/*.csv)

| Author     |   Papers |   Surveys |   Magazine overviews |   Non-papers | Papers excluded   | Citation reduction   | h-index   |   h-index drop | i10-index reduction   | Citation reduction (+ magazine)   |   h-index drop (+ magazine) |
|------------|----------|-----------|----------------------|--------------|-------------------|----------------------|-----------|----------------|-----------------------|-----------------------------------|-----------------------------|
| Zhu Han    |     1353 |       114 |                   21 |           17 | 8.43%             | 21.87%               | 128 → 117 |             11 | 9.26%                 | 25.20%                            |                        14   |
| L. Hanzo   |     2710 |       180 |                   23 |           10 | 6.64%             | 31.77%               | 131 → 108 |             23 | 10.27%                | 34.77%                            |                        26   |
| E. Hossain |      733 |       125 |                    9 |           28 | 17.05%            | 37.53%               | 107 → 80  |             27 | 22.51%                | 40.94%                            |                        29   |
| D. Niyato  |     2514 |       350 |                   68 |           44 | 13.92%            | 38.61%               | 143 → 114 |             29 | 18.34%                | 41.91%                            |                        32   |
| F. Yu      |      985 |        86 |                   13 |           33 | 8.73%             | 30.19%               | 101 → 86  |             15 | 11.96%                | 33.97%                            |                        17   |
| Average    |          |           |                      |              | 10.95%            | 31.99%               |           |             21 | 14.47%                | 35.36%                            |                        23.6 |

Books/editorials are not counted as surveys. '(+ magazine)' also excludes magazine articles that the classifier flagged but that do not present themselves as surveys.
