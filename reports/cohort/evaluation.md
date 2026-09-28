# Survey classifier evaluation (2026-09-28)

Model: `./distilbert_survey_model` — DistilBERT threshold 0.96

Learned hybrid coefficients: `{'bert_logit': 0.493, 'title_terms': 2.242, 'abstract_survey_cues': 1.404, 'abstract_research_cues': -0.298, 'abstract_missing': 0.262, 'log_refs': 0.807, 'refs_missing': 2.985}`

## Table II(a) — held-out test split (n = 1925, 963 surveys)

| Method                             | Acc.   | Prec.   | Recall   | F1    |
|------------------------------------|--------|---------|----------|-------|
| Keyword only (title)               | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| TF-IDF + SVM                       | 94.8%  | 96.5%   | 92.8%    | 94.7% |
| DistilBERT only                    | 95.3%  | 97.4%   | 93.1%    | 95.2% |
| Hybrid (OR, paper)                 | 95.3%  | 97.4%   | 93.1%    | 95.2% |
| Learned hybrid (title+abstract)    | 94.9%  | 98.0%   | 91.7%    | 94.7% |
| Learned hybrid (+ reference count) | 95.4%  | 97.8%   | 92.9%    | 95.3% |
| Indexer type ('Review')            | 52.2%  | 100.0%  | 4.4%     | 8.4%  |

## Table II(a2) — same test split, title only (abstract removed)

| Method                             | Acc.   | Prec.   | Recall   | F1    |
|------------------------------------|--------|---------|----------|-------|
| Keyword only (title)               | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| TF-IDF + SVM                       | 86.4%  | 90.1%   | 81.9%    | 85.8% |
| DistilBERT only                    | 88.7%  | 94.5%   | 82.2%    | 88.0% |
| Hybrid (OR, paper)                 | 88.7%  | 94.5%   | 82.2%    | 88.0% |
| Learned hybrid (title+abstract)    | 88.7%  | 94.8%   | 81.8%    | 87.8% |
| Learned hybrid (+ reference count) | 90.1%  | 95.0%   | 84.7%    | 89.6% |
| Indexer type ('Review')            | 52.2%  | 100.0%  | 4.4%     | 8.4%  |

## Table IV — impact of excluding detected surveys, survey authors (learned hybrid; data/cohort_openalex/survey/*.csv)

| Author           |   Papers |   Surveys |   Magazine overviews |   Non-papers | Papers excluded   | Citation reduction   | h-index   |   h-index drop | i10-index reduction   | Citation reduction (+ magazine)   |   h-index drop (+ magazine) |
|------------------|----------|-----------|----------------------|--------------|-------------------|----------------------|-----------|----------------|-----------------------|-----------------------------------|-----------------------------|
| Zhu Han          |     2552 |       157 |                   43 |          246 | 6.15%             | 18.94%               | 142 → 129 |           13   | 7.94%                 | 23.50%                            |                        14   |
| Lajos Hanzo      |     2113 |       213 |                   28 |           98 | 10.08%            | 32.64%               | 127 → 106 |           21   | 10.77%                | 39.54%                            |                        28   |
| Ekram Hossain    |      809 |       119 |                   15 |          144 | 14.71%            | 28.52%               | 104 → 85  |           19   | 17.94%                | 36.17%                            |                        24   |
| Dusit Tao Niyato |     2491 |       265 |                   62 |          195 | 10.64%            | 34.56%               | 143 → 121 |           22   | 12.63%                | 39.39%                            |                        27   |
| Fei Richard Yu   |     1327 |        99 |                    9 |           68 | 7.46%             | 28.89%               | 118 → 100 |           18   | 11.11%                | 31.38%                            |                        21   |
| Average          |          |           |                      |              | 9.81%             | 28.71%               |           |           18.6 | 12.08%                | 34.00%                            |                        22.8 |

Books/editorials are not counted as surveys. '(+ magazine)' also excludes magazine articles that the classifier flagged but that do not present themselves as surveys.

## Table IV(b) — survey authors, papers excluded by each method (averages over authors)

| Method                                                  | Papers excluded   | Citations   |   Δh (avg) |   Δh han |   Δh hanzo |   Δh hossain |   Δh niyato |   Δh yu |
|---------------------------------------------------------|-------------------|-------------|------------|----------|------------|--------------|-------------|---------|
| DistilBERT only (no rules)                              | 19.0%             | 38.9%       |       27.6 |       19 |         37 |           29 |          31 |      22 |
| DistilBERT + rules                                      | 10.8%             | 29.2%       |       18.8 |       14 |         21 |           19 |          22 |      18 |
| Learned hybrid (no rules)                               | 16.1%             | 36.6%       |       25   |       17 |         32 |           26 |          28 |      22 |
| Learned hybrid + rules (main)                           | 9.8%              | 28.7%       |       18.6 |       13 |         21 |           19 |          22 |      18 |
| Learned hybrid + rules, magazine overviews also removed | 11.4%             | 34.0%       |       22.8 |       14 |         28 |           24 |          27 |      21 |
| Learned hybrid + rules, type hidden                     | 12.8%             | 31.1%       |       20.4 |       14 |         24 |           22 |          23 |      19 |
| Indexer type ('Review')                                 | 0.0%              | 0.0%        |        0.2 |        1 |          0 |            0 |           0 |       0 |

'Rules' = title-only, non-paper and magazine rules (Section III-A.4); 'no rules' excludes every paper the classifier flags.

## Table IV — impact of excluding detected surveys, comparison cohort (learned hybrid; data/cohort_openalex/comparison/*.csv)

| Author               |   Papers |   Surveys |   Magazine overviews |   Non-papers | Papers excluded   | Citation reduction   | h-index   |   h-index drop | i10-index reduction   | Citation reduction (+ magazine)   |   h-index drop (+ magazine) |
|----------------------|----------|-----------|----------------------|--------------|-------------------|----------------------|-----------|----------------|-----------------------|-----------------------------------|-----------------------------|
| Zhiguo Ding          |     1053 |        51 |                   14 |           49 | 4.84%             | 17.20%               | 118 → 112 |            6   | 4.93%                 | 23.27%                            |                        10   |
| Robert W. Heath      |     1047 |        44 |                   20 |           58 | 4.20%             | 10.87%               | 133 → 130 |            3   | 2.91%                 | 19.42%                            |                         8   |
| Shi Hong Jin         |     1247 |        60 |                    9 |           12 | 4.81%             | 10.08%               | 99 → 93   |            6   | 5.99%                 | 11.99%                            |                         8   |
| Ying‐Chang Liang     |      908 |        56 |                    5 |           28 | 6.17%             | 23.91%               | 111 → 100 |           11   | 7.91%                 | 25.77%                            |                        12   |
| Derrick Wing Kwan Ng |      771 |        34 |                    7 |           31 | 4.41%             | 17.57%               | 104 → 97  |            7   | 6.73%                 | 19.86%                            |                         9   |
| Average              |          |           |                      |              | 4.89%             | 15.92%               |           |            6.6 | 5.69%                 | 20.06%                            |                         9.4 |

Books/editorials are not counted as surveys. '(+ magazine)' also excludes magazine articles that the classifier flagged but that do not present themselves as surveys.

## Table IV(b) — comparison cohort, papers excluded by each method (averages over authors)

| Method                                                  | Papers excluded   | Citations   |   Δh (avg) |   Δh ding |   Δh heath |   Δh jin |   Δh liang |   Δh ng |
|---------------------------------------------------------|-------------------|-------------|------------|-----------|------------|----------|------------|---------|
| DistilBERT only (no rules)                              | 9.4%              | 22.6%       |       11   |        13 |         11 |        8 |         13 |      10 |
| DistilBERT + rules                                      | 5.6%              | 16.3%       |        7   |         6 |          4 |        6 |         11 |       8 |
| Learned hybrid (no rules)                               | 7.3%              | 20.3%       |        9.4 |        10 |          8 |        8 |         12 |       9 |
| Learned hybrid + rules (main)                           | 4.9%              | 15.9%       |        6.6 |         6 |          3 |        6 |         11 |       7 |
| Learned hybrid + rules, magazine overviews also removed | 6.0%              | 20.1%       |        9.4 |        10 |          8 |        8 |         12 |       9 |
| Learned hybrid + rules, type hidden                     | 5.3%              | 16.1%       |        6.6 |         6 |          3 |        6 |         11 |       7 |
| Indexer type ('Review')                                 | 0.0%              | 0.0%        |        0   |         0 |          0 |        0 |          0 |       0 |

'Rules' = title-only, non-paper and magazine rules (Section III-A.4); 'no rules' excludes every paper the classifier flags.

## Table IV(c) — survey authors vs. comparison cohort (averages over authors)

| Method                                                  | Papers: survey authors   | Papers: comparison   | Citations: survey authors   | Citations: comparison   |   Δh: survey authors |   Δh: comparison |
|---------------------------------------------------------|--------------------------|----------------------|-----------------------------|-------------------------|----------------------|------------------|
| DistilBERT only (no rules)                              | 19.0%                    | 9.4%                 | 38.9%                       | 22.6%                   |                 27.6 |             11   |
| DistilBERT + rules                                      | 10.8%                    | 5.6%                 | 29.2%                       | 16.3%                   |                 18.8 |              7   |
| Learned hybrid (no rules)                               | 16.1%                    | 7.3%                 | 36.6%                       | 20.3%                   |                 25   |              9.4 |
| Learned hybrid + rules (main)                           | 9.8%                     | 4.9%                 | 28.7%                       | 15.9%                   |                 18.6 |              6.6 |
| Learned hybrid + rules, magazine overviews also removed | 11.4%                    | 6.0%                 | 34.0%                       | 20.1%                   |                 22.8 |              9.4 |
| Learned hybrid + rules, type hidden                     | 12.8%                    | 5.3%                 | 31.1%                       | 16.1%                   |                 20.4 |              6.6 |
| Indexer type ('Review')                                 | 0.0%                     | 0.0%                 | 0.0%                        | 0.0%                    |                  0.2 |              0   |
