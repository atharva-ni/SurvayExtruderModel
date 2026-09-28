# Survey classifier evaluation (2026-09-29)

Model: `./distilbert_survey_model` — DistilBERT threshold 0.96

Validation-fitted hybrid coefficients: `{'bert_logit': 0.493, 'title_terms': 2.242, 'abstract_survey_cues': 1.404, 'abstract_research_cues': -0.298, 'abstract_missing': 0.262, 'log_refs': 0.807, 'refs_missing': 2.985}`

Learned hybrid (src/combiner.py): C = 0.1, threshold 0.74

## Table II(a) — held-out test split (n = 1925, 963 surveys)

| Method                                       | Acc.   | Prec.   | Recall   | F1    |
|----------------------------------------------|--------|---------|----------|-------|
| Keyword only (title)                         | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| TF-IDF + SVM                                 | 94.8%  | 96.5%   | 92.8%    | 94.7% |
| DistilBERT only                              | 95.3%  | 97.4%   | 93.1%    | 95.2% |
| Hybrid (OR, paper)                           | 95.3%  | 97.4%   | 93.1%    | 95.2% |
| Validation-fitted hybrid (title+abstract)    | 94.9%  | 98.0%   | 91.7%    | 94.7% |
| Validation-fitted hybrid (+ reference count) | 95.4%  | 97.8%   | 92.9%    | 95.3% |
| Learned hybrid                               | 93.5%  | 98.4%   | 88.5%    | 93.2% |
| Indexer type ('Review')                      | 52.2%  | 100.0%  | 4.4%     | 8.4%  |

## Table II(a2) — same test split, title only (abstract removed)

| Method                                       | Acc.   | Prec.   | Recall   | F1    |
|----------------------------------------------|--------|---------|----------|-------|
| Keyword only (title)                         | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| TF-IDF + SVM                                 | 86.4%  | 90.1%   | 81.9%    | 85.8% |
| DistilBERT only                              | 88.7%  | 94.5%   | 82.2%    | 88.0% |
| Hybrid (OR, paper)                           | 88.7%  | 94.5%   | 82.2%    | 88.0% |
| Validation-fitted hybrid (title+abstract)    | 88.7%  | 94.8%   | 81.8%    | 87.8% |
| Validation-fitted hybrid (+ reference count) | 90.1%  | 95.0%   | 84.7%    | 89.6% |
| Learned hybrid                               | 86.6%  | 94.1%   | 78.2%    | 85.4% |
| Indexer type ('Review')                      | 52.2%  | 100.0%  | 4.4%     | 8.4%  |

## Table IV — impact of excluding detected surveys, survey authors (learned hybrid + rules; data/cohort_openalex/survey/*.csv)

| Author           |   Papers |   Surveys |   Non-papers |   In combiner training |   Citations |   Citations* | Papers excluded   | Citation reduction   |   h-index |   h-index* |   h-index drop | h-index change   | i10-index reduction   |
|------------------|----------|-----------|--------------|------------------------|-------------|--------------|-------------------|----------------------|-----------|------------|----------------|------------------|-----------------------|
| Zhu Han          |   2552   |     173   |        246   |                  278   |     97673   |      75830   | 6.8%              | 22.4%                |     142   |      128   |           14   | -9.9%            | 9.0%                  |
| Lajos Hanzo      |   2113   |     198   |         98   |                  115   |     84963   |      53769   | 9.4%              | 36.7%                |     127   |      101   |           26   | -20.5%           | 11.0%                 |
| Ekram Hossain    |    809   |     120   |        144   |                   81   |     39279   |      25971   | 14.8%             | 33.9%                |     104   |       82   |           22   | -21.2%           | 19.7%                 |
| Dusit Tao Niyato |   2491   |     281   |        195   |                  366   |    101886   |      64322   | 11.3%             | 36.9%                |     143   |      118   |           25   | -17.5%           | 14.3%                 |
| Fei Richard Yu   |   1327   |      89   |         68   |                  119   |     56118   |      39712   | 6.7%              | 29.2%                |     118   |       99   |           19   | -16.1%           | 10.7%                 |
| Average          |   1858.4 |     172.2 |        150.2 |                  191.8 |     75983.8 |      51920.8 | 9.8%              | 31.8%                |     126.8 |      105.6 |           21.2 | -17.0%           | 12.9%                 |

* Excluding papers categorized as surveys. Books/editorials are not counted as surveys. 'In combiner training': profile papers among the labeled papers the learned hybrid was fitted on.

## Table IV(b) — survey authors, papers excluded by each method (averages over authors)

| Method                                                                | Papers excluded   | Citations   |   Δh (avg) |   Δh han |   Δh hanzo |   Δh hossain |   Δh niyato |   Δh yu |
|-----------------------------------------------------------------------|-------------------|-------------|------------|----------|------------|--------------|-------------|---------|
| DistilBERT, no rules                                                  | 19.0%             | 38.9%       |       27.6 |       19 |         37 |           29 |          31 |      22 |
| DistilBERT + rules                                                    | 13.5%             | 36.1%       |       25.8 |       18 |         33 |           26 |          30 |      22 |
| Learned hybrid, no rules                                              | 13.5%             | 33.6%       |       22.8 |       14 |         29 |           25 |          26 |      20 |
| Learned hybrid + rules (main)                                         | 9.8%              | 31.8%       |       21.2 |       14 |         26 |           22 |          25 |      19 |
| same, type hidden                                                     | 12.0%             | 33.5%       |       22.8 |       14 |         29 |           25 |          26 |      20 |
| Validation-fitted hybrid + rules incl. magazine rule (earlier system) | 9.8%              | 28.7%       |       18.6 |       13 |         21 |           19 |          22 |      18 |
| Indexer type ('Review')                                               | 0.0%              | 0.0%        |        0.2 |        1 |          0 |            0 |           0 |       0 |

'Rules' = title-only and non-paper rules; 'no rules' excludes every paper the classifier flags.

## Table IV — impact of excluding detected surveys, comparison cohort (learned hybrid + rules; data/cohort_openalex/comparison/*.csv)

| Author               |   Papers |   Surveys |   Non-papers |   In combiner training |   Citations |   Citations* | Papers excluded   | Citation reduction   |   h-index |   h-index* |   h-index drop | h-index change   | i10-index reduction   |
|----------------------|----------|-----------|--------------|------------------------|-------------|--------------|-------------------|----------------------|-----------|------------|----------------|------------------|-----------------------|
| Zhiguo Ding          |   1053   |      60   |         49   |                   92   |     66403   |       48,441 | 5.7%              | 27.0%                |       118 |      108   |           10   | -8.5%            | 6.8%                  |
| Robert W. Heath      |   1047   |      65   |         58   |                   85   |     80025   |       63,928 | 6.2%              | 20.1%                |       133 |      125   |            8   | -6.0%            | 6.2%                  |
| Shi Hong Jin         |   1247   |      56   |         12   |                   89   |     46427   |       40,815 | 4.5%              | 12.1%                |        99 |       91   |            8   | -8.1%            | 6.7%                  |
| Ying‐Chang Liang     |    908   |      52   |         28   |                   70   |     54512   |       41,423 | 5.7%              | 24.0%                |       111 |      101   |           10   | -9.0%            | 6.6%                  |
| Derrick Wing Kwan Ng |    771   |      33   |         31   |                   51   |     41044   |       33,433 | 4.3%              | 18.5%                |       104 |       96   |            8   | -7.7%            | 6.0%                  |
| Average              |   1005.2 |      53.2 |         35.6 |                   77.4 |     57682.2 |       45,608 | 5.3%              | 20.4%                |       113 |      104.2 |            8.8 | -7.9%            | 6.5%                  |

* Excluding papers categorized as surveys. Books/editorials are not counted as surveys. 'In combiner training': profile papers among the labeled papers the learned hybrid was fitted on.

## Table IV(b) — comparison cohort, papers excluded by each method (averages over authors)

| Method                                                                | Papers excluded   | Citations   |   Δh (avg) |   Δh ding |   Δh heath |   Δh jin |   Δh liang |   Δh ng |
|-----------------------------------------------------------------------|-------------------|-------------|------------|-----------|------------|----------|------------|---------|
| DistilBERT, no rules                                                  | 9.4%              | 22.6%       |       11   |        13 |         11 |        8 |         13 |      10 |
| DistilBERT + rules                                                    | 7.5%              | 22.2%       |       10.8 |        13 |         10 |        8 |         13 |      10 |
| Learned hybrid, no rules                                              | 6.6%              | 20.7%       |        9.2 |        10 |         10 |        8 |         10 |       8 |
| Learned hybrid + rules (main)                                         | 5.3%              | 20.4%       |        8.8 |        10 |          8 |        8 |         10 |       8 |
| same, type hidden                                                     | 5.9%              | 20.6%       |        9.2 |        10 |         10 |        8 |         10 |       8 |
| Validation-fitted hybrid + rules incl. magazine rule (earlier system) | 4.9%              | 15.9%       |        6.6 |         6 |          3 |        6 |         11 |       7 |
| Indexer type ('Review')                                               | 0.0%              | 0.0%        |        0   |         0 |          0 |        0 |          0 |       0 |

'Rules' = title-only and non-paper rules; 'no rules' excludes every paper the classifier flags.

## Table IV(c) — survey authors vs. comparison cohort (averages over authors)

| Method                                                                | Papers: survey authors   | Papers: comparison   | Citations: survey authors   | Citations: comparison   |   Δh: survey authors |   Δh: comparison |
|-----------------------------------------------------------------------|--------------------------|----------------------|-----------------------------|-------------------------|----------------------|------------------|
| DistilBERT, no rules                                                  | 19.0%                    | 9.4%                 | 38.9%                       | 22.6%                   |                 27.6 |             11   |
| DistilBERT + rules                                                    | 13.5%                    | 7.5%                 | 36.1%                       | 22.2%                   |                 25.8 |             10.8 |
| Learned hybrid, no rules                                              | 13.5%                    | 6.6%                 | 33.6%                       | 20.7%                   |                 22.8 |              9.2 |
| Learned hybrid + rules (main)                                         | 9.8%                     | 5.3%                 | 31.8%                       | 20.4%                   |                 21.2 |              8.8 |
| same, type hidden                                                     | 12.0%                    | 5.9%                 | 33.5%                       | 20.6%                   |                 22.8 |              9.2 |
| Validation-fitted hybrid + rules incl. magazine rule (earlier system) | 9.8%                     | 4.9%                 | 28.7%                       | 15.9%                   |                 18.6 |              6.6 |
| Indexer type ('Review')                                               | 0.0%                     | 0.0%                 | 0.0%                        | 0.0%                    |                  0.2 |              0   |
