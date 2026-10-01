# Survey Excluder evaluation (2026-09-29)

Our models, both in `./distilbert_survey_model`: the fine-tuned DistilBERT (threshold 0.96) and the learned hybrid built on it, the main model (src/learned_hybrid.py; C = 0.1, threshold 0.73).

## Main results — our models

| Evaluation set                                         | Model                        |   Papers | Accuracy   | Precision   | Recall   | F1    |
|--------------------------------------------------------|------------------------------|----------|------------|-------------|----------|-------|
| Held-out test split (journal papers, title + abstract) | Learned hybrid (ours)        |     1925 | 93.6%      | 98.1%       | 88.9%    | 93.2% |
| Held-out test split (journal papers, title + abstract) | Fine-tuned DistilBERT (ours) |     1925 | 95.3%      | 97.4%       | 93.1%    | 95.2% |
| Held-out test split, title only                        | Learned hybrid (ours)        |     1925 | 86.9%      | 93.6%       | 79.2%    | 85.8% |
| Held-out test split, title only                        | Fine-tuned DistilBERT (ours) |     1925 | 88.7%      | 94.5%       | 82.2%    | 88.0% |

Survey = positive class. The learned hybrid is the main model. Author-profile papers: the learned hybrid is fitted on these papers, so it is scored by nested cross-validation; 95% CIs from stratified bootstrap. Papers from venues not used in training: reports/external_evaluation.md.

## Our models and other methods — held-out test split (n = 1925, 963 surveys)

| Method                       | Acc.   | Prec.   | Recall   | F1    |
|------------------------------|--------|---------|----------|-------|
| Learned hybrid (ours)        | 93.6%  | 98.1%   | 88.9%    | 93.2% |
| Fine-tuned DistilBERT (ours) | 95.3%  | 97.4%   | 93.1%    | 95.2% |
| TF-IDF + SVM                 | 94.8%  | 96.5%   | 92.8%    | 94.7% |
| Keyword only                 | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| Original tool                | 80.3%  | 95.6%   | 63.4%    | 76.3% |
| Indexer type (Review)        | 52.2%  | 100.0%  | 4.4%     | 8.4%  |
| Smyth & Cunningham rule      | 52.2%  | 100.0%  | 4.4%     | 8.4%  |

## Our models and other methods — held-out test split, title only (n = 1925, 963 surveys)

| Method                       | Acc.   | Prec.   | Recall   | F1    |
|------------------------------|--------|---------|----------|-------|
| Learned hybrid (ours)        | 86.9%  | 93.6%   | 79.2%    | 85.8% |
| Fine-tuned DistilBERT (ours) | 88.7%  | 94.5%   | 82.2%    | 88.0% |
| TF-IDF + SVM                 | 86.4%  | 90.1%   | 81.9%    | 85.8% |
| Keyword only                 | 78.7%  | 100.0%  | 57.4%    | 73.0% |
| Original tool                | 80.3%  | 95.6%   | 63.4%    | 76.3% |
| Indexer type (Review)        | 52.2%  | 100.0%  | 4.4%     | 8.4%  |
| Smyth & Cunningham rule      | 52.2%  | 100.0%  | 4.4%     | 8.4%  |

## Survey exclusion per author — Cohort B (prolific survey authors) (data/cohort_openalex/survey/*.csv)

| Author           |   Papers |   Surveys |   Non-papers |   In hybrid training |   Citations |   Citations* | Papers excluded   | Citation reduction   |   h-index |   h-index* |   h-index drop | h-index change   | i10-index reduction   |
|------------------|----------|-----------|--------------|----------------------|-------------|--------------|-------------------|----------------------|-----------|------------|----------------|------------------|-----------------------|
| Zhu Han          |   2552   |     174   |        246   |                367   |     97673   |      75510   | 6.8%              | 22.7%                |     142   |      128   |             14 | -9.9%            | 9.3%                  |
| Lajos Hanzo      |   2113   |     207   |         98   |                152   |     84963   |      53915   | 9.8%              | 36.5%                |     127   |      102   |             25 | -19.7%           | 11.3%                 |
| Ekram Hossain    |    809   |     122   |        144   |                104   |     39279   |      25964   | 15.1%             | 33.9%                |     104   |       82   |             22 | -21.2%           | 19.7%                 |
| Dusit Tao Niyato |   2491   |     293   |        195   |                499   |    101886   |      64053   | 11.8%             | 37.1%                |     143   |      118   |             25 | -17.5%           | 14.6%                 |
| Fei Richard Yu   |   1327   |      90   |         68   |                171   |     56118   |      39627   | 6.8%              | 29.4%                |     118   |       99   |             19 | -16.1%           | 10.8%                 |
| Average          |   1858.4 |     177.2 |        150.2 |                258.6 |     75983.8 |      51813.8 | 10.0%             | 31.9%                |     126.8 |      105.8 |             21 | -16.9%           | 13.1%                 |

* Without the surveys detected by the learned hybrid. Books/editorials are not counted as surveys. 'In hybrid training': profile papers among the labeled papers the learned hybrid was fitted on.

## Survey share corrected for classification error — Cohort B (prolific survey authors)

| Author           | Papers flagged   | Est. true survey share of papers (95% CI)   | Citations flagged   | Est. true survey share of citations (95% CI)   |
|------------------|------------------|---------------------------------------------|---------------------|------------------------------------------------|
| Zhu Han          | 6.8%             | 7.6% (6–9%)                                 | 22.7%               | 21.5% (20–23%)                                 |
| Lajos Hanzo      | 9.8%             | 11.4% (9–14%)                               | 36.5%               | 35.5% (33–38%)                                 |
| Ekram Hossain    | 15.1%            | 18.1% (15–22%)                              | 33.9%               | 35.3% (33–39%)                                 |
| Dusit Tao Niyato | 11.8%            | 13.9% (11–17%)                              | 37.1%               | 37.0% (35–39%)                                 |
| Fei Richard Yu   | 6.8%             | 7.6% (6–9%)                                 | 29.4%               | 27.3% (25–29%)                                 |
| Average          | 10.0%            | 11.7% (10–15%)                              | 31.9%               | 31.3% (29–34%)                                 |

Adjusted classify-and-count with the learned hybrid's reweighted recall (79.8%) and false-positive rate (0.80%) from its cross-validated predictions on the 323 hand-labeled papers (`data/eval_to_label.csv`); CIs: 2000 stratified bootstrap resamples of those papers. Assumes the error rates measured on the hand-labeled papers (Cohort B's Semantic Scholar profiles, papers with abstracts) hold for every profile, and that citations do not depend on whether a paper is misclassified.

## Our models and other methods — papers each would exclude, Cohort B (prolific survey authors) (averages over authors)

| Method                       | Papers excluded   | Citations   |   Δh (avg) |   Δh han |   Δh hanzo |   Δh hossain |   Δh niyato |   Δh yu |
|------------------------------|-------------------|-------------|------------|----------|------------|--------------|-------------|---------|
| Learned hybrid (ours)        | 10.0%             | 31.9%       |       21   |       14 |         25 |           22 |          25 |      19 |
| Fine-tuned DistilBERT (ours) | 13.5%             | 36.1%       |       25.8 |       18 |         33 |           26 |          30 |      22 |
| Original tool                | 9.1%              | 23.1%       |       13.8 |       10 |          9 |           17 |          18 |      15 |
| Indexer type (Review)        | 0.0%              | 0.0%        |        0.2 |        1 |          0 |            0 |           0 |       0 |
| Smyth & Cunningham rule      | 0.0%              | 0.0%        |        0.2 |        1 |          0 |            0 |           0 |       0 |

Both of our models include the title-only and non-paper rules.

## Survey exclusion per author — Cohort A (comparison authors) (data/cohort_openalex/comparison/*.csv)

| Author               |   Papers |   Surveys |   Non-papers |   In hybrid training |   Citations |   Citations* | Papers excluded   | Citation reduction   |   h-index |   h-index* |   h-index drop | h-index change   | i10-index reduction   |
|----------------------|----------|-----------|--------------|----------------------|-------------|--------------|-------------------|----------------------|-----------|------------|----------------|------------------|-----------------------|
| Zhiguo Ding          |   1053   |      60   |         49   |                126   |     66403   |      48252   | 5.7%              | 27.3%                |       118 |        108 |             10 | -8.5%            | 7.2%                  |
| Robert W. Heath      |   1047   |      67   |         58   |                130   |     80025   |      63769   | 6.4%              | 20.3%                |       133 |        124 |              9 | -6.8%            | 6.3%                  |
| Shi Hong Jin         |   1247   |      58   |         12   |                136   |     46427   |      40744   | 4.7%              | 12.2%                |        99 |         91 |              8 | -8.1%            | 6.9%                  |
| Ying‐Chang Liang     |    908   |      54   |         28   |                101   |     54512   |      41364   | 5.9%              | 24.1%                |       111 |        101 |             10 | -9.0%            | 6.6%                  |
| Derrick Wing Kwan Ng |    771   |      34   |         31   |                 73   |     41044   |      33419   | 4.4%              | 18.6%                |       104 |         96 |              8 | -7.7%            | 6.2%                  |
| Average              |   1005.2 |      54.6 |         35.6 |                113.2 |     57682.2 |      45509.6 | 5.4%              | 20.5%                |       113 |        104 |              9 | -8.0%            | 6.6%                  |

* Without the surveys detected by the learned hybrid. Books/editorials are not counted as surveys. 'In hybrid training': profile papers among the labeled papers the learned hybrid was fitted on.

## Survey share corrected for classification error — Cohort A (comparison authors)

| Author               | Papers flagged   | Est. true survey share of papers (95% CI)   | Citations flagged   | Est. true survey share of citations (95% CI)   |
|----------------------|------------------|---------------------------------------------|---------------------|------------------------------------------------|
| Zhiguo Ding          | 5.7%             | 6.2% (5–8%)                                 | 27.3%               | 24.7% (22–27%)                                 |
| Robert W. Heath      | 6.4%             | 7.1% (6–9%)                                 | 20.3%               | 19.2% (17–21%)                                 |
| Shi Hong Jin         | 4.7%             | 4.9% (4–6%)                                 | 12.2%               | 11.1% (10–13%)                                 |
| Ying‐Chang Liang     | 5.9%             | 6.5% (5–8%)                                 | 24.1%               | 22.2% (20–24%)                                 |
| Derrick Wing Kwan Ng | 4.4%             | 4.6% (4–6%)                                 | 18.6%               | 16.2% (14–18%)                                 |
| Average              | 5.4%             | 5.9% (5–7%)                                 | 20.5%               | 18.7% (17–20%)                                 |

Adjusted classify-and-count with the learned hybrid's reweighted recall (79.8%) and false-positive rate (0.80%) from its cross-validated predictions on the 323 hand-labeled papers (`data/eval_to_label.csv`); CIs: 2000 stratified bootstrap resamples of those papers. Assumes the error rates measured on the hand-labeled papers (Cohort B's Semantic Scholar profiles, papers with abstracts) hold for every profile, and that citations do not depend on whether a paper is misclassified.

## Our models and other methods — papers each would exclude, Cohort A (comparison authors) (averages over authors)

| Method                       | Papers excluded   | Citations   |   Δh (avg) |   Δh ding |   Δh heath |   Δh jin |   Δh liang |   Δh ng |
|------------------------------|-------------------|-------------|------------|-----------|------------|----------|------------|---------|
| Learned hybrid (ours)        | 5.4%              | 20.5%       |        9   |        10 |          9 |        8 |         10 |       8 |
| Fine-tuned DistilBERT (ours) | 7.5%              | 22.2%       |       10.8 |        13 |         10 |        8 |         13 |      10 |
| Original tool                | 4.5%              | 9.8%        |        4.2 |         6 |          2 |        3 |          7 |       3 |
| Indexer type (Review)        | 0.0%              | 0.0%        |        0   |         0 |          0 |        0 |          0 |       0 |
| Smyth & Cunningham rule      | 0.0%              | 0.0%        |        0   |         0 |          0 |        0 |          0 |       0 |

Both of our models include the title-only and non-paper rules.

## Cohort B vs. Cohort A (averages over authors)

| Method                       | Papers: Cohort B   | Papers: Cohort A   | Citations: Cohort B   | Citations: Cohort A   |   Δh: Cohort B |   Δh: Cohort A |
|------------------------------|--------------------|--------------------|-----------------------|-----------------------|----------------|----------------|
| Learned hybrid (ours)        | 10.0%              | 5.4%               | 31.9%                 | 20.5%                 |           21   |            9   |
| Fine-tuned DistilBERT (ours) | 13.5%              | 7.5%               | 36.1%                 | 22.2%                 |           25.8 |           10.8 |
| Original tool                | 9.1%               | 4.5%               | 23.1%                 | 9.8%                  |           13.8 |            4.2 |
| Indexer type (Review)        | 0.0%               | 0.0%               | 0.0%                  | 0.0%                  |            0.2 |            0   |
| Smyth & Cunningham rule      | 0.0%               | 0.0%               | 0.0%                  | 0.0%                  |            0.2 |            0   |

Corrected for classification error: surveys are 11.7% (10–15%) vs. 5.9% (5–7%) of papers and 31.3% (29–34%) vs. 18.7% (17–20%) of citations. The two cohorts' intervals share the same error-rate uncertainty, so they are not independent.
