# Survey Excluder evaluation (2026-09-29)

Our models, both in `./distilbert_survey_model`: the fine-tuned DistilBERT (threshold 0.96) and the learned hybrid built on it, the main model (src/learned_hybrid.py; C = 0.1, threshold 0.73).

## Main results — our models

| Evaluation set                                                    | Model                        |   Papers | Accuracy   | Precision      | Recall          | F1    |
|-------------------------------------------------------------------|------------------------------|----------|------------|----------------|-----------------|-------|
| Held-out test split (journal papers, title + abstract)            | Learned hybrid (ours)        |     1925 | 93.6%      | 98.1%          | 88.9%           | 93.2% |
| Held-out test split (journal papers, title + abstract)            | Fine-tuned DistilBERT (ours) |     1925 | 95.3%      | 97.4%          | 93.1%           | 95.2% |
| Held-out test split, title only                                   | Learned hybrid (ours)        |     1925 | 86.9%      | 93.6%          | 79.2%           | 85.8% |
| Held-out test split, title only                                   | Fine-tuned DistilBERT (ours) |     1925 | 88.7%      | 94.5%          | 82.2%           | 88.0% |
| Author-profile papers, blind-labeled (reweighted to the real mix) | Learned hybrid (ours)        |      323 | 97.6%      | 89.7% (84–95%) | 79.8% (64–96%)  | 84.4% |
| Author-profile papers, blind-labeled (reweighted to the real mix) | Fine-tuned DistilBERT (ours) |      323 | 94.1%      | 58.4% (44–74%) | 92.2% (78–100%) | 71.5% |

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

## Our models and other methods — author-profile papers, blind-labeled (n = 323, 111 surveys; labels: first author)

| Method                       | Acc.   | Prec.   | Recall   | F1    | F1 (H+M labels)   | Prec. (rw)   | Recall (rw)   | Flagged (rw)   | Prec. (rw) 95% CI   | Recall (rw) 95% CI   |
|------------------------------|--------|---------|----------|-------|-------------------|--------------|---------------|----------------|---------------------|----------------------|
| Learned hybrid (ours)        | 94.2%  | 89.8%   | 93.7%    | 91.7% | 94.9%             | 89.7%        | 79.8%         | 7.1%           | 84–95%              | 64–96%               |
| Fine-tuned DistilBERT (ours) | 87.0%  | 73.2%   | 98.2%    | 83.8% | 87.9%             | 58.4%        | 92.2%         | 12.6%          | 44–74%              | 78–100%              |
| TF-IDF + SVM                 | 88.9%  | 76.6%   | 97.3%    | 85.7% | 88.2%             | 57.6%        | 91.5%         | 12.7%          | 44–75%              | 78–100%              |
| Keyword only                 | 76.2%  | 97.2%   | 31.5%    | 47.6% | 54.3%             | 97.2%        | 23.3%         | 1.9%           | 91–100%             | 17–33%               |
| Original tool                | 80.5%  | 87.5%   | 50.5%    | 64.0% | 68.0%             | 63.9%        | 43.8%         | 5.5%           | 43–94%              | 31–59%               |
| Indexer type (Review)        | 84.5%  | 76.1%   | 80.2%    | 78.1% | 86.5%             | 60.9%        | 59.4%         | 7.8%           | 47–78%              | 44–80%               |
| Smyth & Cunningham rule      | 81.4%  | 78.0%   | 64.0%    | 70.3% | 77.1%             | 64.3%        | 47.4%         | 5.9%           | 48–83%              | 35–65%               |
| LLM labels (Claude)          | 91.3%  | 85.5%   | 90.1%    | 87.7% | 94.7%             | 80.4%        | 73.2%         | 7.3%           | 67–92%              | 56–93%               |

(rw) = reweighted by sampling stratum to the real mix of papers in the profiles; estimated true survey rate: 8.0% (95% CI 6–11%). CIs: 2000 bootstrap resamples within each stratum. Learned hybrid: fitted on these papers, so scored by nested 10-fold cross-validation, averaged over 5 repetitions (reweighted precision per repetition: 92.6%, 83.2%, 91.1%, 89.7%, 91.9%; recall: 83.0%, 75.2%, 82.3%, 75.2%, 83.0%).

## Author-profile papers — sampling design

| Stratum      | Definition                                         |   Pool |   Sampled |   Labeled |   Surveys |   Weight | Share of est. survey rate   |
|--------------|----------------------------------------------------|--------|-----------|-----------|-----------|----------|-----------------------------|
| abstract_cue | survey phrasing in the abstract, none in the title |    120 |       120 |       117 |        63 |    1     | 3.4%                        |
| random       | everything else                                    |   1717 |       160 |       159 |         4 |   10.731 | 2.3%                        |
| title_kw     | survey term in the title                           |     48 |        48 |        47 |        44 |    1     | 2.4%                        |

Pool = profile papers with an abstract of at least 30 words, deduplicated by title, not in the training set. Weight = pool / sampled. Estimated survey rate = weighted surveys / weighted labeled papers.

## Author-profile papers — label agreement: first author (blind) vs. LLM

323 papers labeled survey or research by both: agreement 91.3%, Cohen's kappa 0.81; 28 disagreements (the first author's label is used). All papers were labeled blind by the first author.

## Survey exclusion per author — Cohort B (prolific survey authors) (data/proauthor_s2/*.csv)

| Author     |   Papers |   Surveys |   Non-papers |   In hybrid training |   Citations |   Citations* | Papers excluded   | Citation reduction   |   h-index |   h-index* |   h-index drop | h-index change   | i10-index reduction   |
|------------|----------|-----------|--------------|----------------------|-------------|--------------|-------------------|----------------------|-----------|------------|----------------|------------------|-----------------------|
| Zhu Han    |    1,353 |      91   |         17   |                  151 |     70229   |      54784   | 6.7%              | 22.0%                |       128 |        117 |             11 | -8.6%            | 7.8%                  |
| L. Hanzo   |    2,710 |     168   |         10   |                  162 |     88856   |      62656   | 6.2%              | 29.5%                |       131 |        109 |             22 | -16.8%           | 9.4%                  |
| E. Hossain |      733 |     105   |         28   |                   82 |     40090   |      26169   | 14.3%             | 34.7%                |       107 |         83 |             24 | -22.4%           | 18.5%                 |
| D. Niyato  |    2,514 |     308   |         44   |                  519 |    101777   |      64333   | 12.3%             | 36.8%                |       143 |        116 |             27 | -18.9%           | 16.2%                 |
| F. Yu      |      985 |      80   |         33   |                  116 |     40612   |      27780   | 8.1%              | 31.6%                |       101 |         85 |             16 | -15.8%           | 12.1%                 |
| Average    |    1,659 |     150.4 |         26.4 |                  206 |     68312.8 |      47144.4 | 9.5%              | 30.9%                |       122 |        102 |             20 | -16.5%           | 12.8%                 |

* Without the surveys detected by the learned hybrid. Books/editorials are not counted as surveys. 'In hybrid training': profile papers among the labeled papers the learned hybrid was fitted on.

## Survey share corrected for classification error — Cohort B (prolific survey authors)

| Author     | Papers flagged   | Est. true survey share of papers (95% CI)   | Citations flagged   | Est. true survey share of citations (95% CI)   |
|------------|------------------|---------------------------------------------|---------------------|------------------------------------------------|
| Zhu Han    | 6.7%             | 7.5% (6–9%)                                 | 22.0%               | 20.8% (19–23%)                                 |
| L. Hanzo   | 6.2%             | 6.8% (6–8%)                                 | 29.5%               | 27.0% (25–29%)                                 |
| E. Hossain | 14.3%            | 17.1% (14–21%)                              | 34.7%               | 35.8% (33–39%)                                 |
| D. Niyato  | 12.3%            | 14.5% (12–18%)                              | 36.8%               | 36.9% (35–39%)                                 |
| F. Yu      | 8.1%             | 9.3% (8–12%)                                | 31.6%               | 30.2% (28–32%)                                 |
| Average    | 9.5%             | 11.1% (9–14%)                               | 30.9%               | 30.1% (28–32%)                                 |

Adjusted classify-and-count with the learned hybrid's reweighted recall (79.8%) and false-positive rate (0.80%) from its cross-validated predictions on the 323 hand-labeled papers (`data/eval_to_label.csv`); CIs: 2000 stratified bootstrap resamples of those papers. Assumes the error rates measured on the hand-labeled papers (Cohort B's Semantic Scholar profiles, papers with abstracts) hold for every profile, and that citations do not depend on whether a paper is misclassified.

## Our models and other methods — papers each would exclude, Cohort B (prolific survey authors) (averages over authors)

| Method                       | Papers excluded   | Citations   |   Δh (avg) |   Δh han |   Δh hanzo |   Δh hossain |   Δh niyato |   Δh yu |
|------------------------------|-------------------|-------------|------------|----------|------------|--------------|-------------|---------|
| Learned hybrid (ours)        | 9.5%              | 30.9%       |       20   |       11 |         22 |           24 |          27 |      16 |
| Fine-tuned DistilBERT (ours) | 13.4%             | 36.5%       |       24.2 |       14 |         29 |           29 |          32 |      17 |
| Original tool                | 8.8%              | 22.5%       |       14.2 |        7 |         10 |           21 |          20 |      13 |
| Indexer type (Review)        | 10.0%             | 31.4%       |       21   |       10 |         27 |           25 |          27 |      16 |
| Smyth & Cunningham rule      | 5.3%              | 23.0%       |       15.2 |        8 |         19 |           16 |          20 |      13 |

Both of our models include the title-only and non-paper rules.
