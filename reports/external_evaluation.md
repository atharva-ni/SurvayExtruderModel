# External evaluation on Kaggle arXiv papers from unseen venues (2026-09-29)

Test set: `data/kaggle_arxiv_test.csv` — text (title, abstract) from the Kaggle arXiv snapshot, labels from the venue where each paper was published; none of the venues is used in training. Model: `./distilbert_survey_model`. CIs: 2000 bootstrap resamples. 'Prec. at 8.0% surveys' is the precision expected when surveys are as rare as in real author profiles.

## Test set — title + abstract (n = 970, 485 surveys)

| Method                       | Acc.   | Prec.   | Recall   | F1    | F1 95% CI   | ΔF1 vs learned hybrid (95% CI)   | Prec. at 8.0% surveys   |
|------------------------------|--------|---------|----------|-------|-------------|----------------------------------|-------------------------|
| Learned hybrid (ours)        | 89.7%  | 98.5%   | 80.6%    | 88.7% | 86.4%–90.7% |                                  | 85.0%                   |
| Fine-tuned DistilBERT (ours) | 92.4%  | 96.6%   | 87.8%    | 92.0% | 90.2%–93.8% | +3.3 (+1.6 to +5.0)              | 71.2%                   |
| TF-IDF + SVM                 | 88.7%  | 98.2%   | 78.8%    | 87.4% | 85.0%–89.6% | -1.3 (-3.4 to +0.6)              | 82.6%                   |
| Keyword only                 | 61.5%  | 100.0%  | 23.1%    | 37.5% | 32.7%–42.6% | -51.2 (-56.1 to -46.4)           | 100.0%                  |
| Original tool                | 63.5%  | 97.1%   | 27.8%    | 43.3% | 38.1%–47.7% | -45.4 (-50.2 to -40.7)           | 74.6%                   |
| Indexer type (Review)        | 60.8%  | 100.0%  | 21.6%    | 35.6% | 30.7%–40.4% | -53.1 (-58.3 to -47.9)           | 100.0%                  |
| Smyth & Cunningham rule      | 60.8%  | 100.0%  | 21.6%    | 35.6% | 30.7%–40.4% | -53.1 (-58.3 to -47.9)           | 100.0%                  |

## Recall on surveys per venue (title + abstract)

| Venue                                                           |   Surveys | Learned hybrid (ours)   | Fine-tuned DistilBERT (ours)   |
|-----------------------------------------------------------------|-----------|-------------------------|--------------------------------|
| Annual Reviews in Control                                       |        84 | 66.7%                   | 76.2%                          |
| Annual Review of Statistics and Its Application                 |        57 | 89.5%                   | 89.5%                          |
| Foundations and Trends in Machine Learning                      |        49 | 77.6%                   | 91.8%                          |
| Archives of Computational Methods in Engineering                |        47 | 78.7%                   | 85.1%                          |
| WIREs Data Mining and Knowledge Discovery                       |        47 | 83.0%                   | 87.2%                          |
| Annual Review of Control, Robotics, and Autonomous Systems      |        30 | 90.0%                   | 90.0%                          |
| Statistics Surveys                                              |        26 | 69.2%                   | 80.8%                          |
| IEEE Reviews in Biomedical Engineering                          |        24 | 95.8%                   | 100.0%                         |
| Foundations and Trends in Signal Processing                     |        16 | 81.2%                   | 81.2%                          |
| Foundations and Trends in Computer Graphics and Vision          |        15 | 86.7%                   | 93.3%                          |
| Foundations and Trends in Communications and Information Theory |        14 | 71.4%                   | 92.9%                          |
| Annual Review of Biomedical Data Science                        |        12 | 91.7%                   | 100.0%                         |
| Foundations and Trends in Information Retrieval                 |        11 | 100.0%                  | 100.0%                         |
| Foundations and Trends in Optimization                          |        11 | 54.5%                   | 81.8%                          |
| Foundations and Trends in Robotics                              |        10 | 90.0%                   | 100.0%                         |
| Foundations and Trends in Theoretical Computer Science          |         8 | 100.0%                  | 87.5%                          |
| Foundations and Trends in Databases                             |         4 | 75.0%                   | 100.0%                         |
| Foundations and Trends in Networking                            |         4 | 100.0%                  | 100.0%                         |
| Foundations and Trends in Privacy and Security                  |         4 | 75.0%                   | 100.0%                         |
| Foundations and Trends in Programming Languages                 |         4 | 100.0%                  | 100.0%                         |
| Foundations and Trends in Systems and Control                   |         3 | 100.0%                  | 100.0%                         |
| Foundations and Trends in Human-Computer Interaction            |         2 | 50.0%                   | 100.0%                         |
| Foundations and Trends in Web Science                           |         2 | 100.0%                  | 100.0%                         |
| Foundations and Trends in Electric Energy Systems               |         1 | 100.0%                  | 100.0%                         |

## F1 per topic group (title + abstract)

| Group   |   Papers | Learned hybrid (ours)   | Fine-tuned DistilBERT (ours)   |
|---------|----------|-------------------------|--------------------------------|
| biomed  |       72 | 97.1%                   | 97.3%                          |
| comm    |       78 | 86.1%                   | 92.1%                          |
| control |      254 | 85.6%                   | 89.3%                          |
| ml      |      376 | 88.2%                   | 93.1%                          |
| stats   |      166 | 89.6%                   | 90.6%                          |
| theory  |       24 | 100.0%                  | 95.7%                          |

## Test set — title only (n = 970, 485 surveys)

| Method                       | Acc.   | Prec.   | Recall   | F1    | F1 95% CI   | ΔF1 vs learned hybrid (95% CI)   | Prec. at 8.0% surveys   |
|------------------------------|--------|---------|----------|-------|-------------|----------------------------------|-------------------------|
| Learned hybrid (ours)        | 77.1%  | 90.7%   | 60.4%    | 72.5% | 68.8%–75.9% |                                  | 45.9%                   |
| Fine-tuned DistilBERT (ours) | 80.1%  | 90.8%   | 67.0%    | 77.1% | 73.8%–80.1% | +4.6 (+2.2 to +7.0)              | 46.1%                   |
| TF-IDF + SVM                 | 71.6%  | 83.4%   | 54.0%    | 65.6% | 61.7%–69.3% | -6.9 (-10.4 to -3.6)             | 30.5%                   |
| Keyword only                 | 61.5%  | 100.0%  | 23.1%    | 37.5% | 32.7%–42.6% | -35.0 (-39.8 to -30.2)           | 100.0%                  |
| Original tool                | 63.5%  | 97.1%   | 27.8%    | 43.3% | 38.1%–47.7% | -29.3 (-34.1 to -24.7)           | 74.6%                   |
| Indexer type (Review)        | 60.8%  | 100.0%  | 21.6%    | 35.6% | 30.7%–40.4% | -36.9 (-42.8 to -30.7)           | 100.0%                  |
| Smyth & Cunningham rule      | 60.8%  | 100.0%  | 21.6%    | 35.6% | 30.7%–40.4% | -36.9 (-42.8 to -30.7)           | 100.0%                  |
