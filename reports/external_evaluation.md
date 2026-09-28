# External evaluation on Kaggle arXiv papers from unseen venues (2026-09-29)

Test set: `data/kaggle_arxiv_test.csv` — text (title, abstract) from the Kaggle arXiv snapshot, labels from the venue where each paper was published; none of the venues is used in training. Model: `./distilbert_survey_model`. CIs: 2000 bootstrap resamples. 'Prec. at 8.0% surveys' is the precision expected when surveys are as rare as in real author profiles.

## Test set — title + abstract (n = 970, 485 surveys)

| Method                                                        | Acc.   | Prec.   | Recall   | F1    | F1 95% CI   | ΔF1 vs DistilBERT (95% CI)   | Prec. at 8.0% surveys   |
|---------------------------------------------------------------|--------|---------|----------|-------|-------------|------------------------------|-------------------------|
| Keyword only (title)                                          | 61.5%  | 100.0%  | 23.1%    | 37.5% | 32.7%–42.6% | -54.5 (-59.5 to -49.5)       | 100.0%                  |
| TF-IDF + SVM                                                  | 88.7%  | 98.2%   | 78.8%    | 87.4% | 85.0%–89.6% | -4.6 (-6.7 to -2.6)          | 82.6%                   |
| DistilBERT only                                               | 92.4%  | 96.6%   | 87.8%    | 92.0% | 90.2%–93.8% |                              | 71.2%                   |
| Hybrid (OR, paper)                                            | 92.5%  | 96.6%   | 88.0%    | 92.1% | 90.4%–93.9% | +0.1 (+0.0 to +0.4)          | 71.2%                   |
| Validation-fitted hybrid (title+abstract)                     | 91.9%  | 97.2%   | 86.2%    | 91.4% | 89.5%–93.2% | -0.6 (-1.5 to +0.2)          | 75.2%                   |
| Validation-fitted hybrid (+ reference count)                  | 92.2%  | 96.6%   | 87.4%    | 91.8% | 89.9%–93.5% | -0.2 (-1.2 to +0.7)          | 71.1%                   |
| Learned hybrid                                                | 88.4%  | 98.7%   | 77.7%    | 87.0% | 84.6%–89.3% | -5.0 (-6.9 to -3.1)          | 86.8%                   |
| Indexer type ('Review')                                       | 60.8%  | 100.0%  | 21.6%    | 35.6% | 30.7%–40.4% | -56.4 (-61.6 to -51.3)       | 100.0%                  |
| Earlier model: distilbert_survey_model_synthetic (OR, th=0.8) | 69.3%  | 87.0%   | 45.4%    | 59.6% | 55.4%–63.6% | -32.4 (-36.6 to -28.0)       | 36.7%                   |

## Recall on surveys per venue (title + abstract)

| Venue                                                           |   Surveys | DistilBERT only   | Validation-fitted hybrid (+ reference count)   | Learned hybrid   |
|-----------------------------------------------------------------|-----------|-------------------|------------------------------------------------|------------------|
| Annual Reviews in Control                                       |        84 | 76.2%             | 76.2%                                          | 64.3%            |
| Annual Review of Statistics and Its Application                 |        57 | 89.5%             | 89.5%                                          | 91.2%            |
| Foundations and Trends in Machine Learning                      |        49 | 91.8%             | 89.8%                                          | 75.5%            |
| Archives of Computational Methods in Engineering                |        47 | 85.1%             | 83.0%                                          | 76.6%            |
| WIREs Data Mining and Knowledge Discovery                       |        47 | 87.2%             | 87.2%                                          | 78.7%            |
| Annual Review of Control, Robotics, and Autonomous Systems      |        30 | 90.0%             | 93.3%                                          | 90.0%            |
| Statistics Surveys                                              |        26 | 80.8%             | 84.6%                                          | 65.4%            |
| IEEE Reviews in Biomedical Engineering                          |        24 | 100.0%            | 100.0%                                         | 95.8%            |
| Foundations and Trends in Signal Processing                     |        16 | 81.2%             | 81.2%                                          | 75.0%            |
| Foundations and Trends in Computer Graphics and Vision          |        15 | 93.3%             | 93.3%                                          | 80.0%            |
| Foundations and Trends in Communications and Information Theory |        14 | 92.9%             | 92.9%                                          | 50.0%            |
| Annual Review of Biomedical Data Science                        |        12 | 100.0%            | 100.0%                                         | 91.7%            |
| Foundations and Trends in Information Retrieval                 |        11 | 100.0%            | 100.0%                                         | 100.0%           |
| Foundations and Trends in Optimization                          |        11 | 81.8%             | 72.7%                                          | 45.5%            |
| Foundations and Trends in Robotics                              |        10 | 100.0%            | 100.0%                                         | 80.0%            |
| Foundations and Trends in Theoretical Computer Science          |         8 | 87.5%             | 87.5%                                          | 87.5%            |
| Foundations and Trends in Databases                             |         4 | 100.0%            | 100.0%                                         | 75.0%            |
| Foundations and Trends in Networking                            |         4 | 100.0%            | 100.0%                                         | 100.0%           |
| Foundations and Trends in Privacy and Security                  |         4 | 100.0%            | 75.0%                                          | 100.0%           |
| Foundations and Trends in Programming Languages                 |         4 | 100.0%            | 100.0%                                         | 75.0%            |
| Foundations and Trends in Systems and Control                   |         3 | 100.0%            | 100.0%                                         | 100.0%           |
| Foundations and Trends in Human-Computer Interaction            |         2 | 100.0%            | 100.0%                                         | 50.0%            |
| Foundations and Trends in Web Science                           |         2 | 100.0%            | 100.0%                                         | 100.0%           |
| Foundations and Trends in Electric Energy Systems               |         1 | 100.0%            | 100.0%                                         | 100.0%           |

## F1 per topic group (title + abstract)

| Group   |   Papers | DistilBERT only   | Validation-fitted hybrid (+ reference count)   | Learned hybrid   |
|---------|----------|-------------------|------------------------------------------------|------------------|
| biomed  |       72 | 97.3%             | 100.0%                                         | 97.1%            |
| comm    |       78 | 92.1%             | 90.7%                                          | 82.4%            |
| control |      254 | 89.3%             | 89.7%                                          | 84.0%            |
| ml      |      376 | 93.1%             | 91.7%                                          | 86.2%            |
| stats   |      166 | 90.6%             | 91.2%                                          | 89.6%            |
| theory  |       24 | 95.7%             | 95.7%                                          | 90.9%            |

## Test set — title only (n = 970, 485 surveys)

| Method                                                        | Acc.   | Prec.   | Recall   | F1    | F1 95% CI   | ΔF1 vs DistilBERT (95% CI)   | Prec. at 8.0% surveys   |
|---------------------------------------------------------------|--------|---------|----------|-------|-------------|------------------------------|-------------------------|
| Keyword only (title)                                          | 61.5%  | 100.0%  | 23.1%    | 37.5% | 32.7%–42.6% | -39.6 (-44.6 to -34.7)       | 100.0%                  |
| TF-IDF + SVM                                                  | 71.6%  | 83.4%   | 54.0%    | 65.6% | 61.7%–69.3% | -11.5 (-14.9 to -8.0)        | 30.5%                   |
| DistilBERT only                                               | 80.1%  | 90.8%   | 67.0%    | 77.1% | 73.8%–80.1% |                              | 46.1%                   |
| Hybrid (OR, paper)                                            | 80.1%  | 90.8%   | 67.0%    | 77.1% | 73.8%–80.1% | +0.0 (+0.0 to +0.0)          | 46.1%                   |
| Validation-fitted hybrid (title+abstract)                     | 80.0%  | 91.2%   | 66.4%    | 76.8% | 73.6%–79.8% | -0.3 (-0.8 to +0.3)          | 47.5%                   |
| Validation-fitted hybrid (+ reference count)                  | 80.9%  | 89.9%   | 69.7%    | 78.5% | 75.3%–81.4% | +1.4 (+0.0 to +2.9)          | 43.6%                   |
| Learned hybrid                                                | 76.3%  | 91.0%   | 58.4%    | 71.1% | 67.4%–74.6% | -6.0 (-8.5 to -3.5)          | 46.8%                   |
| Indexer type ('Review')                                       | 60.8%  | 100.0%  | 21.6%    | 35.6% | 30.7%–40.4% | -41.5 (-47.1 to -35.7)       | 100.0%                  |
| Earlier model: distilbert_survey_model_synthetic (OR, th=0.8) | 65.5%  | 95.7%   | 32.4%    | 48.4% | 43.4%–53.3% | -28.7 (-33.3 to -24.3)       | 66.1%                   |
