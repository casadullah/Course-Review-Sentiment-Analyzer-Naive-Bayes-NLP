# Course Review Sentiment Analyzer — Naive Bayes NLP

## What this project does
Predicts whether an online course review is positive 
or negative using Naive Bayes classification applied 
to 12,000+ real course reviews — without looking at 
the star rating. Pure text-to-sentiment prediction.

## The real-world challenge
The dataset was severely imbalanced — 89% of reviews 
were 4 or 5 stars. A naive model predicting "good" 
for everything would score 96% accuracy while being 
completely useless. This project solves that problem 
systematically.

## Complete pipeline built
- Custom CSV parser to handle malformed rows 
  (30+ real broken rows in production data)
- Data type conversion (ratings stored as text)
- EDA: discovered severe class imbalance 
  (mean rating 4.84, median 5.0)
- Regex-based linguistic feature analysis
- Text normalization (lowercase, remove punctuation)
- TF-IDF vectorization (text → numbers)
- Multinomial Naive Bayes classification
- Oversampling with sklearn resample to fix 
  class imbalance
- Multi-class (1–5 stars) and binary 
  (good/bad) models compared
- Final evaluation on completely unseen test set

## Key findings
- Comment length negatively correlates with rating 
  (r = -0.28): unhappy customers write more
- 6,147 reviews contain positive emphasis words 
  vs only 23 with negative words — confirming 
  extreme class imbalance
- Binary classification after upsampling achieves 
  84.7% accuracy on test set
- Bad review recall: 0.769 — model catches 77% of 
  negative reviews (the most important metric for 
  business use)

## Model comparison
| Model | Accuracy | Bad recall |
|-------|----------|-----------|
| Multi-class raw | 96% (misleading) | ~0% |
| Multi-class upsampled | 74% | 19-55% |
| Binary raw | 96% (misleading) | 0% |
| Binary upsampled | 84.7% | 76.9% |

## Tools used
Python · Pandas · NumPy · Sklearn · NLTK · 
Matplotlib · Seaborn · Regex

## How to run
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook naive-bayes-sentiment-analysis.ipynb
