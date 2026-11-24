# Movie Genre Prediction from Plot Descriptions

## Abstract

## 1. Introduction

## 2. Background / Related Work

## 3. Data
- Sources, retrieval date, license
- What one row is (one movie)
- Genre mapping (see `config.py`) and the priority rule for movies with several genres
- Row counts before and after cleaning, class balance
- Overlap between IMDb and Rotten Tomatoes and how often their labels agree

## 4. Methodology
- Baseline: majority class. Models: Naive Bayes, Logistic Regression, Random Forest
- TF-IDF inside a Pipeline, fit on training data only
- 80/20 stratified split, seed 42
- Cross-source test removes movies seen in IMDb training
- Primary metric: macro-F1 (classes are imbalanced)

## 5. Results

## 6. Discussion 

## 7. Conclusion

## References

## Contributions and AI Assistance
