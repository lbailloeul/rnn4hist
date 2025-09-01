# SPS Data Analysis Report
*Generated: 2025-09-01 11:45:39*

## Executive Summary

This report presents a comprehensive analysis of 1045 SPS histogram fitting quality samples,
with 780 GOOD samples (74.6%) and 265 BAD samples (25.4%).

## 🎯 Classification Boundaries

### Best Single Feature Classifier: **Mu_p**
- **Accuracy**: 79.7%
- **Threshold**: 2.713
- **Rule**: Lower values indicate GOOD
- **Effect Size (Cohen's d)**: 0.773

### All Features Performance:

**1. Mu_p**
- Accuracy: 79.7%
- Threshold: 2.713 (Lower = GOOD)
- Effect size: 0.773 (Medium)

**2. Sigma_0**
- Accuracy: 77.0%
- Threshold: 2.778 (Lower = GOOD)
- Effect size: 0.487 (Small)

**3. Sigma_Gain**
- Accuracy: 74.6%
- Threshold: 5.988 (Lower = GOOD)
- Effect size: 0.600 (Medium)

## 🔄 Principal Component Analysis

### Variance Explanation:

- **PC1**: 34.9% (cumulative: 34.9%)
- **PC2**: 33.9% (cumulative: 68.8%)
- **PC3**: 31.2% (cumulative: 100.0%)

### Class Separation Quality:

- **PC1**: Fisher ratio = 0.000 (Poor separation)
- **PC2**: Fisher ratio = 0.045 (Poor separation)
- **PC3**: Fisher ratio = 0.716 (Excellent separation)

**Best separating component**: PC3 (Fisher ratio: 0.716)

## 🎲 Clustering Analysis

### Optimal Clustering: 7 clusters
- **Silhouette Score**: 0.296

### Cluster Purity Analysis:

**Cluster 0**:
- Size: 154 samples
- Composition: 144 GOOD, 10 BAD
- Purity: 93.5% (majority: GOOD)

**Cluster 1**:
- Size: 156 samples
- Composition: 145 GOOD, 11 BAD
- Purity: 92.9% (majority: GOOD)

**Cluster 2**:
- Size: 143 samples
- Composition: 54 GOOD, 89 BAD
- Purity: 62.2% (majority: BAD)

**Cluster 3**:
- Size: 146 samples
- Composition: 110 GOOD, 36 BAD
- Purity: 75.3% (majority: GOOD)

**Cluster 4**:
- Size: 129 samples
- Composition: 120 GOOD, 9 BAD
- Purity: 93.0% (majority: GOOD)

**Cluster 5**:
- Size: 155 samples
- Composition: 84 GOOD, 71 BAD
- Purity: 54.2% (majority: GOOD)

**Cluster 6**:
- Size: 162 samples
- Composition: 123 GOOD, 39 BAD
- Purity: 75.9% (majority: GOOD)

## 🔍 Supervised Classification Analysis

### Cluster Purity Assessment:
- **Average cluster purity**: 0.782
- **Minimum cluster purity**: 0.542
- **Purity threshold**: 0.85
- **Recommendation**: Supervised learning approach

### Supervised Classifier Performance:

**Support Vector Machine (SVM)**:
- Accuracy: 0.924
- Precision: 0.930
- Recall: 0.970
- F1-Score: 0.950

**Random Forest**:
- Accuracy: 0.901
- Precision: 0.921
- Recall: 0.949
- F1-Score: 0.935

**Feature Importance (Random Forest)**:
- Mu_p: 0.479
- Sigma_Gain: 0.268
- Sigma_0: 0.253

## 💡 Recommendations

### For Quality Control Implementation:

1. **Primary Strategy**: Use PCA transformation for classification
2. **Best Component**: Focus on PC3
3. **Multi-feature Approach**: Combine features using PCA weights

### Next Steps:

- Validate findings on independent test set
- Consider ensemble methods for improved performance
- Implement real-time quality monitoring based on findings
- Regular model retraining as new data becomes available

---
*Analysis completed with 3 features: Sigma_Gain, Sigma_0, Mu_p*