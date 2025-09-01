# SPS Data Analysis Report
## Dataset Overview
- Total samples: 1045
- GOOD samples: 780
- BAD samples: 265
- Features analyzed: Sigma_Gain, Sigma_0, Mu_p

## Optimal Decision Boundaries
### Sigma_Gain
- **Threshold**: 5.988
- **Rule**: Lower values indicate GOOD
- **Classification Accuracy**: 74.6%
- **GOOD range**: [0.503, 5.988]
- **BAD range**: [0.581, 5.926]

### Sigma_0
- **Threshold**: 2.777
- **Rule**: Lower values indicate GOOD
- **Classification Accuracy**: 77.0%
- **GOOD range**: [0.301, 2.995]
- **BAD range**: [0.314, 2.989]

### Mu_p
- **Threshold**: 2.709
- **Rule**: Lower values indicate GOOD
- **Classification Accuracy**: 79.9%
- **GOOD range**: [0.104, 3.463]
- **BAD range**: [0.105, 3.498]

## Key Findings
- **Best single feature**: Mu_p with 79.9% accuracy
- **Feature Insights**:
  - Sigma_Gain: GOOD samples average 3.063, BAD samples average 3.992 (difference: 30.3%)
  - Sigma_0: GOOD samples average 1.560, BAD samples average 1.939 (difference: 24.3%)
  - Mu_p: GOOD samples average 1.688, BAD samples average 2.398 (difference: 42.1%)
