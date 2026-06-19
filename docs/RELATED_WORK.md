# Related Work

*Scope: only studies using NHANES datasets are compared. UK Biobank, CHNS,
EHR, retinal-image, genetics, and multimodal non-NHANES papers are
excluded.*

## Table 1. NHANES-only comparable diabetes prediction studies

| Paper | Year | Dataset | N | Non-Glycemic | Model | Explainability | External Validation | Calibration | Risk Stratification | Fairness | AUC / Main Result | Key Limitation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yang et al. [3] | 2026 | NHANES 2013–2018, high-risk adults | 2,355 | Partial | RF, XGBoost, DT, MLP, AdaBoost | SHAP | No | Not reported | No | No | XGBoost AUC 0.903, accuracy 0.815, sensitivity 0.962; RF AUC 0.896 | High-risk-only cohort; no calibration, temporal validation, or risk stratification |
| Qin et al. [1] | 2022 | NHANES 1999–2020 | 17,833 | Lifestyle-focused | CatBoost, XGBoost, RF, LR, SVM | Feature importance / SHAP | No | No | No | No | CatBoost accuracy 82.1%, AUC 0.83 | Low specificity; no temporal validation or calibration |
| Riveros Perez & Avella-Molano [2] | 2025 | NHANES 2007–2018 | 29,509 | Yes, lifestyle/anthropometric | LR, SVM, RF, XGBoost, CatBoost | Feature importance | No | Not reported | No | No | XGBoost AUC 0.8168 | Low sensitivity; no calibration, temporal validation, or risk stratification |
| Cichosz et al. [4] | 2024 | NHANES 2005–2018 | 45,431 | Partial | 5-model ML comparison (incl. AutoML) | Not reported | Not reported | Not reported | No | No | Not reported in accessible record | Focused on undiagnosed diabetes; full metric table needs verification |
| **This work** | — | Multi-cycle NHANES (train G–J, test L) | Train 23,825; test 8,153 | Yes — 44 leakage-free non-glycemic variables | LightGBM, XGBoost, GBM, RF, FT-Transformer, SVM, stacking | SHAP | Yes, temporal external validation | Yes, Platt calibration | Yes, 3-tier | Yes | Stacking AUC ≈0.895; survey-weighted AUC ≈0.9034; PR-AUC ≈0.493; ECE ≈0.014 | Needs prospective external validation |

## Table 2. Methodological feature comparison matrix

| Methodological Component | Qin 2022 | Riveros Perez 2025 | Yang 2026 | This Work |
|---|---|---|---|---|
| NHANES dataset | ✓ | ✓ | ✓ | ✓ |
| Population-level screening | ✓ | ✓ | No — high-risk only | ✓ |
| Non-glycemic restriction | Partial | ✓ | Partial | ✓ |
| Leakage prevention | ✗ | ✗ | ✗ | ✓ |
| Stacking ensemble | ✗ | ✗ | ✗ | ✓ |
| FT-Transformer | ✗ | ✗ | ✗ | ✓ |
| SHAP explainability | Partial | Partial | ✓ | ✓ |
| Temporal external validation | ✗ | ✗ | ✗ | ✓ |
| Calibration analysis | ✗ | ✗ | ✗ | ✓ |
| PR-AUC reporting | ✗ | ✗ | Not reported | ✓ |
| Risk stratification | ✗ | ✗ | ✗ | ✓ |
| Fairness / subgroup analysis | ✗ | ✗ | ✗ | ✓ |
| DeLong testing | ✗ | ✗ | ✗ | ✓ |

## Ranking by similarity to this study

1. **Riveros Perez & Avella-Molano (2025)** — closest, since it uses NHANES, lifestyle/anthropometric variables, and a non-invasive diabetes-prediction framing.
2. **Qin et al. (2022)** — close, since it uses NHANES, lifestyle features, and multiple ML model families.
3. **Yang et al. (2026)** — methodologically strong (XGBoost/RF, SHAP) but less comparable, since the cohort is restricted to a high-risk subgroup.
4. **Cichosz et al. (2024)** — relevant given the large NHANES sample, but less directly comparable since exact performance and feature structure require further verification.

## Remaining research gaps

Existing NHANES-based diabetes studies mainly optimize discrimination,
especially accuracy or ROC-AUC. They rarely integrate all translational
requirements together: leakage-free non-glycemic feature design, temporal
validation, calibration, PR-AUC, subgroup fairness, risk stratification,
DeLong testing, and deployment-oriented thresholds.

## Novelty statement

To the best of our knowledge, no previous NHANES-based diabetes
prediction study has simultaneously integrated leakage-aware non-glycemic
feature restriction, heterogeneous stacking, FT-Transformer-enhanced
tabular learning, temporal external validation, Platt-calibrated risk
estimation, SHAP explainability, three-tier risk stratification, subgroup
fairness analysis, and statistical model comparison within a single
population-level framework.

## Related work (IEEE style)

Existing NHANES-based diabetes prediction studies demonstrate the
feasibility of machine learning for population-level screening, but their
methodological scope remains incomplete. Qin et al. used NHANES
1999–2020 data and compared CatBoost, XGBoost, Random Forest, logistic
regression, and SVM for lifestyle-based diabetes prediction. CatBoost
achieved the best reported performance, with 82.1% accuracy and an AUC of
0.83. However, the reported specificity was limited, and the study did
not include temporal validation, probability calibration, PR-AUC
evaluation, formal risk stratification, or subgroup fairness assessment
[1]. This limits its translational relevance, since discrimination alone
is insufficient for clinical screening under imbalanced disease
prevalence.

Riveros Perez and Avella-Molano later evaluated machine-learning
algorithms using NHANES 2007–2018 lifestyle and anthropometric variables
in 29,509 non-pregnant adults [2]. This work is highly relevant because
it focused on non-invasive diabetes prediction using routinely obtainable
lifestyle-related variables. XGBoost achieved the highest reported AUC
among the evaluated models. However, the study noted limited sensitivity,
and calibration analysis, temporal validation, risk stratification, and
fairness assessment were not reported. Although this study aligns closely
with the non-invasive screening motivation, its deployment readiness
remains limited.

Yang et al. developed machine-learning models for diabetes prediction
among high-risk adults using NHANES 2013–2018 data [3]. Their XGBoost
model achieved strong discrimination, with an AUC of 0.903 and
sensitivity of 0.962, while Random Forest achieved an AUC of 0.896. This
study is notable for incorporating SHAP-based interpretation and
demonstrating strong model performance. However, the analysis was
restricted to a high-risk subgroup of 2,355 participants rather than the
general NHANES population, and calibration, temporal validation, and
deployment-oriented risk stratification were not reported, limiting its
applicability to broad population-level screening.

Other NHANES-based diabetes detection studies have explored simple
clinical-variable prediction and undiagnosed-diabetes identification
using machine-learning models [4]. Several of these studies either
require additional table-level verification for exact metrics or include
biomarker-dependent definitions that reduce comparability with
non-glycemic screening frameworks. These limitations highlight the need
for NHANES-based models that evaluate not only discrimination but also
calibration, subgroup robustness, leakage prevention, threshold behavior,
and clinical deployment readiness.

Compared with existing NHANES-based studies, the framework in this
repository provides a broader translational evaluation. It uses 44
leakage-free non-glycemic variables and integrates LightGBM, XGBoost,
GBM, Random Forest, FT-Transformer, SVM, and a heterogeneous stacking
ensemble with a logistic-regression meta-learner. Beyond discrimination
performance, it includes temporal external validation, Platt-calibrated
probability estimation, SHAP explainability, PR-AUC evaluation,
three-tier risk stratification, subgroup fairness analysis, leakage
prevention, and DeLong statistical testing. The contribution is therefore
not merely another NHANES diabetes classifier, but a clinically oriented
and deployment-aware framework for population-level Type 2 Diabetes risk
stratification.

## References

[1] Y. Qin, J. Wu, W. Xiao, K. Wang, A. Huang, B. Liu, J. Yu, C. Li, F. Yu,
and Z. Ren, "Machine Learning Models for Data-Driven Prediction of
Diabetes by Lifestyle Type," *Int. J. Environ. Res. Public Health*, vol.
19, no. 22, p. 15027, 2022. https://doi.org/10.3390/ijerph192215027

[2] E. Riveros Perez and B. Avella-Molano, "Learning from the machine: is
diabetes in adults predicted by lifestyle variables? A retrospective
predictive modelling study of NHANES 2007–2018," *BMJ Open*, vol. 15, no.
3, p. e096595, 2025. https://doi.org/10.1136/bmjopen-2024-096595

[3] Yang et al., "Machine learning predicts diabetes risk in high-risk
populations: analysis of National Health and Nutrition Examination
Survey data," *Archives of Medical Science*, 2026.

[4] S. L. Cichosz, C. Bender, and O. Hejlesen, "A Comparative Analysis of
Machine Learning Models for the Detection of Undiagnosed Diabetes
Patients," *Diabetology*, vol. 5, no. 1, pp. 1–11, 2024.
https://doi.org/10.3390/diabetology5010001

---

*Citations verified against publisher/PubMed records as of June 2026. The
Cichosz et al. journal (Diabetology) was corrected from an earlier draft
that omitted it.*
