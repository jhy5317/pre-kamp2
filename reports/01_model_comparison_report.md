# 1차 보고서: 열처리 공정 불량비율 예측 모델 비교 평가

**작성일**: 2026-09-16
**프로젝트**: pre-kamp2 (2026 K-인공지능 제조데이터 경진대회 6회 연습 프로젝트)
**대상 데이터**: KAMP 열처리 공정최적화 AI 데이터셋 (배정번호 136건)

---

## 1. 개요

본 보고서는 `01_baseline_linear_regression.ipynb` → `02_preprocessing_and_skew_correction.ipynb` → `03_model_comparison.ipynb`로 이어지는 파이프라인의 최종 산출물로, 열처리 공정 센서 데이터를 이용해 배정번호(배치)별 **불량비율**을 예측하는 회귀 모델 7종을 동일한 조건에서 비교 평가한 결과를 정리한다.

### 1.1 파이프라인 요약

| 단계 | 노트북 | 주요 내용 |
|---|---|---|
| 01 | 01_baseline_linear_regression | 원본 센서 CSV(2,939,722행) + 라벨(136건) 병합, 배치 단위 min/max 집계(38개 피처), 다중공선성 제거(27개 피처), Train:Val:Test = 108:14:14 분할, SGDRegressor 베이스라인 확정 |
| 02 | 02_preprocessing_and_skew_correction | 01에서 잘못 포함된 `Unnamed: 0` 파생 컬럼을 원본 행 범위 정보로 분리·저장 후 제거(36개 피처 기준), 독립변수 간 상관관계 히트맵, 다중공선성 재점검, 왜도(skew) 진단 및 Yeo-Johnson 파워변환 |
| 03 (본 보고서 대상) | 03_model_comparison | 02의 전처리 데이터에 MinMaxScaler 추가 적용, 7개 회귀 모델(Linear, SGD, Ridge, Lasso, DecisionTree, RandomForest, GradientBoosting) 학습·검증·최종 테스트 평가 |

### 1.2 데이터 분할 원칙

전 과정에서 **Train : Val : Test = 108 : 14 : 14 (8:1:1)** 분할을 고정 `random_state=42`로 유지했으며, Test set은 각 노트북에서 **최종 평가 시 단 한 번만** 사용해 데이터 누수(leakage)를 방지했다. 하이퍼파라미터 탐색은 K-Fold 교차검증이 아닌 **Validation set 기반 그리드서치**(`grid_search_by_val`)로 수행했다 — 표본이 108건으로 작아 K-Fold보다 명시적 3분할 방식이 이 프로젝트의 일관된 설계 원칙이기 때문이다.

---

## 2. 모델별 평가 결과

### 2.1 Validation 성능 (전체 모델)

| 모델 | Train R² | Val R² | Val RMSE | Val MAE |
|---|---|---|---|---|
| SGDRegressor | 0.0326 | **0.0604** | 0.0261 | 0.0195 |
| Lasso | 0.1195 | 0.0595 | 0.0261 | 0.0181 |
| DecisionTree | 0.8353 | 0.0546 | 0.0262 | 0.0184 |
| Ridge | 0.0397 | 0.0124 | 0.0267 | 0.0201 |
| GradientBoosting | 0.0632 | -0.0088 | 0.0270 | 0.0205 |
| LinearRegression | 0.2375 | -0.0855 | 0.0280 | 0.0238 |
| RandomForest | 0.7867 | -0.3544 | 0.0313 | 0.0242 |

### 2.2 Test 성능 (최종 평가, R² 내림차순)

| 순위 | 모델 | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| 1 | **Lasso** | **0.1186** | 0.0510 | 0.0328 |
| 2 | Ridge | 0.0368 | 0.0534 | 0.0333 |
| 3 | SGDRegressor | 0.0305 | 0.0535 | 0.0332 |
| 4 | LinearRegression | 0.0036 | 0.0543 | 0.0346 |
| 5 | GradientBoosting | -0.0014 | 0.0544 | 0.0333 |
| 6 | DecisionTree | -0.1329 | 0.0579 | 0.0344 |
| 7 | RandomForest | -0.3150 | 0.0623 | 0.0413 |

> 01 노트북 베이스라인(SGDRegressor) Test 성능: **R²=0.0414, RMSE=0.0532, MAE=0.0334**

### 2.3 선정된 하이퍼파라미터

| 모델 | 최적 하이퍼파라미터 (Val 기준) |
|---|---|
| Ridge | alpha = 31.62 |
| Lasso | alpha = 0.000681 (26개 피처 중 19개 계수를 0으로 축소) |
| DecisionTree | max_depth = 4 |
| RandomForest | n_estimators = 100 (조기 종료 시 최적 체크포인트, 최대 200까지 학습 후 patience로 종료) |
| GradientBoosting | n_estimators = 1 (2번째 트리부터 Val 성능 악화, 사실상 부스팅 효과 없음) |

![Test R² 모델 비교](../figures/19_model_comparison_test_r2.png)

---

## 3. 핵심 발견

### 3.1 Lasso가 최종 1위 — 베이스라인 대비 약 2.9배 개선

**Lasso(alpha=0.000681)가 Test R²=0.1186으로 전체 7개 모델 중 최고 성능**을 기록했다. 이는 01 베이스라인 SGDRegressor의 Test R²=0.0414 대비 약 **2.86배**의 개선이다. Lasso는 26개 피처 중 19개(73%)의 계수를 0으로 축소해 사실상 7개 피처만으로 예측을 수행했는데, 이 자동 피처 선택이 작은 표본(train 108건) 환경에서 과적합을 억제하는 데 효과적이었던 것으로 보인다. Ridge(Test R²=0.0368)와 SGDRegressor(0.0305)도 정규화가 없는 LinearRegression(0.0036)보다 뚜렷이 우수했다는 점이 이 해석을 뒷받침한다.

### 3.2 LinearRegression 계수 불안정성 — 잔존 다중공선성의 증거

LinearRegression의 계수 크기를 시각화한 결과, 한 피처(건조로 온도 2 Zone_min)의 계수가 약 **1.6 × 10¹³**으로 나머지 15개 계수를 압도적으로 초과하는 현상이 관찰됐다.

![LinearRegression 계수 크기](../figures/17_feature_importance_linear.png)

이는 02 단계에서 |r|≥0.8 기준으로 다중공선성을 제거했음에도 **잔존 다중공선성 또는 준완전공선성(near-collinearity)이 남아 있음을 보여주는 실증적 증거**다. 정규화가 없는 최소제곱법(OLS)은 공선성이 있는 변수 조합에서 계수가 극단적으로 팽창할 수 있는데, 바로 이 현상이 관측된 것이다. 이 계수 그래프는 해석용으로 신뢰하기 어려우며, 실제 피처 영향력 해석에는 **Ridge/Lasso 계수(정규화로 안정화됨) 또는 RandomForest의 `feature_importances_`를 사용하는 것이 타당**하다.

### 3.3 트리 계열 모델의 심각한 과적합

DecisionTree, RandomForest, GradientBoosting 세 트리 모델 모두 Train R²는 높지만(0.79~0.84, GradientBoosting 제외) Val/Test R²는 음수를 기록해 **과적합이 뚜렷하게 나타났다.**

- **RandomForest**: Train R²=0.7867 vs Test R²=**-0.3150**(전체 최하위). `max_depth`를 제한 없이(`None`) 학습한 것이 원인으로 추정되며, DecisionTree는 그리드서치로 `max_depth=4`를 탐색·적용한 반면 RandomForest는 트리 개수(`n_estimators`)만 체크포인트 방식으로 조절했을 뿐 개별 트리 깊이는 제한하지 않았다. 이는 코드상의 설계 한계로, 다음 단계에서 `max_depth`도 함께 튜닝할 필요가 있다.
- **GradientBoosting**: `staged_predict`로 재구성한 학습 곡선에서 Val 손실이 **1번째 트리에서 최소, 2번째 트리부터 악화**되는 것으로 나타나 `best_n_estimators=1`이 선택됐다. 사실상 부스팅의 이점을 전혀 활용하지 못한 것으로, 108건이라는 매우 작은 학습 표본 크기가 원인으로 판단된다.
- **DecisionTree**: `max_depth=4`로 제한했음에도 Test R²=-0.1329로 음수를 기록해, 단일 트리 구조 자체가 이 데이터의 노이즈에 취약함을 시사한다.

![RandomForest 학습 곡선](../figures/14_random_forest_training_curve.png)
![GradientBoosting 학습 곡선](../figures/15_gradient_boosting_training_curve.png)

### 3.4 피처 중요도 비교

정규화가 없는 LinearRegression의 계수는 위 3.2절의 사유로 신뢰하기 어렵다. 반면 RandomForest의 `feature_importances_`는 값이 고르게 분포되어 있으며, **"솔트 컨베이어 온도 2 Zone_max"**가 가장 중요한 피처로 나타났다. RandomForest 자체의 예측 성능은 낮지만, 트리 기반 중요도 지표는 스케일에 영향을 받지 않고 안정적이라는 점에서 피처 해석 목적으로는 참고할 가치가 있다.

![RandomForest/GradientBoosting 피처 중요도](../figures/18_feature_importance_tree.png)

> 참고: Ridge, Lasso, GradientBoosting의 피처 중요도는 03 노트북 실행 결과에 셀 출력으로 포함되어 있으나 별도 PNG 파일로는 저장되지 않았다. 향후 버전에서 저장 로직을 추가할 필요가 있다.

### 3.5 최고 모델의 예측 vs 실제

![Lasso 예측 vs 실제 (Test)](../figures/20_best_model_test_pred_vs_actual.png)

---

## 4. 한계 (Limitations)

1. **표본 크기가 매우 작다.** 전체 136개 배치 중 Test set은 14건에 불과해, R² 및 RMSE 추정치의 분산이 크고 배치 구성이 조금만 달라져도 순위가 바뀔 수 있다. 현재 결과는 일반화 성능의 안정적 추정치라기보다 하나의 스냅샷으로 해석해야 한다.
2. **잔존 다중공선성.** |r|≥0.8 기준의 pairwise 제거만으로는 완전히 해소되지 않았으며(3.2절), VIF(분산팽창지수) 기반의 추가 점검이 필요하다.
3. **트리 계열 모델의 하이퍼파라미터 탐색이 불균형하다.** RandomForest는 `max_depth`를 탐색하지 않았고, GradientBoosting은 `learning_rate`/`max_depth`를 고정값(0.05, 3)으로 사용했다. 공정한 비교를 위해서는 모든 모델에 동등한 수준의 탐색이 필요하다.
4. **Validation 기반 탐색의 분산.** K-Fold 대신 단일 Validation set(14건)으로 하이퍼파라미터를 선택했기 때문에, 선택된 하이퍼파라미터 자체도 표본 변동에 민감할 수 있다.

---

## 5. 다음 단계 제안

1. **RandomForest `max_depth` 튜닝**: `grid_search_by_val`을 확장해 `n_estimators`와 `max_depth`를 함께 탐색.
2. **Lasso 선택 피처만으로 재학습**: Lasso가 0이 아닌 계수를 남긴 7개 피처만 사용해 다른 모델(Ridge, RandomForest 등)을 재학습하고 성능 변화를 확인.
3. **K-Fold 교차검증 도입 검토**: 표본이 작으므로 Leave-One-Out 또는 5-Fold CV로 하이퍼파라미터 선택의 안정성을 재검증.
4. **VIF 기반 다중공선성 재점검**: 3.2절의 계수 불안정성 문제를 근본적으로 해소하기 위해 pairwise 상관계수 대신 VIF를 도입.
5. **부스팅 계열 추가 검토**: XGBoost, LightGBM 등 정규화 내장 부스팅 모델을 소표본에 맞는 설정(낮은 `max_depth`, 강한 L1/L2)으로 시도.
6. **Ridge/Lasso/GradientBoosting 피처 중요도 PNG 저장**: 03 노트북에 저장 로직 추가.

---

## 6. 결론

7개 모델 비교 결과 **Lasso가 Test R²=0.1186으로 최고 성능**을 기록했으며, 이는 01 베이스라인(SGDRegressor, R²=0.0414) 대비 약 2.9배 개선이다. 정규화 계열(Lasso, Ridge, SGD)이 정규화 없는 모델(LinearRegression) 및 트리 계열 모델을 모두 상회한 것은, 이 데이터셋이 **작은 표본 크기와 잔존 다중공선성으로 인해 정규화가 필수적인 환경**임을 시사한다. 트리 계열 모델(특히 RandomForest)의 심각한 과적합은 하이퍼파라미터 탐색 범위 확대로 개선 여지가 있으나, 현재로서는 Lasso가 가장 신뢰할 수 있는 선택이다.

---

*본 보고서는 03_model_comparison.ipynb의 실제 실행 결과(2026-09-16 로컬 실행)를 기반으로 작성되었다.*
