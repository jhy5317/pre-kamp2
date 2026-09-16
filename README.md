# 열처리 공정최적화 AI 데이터셋 - 선형회귀 베이스라인

2026년도 K-인공지능 제조데이터 경진대회(6회) 참가 준비를 위한 연습 프로젝트입니다.
KAMP(중소벤처기업부·KAIST)가 제공하는 "열처리 공정최적화 AI 데이터셋"과 가이드북을 바탕으로,
데이터 비교 → 라벨 연관관계 분석 → 전처리 → Train/Val/Test 분할 → 선형회귀 베이스라인 학습 →
평가/시각화 → 전체 EDA(Feature Importance 포함) 까지의 파이프라인을 `01` 노트북에서 구현했고,
`02` 노트북에서 전처리를 한 단계 더 보강했습니다 (원본 행 번호 컬럼 분리, 독립변수 상관관계
히트맵, 왜도(skew) 보정). `03` 노트북에서는 이 개선된 데이터로 LinearRegression/SGDRegressor를
재평가하고, 정규화 회귀(Ridge/Lasso)와 트리 계열 모델(Decision Tree/Random Forest/Gradient
Boosting)까지 확장해 성능을 비교합니다. 03 실행 결과, **Lasso가 Test R²=0.1186으로 7개 모델 중
최고 성능**을 기록했으며(01 베이스라인 대비 약 2.9배 개선), 이를 바탕으로 한 1차 보고서
(`reports/01_model_comparison_report.md` / `.pdf`)를 작성 완료했습니다. 1차 보고서에서 찾은
개선 여지를 실제로 구현하기 위해 `04_advanced_modeling.ipynb`를 추가해 실행 완료했고(2차 보고서
`reports/02_advanced_modeling_report.md` / `.pdf` 작성 완료), RandomForest `max_depth` 튜닝,
Lasso 선택 피처 재학습, XGBoost/LightGBM 추가, K-Fold 안정성 검증, VIF 재점검, 남은 피처
중요도 PNG 저장까지 6가지 개선을 다뤘습니다 (03은 1차 보고서의 근거이므로 수정하지 않고 그대로
보존).

이후 04의 결과를 검증하는 과정에서 **02의 왜도(skew) 보정 단계에 있던 버그**를 발견했습니다:
일부 컬럼에서 Yeo-Johnson 변환이 예외 없이 "성공"하고도 극단적인 λ 때문에 변환값이 수치적으로
상수에 가깝게 "조용히 붕괴"해, 사실상 정보가 사라진 컬럼이 6개나 있었습니다 (01의 LinearRegression
계수 폭주의 실제 원인이기도 했습니다 — VIF로는 잡히지 않는 절편과의 공선성 문제). `02`에 붕괴
가드를 추가해 수정했고(`src/training_utils.py`의 `correct_skew_with_power_transform()`), 이
수정을 반영해 재학습·레버리지 분석·Test 재검증을 수행하는 `05_improved_modeling.ipynb`를
추가했습니다 (03/04는 각 보고서의 근거이므로 그대로 보존하고, 05가 새 체크포인트/산출물에는
모두 `v3_` 접두사를 붙여 기존 파일과 구분합니다).

2차 보고서 개정판 작성 후 "더 시도해볼만한 방법"으로 제안했던 5가지 — 가중회귀, 강건회귀
(Huber/RANSAC), 이항 GLM, 반복 K-Fold를 통한 정직한 성능 재추정, Zone 온도/OP 변동폭
도메인 파생 피처 — 를 모두 실행하는 `06_further_experiments.ipynb`를 추가했습니다 (산출물은
`v4_` 접두사 사용). 결론적으로 **이 5가지 중 어느 것도 04/05의 Lasso를 확실하게 뛰어넘지는
못했지만**, 배치 크기 기반 가중회귀는 세 선형모델 전부에서 일관되게 (작지만) 성능을
개선했고, 반대로 정규화 없는 이항 GLM은 배치별 표본 크기 편차 때문에 계수가 발산하는
심각한 수치적 불안정성을 보였습니다 (자세한 내용은 06 노트북 및 아래 "06 노트북" 절 참고).

## 폴더 구조

```
pre-kamp2/
├── dataset/                              # (기존) 원본 데이터
│   ├── raw_total_data.csv                # 원본 센서 데이터 (약 294만 행 x 20열, ~481MB)
│   ├── 품질전처리후데이터.csv              # 품질전처리(결측치 제거 등) 완료된 센서 데이터 (~502MB)
│   └── label.xlsx                        # 배정번호별 양품/불량 수량 라벨 데이터 (136행 x 22열)
├── notebooks/
│   ├── 01_baseline_linear_regression.ipynb          # 01: 비교/병합/전처리/분할/베이스라인 학습/평가
│   ├── 02_preprocessing_and_skew_correction.ipynb   # 02: Unnamed 분리, 상관관계 히트맵, skew 보정
│   ├── 03_model_comparison.ipynb                    # 03: 회귀/정규화회귀/트리모델 학습·비교
│   ├── 04_advanced_modeling.ipynb                   # 04: RF 튜닝/Lasso 피처선택/XGB·LGBM/K-Fold/VIF
│   ├── 05_improved_modeling.ipynb                   # 05: 02 버그 수정 반영 재학습/레버리지 분석/Test 재검증
│   └── 06_further_experiments.ipynb                 # 06: 가중회귀/강건회귀/이항GLM/반복K-Fold/도메인피처
├── src/
│   └── training_utils.py                 # 전처리/학습(체크포인트)/평가/시각화 공통 함수 모음
├── data_processed/                       # (노트북 실행 시 자동 생성) 중간 산출물 저장 폴더
│   ├── df_total.csv                      # [01] 배정번호 단위로 집계+라벨 병합된 최종 데이터셋
│   ├── X_train.csv / X_val.csv           # [01] 학습/검증용 독립변수 (Unnamed 컬럼 포함, 01 기준)
│   ├── X_test_DO_NOT_OPEN.csv            # [01] 테스트셋 (분석 과정에서 열어보지 않을 것)
│   ├── feature_importance.csv            # [01] 회귀계수 기반 변수 중요도 표
│   ├── original_row_range.csv            # [02] 배정번호별 원본 CSV 행 번호 범위 (Unnamed: 0 분리 저장)
│   ├── X_train_skew_corrected.csv        # [02] Unnamed 제외 + skew 보정 완료된 학습용 독립변수
│   ├── X_val_skew_corrected.csv          # [02] 〃 검증용
│   ├── X_test_skew_corrected_DO_NOT_OPEN.csv  # [02] 〃 테스트용 (열어보지 않을 것)
│   ├── y_train.csv / y_val.csv / y_test_DO_NOT_OPEN.csv  # [02] 종속변수(불량비율) 분할본
│   ├── model_comparison_val_metrics.csv  # [03] 모델별 train/val 지표
│   ├── model_comparison_test_metrics.csv # [03] 모델별 최종 test 지표 (test는 03에서 1회만 사용)
│   ├── model_comparison_best_hyperparams.csv  # [03] Ridge/Lasso/트리 모델 최적 하이퍼파라미터
│   ├── vif_train.csv                     # [04] train 데이터 피처별 VIF(분산팽창지수)
│   ├── kfold_stability_results.csv       # [04] 최종 후보 모델 K-Fold 안정성 검증 결과
│   ├── lasso_selected_features.json      # [04] Lasso가 0이 아닌 계수로 선택한 피처 목록
│   ├── model_comparison_test_metrics_v2.csv  # [04] 03의 7개 모델 + 04 신규 모델 통합 test 지표
│   ├── v3_*.csv / v3_*.json              # [05] 02 버그 수정 반영 후 재계산된 VIF/레버리지 분석/
│   │                                      #      Lasso 선택 피처(버전별)/Val·Test 지표/K-Fold 결과
│   │                                      #      (★ X_train_skew_corrected.csv 등 02가 만드는
│   │                                      #      파일 자체는 02를 재실행하면 덮어써지며 버전 구분이
│   │                                      #      없음 — 05는 이 "수정된 이후" 데이터를 사용합니다)
│   └── v4_*.csv                          # [06] 가중치 분포/반복 K-Fold 실험 비교표/최종 Test 비교표
├── checkpoints/                          # (노트북 실행 시 자동 생성) 학습 체크포인트 저장 폴더
│   ├── sgd_baseline.joblib               # [01] SGDRegressor 체크포인트
│   ├── power_transformer.joblib          # [02] 학습된 Yeo-Johnson PowerTransformer (train에만 fit)
│   ├── sgd_v2_skewcorrected.joblib       # [03] 02 데이터로 재학습한 SGDRegressor 체크포인트
│   ├── random_forest.joblib              # [03] Random Forest 체크포인트 (트리 증가 진행상황 포함)
│   ├── final_minmax_scaler.joblib        # [03] 최종 MinMaxScaler (train에만 fit, 04에서도 재사용)
│   ├── linear_regression / ridge / lasso / decision_tree / gradient_boosting .joblib  # [03] 학습 완료 모델
│   ├── rf_maxdepth_cand*.joblib          # [04] RandomForest max_depth 후보별 체크포인트
│   ├── random_forest_tuned.joblib        # [04] 최적 max_depth로 튜닝된 RandomForest
│   ├── ridge_lasso_subset.joblib / rf_lasso_subset.joblib  # [04] Lasso 선택 피처로 재학습한 모델
│   ├── xgboost_cand*.joblib / lightgbm_cand*.joblib  # [04] XGBoost/LightGBM 하이퍼파라미터 후보별 체크포인트
│   ├── xgboost.joblib / lightgbm.joblib  # [04] 최종 XGBoost/LightGBM 모델
│   ├── v3_*.joblib                       # [05] 02 버그 수정 반영 후 새로 fit한 MinMaxScaler +
│   │                                      #      버전별(all/drop_highlev) Lasso/Ridge/ElasticNet
│   │                                      #      모델, RF/XGBoost/LightGBM 후보별 체크포인트
│   └── v4_final_minmax_scaler.joblib     # [06] range 피처가 추가된 06 전용 피처셋으로 새로 fit한
│                                          #      MinMaxScaler
├── figures/                              # (노트북 실행 시 자동 생성) 저장된 그래프(PNG) 모음
│   ├── 01_~09_*.png                      # [01] 결측치 비교 ~ Feature Importance
│   ├── 10_~12_*.png                      # [02] 독립변수 상관관계 히트맵(전/후), skew 보정 전후 분포
│   ├── 13_~20_*.png                      # [03] SGD/RF/GB 학습곡선, 모델별 R² 비교, Feature Importance 등
│   ├── 21_~28_*.png                      # [04] RF튜닝/XGB/LGBM 학습곡선, VIF, 통합 R² 비교,
│   │                                      #      Ridge/Lasso/GB 피처 중요도
│   ├── 29_~35_*.png                      # [05] VIF(수정후), 레버리지 분포, RF 학습곡선, Val/Test
│   │                                      #      R² 비교, 최고모델 예측-실제/잔차
│   └── 26_~27_*.png                      # [06] 배치 크기별 표본 가중치 분포, 실험 A~D + 기존 후보
│                                          #      반복 K-Fold 비교(오차막대=95% 신뢰구간)
├── reports/
│   ├── 01_model_comparison_report.md     # 03 실행 결과 기반 1차 보고서 (원본, 마크다운)
│   ├── 01_model_comparison_report.pdf    # 〃 PDF 버전 (한글 폰트: NanumGothic)
│   ├── 02_advanced_modeling_report.md    # 04 실행 결과 기반 2차 보고서
│   └── 02_advanced_modeling_report.pdf   # 〃 PDF 버전
├── requirements.txt                       # 이 프로젝트 실행에 필요한 라이브러리 버전
├── requirements-windows.txt               # (기존) 딥러닝 학습용 전체 가상환경 requirements
└── README.md                              # 이 문서
```

## 사용법

1. 기존에 만들어둔 가상환경(`.venv`)을 활성화합니다. (Windows 예시)
   ```
   .venv\Scripts\activate
   ```
2. 이 프로젝트 실행에 필요한 패키지를 설치합니다.
   ```
   pip install -r requirements.txt
   ```
3. Jupyter Lab(또는 Notebook)을 실행합니다.
   ```
   jupyter lab
   ```
4. `notebooks/01_baseline_linear_regression.ipynb` 를 열어 **위에서부터 순서대로** 셀을
   실행합니다. (섹션 구성은 아래 "분석 파이프라인 개요" 참고)
5. 01 실행이 끝나 `data_processed/df_total.csv`가 생성된 뒤,
   `notebooks/02_preprocessing_and_skew_correction.ipynb` 를 열어 마찬가지로 위에서부터
   순서대로 실행합니다. (02는 01의 `df_total.csv`만 불러오므로 481MB/502MB 원본 CSV를
   다시 읽지 않습니다.)
6. 02 실행이 끝나 `data_processed/X_train_skew_corrected.csv` 등이 생성된 뒤,
   `notebooks/03_model_comparison.ipynb` 를 열어 마찬가지로 순서대로 실행합니다.
   (역시 481MB/502MB 원본 CSV를 다시 읽지 않습니다. Random Forest/SGD 셀은 중간에
   끊겨도 같은 셀을 다시 실행하면 체크포인트에서 이어서 학습합니다.)
7. 03 실행이 끝나 `checkpoints/final_minmax_scaler.joblib`, `checkpoints/{ridge,lasso,
   gradient_boosting}.joblib`, `data_processed/model_comparison_test_metrics.csv` 가
   생성된 뒤, `notebooks/04_advanced_modeling.ipynb` 를 열어 순서대로 실행합니다.
   RandomForest/XGBoost/LightGBM 하이퍼파라미터 탐색 셀들은 후보마다 별도 체크포인트를
   사용하므로, 중간에 끊겨도 같은 셀을 다시 실행하면 끝난 후보는 건너뛰고 이어서
   진행합니다. (`xgboost`/`lightgbm`/`statsmodels`가 새로 필요하므로 실행 전
   `pip install -r requirements.txt` 로 가상환경을 갱신하세요.)
8. 04에서 발견된 02의 왜도 보정 버그를 고쳤으므로, **`notebooks/02_preprocessing_and_
   skew_correction.ipynb`를 커널을 재시작한 뒤 처음부터 다시 실행**해 `data_processed/
   X_train_skew_corrected.csv` 등을 갱신합니다. (★ Jupyter는 같은 커널 세션에서
   `import training_utils`를 다시 실행해도 파일이 바뀐 걸 자동으로 반영하지 않으므로,
   `src/training_utils.py`가 바뀐 뒤에는 반드시 커널을 재시작해야 합니다.) 9-1 셀
   출력에 "N개는 CollapsedTransform..."이라는 메시지가 나오면 정상 작동한 것입니다.
9. 02 재실행이 끝난 뒤 `notebooks/05_improved_modeling.ipynb` 를 열어 순서대로
   실행합니다. `statsmodels.api`를 사용하므로 `requirements.txt`의 `scipy==1.12.0` 핀이
   설치돼 있어야 합니다(오래된 가상환경이라면 `pip install scipy==1.12.0`을 따로
   실행). 03/04는 그대로 보존되므로, 05가 만드는 모든 체크포인트/산출물 파일명에는
   `v3_` 접두사가 붙어 기존 파일을 덮어쓰지 않습니다.
10. 05 실행이 끝난 뒤 `notebooks/06_further_experiments.ipynb` 를 열어 순서대로
    실행합니다. 새로 설치해야 할 패키지는 없습니다(numpy/pandas/scikit-learn/statsmodels/
    tqdm 모두 기존 `requirements.txt`에 이미 포함). 반복 K-Fold(5-Fold×20회) 평가를 여러
    후보에 반복하므로 전체 실행에 1~2분 정도 걸릴 수 있습니다(로컬 PC 사양에 따라 다름).
    `random_state=42`로 전 과정이 고정되어 있어, 다시 실행해도 같은 결과가 나와야 합니다.
    03/04/05는 그대로 보존되므로, 06이 만드는 모든 체크포인트/산출물 파일명에는 `v4_`
    접두사가 붙어 기존 파일을 덮어쓰지 않습니다.

### 진행 현황 (2026-09-16 기준)

- `01_baseline_linear_regression.ipynb`: 로컬 PC에서 실행 완료. LinearRegression과
  SGDRegressor(체크포인트 재개 포함)를 비교해 SGDRegressor를 최종 베이스라인으로 선택했고,
  Test 성능은 R²=0.0414, RMSE=0.0532, MAE=0.0334 였습니다. 다만 학습에 사용된 독립변수
  중 `Unnamed: 0_max`(원본 CSV 행 번호가 실수로 섞여 들어간 컬럼)가 포함되어 있던 문제를
  02에서 바로잡았습니다.
- `02_preprocessing_and_skew_correction.ipynb`: 로컬 PC에서 실행 완료. **(2026-09-16 버그
  수정 후 재실행)** 일부 컬럼(6개)에서 Yeo-Johnson 변환이 예외 없이 "성공"하고도 극단적인
  λ 때문에 변환값이 수치적으로 상수에 가깝게 "조용히 붕괴"하는 문제를 발견해, 변환 후
  표준편차를 확인하는 가드를 `correct_skew_with_power_transform()`에 추가했습니다. 수정
  전에는 설계행렬 조건수가 7.28×10¹⁸(사실상 특이행렬)이었는데, 수정 후 1.27×10⁷로
  정상화되었습니다 (자세한 내용은 9-1-A 셀, 05 노트북 3-1절 참고).
- `03_model_comparison.ipynb`: 로컬 PC에서 실행 완료 (셀 1~15 오류 없이 순차 실행).
  01의 LinearRegression/SGDRegressor 재평가에 더해 Ridge/Lasso, Decision Tree/Random
  Forest/Gradient Boosting까지 총 7개 모델을 학습·비교했습니다. **최종 선정 모델은
  Lasso(alpha≈0.00068), Test R²=0.1186**로 01 베이스라인(R²=0.0414) 대비 약 2.9배
  개선되었습니다. LinearRegression 회귀계수 중 하나가 비정상적으로 크게(~1.6×10¹³) 나타나
  잔존 다중공선성이 있음을 확인했고, Random Forest/Gradient Boosting은 표본 크기(108건)
  대비 과적합이 뚜렷했습니다. 자세한 내용은 1차 보고서를 참고하세요.
- 1차 보고서 (`reports/01_model_comparison_report.md`, `.pdf`): **작성 완료.** 7개 모델
  val/test 비교표, 핵심 발견(Lasso 최고 성능, LinearRegression 계수 불안정성, 트리모델
  과적합), 한계, 다음 단계 제안을 담고 있습니다.
- `04_advanced_modeling.ipynb`: **로컬 PC에서 실행 완료** (셀 1~20 오류 없이 순차 실행).
  RandomForest `max_depth` 튜닝, Lasso 선택 피처 재학습, XGBoost/LightGBM 추가, K-Fold
  안정성 검증, VIF 재점검, Ridge/Lasso/GradientBoosting 피처 중요도 PNG 저장까지 1차
  보고서의 "다음 단계 제안" 6가지를 모두 다뤘습니다. 이 결과를 바탕으로 2차 보고서
  (`reports/02_advanced_modeling_report.md`, `.pdf`)를 작성 완료했습니다.
- 2차 보고서 (`reports/02_advanced_modeling_report.md`, `.pdf`): **작성 완료.**
- `05_improved_modeling.ipynb`: **로컬 PC에서 실행 완료** (셀 1~22 오류 없이 순차 실행).
  2차 보고서 이후 성능 개선을 탐색하는 과정에서 02의 왜도 보정 버그를 발견해 고쳤고, 그
  수정이 반영된 데이터로 VIF/조건수 재확인, 레버리지(leverage) 분석(고-레버리지 관측치
  3개를 배정번호까지 특정: 131758 / 118005 / 122460), 전체(108행) vs 고-레버리지 제외
  (105행) 두 버전으로 Ridge/ElasticNet/RandomForest/XGBoost/LightGBM 재학습, Test 최종
  검증까지 수행했습니다. **Val R²는 크게 개선(Lasso 0.0595 → 0.1103, 고-레버리지 3행
  제외)되었지만 Test R²는 기존과 거의 동일(0.1186 → 0.1176)** — 이는 전처리 버그가 있던
  기존 모델은 Val과 Test 성능이 서로 크게 어긋나 있었던 반면(불안정성의 신호), 수정 후에는
  Val·Test가 훨씬 일관되게 맞아떨어진다는 뜻으로 해석했습니다. K-Fold 재확인 결과는
  분산이 커서(표본 108~119건) 이 개선폭을 통계적으로 단정하기는 이르다는 한계도
  함께 기록했습니다.
- `06_further_experiments.ipynb`: **로컬 검증 환경(사용자 환경과 동일한 고정 버전 조합)에서
  실행 완료** (전 셀 오류 없이 순차 실행 확인, `random_state=42`로 재현 가능). 가중회귀·
  강건회귀·이항 GLM·반복 K-Fold·도메인 파생 피처(range) 5가지를 모두 시도했습니다.
  **핵심 결론: 5가지 중 어느 것도 04/05의 Lasso를 확실하게 뛰어넘지 못했고, 반복
  K-Fold(5-Fold×20회) 기준으로는 이 노트북에서 평가한 모든 후보(04/05의 기존 Lasso
  포함)의 평균 R²가 음수였습니다.** 다만 배치 크기 기반 가중치는 Ridge/Lasso/ElasticNet
  세 모델 전부에서 일관되게 비가중 버전보다 나은 성능을 보여, 이 노트북에서 찾은 유일한
  신뢰할 만한 개선이었습니다. 반대로 정규화 없는 이항 GLM은 배치별 표본 크기가 수십~6만
  건까지 차이 나는 데이터 특성 때문에 반복 K-Fold 평가 중 계수가 발산하는 심각한 수치적
  불안정성을 보였습니다(L2 정규화로 해소는 했으나 성능 자체는 가중회귀보다 낮았음). 자세한
  내용은 아래 "06 노트북" 절 참고.

## 분석 파이프라인 개요 (노트북 섹션 순서)

1. **원본 vs 품질전처리 데이터 비교** — 두 CSV의 스키마, dtype, 결측치, 배정번호 수,
   수집 기간을 비교합니다.
2. **label.xlsx 연관관계 분석** — 라벨의 양품/불량수량으로 `불량비율`(%) 파생변수를 만들고,
   센서 데이터를 배정번호 단위로 집계(min/max)한 뒤 라벨과 병합, 상관관계 히트맵으로
   연관관계를 확인합니다.
3. **전처리 파이프라인** — 상관계수 0.8 이상인 다중공선성 변수쌍을 자동으로 제거합니다.
4. **Train : Val : Test = 8 : 1 : 1 분할** — `src/training_utils.split_train_val_test()`
   함수로 분할하며, **Test 세트는 노트북의 "최종 테스트 평가" 셀 단 한 번만 사용**합니다.
   (그 전까지는 EDA/스케일링/튜닝 어디에도 사용하지 않습니다.)
5. **선형회귀 베이스라인 학습**
   - `LinearRegression`(closed-form, 즉시 수렴) 과
   - `SGDRegressor`(epoch 단위 반복학습)를 함께 학습합니다.
   - 반복학습은 **진행상황을 progress bar(tqdm)로 표시**하고, **매 10 epoch마다 체크포인트를
     저장**합니다 (`checkpoints/sgd_baseline.joblib`). 커널이 죽거나 중간에 실행이 끊겨도
     같은 셀을 다시 실행하면 저장된 체크포인트부터 이어서 학습합니다(resume).
     처음부터 다시 학습하려면 체크포인트 파일을 삭제하세요.
6. **모델 평가 및 지표 시각화** — R², RMSE, MAE 계산, 예측-실제 산점도, 잔차 플롯,
   학습 곡선을 그립니다.
7. **최종 테스트 평가** — 선택된 최종 모델로 Test 세트에 대해 단 한 번 성능을 확인합니다.
8. **전체 EDA + Feature Importance** — 모든 처리가 끝난 최종 데이터셋에 대한 분포 확인과,
   회귀계수 절댓값 기반 변수 중요도(어떤 공정 변수가 불량비율에 영향을 크게 주는지)를
   시각화합니다.

## 02 노트북: 전처리 보강 (Unnamed 분리 · 상관관계 히트맵 · Skew 보정)

01 노트북의 `data_processed/df_total.csv`를 이어받아 모델을 새로 학습하지 않고 아래 세
가지만 다룹니다. (모델 재학습은 03에서 진행)

1. **`Unnamed: 0`(원본 행 번호) 분리** — `품질전처리후데이터.csv`를 저장할 때 pandas
   인덱스가 그대로 컬럼으로 딸려 들어간 것으로, 01에서는 이 컬럼이 실수로 최종 모델 입력
   변수(`Unnamed: 0_max`)에 포함되어 있었습니다. 배정번호별 원본 행 범위는
   `data_processed/original_row_range.csv`로 따로 저장하고, 분석/모델링 데이터에서는
   완전히 제외합니다.
2. **독립변수 상관관계 히트맵** — 다중공선성 제거 전/후로 독립변수끼리의 상관관계
   히트맵을 각각 그려 비교합니다 (`figures/10_...`, `figures/11_...`).
3. **왜도(Skew) 보정** — `X_train` 기준으로 각 변수의 왜도를 진단하고, 치우친 변수에는
   Yeo-Johnson 파워변환(`sklearn.preprocessing.PowerTransformer`)을 적용합니다. 0/음수
   값에서도 안전하게 동작하고 변수마다 최적 변환 강도를 데이터로부터 자동으로 찾아준다는
   점에서 로그변환 대신 선택했습니다. 변환 파라미터는 train에만 `fit`하고 val/test는
   `transform`만 적용해 데이터 누수를 방지합니다 (`checkpoints/power_transformer.joblib`
   에 저장, 03에서 재사용 가능).

## 03 노트북: 모델 비교 (회귀 재평가 · 정규화 회귀 · 트리 계열)

02의 전처리 개선판 데이터에 `MinMaxScaler`(train에만 fit)를 한 번 더 적용한 뒤, 아래
7개 모델을 학습·비교합니다. (test는 "최종 테스트 평가" 셀에서 전체 모델을 한 번에
평가할 때만 사용)

1. **LinearRegression / SGDRegressor** — 01과 동일한 모델을 새 데이터로 재학습해
   전처리 개선 효과를 확인합니다.
2. **Ridge(L2) / Lasso(L1)** — 회귀계수에 페널티를 줘 과적합을 억제하는 정규화 회귀.
   Lasso는 규제가 강해지면 일부 계수를 정확히 0으로 만들어 자동 변수 선택 효과도
   있습니다. `alpha`는 val 성능 기준으로 탐색합니다.
3. **Decision Tree / Random Forest / Gradient Boosting** — 대표적인 트리 계열 3종.
   Random Forest는 SGD처럼 트리를 조금씩(`warm_start`) 늘려가며 tqdm 진행률 표시줄과
   체크포인트(`checkpoints/random_forest.joblib`)로 중단 후 재개가 가능합니다.
   Gradient Boosting은 `staged_predict()`로 부스팅 반복별 학습곡선을 사후에 재구성해
   최적 반복 수를 찾습니다.

## 04 노트북: 심화 모델링 (RandomForest 튜닝 · Lasso 피처선택 · XGBoost/LightGBM · K-Fold · VIF)

1차 보고서(`reports/01_model_comparison_report.md`)의 "다음 단계 제안"을 구현합니다.
03이 저장해 둔 `final_minmax_scaler.joblib`(train에만 fit)을 그대로 재사용해, 새로
학습하는 모델들을 03의 7개 모델과 동일한 전처리 위에서 비교합니다. **03 노트북은
1차 보고서의 근거이므로 수정하지 않고 그대로 보존**하고, 개선사항은 모두 04에서
새로 구현합니다.

1. **RandomForest `max_depth` 튜닝** — 03에서 탐색하지 않았던 `max_depth`를
   `grid_search_by_val_checkpoint()`(신규)로 탐색해 과적합(Test R²=-0.3150)을 개선합니다.
2. **Lasso 선택 피처만 재학습** — `select_nonzero_lasso_features()`(신규)로 03의 Lasso가
   선택한 피처만 추려 Ridge/RandomForest를 다시 학습하고, 전체 피처 사용 대비 성능을
   비교합니다.
3. **XGBoost / LightGBM** — `train_xgboost_with_checkpoint()` / `train_lightgbm_with_
   checkpoint()`(신규, 이전 booster에 이어서 학습하는 방식으로 체크포인트 재개 구현)로
   정규화가 내장된 부스팅 모델을 추가합니다.
4. **K-Fold 안정성 검증** — `kfold_stability_check()`(신규)로 최종 후보 모델들이 5-Fold
   교차검증에서도 비슷한 성능을 내는지 확인합니다 (하이퍼파라미터 탐색은 기존처럼
   Validation set 기준을 유지하고, K-Fold는 사후 검증용으로만 병행합니다).
5. **VIF 재점검** — `compute_vif()`(신규, statsmodels 기반)로 02의 pairwise 상관계수
   제거만으로 해소되지 않은 다중공선성이 남아있는지 정량적으로 확인합니다.
6. **피처 중요도 PNG 보완** — 03에서 셀 출력으로만 남아있던 Ridge/Lasso/GradientBoosting
   피처 중요도를 파일로 저장합니다.

신규 함수는 모두 `src/training_utils.py` 섹션 11에 있으며, 03에서 새 데이터를 열 때마다
Test는 "최종 테스트 비교" 셀에서 04가 새로 학습한 모델에 한해서만 한 번 사용하고, 03의
7개 모델 test 지표는 저장된 CSV를 그대로 재사용합니다(다시 예측하지 않음).

## 02 노트북 버그 수정 + 05 노트북: 전처리 근본 수정 반영 재학습

04 결과를 검토하며 성능을 더 끌어올릴 방법을 찾던 중, 02의 왜도 보정 단계에서 **"조용한
붕괴(silent collapse)" 버그**를 발견했습니다.

- **증상**: `건조로 온도 1/2 Zone_min` 등 6개 컬럼이 Yeo-Johnson 변환 후 108개 행 전부
  거의 동일한 값(표준편차 1e-15~1e-16 수준)으로 붕괴해, 사실상 정보가 없는 상수 컬럼이
  되어 있었습니다.
- **원인**: 값이 좁은 양수 범위(예: 97~100)에 몰려있는 컬럼에서, scipy의 Yeo-Johnson λ
  최적화가 예외를 던지지 않고 "성공"하면서도 -3~-7 수준의 극단적인 λ를 골라, 변환식
  `((x+1)^λ - 1) / λ`의 `(x+1)^λ` 항이 수치적으로 0에 언더플로우된 것이었습니다.
- **영향**: 이 근-상수 컬럼이 회귀식의 절편(intercept)과 거의 완벽한 공선성을 이루면서
  설계행렬 조건수가 극단적으로 악화되었습니다 — 01의 LinearRegression 계수 폭주
  (~1.6×10¹³)의 실제 원인이었습니다. VIF는 "다른 피처와의" 공선성만 측정하므로 이
  문제를 잡아내지 못합니다.
- **수정**: `src/training_utils.py`의 `correct_skew_with_power_transform()`에 변환 후
  train 표준편차가 임계값(기본 1e-3) 미만이면 원본값을 유지하는 가드(`collapse_std_
  threshold`)를 추가했습니다. 특정 컬럼을 하드코딩하지 않고 일반적인 수치 기준으로
  판단하므로, 앞으로 비슷한 문제가 다른 컬럼에서 생기더라도 자동으로 감지됩니다.

`05_improved_modeling.ipynb`는 이 수정이 반영된 데이터로:

1. 조건수/VIF를 재계산해 수정 효과를 검증하고,
2. Lasso를 다시 학습해 신뢰할 수 있는 피처를 재선택하고,
3. 레버리지(leverage) 분석으로 회귀에 과도한 영향을 주는 관측치를 배정번호까지 역추적하고
   (`position -> X_train.index[position] -> df_total.iloc[label]` 순서로 정확히 매핑 —
   위치와 라벨을 혼동하지 않도록 주의 깊게 구현),
4. 전체(108행)/고-레버리지 3행 제외(105행) 두 버전으로 Ridge/ElasticNet/RandomForest/
   XGBoost/LightGBM을 체크포인트+진행률 표시줄과 함께 재학습하고,
5. Test로 최종 검증합니다 (이 노트북에서 처음 Test를 엽니다).

03/04는 각각 1차/2차 보고서의 근거이므로 수정하지 않고 그대로 보존하며, 05가 만드는 모든
체크포인트/산출물 파일명에는 `v3_` 접두사를 붙여 기존 파일과 절대 겹치지 않도록 했습니다.

## 06 노트북: 가중회귀 · 강건회귀 · 이항 GLM · 반복 K-Fold · 도메인 파생 피처

2차 보고서 개정판 작성 후 "더 시도해볼만한 방법이 있는가?"라는 질문에 답하며 제안했던
5가지를 `06_further_experiments.ipynb`에서 모두 실행했습니다.

1. **가중회귀(Weighted Regression)** — 배치마다 표본 크기(양품+불량수량)가 수십 건에서
   6만 건 이상까지 차이 나는데, 지금까지 모든 모델이 108개 행을 동일한 신뢰도로 취급해
   왔다는 점을 보정합니다. `compute_sample_weights()`(신규)로 `sqrt(양품+불량수량)`에
   비례하는(평균 1로 정규화된) 표본 가중치를 계산하고, `grid_search_by_val()`과
   `repeated_kfold_evaluate()`에 `sample_weight` 인자로 전달합니다.
2. **강건회귀(Robust Regression)** — sklearn의 `HuberRegressor`/`RANSACRegressor`로,
   이상치(특히 05에서 수동으로 찾아 제외했던 고-레버리지 관측치)에 덜 민감한 적합을
   시도합니다.
3. **이항 GLM(Binomial GLM)** — 불량비율을 연속값이 아니라 원본 양품/불량 카운트에서 나온
   이항비율로 다시 정의해 `fit_binomial_glm()`/`predict_binomial_glm()`(신규,
   statsmodels 기반)으로 적합합니다. **중요한 발견**: 배치별 표본 크기 편차가 너무 커서,
   정규화 없는(alpha=0) GLM은 반복 K-Fold 평가 중 일부 fold에서 계수가 발산해 예측이
   수치적으로 폭발했습니다(평균 R² -7.1×10⁴). L2 정규화(`fit_regularized`, alpha=0.5~1)로
   해소했습니다.
4. **반복 K-Fold(Repeated K-Fold)** — 04/05의 사후 K-Fold(1회, 5개 fold 점수)만으로는
   fold 분할 방식에 따른 우연을 완전히 배제하기 어렵다는 판단에 따라,
   `repeated_kfold_evaluate()`(신규)로 5-Fold를 20회 반복(총 100개 fold 점수)해 95%
   신뢰구간까지 포함한 평가 기준으로 승격했습니다. 이 노트북에서는 하이퍼파라미터 선택과
   최종 후보 비교 모두 반복 K-Fold를 1차 기준으로 사용합니다(Val 14건 단일 분할 대신).
5. **도메인 파생 피처** — `add_range_features()`(신규)로 min/max 쌍이 모두 남아있는 센서에
   대해 "최솟값-최댓값 변동폭(range)" 피처를 추가합니다. ★ `range = max - min`은 완전
   공선성(정확한 선형관계)을 만들기 때문에, 이 함수는 range를 추가하는 대신 대응하는 max
   컬럼을 제거합니다(min은 유지) — 개발 중 이 처리를 하지 않았을 때 설계행렬 조건수가
   다시 ~1e18까지 폭발하는 것을 실제로 확인하고 고친 것입니다.

**핵심 결론**: 이 5가지 중 어느 것도 04/05의 Lasso를 확실하게 뛰어넘지 못했습니다. 반복
K-Fold(5-Fold×20회, train+val 105~122건) 기준으로는 이 노트북에서 평가한 모든 후보 —
04/05의 기존 Lasso를 포함해서 — 의 평균 R²가 음수였습니다. 다만 가중회귀는 Ridge/Lasso/
ElasticNet 세 모델 전부에서 일관되게 비가중 버전보다 나은 성능을 보여(예: Lasso -0.149 →
-0.099), 이 노트북에서 찾은 유일하게 신뢰할 만한 개선이었습니다. 강건회귀와 (정규화한)
이항 GLM은 가중회귀보다 낮은 성능을 보였고, range 피처는 효과가 불확실했습니다(신뢰구간이
일부 겹침). 최종 Test 검증(반복 K-Fold 상위 5개 후보만 대상으로, Test는 이 셀에서 처음
엶)에서도 Test R²는 0.02~0.03 수준으로, 04/05가 보고했던 0.1176~0.1186보다 낮았습니다 —
이는 04/05의 그 수치가 Val 14건 단일 분할에서 우연히 잘 맞은 결과였을 가능성을 시사합니다.
03/04/05는 그대로 보존되며, 06이 만드는 모든 체크포인트/산출물 파일명에는 `v4_` 접두사를
붙였습니다. 신규 함수는 모두 `src/training_utils.py` 섹션 12에 있습니다.

## 참고

- 데이터 출처: 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 열처리 공정최적화 AI
  데이터셋, 스마트제조혁신추진단(㈜임픽스), 2022.12.23., www.kamp-ai.kr
- 원본 가이드북은 **선형회귀**로 배정번호별 불량비율(연속값)을 예측하는 예제입니다. 이
  노트북도 동일하게 회귀(선형회귀) 문제로 다룹니다. (로지스틱 회귀로 분류 문제화하는 방안도
  검토했으나, 목표변수를 임의로 이진화해야 하는 문제가 있어 이번 베이스라인에서는 원본
  가이드북 방식(회귀)을 그대로 따르기로 결정했습니다.)
