# -*- coding: utf-8 -*-
"""
training_utils.py
==================
열처리 공정최적화 AI 데이터셋 - 선형회귀 베이스라인 파이프라인에서 공통으로 사용하는
전처리 / 학습(체크포인트 포함) / 평가 / 시각화 유틸리티 함수 모음.

이 모듈은 01_baseline_linear_regression.ipynb 노트북에서 import 되어 사용됩니다.
경진대회(KAMP AI) 연습을 위한 학습용 코드이므로, 각 함수에는 동작 설명 주석을
꼼꼼히 남겨두었습니다.

주요 구성
---------
0) 한글 폰트 설정 유틸          : set_korean_font
1) 데이터 품질 비교 유틸        : compare_missing, compare_schema
2) 배정번호 단위 집계 유틸      : aggregate_min_max
3) 다중공선성 제거 유틸         : remove_multicollinear_features
4) 데이터 분할 유틸 (8:1:1)     : split_train_val_test
5) 체크포인트 매니저            : CheckpointManager
6) 진행상황 표시 + 재개 가능한 학습 루프 : train_sgd_regressor_with_checkpoint
7) 평가 지표 계산               : evaluate_regression
8) 시각화 함수들                : plot_pred_vs_actual, plot_residuals,
                                  plot_training_curve, plot_feature_importance,
                                  plot_missing_comparison
9) EDA 유틸 (02 노트북)         : plot_correlation_heatmap, check_skewness,
                                  correct_skew_with_power_transform,
                                  plot_skew_before_after
10) 모델 비교 유틸 (03 노트북)  : train_random_forest_with_checkpoint,
                                  staged_learning_curve, grid_search_by_val,
                                  plot_model_comparison
11) 심화 개선 유틸 (04 노트북)  : compute_vif, select_nonzero_lasso_features,
                                  grid_search_by_val_checkpoint,
                                  train_xgboost_with_checkpoint,
                                  train_lightgbm_with_checkpoint,
                                  kfold_stability_check
"""

from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 0) 한글 폰트 설정 유틸 (그래프 한글 깨짐 방지)
# ---------------------------------------------------------------------------

def set_korean_font(preferred: str = "NanumGothic", verbose: bool = True) -> str:
    """matplotlib 그래프에서 한글(변수명, 제목 등)이 네모(□)로 깨지는 문제를 방지하기
    위해 나눔고딕(NanumGothic) 폰트를 찾아 적용한다.

    동작 방식
    ---------
    1) 시스템에 설치된 폰트 목록(matplotlib.font_manager)에서 'preferred'와 정확히
       일치하는 폰트를 우선 찾는다.
    2) 정확히 일치하지 않으면, 폰트 이름에 'nanum' 과 'gothic' 이 모두 포함된 폰트를
       대소문자 구분 없이 검색한다 (설치 환경에 따라 'Nanum Gothic' 처럼 표기가 다를
       수 있기 때문).
    3) 그래도 못 찾으면 나눔고딕이 설치되어 있지 않은 것으로 보고, 콘솔에 설치 안내
       메시지를 출력한 뒤 기본 sans-serif 폰트로 대체한다 (프로그램이 죽지 않도록).
    4) 마지막으로 그래프의 마이너스(-) 부호가 깨지는 문제도 함께 방지한다
       (한글 폰트를 쓰면 유니코드 마이너스 기호가 깨지는 경우가 있음).

    Parameters
    ----------
    preferred : 우선적으로 사용할 폰트 이름 (기본값: "NanumGothic")
    verbose : True 면 적용 결과를 print 로 출력

    Returns
    -------
    실제로 적용된 폰트 이름 (찾지 못한 경우 "sans-serif")
    """
    installed_names = {f.name for f in fm.fontManager.ttflist}

    # 1) 정확히 일치하는 이름 우선 탐색
    candidates = [preferred, "Nanum Gothic", "NanumGothicOTF", "NanumGothic Coding"]
    found = next((name for name in candidates if name in installed_names), None)

    # 2) 대소문자 구분 없이 'nanum' + 'gothic' 포함 여부로 재탐색
    if found is None:
        found = next(
            (name for name in installed_names
             if "nanum" in name.lower() and "gothic" in name.lower()),
            None,
        )

    if found:
        plt.rcParams["font.family"] = found
        applied = found
        if verbose:
            print(f"[한글 폰트] '{found}' 적용 완료.")
    else:
        plt.rcParams["font.family"] = "sans-serif"
        applied = "sans-serif"
        if verbose:
            print("[한글 폰트] ⚠️ 나눔고딕(NanumGothic) 폰트를 찾지 못했습니다. "
                  "그래프의 한글이 네모(□)로 깨질 수 있습니다.")
            print("  설치 방법 (Windows):")
            print("   1) https://hangeul.naver.com/font 또는 "
                  "https://fonts.google.com/specimen/Nanum+Gothic 에서 '나눔고딕' 다운로드")
            print("   2) 다운로드한 폰트 파일을 더블클릭 후 '설치' 클릭")
            print("   3) Jupyter 커널을 재시작(Kernel > Restart)한 뒤 이 셀을 다시 실행")

    # 한글 폰트 적용 시 마이너스(-) 기호가 깨지는 문제 방지
    plt.rcParams["axes.unicode_minus"] = False
    return applied


# ---------------------------------------------------------------------------
# 1) 데이터 품질 비교 유틸
# ---------------------------------------------------------------------------

def compare_missing(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """raw 데이터와 품질전처리 후 데이터의 결측치 개수를 컬럼별로 비교한다.

    Parameters
    ----------
    df_raw : 원본(raw_total_data) 데이터프레임
    df_clean : 품질전처리 후 데이터프레임

    Returns
    -------
    두 데이터셋의 결측치 개수, 결측 비율(%), 차이를 담은 비교표(DataFrame).
    컬럼 집합이 서로 다를 수 있으므로 union 을 기준으로 비교한다.
    """
    raw_na = df_raw.isna().sum()
    clean_na = df_clean.isna().sum()
    all_cols = sorted(set(raw_na.index) | set(clean_na.index))

    rows = []
    for col in all_cols:
        raw_cnt = int(raw_na.get(col, np.nan)) if col in raw_na.index else np.nan
        clean_cnt = int(clean_na.get(col, np.nan)) if col in clean_na.index else np.nan
        raw_pct = raw_cnt / len(df_raw) * 100 if col in df_raw.columns else np.nan
        clean_pct = clean_cnt / len(df_clean) * 100 if col in df_clean.columns else np.nan
        rows.append({
            "column": col,
            "raw_missing_count": raw_cnt,
            "raw_missing_pct": raw_pct,
            "clean_missing_count": clean_cnt,
            "clean_missing_pct": clean_pct,
            "in_raw_only": col not in df_clean.columns,
            "in_clean_only": col not in df_raw.columns,
        })
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def compare_schema(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """raw 데이터와 품질전처리 후 데이터의 스키마(컬럼, dtype, 행 수)를 비교한다.

    행(row) 수, 컬럼 수, 컬럼별 dtype 변화, 메모리 사용량 등을 한 번에 확인할 수
    있도록 요약 리포트를 만든다.
    """
    summary = {
        "raw_shape": df_raw.shape,
        "clean_shape": df_clean.shape,
        "raw_memory_MB": round(df_raw.memory_usage(deep=True).sum() / 1e6, 2),
        "clean_memory_MB": round(df_clean.memory_usage(deep=True).sum() / 1e6, 2),
        "columns_only_in_raw": sorted(set(df_raw.columns) - set(df_clean.columns)),
        "columns_only_in_clean": sorted(set(df_clean.columns) - set(df_raw.columns)),
        "row_count_diff": df_raw.shape[0] - df_clean.shape[0],
    }

    common_cols = sorted(set(df_raw.columns) & set(df_clean.columns))
    dtype_rows = []
    for col in common_cols:
        dtype_rows.append({
            "column": col,
            "raw_dtype": str(df_raw[col].dtype),
            "clean_dtype": str(df_clean[col].dtype),
            "dtype_changed": str(df_raw[col].dtype) != str(df_clean[col].dtype),
        })
    dtype_df = pd.DataFrame(dtype_rows)
    return summary, dtype_df


# ---------------------------------------------------------------------------
# 2) 배정번호 단위 집계 유틸 (가이드북 방식: 배정번호별 min/max 통계량 산출)
# ---------------------------------------------------------------------------

def aggregate_min_max(df: pd.DataFrame, id_col: str = "배정번호",
                       time_col: str = "TAG_MIN") -> pd.DataFrame:
    """센서 원시 데이터(초 단위)를 배정번호(batch id) 단위로 집계한다.

    각 배정번호(batch) 별로 센서 값들의 최솟값/최댓값을 구해 '_min', '_max'
    접미사를 붙인 뒤 옆으로 이어 붙인다 (가이드북 [코드 7],[코드 8],[코드 10] 로직).
    time_col(TAG_MIN)은 통계 집계 대상이 아니므로 제외한다.

    Parameters
    ----------
    df : TAG_MIN, 배정번호, 센서 컬럼들을 포함한 원시(혹은 품질전처리) 데이터
    id_col : 배정번호 컬럼명
    time_col : 시간 컬럼명 (집계에서 제외)

    Returns
    -------
    배정번호 1행 = 1 batch 인 wide-format DataFrame (컬럼: 배정번호, X_min, X_max, ...)
    """
    feature_cols = [c for c in df.columns if c not in (id_col, time_col)]

    df_min = df.groupby(id_col)[feature_cols].min().reset_index()
    df_min.columns = [id_col] + [f"{c}_min" for c in feature_cols]

    df_max = df.groupby(id_col)[feature_cols].max().reset_index()
    df_max.columns = [id_col] + [f"{c}_max" for c in feature_cols]

    df_agg = pd.merge(df_min, df_max, on=id_col, how="inner")
    return df_agg


def make_defect_rate(df_label: pd.DataFrame, good_col: str = "양품수량",
                      bad_col: str = "불량수량", out_col: str = "불량비율") -> pd.DataFrame:
    """라벨 데이터의 양품수량/불량수량으로 '불량비율'(%) 파생변수를 만든다."""
    df_label = df_label.copy()
    df_label[out_col] = df_label[bad_col] / (df_label[bad_col] + df_label[good_col]) * 100
    return df_label


# ---------------------------------------------------------------------------
# 3) 다중공선성 제거 유틸
# ---------------------------------------------------------------------------

def remove_multicollinear_features(X: pd.DataFrame, threshold: float = 0.8) -> tuple[pd.DataFrame, list[dict]]:
    """독립변수 간 상관계수가 threshold 이상인 변수쌍 중 하나를 반복적으로 제거한다.

    가이드북 4)-2 단계처럼 '상관관계가 높은 두 변수 중 하나만 남긴다'는 규칙을
    일반화한 함수. 매 반복(iteration)마다 상관행렬에서 절대값이 가장 큰 쌍을 찾아
    그 중 다른 변수들과의 평균 절대 상관관계가 더 큰 쪽(=중복 정보가 더 많은 쪽)을
    제거한다. threshold 미만이 될 때까지 반복한다.

    Parameters
    ----------
    X : 독립변수만 포함된 DataFrame (수치형)
    threshold : 이 값 이상으로 상관된 변수쌍이 있으면 하나를 제거

    Returns
    -------
    (변수가 제거된 DataFrame, 제거 로그(list of dict: 제거된 컬럼/사유/상관계수))
    """
    X = X.copy()
    drop_log = []

    while True:
        corr = X.corr().abs()
        np.fill_diagonal(corr.values, 0.0)
        max_corr = corr.values.max()
        if max_corr < threshold or X.shape[1] <= 1:
            break

        # 임계값을 넘는 쌍 중 상관계수가 가장 큰 쌍을 찾는다
        i, j = np.unravel_index(np.argmax(corr.values), corr.shape)
        col_i, col_j = corr.index[i], corr.columns[j]

        # 두 변수 중, 나머지 변수들과의 평균 절대 상관관계가 더 큰(중복정보가 많은) 변수를 제거
        mean_corr_i = corr[col_i].mean()
        mean_corr_j = corr[col_j].mean()
        drop_col = col_i if mean_corr_i >= mean_corr_j else col_j
        keep_col = col_j if drop_col == col_i else col_i

        drop_log.append({
            "dropped": drop_col,
            "kept": keep_col,
            "correlation": float(max_corr),
        })
        X = X.drop(columns=[drop_col])

    return X, drop_log


# ---------------------------------------------------------------------------
# 4) 데이터 분할 유틸 (train : val : test = 8 : 1 : 1)
# ---------------------------------------------------------------------------

def split_train_val_test(X: pd.DataFrame, y: pd.Series, ratios=(0.8, 0.1, 0.1),
                          random_state: int = 42):
    """전체 데이터를 train : val : test = 8 : 1 : 1 비율로 분할한다.

    주의(★매우 중요★): 이 함수가 반환하는 test 세트(X_test, y_test)는
    '최종 성능 평가' 셀 이외의 어떤 단계에서도 절대 열어보거나 학습/스케일링에
    사용해서는 안 된다. (데이터 누수 방지 원칙)

    내부적으로 train_test_split 을 두 번 사용한다:
      1) 전체 -> (train+val) : test  = 0.9 : 0.1
      2) (train+val) -> train : val  = 0.8/0.9 : 0.1/0.9  (전체 기준 8:1)

    Parameters
    ----------
    X, y : 독립변수, 종속변수(불량비율)
    ratios : (train, val, test) 비율. 합이 1.0 이어야 한다.
    random_state : 재현성을 위한 시드값

    Returns
    -------
    dict(X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios 합은 1.0 이어야 합니다."

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )

    # trainval 내부에서 val 이 차지해야 할 비율로 재계산
    val_ratio_within_trainval = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio_within_trainval, random_state=random_state
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# 5) 체크포인트 매니저 (중단 후 재개를 위한 저장/복원)
# ---------------------------------------------------------------------------

class CheckpointManager:
    """학습 중 중간에 끊기는 상황(커널 재시작, 정전, 실수로 셀 중단 등)을 대비해
    모델/스케일러/학습 진행상태를 주기적으로 디스크에 저장하고, 다시 불러와
    이어서 학습(resume)할 수 있게 해주는 간단한 매니저.

    저장 파일: {checkpoint_dir}/{name}.joblib
    저장 내용: {"epoch", "model", "scaler", "best_val_loss", "history"} 등 dict
    """

    def __init__(self, checkpoint_dir: str, name: str = "sgd_baseline"):
        self.checkpoint_dir = checkpoint_dir
        self.name = name
        os.makedirs(checkpoint_dir, exist_ok=True)

    @property
    def path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"{self.name}.joblib")

    def save(self, state: dict) -> None:
        """state(dict)를 통째로 joblib 으로 직렬화하여 저장한다."""
        tmp_path = self.path + ".tmp"
        joblib.dump(state, tmp_path)
        os.replace(tmp_path, self.path)  # 저장 도중 중단되어도 기존 체크포인트가 깨지지 않도록 원자적 교체

    def load(self) -> Optional[dict]:
        """저장된 체크포인트가 있으면 불러오고, 없으면 None 을 반환한다."""
        if os.path.exists(self.path):
            return joblib.load(self.path)
        return None

    def exists(self) -> bool:
        return os.path.exists(self.path)


# ---------------------------------------------------------------------------
# 6) 진행상황 표시 + 체크포인트 재개가 가능한 학습 루프
# ---------------------------------------------------------------------------

def train_sgd_regressor_with_checkpoint(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    checkpoint_mgr: CheckpointManager,
    n_epochs: int = 200,
    checkpoint_every: int = 10,
    patience: int = 30,
    random_state: int = 42,
    resume: bool = True,
) -> dict:
    """SGDRegressor 를 epoch 단위 반복학습(partial_fit)으로 학습시키면서,
    (1) tqdm 진행률 표시줄로 학습 진행상황을 보여주고,
    (2) 매 checkpoint_every epoch 마다 모델 상태를 저장하며,
    (3) resume=True 이고 기존 체크포인트가 있으면 그 지점부터 이어서 학습한다.

    선형회귀(LinearRegression)는 최소제곱법으로 한 번에(closed-form) 풀리기 때문에
    '학습 진행상황'이라는 개념이 없다. 그래서 동일한 선형모델을 확률적 경사하강법
    (SGD)으로 반복 학습시켜, 중간에 끊겨도 재개할 수 있는 학습 과정을 보여준다.
    (최종 베이스라인 비교에는 LinearRegression 결과도 함께 사용한다.)

    Parameters
    ----------
    X_train, y_train, X_val, y_val : 스케일링이 완료된 학습/검증 데이터
    checkpoint_mgr : CheckpointManager 인스턴스
    n_epochs : 최대 epoch 수
    checkpoint_every : 몇 epoch 마다 체크포인트를 저장할지
    patience : 검증 손실이 이만큼 연속으로 개선되지 않으면 조기 종료(early stopping)
    random_state : 재현성 시드
    resume : True 면 기존 체크포인트가 있을 때 이어서 학습

    Returns
    -------
    dict(model=최종(=best) 모델, history=epoch별 train/val loss 기록 DataFrame)
    """
    start_epoch = 0
    history = []
    best_val_loss = np.inf
    best_model = None
    no_improve_count = 0

    model = SGDRegressor(
        random_state=random_state,
        max_iter=1,       # partial_fit 을 직접 반복 호출할 것이므로 내부 반복은 1로 고정
        warm_start=True,
        learning_rate="invscaling",
        eta0=0.01,
    )

    # ---- 체크포인트에서 이어서 학습(resume) ----
    if resume and checkpoint_mgr.exists():
        state = checkpoint_mgr.load()
        model = state["model"]
        start_epoch = state["epoch"] + 1
        history = state["history"]
        best_val_loss = state["best_val_loss"]
        best_model = state.get("best_model", model)
        print(f"[체크포인트 발견] epoch {start_epoch} 부터 이어서 학습을 재개합니다. "
              f"(기존 best_val_loss={best_val_loss:.4f})")
    else:
        # 최초 1회 partial_fit 호출 시 필요한 초기화
        model.partial_fit(X_train, y_train)

    pbar = tqdm(range(start_epoch, n_epochs), desc="SGD 학습 진행", unit="epoch")
    for epoch in pbar:
        model.partial_fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_loss = mean_squared_error(y_train, train_pred)
        val_loss = mean_squared_error(y_val, val_pred)

        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        pbar.set_postfix({"train_mse": f"{train_loss:.3f}", "val_mse": f"{val_loss:.3f}"})

        # ---- best 모델 갱신 & early stopping 카운트 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = _clone_sklearn_model(model)
            no_improve_count = 0
        else:
            no_improve_count += 1

        # ---- 주기적 체크포인트 저장 (중단 대비) ----
        if (epoch + 1) % checkpoint_every == 0 or epoch == n_epochs - 1:
            checkpoint_mgr.save({
                "epoch": epoch,
                "model": model,
                "best_model": best_model,
                "best_val_loss": best_val_loss,
                "history": history,
            })

        if no_improve_count >= patience:
            print(f"\n[조기 종료] epoch {epoch} : 검증 손실이 {patience}회 연속 개선되지 않아 학습을 종료합니다.")
            checkpoint_mgr.save({
                "epoch": epoch,
                "model": model,
                "best_model": best_model,
                "best_val_loss": best_val_loss,
                "history": history,
            })
            break

    return {"model": best_model if best_model is not None else model,
            "history": pd.DataFrame(history)}


def _clone_sklearn_model(model):
    """SGDRegressor 의 현재 학습 상태를 깊은 복사하여 반환한다 (best 모델 스냅샷 저장용)."""
    import copy
    return copy.deepcopy(model)


# ---------------------------------------------------------------------------
# 7) 평가 지표 계산
# ---------------------------------------------------------------------------

def evaluate_regression(y_true, y_pred) -> dict:
    """회귀 모델의 대표 평가지표(R2, RMSE, MAE)를 계산해 dict 로 반환한다."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


# ---------------------------------------------------------------------------
# 8) 시각화 함수들
# ---------------------------------------------------------------------------

def plot_missing_comparison(missing_df: pd.DataFrame, top_n: int = 20):
    """raw vs 품질전처리 데이터의 결측치 비율을 컬럼별 막대그래프로 비교한다."""
    plot_df = missing_df.copy()
    plot_df = plot_df.sort_values("raw_missing_pct", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos - 0.2, plot_df["raw_missing_pct"], height=0.4, label="raw_total_data")
    ax.barh(y_pos + 0.2, plot_df["clean_missing_pct"], height=0.4, label="품질전처리후데이터")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["column"])
    ax.invert_yaxis()
    ax.set_xlabel("결측치 비율 (%)")
    ax.set_title("컬럼별 결측치 비율 비교 (raw vs 품질전처리후)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(y_true, y_pred, title: str = "예측값 vs 실제값"):
    """실제값 대비 예측값 산점도 (대각선에 가까울수록 예측이 정확함)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", label="y = x (완벽한 예측)")
    ax.set_xlabel("실제 불량비율 (%)")
    ax.set_ylabel("예측 불량비율 (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, title: str = "잔차(Residual) 플롯"):
    """잔차(실제값-예측값)가 0 주변에 무작위로 퍼져 있는지 확인하는 플롯."""
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolor="k")
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].set_xlabel("예측값")
    axes[0].set_ylabel("잔차 (실제-예측)")
    axes[0].set_title("잔차 vs 예측값")

    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set_title("잔차 분포")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_training_curve(history: pd.DataFrame, title: str = "학습 곡선 (Training Curve)",
                         xlabel: str = "Epoch"):
    """반복학습(SGD의 epoch, RandomForest의 트리 추가 step, GradientBoosting의 부스팅
    반복 등)에서 단계별 train/val MSE 변화를 그린다 (학습 진행상황 시각화).

    history 는 최소한 'epoch', 'train_mse', 'val_mse' 세 컬럼을 가진 DataFrame이어야
    한다 (실제 의미가 epoch이 아니어도 — 예: RandomForest의 step — 호출 전에
    해당 컬럼명을 'epoch'으로 맞춰서 넘기면 된다). title/xlabel 인자로 어떤 학습
    루프의 곡선인지 구분해 표시할 수 있다 (기본값은 01 노트북의 SGD 학습곡선과 동일).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_mse"], label="Train MSE")
    ax.plot(history["epoch"], history["val_mse"], label="Val MSE")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_names, coefficients, top_n: int = 20,
                             title: str = "Feature Importance (|회귀계수|)"):
    """선형회귀 계수의 절댓값 크기로 변수 중요도를 시각화한다.
    계수의 부호(양/음)도 색으로 함께 표현하여, 어떤 변수가 불량비율을
    증가/감소시키는 방향인지 한눈에 볼 수 있게 한다.
    """
    imp_df = pd.DataFrame({"feature": feature_names, "coef": coefficients})
    imp_df["abs_coef"] = imp_df["coef"].abs()
    imp_df = imp_df.sort_values("abs_coef", ascending=False).head(top_n)

    colors = ["#d62728" if c > 0 else "#1f77b4" for c in imp_df["coef"]]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(imp_df))))
    ax.barh(imp_df["feature"][::-1], imp_df["abs_coef"][::-1], color=colors[::-1])
    ax.set_xlabel("|회귀계수| (정규화된 스케일 기준)")
    ax.set_title(title + "  (빨강=불량률 증가 방향, 파랑=불량률 감소 방향)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9) EDA 유틸 (02 노트북): 독립변수 상관관계 히트맵 / 왜도(skew) 진단 및 보정
#    -> 02_preprocessing_and_skew_correction.ipynb 에서 import 되어 사용됩니다.
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, title: str = "변수 간 상관관계 히트맵",
                              figsize: tuple = (14, 12), annot: bool = False) -> plt.Figure:
    """수치형 컬럼들 사이의 피어슨 상관계수 히트맵을 그린다.

    df 에 있는 모든 컬럼을 대상으로 상관행렬을 계산해 RdBu_r 컬러맵으로 시각화한다.
    독립변수 간 다중공선성(서로 강하게 연관된 변수쌍)을 한눈에 파악하는 용도로 쓴다.
    (01 노트북의 상관관계 히트맵은 라벨 변수까지 포함했지만, 이 함수는 독립변수만
    넣어서 '독립변수끼리의' 다중공선성 확인에 집중할 수 있도록 범용으로 만들었다.)

    Parameters
    ----------
    df : 상관관계를 볼 수치형 컬럼들로만 구성된 DataFrame (독립변수만, 식별자/목표변수는 제외 권장)
    title : 그래프 제목
    figsize : 그림 크기
    annot : True 이면 각 셀에 상관계수 숫자를 함께 표시 (변수가 많으면 지저분해지므로 기본 False)

    Returns
    -------
    matplotlib Figure
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap="RdBu_r", center=0, square=True,
                linewidths=0.3, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def check_skewness(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """DataFrame 의 각 수치형 컬럼에 대해 왜도(skewness)를 계산하고,
    절댓값이 threshold 를 넘으면 '치우친(skewed) 변수'로 표시한다.

    왜도(skew)가 0에 가까우면 좌우 대칭(정규분포에 가까움)이고, 절댓값이 클수록
    한쪽으로 꼬리가 길게 치우친 분포임을 의미한다. 통계적으로 흔히 쓰이는 기준으로
    |skew| > 0.5 이면 '약간 치우침', > 1.0 이면 '많이 치우침'으로 본다
    (본 함수 기본 threshold=0.5, 즉 '약간이라도 치우치면' 보정 대상으로 표시).

    ★ 반드시 train 데이터로만 계산할 것 (val/test 포함 시 데이터 누수 발생).

    Parameters
    ----------
    df : 왜도를 계산할 수치형 컬럼들로 구성된 DataFrame (train 데이터만 사용)
    threshold : 이 값을 넘는 |skew| 를 가진 컬럼을 '치우친 변수'로 분류

    Returns
    -------
    컬럼별 skew, abs_skew, is_skewed(bool) 를 담은 DataFrame (abs_skew 내림차순 정렬)
    """
    skew_series = df.skew(numeric_only=True)
    result = pd.DataFrame({
        "column": skew_series.index,
        "skew": skew_series.values,
    })
    result["abs_skew"] = result["skew"].abs()
    result["is_skewed"] = result["abs_skew"] > threshold
    result = result.sort_values("abs_skew", ascending=False).reset_index(drop=True)
    return result


def correct_skew_with_power_transform(X_train: pd.DataFrame, X_val: pd.DataFrame,
                                       X_test: pd.DataFrame, skewed_cols: list,
                                       method: str = "yeo-johnson",
                                       collapse_std_threshold: float = 1e-3) -> dict:
    """치우친(skewed) 변수들에 파워변환(Power Transform)을 적용해 분포를 정규분포에 가깝게 만든다.

    Yeo-Johnson 변환이란?
    ----------------------
    Box-Cox 변환의 확장판으로, 데이터에 0이나 음수가 섞여 있어도 적용할 수 있는
    데이터 기반(data-driven) 단조(monotonic) 변환이다. 컬럼마다 최적의 변환 강도
    (lambda)를 훈련 데이터로부터 자동으로 찾아 학습(fit)하며, log/sqrt 변환처럼
    사람이 직접 변환 종류를 골라야 하는 것과 달리 왜도를 줄이는 방향으로 자동
    보정해준다. 이 프로젝트의 센서 집계값(_min/_max)에는 0에 가깝거나 이론상
    음수가 나올 수 있는 값도 섞여 있어, 0/음수에서 정의되지 않는 log 변환보다
    Yeo-Johnson이 더 안전하다고 판단해 선택했다. (standardize=True 로 설정해
    변환 후 평균 0 · 표준편차 1 로도 함께 맞춘다.)

    ★ 데이터 누수 방지: 변환 파라미터(lambda, 평균/표준편차)는 반드시 train
    데이터에만 fit 하고, val/test 에는 학습된 파라미터로 transform 만 적용한다.
    (01 노트북의 MinMaxScaler와 동일한 원칙)

    Parameters
    ----------
    X_train, X_val, X_test : 8:1:1로 분할된 독립변수 (컬럼 구성이 동일해야 함)
    skewed_cols : check_skewness() 로 찾아낸, 치우침이 심한 컬럼 이름 리스트
                  (이 컬럼들만 변환하고, 나머지 컬럼은 원본값 그대로 둔다)
    method : sklearn PowerTransformer 의 method ("yeo-johnson" 기본값)
    collapse_std_threshold : 변환(및 표준화) 후 train 표준편차가 이 값 미만이면
                  "조용한 붕괴"로 간주해 원본값을 유지한다 (기본 1e-3; 정상적으로
                  변환된 컬럼은 표준화 덕분에 std가 1.0에 매우 가까우므로, 1e-3은
                  정상 변환과 붕괴된 변환을 확실히 구분할 수 있는 보수적인 값)

    ★ 컬럼별로 하나씩 개별 fit 한다 (한 번에 여러 컬럼을 묶어서 fit 하지 않는 이유는
    아래 '왜 컬럼별로 개별 fit 하는가?' 참고). 이 중 극단적으로 치우쳐 있거나
    (예: 대부분 0에 가깝고 소수만 매우 큰 값) 샘플 수가 적어 최적화가 실패하는
    컬럼이 있으면(scipy 최적화 과정에서 BracketError 등 발생), 그 컬럼만 건너뛰고
    원본 값을 그대로 두며 경고 메시지를 출력한다. 나머지 컬럼은 정상적으로 변환된다.

    ★ (중요) '조용한 붕괴(silent collapse)' 방지 가드: scipy 최적화가 예외 없이
    "성공"하더라도, 값이 좁은 양수 범위에 몰려있는 컬럼(예: 97~100 사이의 센서
    집계값)에서는 lambda가 -3~-7 수준의 극단적으로 작은 값으로 수렴하는 경우가
    있다. 이 경우 Yeo-Johnson 공식 ((x+1)^lambda - 1) / lambda 에서 (x+1)^lambda
    항이 수치적으로 0에 언더플로우되어, train 표본 전부가 거의 동일한 값으로
    붕괴해버린다 (표준화 후 std가 1.0이어야 정상인데 1e-15~1e-16 수준으로 떨어짐).
    이렇게 되면 겉보기엔 "변환 성공"이지만 실제로는 해당 컬럼의 정보가 전부
    사라지고, 이 근-상수 컬럼이 회귀식의 절편(intercept)과 거의 완벽한 공선성을
    이루면서 설계행렬의 조건수(condition number)를 극단적으로 악화시킨다
    (LinearRegression 계수 폭주의 실제 원인이었음 — 04/05 노트북 참고). 이는
    scipy 예외를 던지지 않으므로 기존 try/except 만으로는 잡을 수 없어, fit 직후
    train 데이터를 transform 한 결과의 표준편차를 확인하는 별도 가드를 둔다:
    표준편차가 collapse_std_threshold 미만이면 "붕괴된 변환"으로 간주하고
    failed_columns 에 추가한 뒤 원본 값을 그대로 사용한다.

    왜 컬럼별로 개별 fit 하는가?
    ---------------------------
    PowerTransformer 는 여러 컬럼을 한 번에 넣어도 내부적으로 컬럼마다 독립적으로
    lambda 를 최적화하기 때문에 결과 자체는 컬럼별로 fit 하는 것과 동일하다. 다만
    여러 컬럼을 한 번에 fit 하면 그 중 단 하나의 컬럼에서만 최적화가 실패해도
    (scipy.optimize.brent 가 유효한 구간(bracket)을 못 찾는 경우 등) 전체 fit 호출이
    예외를 던지며 멈춰버린다. 컬럼별로 개별 fit 하면 실패한 컬럼만 건너뛰고 나머지는
    정상적으로 진행할 수 있어 훨씬 안전하다.

    Parameters
    ----------
    X_train, X_val, X_test : 8:1:1로 분할된 독립변수 (컬럼 구성이 동일해야 함)
    skewed_cols : check_skewness() 로 찾아낸, 치우침이 심한 컬럼 이름 리스트
                  (이 컬럼들만 변환하고, 나머지 컬럼은 원본값 그대로 둔다)
    method : sklearn PowerTransformer 의 method ("yeo-johnson" 기본값)
    collapse_std_threshold : 위 '조용한 붕괴 방지 가드' 설명 참고 (기본 1e-3)

    Returns
    -------
    dict(
      X_train, X_val, X_test : 변환이 적용된 DataFrame (변환에 실패했거나 애초에
                                치우치지 않은 컬럼은 원본 그대로),
      transformer            : {컬럼명: 학습된 PowerTransformer 객체} 딕셔너리
                                (성공적으로 변환된 컬럼만 포함. 다음 노트북에서
                                재사용하려면 joblib 으로 저장해 둘 것),
      transformed_columns    : 실제로 변환에 성공한 컬럼 이름 리스트,
      failed_columns         : [(컬럼명, 사유), ...] 원본값을 그대로 둔 컬럼 목록
                                (scipy 최적화 자체가 예외를 던진 경우 "실패 원인:
                                에러메시지" 형태, collapse_std_threshold 가드에
                                걸린 경우 "CollapsedTransform: ..." 형태로 구분되며,
                                둘 다 없으면 빈 리스트),
      skew_before_after      : 변환에 성공한 컬럼들의 skew 전/후 비교 DataFrame,
    )
    """
    from sklearn.preprocessing import PowerTransformer

    X_train_out = X_train.copy()
    X_val_out = X_val.copy()
    X_test_out = X_test.copy()

    if len(skewed_cols) == 0:
        print("치우친(skewed) 변수가 없어 파워변환을 적용하지 않습니다.")
        return {
            "X_train": X_train_out, "X_val": X_val_out, "X_test": X_test_out,
            "transformer": {}, "transformed_columns": [], "failed_columns": [],
            "skew_before_after": pd.DataFrame(),
        }

    skew_before = X_train[skewed_cols].skew()

    transformers = {}
    transformed_columns = []
    failed_columns = []

    collapsed_columns = []

    for col in skewed_cols:
        pt_col = PowerTransformer(method=method, standardize=True)
        try:
            pt_col.fit(X_train[[col]])
        except Exception as e:
            # 대부분 '해당 컬럼의 분산이 거의 0이거나(값이 대부분 동일), 값이 한쪽으로
            # 극단적으로 몰려 있어(0 근처에 대부분, 소수만 큰 값 등) scipy 최적화가
            # 유효한 구간(bracket)을 찾지 못한 경우'에 발생한다. 표본 수(train 108개
            # 처럼 적은 경우)가 적을수록 이런 실패가 나기 쉽다.
            failed_columns.append((col, f"{type(e).__name__}: {e}"))
            continue

        # '조용한 붕괴' 가드: 예외 없이 fit 이 끝났어도, 변환된 train 값의 표준편차가
        # 비정상적으로 작으면(정상 변환은 standardize=True 로 인해 std≈1.0 이어야 함)
        # lambda가 극단적으로 작아져 (x+1)^lambda 항이 언더플로우된 것으로 보고
        # 원본값을 유지한다 (자세한 이유는 함수 docstring 참고).
        transformed_train_std = float(np.std(pt_col.transform(X_train[[col]])))
        if transformed_train_std < collapse_std_threshold:
            lam = float(pt_col.lambdas_[0])
            failed_columns.append((
                col,
                f"CollapsedTransform: lambda={lam:.4f} 로 변환 후 train std="
                f"{transformed_train_std:.3e} (<{collapse_std_threshold:.0e}) -> "
                f"수치적으로 상수에 가깝게 붕괴, 원본값 유지"
            ))
            collapsed_columns.append((col, lam, transformed_train_std))
            continue

        transformers[col] = pt_col
        transformed_columns.append(col)

    if failed_columns:
        print(f"[경고] {len(failed_columns)}개 변수는 원본 값을 그대로 둡니다:")
        for col, msg in failed_columns:
            print(f"  - {col}: {msg}")
    if collapsed_columns:
        print(f"\n[참고] 이 중 {len(collapsed_columns)}개는 scipy 최적화 자체는 "
              f"성공했지만 극단적인 lambda로 인해 변환값이 수치적으로 상수로 "
              f"붕괴한 경우입니다 (collapse_std_threshold={collapse_std_threshold:.0e} "
              f"가드로 탐지):")
        for col, lam, std in collapsed_columns:
            print(f"  - {col}: lambda={lam:.4f}, 변환 후 std={std:.3e}")

    if transformed_columns:
        for col in transformed_columns:
            pt_col = transformers[col]
            X_train_out[col] = pt_col.transform(X_train[[col]])
            X_val_out[col] = pt_col.transform(X_val[[col]])
            X_test_out[col] = pt_col.transform(X_test[[col]])

        skew_after = X_train_out[transformed_columns].skew()
        skew_before_after = pd.DataFrame({
            "column": transformed_columns,
            "skew_before": skew_before[transformed_columns].values,
            "skew_after": skew_after.values,
        })
        skew_before_after["abs_skew_before"] = skew_before_after["skew_before"].abs()
        skew_before_after["abs_skew_after"] = skew_before_after["skew_after"].abs()
        skew_before_after = skew_before_after.sort_values(
            "abs_skew_before", ascending=False
        ).reset_index(drop=True)
    else:
        print("모든 변수에서 Yeo-Johnson 변환이 실패해 변환된 컬럼이 없습니다.")
        skew_before_after = pd.DataFrame()

    return {
        "X_train": X_train_out, "X_val": X_val_out, "X_test": X_test_out,
        "transformer": transformers, "transformed_columns": transformed_columns,
        "failed_columns": failed_columns, "skew_before_after": skew_before_after,
    }


def plot_skew_before_after(X_train_before: pd.DataFrame, X_train_after: pd.DataFrame,
                            skewed_cols: list, top_n: int = 6):
    """치우침이 가장 심했던 상위 top_n개 변수에 대해, 파워변환 전/후 분포를
    히스토그램으로 나란히 그려 변환 효과를 눈으로 확인할 수 있게 한다.

    '변환 후' 그래프는 0(표준화된 평균 위치)에 점선을 긋고 x축을 0을 기준으로
    좌우 대칭이 되도록 맞춰서, 데이터가 한쪽으로 치우친 상태에서 **중앙(평균 0)으로
    이동**했다는 느낌을 시각적으로 분명하게 드러낸다. '변환 전' 그래프에도 원래
    평균 위치를 점선으로 표시해 두 그래프를 나란히 비교했을 때 "평균이 어디서
    어디로 이동했는지"가 한눈에 들어오게 했다.

    Parameters
    ----------
    X_train_before : 변환 전 train 데이터 (원본 skew 계산에 사용한 것과 동일해야 함)
    X_train_after  : correct_skew_with_power_transform() 이 반환한 변환 후 train 데이터
    skewed_cols    : check_skewness() 결과에서 abs_skew 내림차순으로 정렬된 컬럼 리스트
                     (앞쪽 top_n 개만 그린다)
    top_n : 그릴 변수 개수

    Returns
    -------
    matplotlib Figure (그릴 변수가 없으면 None)
    """
    cols = skewed_cols[:top_n]
    n = len(cols)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, col in enumerate(cols):
        before_vals = X_train_before[col]
        after_vals = X_train_after[col]

        # 변환 전: 원래 평균 위치를 점선으로 표시 (치우친 정도를 한눈에 보여줌)
        before_mean = before_vals.mean()
        sns.histplot(before_vals, kde=True, ax=axes[i, 0], color="#d62728")
        axes[i, 0].axvline(before_mean, color="black", linestyle="--", linewidth=1.2,
                            label=f"평균={before_mean:.3g}")
        axes[i, 0].set_title(f"{col} (변환 전)")
        axes[i, 0].legend(fontsize=8, loc="upper right")

        # 변환 후: 0(표준화된 중심)을 점선으로 표시하고, x축을 0 기준 좌우 대칭으로
        # 맞춰서 "중앙으로 이동했다"는 느낌을 시각적으로 강조한다.
        sns.histplot(after_vals, kde=True, ax=axes[i, 1], color="#1f77b4")
        axes[i, 1].axvline(0, color="black", linestyle="--", linewidth=1.2, label="중심 (평균=0)")
        half_range = max(abs(after_vals.min()), abs(after_vals.max())) * 1.1
        if half_range > 0:
            axes[i, 1].set_xlim(-half_range, half_range)
        axes[i, 1].set_title(f"{col} (변환 후, Yeo-Johnson)")
        axes[i, 1].legend(fontsize=8, loc="upper right")

    fig.suptitle("치우친(skewed) 변수 파워변환 전/후 분포 비교 (점선 = 평균/중심 위치)", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10) 모델 비교 유틸 (03 노트북): RandomForest 체크포인트 학습 / 부스팅 학습곡선 /
#     검증셋 기준 하이퍼파라미터 탐색 / 모델별 지표 비교 시각화
#     -> 03_model_comparison.ipynb 에서 import 되어 사용됩니다.
# ---------------------------------------------------------------------------

def train_random_forest_with_checkpoint(
    X_train, y_train, X_val, y_val,
    checkpoint_mgr: CheckpointManager,
    max_estimators: int = 300,
    estimators_per_step: int = 10,
    checkpoint_every_steps: int = 5,
    patience_steps: int = 8,
    random_state: int = 42,
    resume: bool = True,
    **rf_kwargs,
) -> dict:
    """RandomForestRegressor를 트리를 조금씩(warm_start) 늘려가며 학습시켜,
    (1) tqdm 진행률 표시줄로 '트리 개수가 늘어남에 따라 검증 성능이 어떻게 변하는지'를
        보여주고,
    (2) 몇 step마다 모델 상태를 저장하며,
    (3) resume=True 이고 기존 체크포인트가 있으면 그 지점부터 이어서 학습한다.

    RandomForest는 트리들이 서로 독립적으로 학습되기 때문에 SGD의 epoch처럼
    '학습이 덜 되었다'는 개념 자체는 없다. 대신 warm_start=True로 트리를 몇 그루씩
    추가해가며(=1 step) val 성능이 더 이상 좋아지지 않는 지점(트리를 더 늘려도
    의미가 없는 지점)을 찾아 조기 종료하고, 중간에 끊겨도 이어서 트리를 추가할 수
    있도록 SGD 학습 루프(train_sgd_regressor_with_checkpoint)와 동일한 구조로 만든
    함수다.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : 학습/검증 데이터 (트리 모델은 스케일링 불필요)
    checkpoint_mgr : CheckpointManager 인스턴스
    max_estimators : 최대 트리 개수
    estimators_per_step : 한 번에 몇 그루씩 추가할지 (SGD의 1 epoch에 해당하는 단위)
    checkpoint_every_steps : 몇 step마다 체크포인트를 저장할지
    patience_steps : 검증 성능(MSE)이 이만큼 연속으로 개선되지 않으면 조기 종료
    random_state : 재현성 시드
    resume : True면 기존 체크포인트가 있을 때 이어서 학습
    **rf_kwargs : RandomForestRegressor에 전달할 추가 파라미터 (max_depth, min_samples_leaf 등)

    Returns
    -------
    dict(model=최종(=best) 모델,
         history=step별 트리 개수/train_mse/val_mse 기록 DataFrame(컬럼: epoch, n_estimators,
                 train_mse, val_mse — plot_training_curve() 와 바로 호환되도록 epoch 컬럼명 사용))
    """
    start_step = 0
    history = []
    best_val_loss = np.inf
    best_model = None
    no_improve_count = 0
    n_estimators_so_far = 0

    model = RandomForestRegressor(
        n_estimators=estimators_per_step,
        warm_start=True,
        random_state=random_state,
        n_jobs=-1,
        **rf_kwargs,
    )

    if resume and checkpoint_mgr.exists():
        state = checkpoint_mgr.load()
        model = state["model"]
        start_step = state["step"] + 1
        history = state["history"]
        best_val_loss = state["best_val_loss"]
        best_model = state.get("best_model", model)
        n_estimators_so_far = state["n_estimators_so_far"]
        print(f"[체크포인트 발견] step {start_step} (트리 {n_estimators_so_far}개)부터 "
              f"이어서 학습을 재개합니다. (기존 best_val_loss={best_val_loss:.4f})")

    total_steps = max(1, max_estimators // estimators_per_step)
    pbar = tqdm(range(start_step, total_steps), desc="RandomForest 학습 진행 (트리 추가)", unit="step")
    for step in pbar:
        n_estimators_so_far = (step + 1) * estimators_per_step
        model.n_estimators = n_estimators_so_far
        model.fit(X_train, y_train)  # warm_start=True 이므로 기존 트리는 유지하고 새 트리만 추가 학습

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_loss = mean_squared_error(y_train, train_pred)
        val_loss = mean_squared_error(y_val, val_pred)

        history.append({"epoch": step, "n_estimators": n_estimators_so_far,
                         "train_mse": train_loss, "val_mse": val_loss})
        pbar.set_postfix({"n_trees": n_estimators_so_far, "val_mse": f"{val_loss:.4f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = _clone_sklearn_model(model)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (step + 1) % checkpoint_every_steps == 0 or step == total_steps - 1:
            checkpoint_mgr.save({
                "step": step, "model": model, "best_model": best_model,
                "best_val_loss": best_val_loss, "history": history,
                "n_estimators_so_far": n_estimators_so_far,
            })

        if no_improve_count >= patience_steps:
            print(f"\n[조기 종료] step {step} (트리 {n_estimators_so_far}개): 검증 손실이 "
                  f"{patience_steps}회 연속 개선되지 않아 학습을 종료합니다.")
            checkpoint_mgr.save({
                "step": step, "model": model, "best_model": best_model,
                "best_val_loss": best_val_loss, "history": history,
                "n_estimators_so_far": n_estimators_so_far,
            })
            break

    return {"model": best_model if best_model is not None else model,
            "history": pd.DataFrame(history)}


def staged_learning_curve(model, X_train, y_train, X_val, y_val) -> pd.DataFrame:
    """GradientBoostingRegressor 처럼 staged_predict() 를 지원하는 부스팅 모델에서,
    부스팅 반복(iteration)마다 train/val MSE가 어떻게 변했는지를 사후에 재구성한다.

    GradientBoosting은 한 번의 fit() 호출로 학습이 끝나지만, 내부적으로는 트리를
    하나씩 순차적으로 추가하며 오차를 줄여나간 것이므로, staged_predict()로 매 단계의
    예측값을 꺼내면 SGD의 epoch별 학습곡선과 동일한 형태의 '학습 진행상황'을 사후에
    재구성할 수 있다 (실시간 진행률 표시줄 대신, 학습이 끝난 뒤 전체 진행 과정을
    한 번에 보여주는 방식).

    Parameters
    ----------
    model : 학습이 완료된 staged_predict 지원 모델 (예: GradientBoostingRegressor)
    X_train, y_train, X_val, y_val : 평가용 데이터

    Returns
    -------
    DataFrame(epoch=부스팅 반복 번호(1부터 시작), train_mse, val_mse)
    -- plot_training_curve() 에 바로 넘길 수 있는 형태
    """
    history = []
    for i, (train_pred, val_pred) in enumerate(
        zip(model.staged_predict(X_train), model.staged_predict(X_val))
    ):
        history.append({
            "epoch": i + 1,
            "train_mse": mean_squared_error(y_train, train_pred),
            "val_mse": mean_squared_error(y_val, val_pred),
        })
    return pd.DataFrame(history)


def grid_search_by_val(model_class, param_grid: list, X_train, y_train, X_val, y_val,
                        fixed_params: dict = None, random_state: int = 42,
                        sample_weight_train=None) -> dict:
    """주어진 하이퍼파라미터 후보들(param_grid)을 하나씩 학습시켜, 검증(val) R2가 가장
    좋은 조합을 찾는다.

    K-Fold 교차검증(cross-validation) 대신, 이 프로젝트에서 이미 만들어 둔 명시적
    train/val/test 8:1:1 분할을 그대로 사용한다 (표본이 136개로 적어 K-Fold도 충분히
    고려할 만하지만, 01/02 노트북과 일관된 방식을 유지하기 위해 val 기준 탐색으로
    통일했다). ★ test 데이터는 이 함수에 전혀 들어오지 않는다 — 하이퍼파라미터 선택은
    반드시 val 로만 한다.

    Parameters
    ----------
    model_class : sklearn 추정기 클래스 (예: Ridge, Lasso, DecisionTreeRegressor)
    param_grid : 시도할 하이퍼파라미터 dict 리스트 (예: [{"alpha": 0.1}, {"alpha": 1.0}])
    X_train, y_train, X_val, y_val : 학습/검증 데이터
    fixed_params : 모든 후보에 공통으로 적용할 고정 파라미터 (예: {"max_iter": 5000})
    random_state : model_class 생성자가 random_state 인자를 받는 경우에만 사용되는 시드
                    (받지 않는 모델이면 자동으로 빼고 재시도한다)
    sample_weight_train : (06 노트북 추가) train 학습(model.fit)에 넘길 표본 가중치.
                    None(기본값)이면 기존과 동일하게 모든 행을 동일 가중치로 학습한다
                    (하위 호환). 지정하면 model.fit(X_train, y_train,
                    sample_weight=sample_weight_train)으로 호출하며(가중치를 지원하지
                    않는 모델 클래스는 TypeError가 발생하므로, 그런 모델에는 사용하지
                    말 것), 평가 지표(train_R2/val_R2 등)는 항상 '가중치 없이(동일
                    가중)' 계산한다 — 이전 노트북들의 비가중 R2와 그대로 비교할 수
                    있게 하기 위함이다. Val 자체를 가중 평가하고 싶다면 결과 모델을
                    받아 별도로 계산할 것.

    Returns
    -------
    dict(best_model=검증 R2가 가장 높았던 모델 (해당 파라미터로 학습 완료된 상태),
         best_params=best_model 의 하이퍼파라미터 dict,
         results=모든 후보의 파라미터 + train/val 지표를 담은 DataFrame (val_R2 내림차순))
    """
    fixed_params = dict(fixed_params or {})
    rows = []
    best_val_r2 = -np.inf
    best_model = None
    best_params = None

    for params in tqdm(param_grid, desc=f"{model_class.__name__} 하이퍼파라미터 탐색"):
        all_params = {**fixed_params, **params}
        try:
            model = model_class(random_state=random_state, **all_params)
        except TypeError:
            # random_state 인자를 받지 않는 모델(예: 일부 설정의 LinearRegression 계열)
            model = model_class(**all_params)

        if sample_weight_train is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight_train)
        else:
            model.fit(X_train, y_train)
        train_metrics = evaluate_regression(y_train, model.predict(X_train))
        val_metrics = evaluate_regression(y_val, model.predict(X_val))

        rows.append({
            **params,
            "train_R2": train_metrics["R2"],
            "val_R2": val_metrics["R2"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
        })

        if val_metrics["R2"] > best_val_r2:
            best_val_r2 = val_metrics["R2"]
            best_model = model
            best_params = params

    results = pd.DataFrame(rows).sort_values("val_R2", ascending=False).reset_index(drop=True)
    return {"best_model": best_model, "best_params": best_params, "results": results}


def plot_model_comparison(metrics_df: pd.DataFrame, metric: str = "R2",
                           title: str = None) -> plt.Figure:
    """여러 모델의 지표(R2/RMSE/MAE)를 막대그래프로 한눈에 비교한다.

    Parameters
    ----------
    metrics_df : 최소한 'model' 컬럼과 metric 로 지정한 지표 컬럼(예: 'R2')이 있는
                 DataFrame. 'split' 컬럼(train/val/test)이 있으면 모델별로 묶어
                 그룹 막대그래프로 표시한다.
    metric : 비교할 지표 컬럼명 ("R2", "RMSE", "MAE" 등)
    title : 그래프 제목 (기본값: f"모델별 {metric} 비교")

    Returns
    -------
    matplotlib Figure
    """
    if title is None:
        title = f"모델별 {metric} 비교"

    fig, ax = plt.subplots(figsize=(10, 5))
    if "split" in metrics_df.columns:
        pivot = metrics_df.pivot(index="model", columns="split", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.legend(title="split")
    else:
        ax.bar(metrics_df["model"], metrics_df[metric])

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 11) 심화 개선 유틸 (04 노트북): VIF 재점검 / Lasso 피처 선택 / 체크포인트 기반
#     하이퍼파라미터 그리드서치 / XGBoost·LightGBM 체크포인트 학습 /
#     K-Fold 안정성 검증
#     -> 04_advanced_modeling.ipynb 에서 import 되어 사용됩니다.
# ---------------------------------------------------------------------------

def compute_vif(X: pd.DataFrame, high_vif_threshold: float = 10.0) -> pd.DataFrame:
    """분산팽창지수(VIF, Variance Inflation Factor)로 다중공선성을 재점검한다.

    02 노트북에서 사용한 pairwise 상관계수(|r|>=threshold) 기준 제거는 '두 변수씩만'
    비교하기 때문에, 세 개 이상의 변수가 함께 선형결합을 이루는 다중공선성
    (예: A ≈ B + C, 각 쌍의 상관계수는 0.8 미만이어도 셋을 합치면 거의 완전한
    선형관계인 경우)은 걸러내지 못할 수 있다. VIF는 각 변수를 나머지 모든 변수로
    회귀했을 때의 결정계수(R²)를 이용해 '이 변수가 다른 변수들로 얼마나 설명되는지'를
    종합적으로 측정한다: VIF = 1 / (1 - R²). 일반적으로 VIF > 10 이면(보수적으로는
    > 5) 다중공선성이 심각한 것으로 간주한다.

    03 노트북에서 LinearRegression 계수 하나가 비정상적으로 크게(~1.6×10¹³) 나온
    현상의 원인을 정량적으로 확인하기 위해 04에서 사용한다.

    ★ VIF 계산 자체가 회귀이므로 반드시 train 데이터로만 계산해야 val/test 누수가
    없다 (01/02/03과 동일한 원칙).

    Parameters
    ----------
    X : VIF를 계산할 수치형 독립변수 DataFrame (train만 사용 권장)
    high_vif_threshold : 이 값을 넘으면 '다중공선성 심각'으로 표시

    Returns
    -------
    컬럼별 VIF를 담은 DataFrame (VIF 내림차순), is_high_vif(bool) 컬럼 포함.
    (완전공선성 등으로 계산이 실패한 컬럼은 VIF=NaN 으로 표시하고 건너뛴다)
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X_const = add_constant(X.astype(float), has_constant="add")
    # add_constant로 추가된 상수항(const)은 VIF 계산 대상(설명변수)이 아니므로 제외
    cols = [c for c in X_const.columns if c != "const"]

    vif_rows = []
    for col in tqdm(cols, desc="VIF 계산"):
        col_idx = X_const.columns.get_loc(col)
        try:
            vif = variance_inflation_factor(X_const.values, col_idx)
        except Exception as e:
            # 완전(또는 거의 완전한) 공선성이 있으면 내부 행렬이 특이(singular)해져
            # 계산이 실패할 수 있다. 이 경우 VIF=NaN 으로 남기고 다음 컬럼으로 진행한다.
            print(f"[경고] {col} VIF 계산 실패 ({type(e).__name__}: {e}) -> NaN 처리")
            vif = np.nan
        vif_rows.append({"feature": col, "VIF": vif})

    result = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False, na_position="first")
    result = result.sort_values("VIF", ascending=False).reset_index(drop=True)
    result["is_high_vif"] = result["VIF"] > high_vif_threshold
    return result


def select_nonzero_lasso_features(lasso_model, feature_names, tol: float = 1e-10) -> list:
    """학습이 끝난 Lasso 모델에서 계수가 0이 아닌(=Lasso가 '선택'한) 피처 이름
    리스트를 반환한다.

    Lasso는 alpha(규제 강도)가 적절하면 중요하지 않은 변수의 계수를 정확히 0으로
    만들어 자동 피처 선택 효과를 낸다 (03 노트북에서 26개 중 19개 계수가 0으로
    축소됨). 이 함수로 골라낸 피처만 사용해 다른 모델을 재학습하면, Lasso가
    선택한 '핵심 변수'만으로도 비슷하거나 더 나은 성능이 나오는지 확인할 수 있다.

    Parameters
    ----------
    lasso_model : 학습이 완료된 sklearn Lasso 인스턴스 (.coef_ 속성 필요)
    feature_names : X의 컬럼 이름 리스트 (lasso_model.coef_ 와 순서가 같아야 함)
    tol : 이 값보다 절댓값이 작은 계수는 '0'으로 간주 (부동소수점 오차 방지)

    Returns
    -------
    0이 아닌 계수를 가진 피처 이름 리스트
    """
    coefs = np.asarray(lasso_model.coef_)
    assert len(coefs) == len(feature_names), (
        "lasso_model.coef_ 길이와 feature_names 길이가 다릅니다. "
        "학습에 사용한 X와 같은 컬럼 순서를 넘겼는지 확인하세요."
    )
    return [f for f, c in zip(feature_names, coefs) if abs(c) > tol]


def grid_search_by_val_checkpoint(train_fn, param_grid: list, X_train, y_train, X_val, y_val,
                                   checkpoint_dir: str, base_name: str,
                                   train_fn_kwargs: dict = None) -> dict:
    """train_random_forest_with_checkpoint / train_xgboost_with_checkpoint /
    train_lightgbm_with_checkpoint 처럼 '(X_train, y_train, X_val, y_val,
    checkpoint_mgr, **kwargs) -> dict(model, history)' 시그니처를 갖는 체크포인트
    학습 함수에 대해, 여러 하이퍼파라미터 후보를 순회하며 검증(val) R2가 가장 좋은
    조합을 찾는다.

    grid_search_by_val()은 한 번의 fit() 으로 끝나는 모델(Ridge/Lasso/DecisionTree)
    용이고, 이 함수는 '진행상황 표시 + 체크포인트 재개가 가능한' 반복학습 루프를
    그대로 감싸는 버전이다. 후보(candidate)마다 서로 다른 체크포인트 파일
    ({checkpoint_dir}/{base_name}_cand{i}.joblib)을 사용하므로, 노트북 실행이
    중간에 끊겨도 이미 끝난 후보는 다시 학습하지 않고 그 후보의 체크포인트에서
    이어서, 또는 다음 후보부터 진행할 수 있다.

    Parameters
    ----------
    train_fn : train_random_forest_with_checkpoint 등 체크포인트 학습 함수
    param_grid : 시도할 하이퍼파라미터 dict 리스트 (예: [{"max_depth": 4}, {"max_depth": 8}])
    X_train, y_train, X_val, y_val : 학습/검증 데이터
    checkpoint_dir : 후보별 체크포인트를 저장할 폴더
    base_name : 체크포인트 파일 이름의 접두사 (예: "rf_maxdepth")
    train_fn_kwargs : 모든 후보에 공통으로 넘길 추가 인자 (예: max_estimators, patience_steps 등)

    Returns
    -------
    dict(best_model=검증 R2가 가장 높았던 모델,
         best_params=best_model 의 하이퍼파라미터 dict,
         best_history=best_model 학습 과정의 history DataFrame (학습곡선 시각화용),
         results=모든 후보의 파라미터 + train/val 지표를 담은 DataFrame (val_R2 내림차순))
    """
    train_fn_kwargs = dict(train_fn_kwargs or {})
    rows = []
    best_val_r2 = -np.inf
    best_model = None
    best_params = None
    best_history = None

    for i, params in enumerate(tqdm(param_grid, desc=f"{base_name} 하이퍼파라미터 탐색(외부 루프)")):
        cm = CheckpointManager(checkpoint_dir, name=f"{base_name}_cand{i}")
        result = train_fn(X_train, y_train, X_val, y_val, cm, **train_fn_kwargs, **params)
        model = result["model"]

        train_metrics = evaluate_regression(y_train, model.predict(X_train))
        val_metrics = evaluate_regression(y_val, model.predict(X_val))
        rows.append({
            **params,
            "train_R2": train_metrics["R2"],
            "val_R2": val_metrics["R2"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
        })

        if val_metrics["R2"] > best_val_r2:
            best_val_r2 = val_metrics["R2"]
            best_model = model
            best_params = params
            best_history = result["history"]

    results = pd.DataFrame(rows).sort_values("val_R2", ascending=False).reset_index(drop=True)
    return {"best_model": best_model, "best_params": best_params,
            "best_history": best_history, "results": results}


def _train_boosting_with_checkpoint(
    model_class, continue_kwarg: str,
    X_train, y_train, X_val, y_val,
    checkpoint_mgr: CheckpointManager,
    max_estimators: int, estimators_per_step: int,
    checkpoint_every_steps: int, patience_steps: int,
    random_state: int, resume: bool,
    desc: str, **model_kwargs,
) -> dict:
    """XGBoost/LightGBM처럼 '이전에 학습된 booster에 이어서(continued training)
    추가 학습'을 지원하는 부스팅 모델을 위한 공용 체크포인트 학습 루프
    (train_xgboost_with_checkpoint, train_lightgbm_with_checkpoint 가 내부적으로 호출).

    RandomForest의 warm_start(기존 모델 객체에 트리를 직접 추가)와 달리, XGBoost와
    LightGBM의 sklearn 래퍼는 매 fit() 호출마다 새 모델을 만들되, 이전 스텝에서
    학습된 booster를 함께 넘기면(XGBoost: fit(..., xgb_model=이전_booster),
    LightGBM: fit(..., init_model=이전_booster)) 그 지점부터 이어서 트리를 추가
    학습한다. 이 함수는 그 방식을 이용해 SGD/RandomForest와 동일한 구조
    (진행률 표시줄 + 주기적 체크포인트 저장 + 조기 종료 + resume)로 학습시킨다.

    Parameters
    ----------
    model_class : XGBRegressor 또는 LGBMRegressor
    continue_kwarg : 이전 booster를 이어받을 때 fit() 에 넘길 키워드 이름
                      ("xgb_model" 또는 "init_model")
    X_train, y_train, X_val, y_val : 학습/검증 데이터 (트리 모델이므로 스케일링 불필요)
    checkpoint_mgr : CheckpointManager 인스턴스
    max_estimators : 최대 트리(부스팅 반복) 개수
    estimators_per_step : 한 번에 몇 라운드씩 추가할지
    checkpoint_every_steps : 몇 step마다 체크포인트를 저장할지
    patience_steps : 검증 손실(MSE)이 이만큼 연속으로 개선되지 않으면 조기 종료
    random_state : 재현성 시드
    resume : True면 기존 체크포인트가 있을 때 이어서 학습
    desc : tqdm 진행률 표시줄에 보여줄 설명 문구
    **model_kwargs : model_class 생성자에 전달할 추가 파라미터 (max_depth, learning_rate 등)

    Returns
    -------
    dict(model=최종(=best) 모델,
         history=step별 트리 개수/train_mse/val_mse 기록 DataFrame)
    """
    start_step = 0
    history = []
    best_val_loss = np.inf
    best_model = None
    no_improve_count = 0
    n_estimators_so_far = 0
    prev_booster = None

    if resume and checkpoint_mgr.exists():
        state = checkpoint_mgr.load()
        prev_booster = state["prev_booster"]
        start_step = state["step"] + 1
        history = state["history"]
        best_val_loss = state["best_val_loss"]
        best_model = state.get("best_model")
        n_estimators_so_far = state["n_estimators_so_far"]
        print(f"[체크포인트 발견] step {start_step} (트리 {n_estimators_so_far}개)부터 "
              f"이어서 학습을 재개합니다. (기존 best_val_loss={best_val_loss:.4f})")

    total_steps = max(1, max_estimators // estimators_per_step)
    pbar = tqdm(range(start_step, total_steps), desc=desc, unit="step")
    for step in pbar:
        model = model_class(n_estimators=estimators_per_step, random_state=random_state, **model_kwargs)
        fit_kwargs = {continue_kwarg: prev_booster} if prev_booster is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)
        # 다음 step에서 이어서 학습할 수 있도록 현재까지의 booster를 꺼내둔다
        prev_booster = model.get_booster() if hasattr(model, "get_booster") else model.booster_

        n_estimators_so_far += estimators_per_step
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_loss = mean_squared_error(y_train, train_pred)
        val_loss = mean_squared_error(y_val, val_pred)

        history.append({"epoch": step, "n_estimators": n_estimators_so_far,
                         "train_mse": train_loss, "val_mse": val_loss})
        pbar.set_postfix({"n_trees": n_estimators_so_far, "val_mse": f"{val_loss:.4f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = _clone_sklearn_model(model)
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (step + 1) % checkpoint_every_steps == 0 or step == total_steps - 1:
            checkpoint_mgr.save({
                "step": step, "prev_booster": prev_booster, "best_model": best_model,
                "best_val_loss": best_val_loss, "history": history,
                "n_estimators_so_far": n_estimators_so_far,
            })

        if no_improve_count >= patience_steps:
            print(f"\n[조기 종료] step {step} (트리 {n_estimators_so_far}개): 검증 손실이 "
                  f"{patience_steps}회 연속 개선되지 않아 학습을 종료합니다.")
            checkpoint_mgr.save({
                "step": step, "prev_booster": prev_booster, "best_model": best_model,
                "best_val_loss": best_val_loss, "history": history,
                "n_estimators_so_far": n_estimators_so_far,
            })
            break

    return {"model": best_model if best_model is not None else model,
            "history": pd.DataFrame(history)}


def train_xgboost_with_checkpoint(
    X_train, y_train, X_val, y_val,
    checkpoint_mgr: CheckpointManager,
    max_estimators: int = 300, estimators_per_step: int = 10,
    checkpoint_every_steps: int = 5, patience_steps: int = 8,
    random_state: int = 42, resume: bool = True,
    **xgb_kwargs,
) -> dict:
    """XGBoost(XGBRegressor)를 RandomForest/SGD와 동일한 '진행상황 표시 + 체크포인트
    재개' 학습 루프로 학습시킨다 (내부적으로 _train_boosting_with_checkpoint 사용).

    XGBoost는 그레이디언트 부스팅에 L1/L2 정규화, 컬럼/행 서브샘플링 등을 추가로
    지원해 sklearn의 GradientBoostingRegressor보다 과적합에 더 강건한 경우가 많다.
    03 노트북에서 GradientBoosting이 108건의 작은 학습 표본 때문에 사실상 부스팅
    효과를 보지 못했던 문제(best_n_estimators=1)를 XGBoost의 정규화 옵션으로
    완화할 수 있는지 확인하기 위해 04에서 추가한다.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : 학습/검증 데이터
    checkpoint_mgr : CheckpointManager 인스턴스
    max_estimators : 최대 부스팅 반복(트리) 수
    estimators_per_step : 한 번에 몇 라운드씩 추가할지
    checkpoint_every_steps : 몇 step마다 체크포인트 저장할지
    patience_steps : 검증 손실이 이만큼 연속 개선 안 되면 조기 종료
    random_state : 재현성 시드
    resume : True면 기존 체크포인트에서 이어서 학습
    **xgb_kwargs : XGBRegressor에 전달할 추가 파라미터
                   (max_depth, learning_rate, reg_alpha, reg_lambda, subsample 등)

    Returns
    -------
    dict(model=최종(=best) 모델, history=step별 학습 기록 DataFrame)
    """
    from xgboost import XGBRegressor

    return _train_boosting_with_checkpoint(
        XGBRegressor, "xgb_model",
        X_train, y_train, X_val, y_val, checkpoint_mgr,
        max_estimators, estimators_per_step, checkpoint_every_steps, patience_steps,
        random_state, resume, desc="XGBoost 학습 진행 (트리 추가)",
        **xgb_kwargs,
    )


def train_lightgbm_with_checkpoint(
    X_train, y_train, X_val, y_val,
    checkpoint_mgr: CheckpointManager,
    max_estimators: int = 300, estimators_per_step: int = 10,
    checkpoint_every_steps: int = 5, patience_steps: int = 8,
    random_state: int = 42, resume: bool = True,
    **lgb_kwargs,
) -> dict:
    """LightGBM(LGBMRegressor)를 RandomForest/SGD와 동일한 '진행상황 표시 + 체크포인트
    재개' 학습 루프로 학습시킨다 (내부적으로 _train_boosting_with_checkpoint 사용).

    LightGBM은 leaf-wise 트리 성장 방식과 정규화(L1/L2, min_child_samples 등)를
    지원해, 작은 표본에서 트리 개수/깊이를 보수적으로 잡으면 과적합을 억제하면서도
    학습 속도가 빠르다는 장점이 있다. XGBoost와 함께 04에서 부스팅 계열 대안으로
    비교한다.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : 학습/검증 데이터
    checkpoint_mgr : CheckpointManager 인스턴스
    max_estimators : 최대 부스팅 반복(트리) 수
    estimators_per_step : 한 번에 몇 라운드씩 추가할지
    checkpoint_every_steps : 몇 step마다 체크포인트 저장할지
    patience_steps : 검증 손실이 이만큼 연속 개선 안 되면 조기 종료
    random_state : 재현성 시드
    resume : True면 기존 체크포인트에서 이어서 학습
    **lgb_kwargs : LGBMRegressor에 전달할 추가 파라미터
                   (max_depth, learning_rate, reg_alpha, reg_lambda, min_child_samples 등)

    Returns
    -------
    dict(model=최종(=best) 모델, history=step별 학습 기록 DataFrame)
    """
    from lightgbm import LGBMRegressor

    # LightGBM은 표본이 매우 작으면 콘솔에 다수의 경고를 출력하는데, verbose=-1로
    # 조용히 시키되(사용자가 명시적으로 넘긴 값이 있으면 그것을 우선한다)
    lgb_kwargs.setdefault("verbose", -1)

    return _train_boosting_with_checkpoint(
        LGBMRegressor, "init_model",
        X_train, y_train, X_val, y_val, checkpoint_mgr,
        max_estimators, estimators_per_step, checkpoint_every_steps, patience_steps,
        random_state, resume, desc="LightGBM 학습 진행 (트리 추가)",
        **lgb_kwargs,
    )


def kfold_stability_check(model_class, params: dict, X, y, n_splits: int = 5,
                           random_state: int = 42, fixed_params: dict = None) -> dict:
    """선정된 최종 하이퍼파라미터로 K-Fold 교차검증을 수행해, Validation set(14건)
    기준으로 선택한 성능이 표본 구성에 따라 얼마나 흔들리는지(분산)를 점검한다.

    이 프로젝트는 01~03에서 일관되게 '명시적 train/val/test 8:1:1 분할'을 하이퍼
    파라미터 선택의 기본 설계 원칙으로 삼아 왔다. 이 함수는 그 원칙을 대체하지
    않는다 — 최종 모델/하이퍼파라미터는 여전히 grid_search_by_val() 등으로 정하고,
    K-Fold는 '그렇게 뽑힌 설정이 fold 구성을 바꿔도 비슷한 성능을 내는지' 확인하는
    사후 안정성 검증(robustness check) 용도로만 병행한다. (표본이 136건으로 작아
    val 14건만으로 낸 R2는 분산이 클 수 있기 때문에 보완이 필요하다)

    Parameters
    ----------
    model_class : sklearn/XGBoost/LightGBM 추정기 클래스 (fit/predict 지원)
    params : 검증할 하이퍼파라미터 dict (grid_search_by_val 등에서 찾은 best_params)
    X, y : train+val을 합친 데이터 (K-Fold가 자체적으로 매 fold마다 검증셋을 새로
           나누므로, 반드시 test는 제외하고 넣을 것 — 데이터 누수 방지)
    n_splits : fold 개수 (기본 5. 표본이 매우 작을 때는 X.shape[0]으로 늘려
               Leave-One-Out 교차검증으로 쓸 수도 있다)
    random_state : KFold의 shuffle 시드
    fixed_params : 모든 fold에 공통으로 적용할 고정 파라미터

    Returns
    -------
    dict(fold_scores=fold별 R2/RMSE/MAE를 담은 DataFrame,
         mean_R2, std_R2, mean_RMSE, std_RMSE, mean_MAE, std_MAE)
    """
    from sklearn.model_selection import KFold

    fixed_params = dict(fixed_params or {})
    all_params = {**fixed_params, **params}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)

    rows = []
    splits = list(kf.split(X_arr))
    for fold, (train_idx, val_idx) in enumerate(
        tqdm(splits, desc=f"{model_class.__name__} K-Fold 안정성 검증")
    ):
        try:
            model = model_class(random_state=random_state, **all_params)
        except TypeError:
            # random_state 인자를 받지 않는 모델(예: 일부 sklearn 선형모델)
            model = model_class(**all_params)
        model.fit(X_arr[train_idx], y_arr[train_idx])
        pred = model.predict(X_arr[val_idx])
        metrics = evaluate_regression(y_arr[val_idx], pred)
        rows.append({"fold": fold, **metrics})

    fold_scores = pd.DataFrame(rows)
    return {
        "fold_scores": fold_scores,
        "mean_R2": fold_scores["R2"].mean(), "std_R2": fold_scores["R2"].std(),
        "mean_RMSE": fold_scores["RMSE"].mean(), "std_RMSE": fold_scores["RMSE"].std(),
        "mean_MAE": fold_scores["MAE"].mean(), "std_MAE": fold_scores["MAE"].std(),
    }


# ---------------------------------------------------------------------------
# 12) 추가 실험 유틸 (06 노트북): 도메인 파생 피처(range) / 배치 크기 기반 표본
#     가중치 / 반복 K-Fold(Repeated K-Fold) 평가 / 이항(Binomial) GLM
#     -> 06_further_experiments.ipynb 에서 import 되어 사용됩니다.
#
#     06은 2차 보고서 부록(02 버그 수정, 05 재검증) 이후에도 "더 시도해볼 만한
#     방법"으로 제안했던 5가지를 모두 실행하기 위해 추가된 유틸 모음이다:
#     (1) 배치 크기 기반 가중회귀, (2) 강건회귀(Huber/RANSAC, sklearn 기본 제공이라
#     별도 유틸 불필요), (3) 이항 GLM, (4) 반복 K-Fold를 통한 정직한 성능 재추정,
#     (5) Zone 온도/OP 변동폭(range) 도메인 파생 피처.
# ---------------------------------------------------------------------------

def add_range_features(X: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """컬럼 이름이 '{센서명}_min' / '{센서명}_max' 쌍으로 모두 존재하는 센서에 대해,
    '{센서명}_range' = {센서명}_max - {센서명}_min 파생변수를 추가한다.

    왜 이 피처가 유용할 수 있는가?
    -------------------------------
    지금까지의 피처는 배정번호(batch) 동안 센서가 도달한 '최솟값'과 '최댓값'만
    각각 담고 있어서, "공정 중 이 값이 얼마나 넓게 변동했는지(process fluctuation
    width)"라는 정보는 두 피처에 암묵적으로만 흩어져 있었다. range를 직접 계산해
    명시적인 피처로 만들면, 모델(특히 선형모델)이 "변동폭이 클수록/작을수록
    불량비율이 어떻게 달라지는지"를 별도의 계수로 직접 학습할 수 있다. 이는 min과
    max를 각각 다른 계수로 학습하는 것보다 더 직접적으로 "안정성(stability)"이라는
    공정 도메인 지식을 인코딩하는 방식이다.

    ★ 주의(설계상 단순화): 이 함수는 (02에서 이미 Yeo-Johnson 변환이 적용된) 변환
    후 min/max 컬럼을 그대로 빼서 range를 만든다. Yeo-Johnson은 단조(monotonic)
    변환이라 순서는 보존되지만 차이(difference)의 물리적 의미까지 보존하지는
    않으므로, 이 range 값은 "원본 센서 단위의 실제 변동폭"이 아니라 "변환된 척도
    상에서의 변동폭"으로 해석해야 한다. 06 노트북에서는 이 점을 감안해 결과를
    "탐색적 시도"로 신중하게 해석한다 (원본 단위로 다시 계산하려면 02 파이프라인을
    처음부터 다시 태워야 하므로, 이번 실험 범위에서는 제외했다).

    ★★ 중요(완전공선성 방지): range = max - min 은 세 컬럼(min, max, range) 사이의
    '정확한' 선형관계(range - max + min = 0)를 만든다. min/max를 그대로 둔 채 range만
    추가하면, 두 컬럼씩 비교하는 pairwise 상관계수(02/04에서 쓰는 |r|>=0.8 기준,
    remove_multicollinear_features)는 이 3개 변수 사이의 완전공선성을 못 잡아낼 수
    있다(각 쌍의 상관계수 자체는 0.8 미만일 수 있기 때문 — 2차 보고서 부록 C에서
    VIF와 pairwise 상관계수의 차이를 다룬 것과 같은 종류의 함정). 실제로 이 함수를
    만들며 min/max를 모두 남긴 채 range를 추가했더니 설계행렬 조건수가 다시
    ~1e18 수준으로 폭발하는 것을 확인했다(VIF도 일부 무한대로 발산). 따라서 이
    함수는 range를 추가하는 대신 **max 컬럼을 제거**한다: (min, range) 두 값만
    있어도 (min, max) 두 값과 정보량은 동일하고(max = min + range로 복원 가능),
    "최솟값 + 변동폭"이라는 조합이 "최솟값 + 최댓값"보다 공정 안정성을 더 직접적으로
    표현한다고 보았다.

    Parameters
    ----------
    X : min/max 접미사가 붙은 피처를 포함한 DataFrame (X_train 등)
    verbose : True면 새로 만든 range 피처 및 제거된 max 컬럼 목록을 출력

    Returns
    -------
    (range 피처가 추가되고 그에 대응하는 max 컬럼은 제거된 새 DataFrame(원본은
     변경하지 않음), 새로 추가된 range 컬럼명 리스트)
    """
    X_out = X.copy()
    min_cols = {c[:-4]: c for c in X.columns if c.endswith("_min")}
    max_cols = {c[:-4]: c for c in X.columns if c.endswith("_max")}
    common_sensors = sorted(set(min_cols) & set(max_cols))

    new_cols = []
    dropped_max_cols = []
    for sensor in common_sensors:
        range_col = f"{sensor}_range"
        X_out[range_col] = X_out[max_cols[sensor]] - X_out[min_cols[sensor]]
        new_cols.append(range_col)
        dropped_max_cols.append(max_cols[sensor])

    # 완전공선성(range = max - min) 방지를 위해 range를 만든 센서의 max 컬럼은 제거
    # (min은 유지 -> (min, range) 조합이 (min, max) 조합과 동일한 정보량을 가짐)
    X_out = X_out.drop(columns=dropped_max_cols)

    if verbose:
        print(f"[range 피처 생성] min/max 쌍이 모두 있는 센서 {len(new_cols)}개에 대해 "
              f"'_range' 피처를 추가하고, 완전공선성 방지를 위해 대응하는 '_max' 컬럼을 "
              f"제거했습니다 (min은 유지):")
        for c, dropped in zip(new_cols, dropped_max_cols):
            print(f"  - {c}  (제거된 컬럼: {dropped})")

    return X_out, new_cols


def compute_sample_weights(df_total: pd.DataFrame, row_labels, good_col: str = "양품수량",
                            bad_col: str = "불량수량", method: str = "sqrt_n") -> np.ndarray:
    """배정번호(batch)별 생산량(양품+불량)을 근거로, 표본별 신뢰도에 비례하는
    표본 가중치(sample_weight)를 계산한다.

    왜 가중치가 필요한가?
    ----------------------
    이 프로젝트의 목표변수 '불량비율'은 배치마다 다른 표본 크기(양품수량+불량수량)로
    계산된 '비율'이다. 예를 들어 118005번 배치는 59,858건 중 4건이 불량이라 불량비율
    추정이 매우 안정적이지만, 표본이 훨씬 작은 배치의 불량비율은 우연에 의해 훨씬
    크게 흔들릴 수 있다(예: 1건만 더 불량이어도 비율이 확 뛴다). 지금까지의 모든
    모델(Ridge/Lasso/RandomForest 등)은 이런 차이를 무시하고 108개 행을 전부
    동일한 신뢰도로 취급해 왔다. 05 노트북에서 찾아낸 고-레버리지 3개 관측치
    중 다수가 상대적으로 표본이 작은 배치였다는 점도 이 가설과 맞아떨어진다.

    Parameters
    ----------
    df_total : 배정번호, 양품수량, 불량수량 컬럼을 포함한 원본 배치 단위 데이터
               (data_processed/df_total.csv 를 그대로 읽은 DataFrame)
    row_labels : 가중치를 계산할 대상 행의 df_total 상 원래 인덱스(라벨) 목록
               (★ 05 노트북에서 확립한 관례와 동일하게, X_train.index 를 그대로
               넘기면 된다 — split_train_val_test 가 train_test_split 을 쓰므로
               X_train.index 는 df_total 의 원래 행 라벨이 섞인 순서를 그대로
               보존하고 있다. 위치(position)가 아니라 라벨이어야 한다는 점에 유의:
               05 노트북 배정번호 인덱싱 오류 사례를 참고할 것.)
    good_col, bad_col : 양품/불량 수량 컬럼명
    method : 가중치 계산 방식.
               "sqrt_n"(기본값) : sqrt(양품수량+불량수량). 이항분포에서 비율 추정치의
                    표준오차가 1/sqrt(n)에 비례한다는 점에 착안해, 표본이 클수록
                    (=추정이 더 정확할수록) 더 큰 가중치를 준다. 제곱근을 쓰는 이유는
                    n을 그대로 쓰면 초대형 배치(예: 59,858건) 하나가 가중합을 거의
                    독점해 사실상 그 한 배치만 학습하는 것과 비슷해지기 때문에,
                    극단적인 쏠림을 완화하기 위함이다.
               "n" : 양품수량+불량수량 그대로 사용 (가중치 쏠림이 훨씬 심함, 참고용).

    Returns
    -------
    row_labels 순서와 동일한 길이의 numpy 배열 (표본 가중치). 평균이 1이 되도록
    정규화해 반환한다 (정규화하지 않으면 가중 손실 함수의 스케일이 alpha 등
    정규화 하이퍼파라미터의 최적값과 뒤엉켜 버리기 때문).
    """
    # ★ df_total 은 반드시 index_col 없이(기본 RangeIndex 0..135로) 읽어야 한다.
    # X_train.index 등은 그 기본 RangeIndex 라벨을 그대로 쓰고 있으므로(05 노트북에서
    # 확립한 관례), df_total 을 배정번호 등으로 index_col 지정해서 읽으면 아래 .loc가
    # 엉뚱한 값을 찾거나 KeyError를 낸다.
    counts = df_total[[good_col, bad_col]]
    n_samples = counts.loc[row_labels, good_col].to_numpy() + counts.loc[row_labels, bad_col].to_numpy()

    if method == "sqrt_n":
        w = np.sqrt(n_samples.astype(float))
    elif method == "n":
        w = n_samples.astype(float)
    else:
        raise ValueError(f"알 수 없는 method: {method} ('sqrt_n' 또는 'n'만 지원)")

    w = w / w.mean()  # 평균 1로 정규화 (가중치 스케일이 정규화 강도 alpha 해석에 영향 주지 않도록)
    return w


def repeated_kfold_evaluate(model_class, params: dict, X, y, n_splits: int = 5,
                             n_repeats: int = 20, random_state: int = 42,
                             fixed_params: dict = None, sample_weight=None,
                             desc: str = None) -> dict:
    """K-Fold를 여러 번(n_repeats) 서로 다른 방식으로 섞어 반복 수행하는 '반복
    K-Fold(Repeated K-Fold)'로, 표본이 작을 때 단일 분할(Val 14건, 또는 K-Fold 1회
    5개 fold)만으로는 알기 어려운 '정직한' 성능 분포(평균 및 신뢰구간)를 추정한다.

    왜 기존 kfold_stability_check() 만으로는 부족한가?
    ----------------------------------------------------
    04/05 노트북의 kfold_stability_check()는 K-Fold를 '1번'만 수행한다(예:
    5-Fold → 5개의 R2 점수). fold를 어떻게 나누느냐(어떤 행이 어느 fold에
    들어가느냐)에 따라 5개 점수 자체가 꽤 달라질 수 있는데, 표본이 108~122건으로
    작을 때는 이 '분할 방식에 따른 우연'의 영향이 특히 크다. 이 함수는 fold
    분할을 random_state를 바꿔가며 n_repeats번 반복해 (n_splits × n_repeats)개의
    점수를 모으므로, '어쩌다 이번 분할에서 잘 나온 것'인지 '실제로 안정적으로 좋은
    것'인지를 훨씬 더 믿을 수 있게 구분해준다. 이 함수는 (a) 하이퍼파라미터/모델
    선택 기준으로도, (b) 최종 후보들의 '정직한' 성능 재추정(신뢰구간 포함)으로도
    쓸 수 있다.

    Parameters
    ----------
    model_class : sklearn 스타일 추정기 클래스 (fit/predict 지원, sample_weight
                  지원 여부는 모델에 따라 다름)
    params : 검증할 하이퍼파라미터 dict
    X, y : 평가에 사용할 전체 데이터 (★ train+val을 합쳐서 넣을 것 — test는 절대
           넣지 않는다. 05까지와 동일한 데이터 누수 방지 원칙)
    n_splits : 한 번의 K-Fold에서 나눌 fold 개수 (기본 5)
    n_repeats : K-Fold를 몇 번 반복할지 (기본 20 -> 총 5*20=100개 fold 점수)
    random_state : RepeatedKFold의 시드 (반복마다 다른 분할이 되도록 내부적으로
                   자동으로 바뀐다)
    fixed_params : 모든 fold에 공통으로 적용할 고정 파라미터
    sample_weight : X, y와 같은 길이의 표본 가중치 배열. None이 아니면 각 fold의
                   train 부분에서 해당하는 가중치만 잘라 model.fit(...,
                   sample_weight=...)로 넘긴다 (평가 지표 자체는 항상 비가중으로
                   계산해 다른 후보들과 비교 가능하게 유지한다).
    desc : tqdm 진행률 표시줄 설명 (기본값: 모델 클래스 이름 기반 자동 생성)

    Returns
    -------
    dict(
      fold_scores : (n_splits*n_repeats)개 행을 가진 DataFrame (repeat, fold, R2, RMSE, MAE),
      mean_R2, std_R2, ci95_R2 : R2의 평균/표준편차/95% 신뢰구간(정규근사, (하한,상한) 튜플),
      mean_RMSE, std_RMSE, mean_MAE, std_MAE : RMSE/MAE도 동일하게 집계,
      n_folds_total : 총 fold(반복) 개수 = n_splits * n_repeats,
    )
    """
    from sklearn.model_selection import RepeatedKFold

    fixed_params = dict(fixed_params or {})
    all_params = {**fixed_params, **params}
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    y_arr = y_arr.ravel() if y_arr.ndim > 1 else y_arr
    w_arr = None
    if sample_weight is not None:
        w_arr = sample_weight.values if hasattr(sample_weight, "values") else np.asarray(sample_weight)

    if desc is None:
        desc = f"{model_class.__name__} 반복 K-Fold({n_splits}x{n_repeats}) 평가"

    rows = []
    splits = list(rkf.split(X_arr))
    for i, (train_idx, val_idx) in enumerate(tqdm(splits, desc=desc)):
        repeat_no = i // n_splits
        fold_no = i % n_splits
        try:
            model = model_class(random_state=random_state, **all_params)
        except TypeError:
            model = model_class(**all_params)

        if w_arr is not None:
            model.fit(X_arr[train_idx], y_arr[train_idx], sample_weight=w_arr[train_idx])
        else:
            model.fit(X_arr[train_idx], y_arr[train_idx])

        pred = model.predict(X_arr[val_idx])
        metrics = evaluate_regression(y_arr[val_idx], pred)
        rows.append({"repeat": repeat_no, "fold": fold_no, **metrics})

    fold_scores = pd.DataFrame(rows)

    def _ci95(series: pd.Series) -> tuple:
        """정규근사 95% 신뢰구간: mean ± 1.96 * (표준오차). 표준오차는 fold 점수들이
        서로 독립이 아니므로(같은 데이터를 반복 재사용) 다소 낙관적으로 좁게 나올 수
        있다는 한계가 있지만, 단일 Val 14건 점수 하나보다는 훨씬 정직한 불확실성
        추정치를 제공한다."""
        m, s, n = series.mean(), series.std(), len(series)
        se = s / np.sqrt(n)
        return (m - 1.96 * se, m + 1.96 * se)

    return {
        "fold_scores": fold_scores,
        "mean_R2": fold_scores["R2"].mean(), "std_R2": fold_scores["R2"].std(),
        "ci95_R2": _ci95(fold_scores["R2"]),
        "mean_RMSE": fold_scores["RMSE"].mean(), "std_RMSE": fold_scores["RMSE"].std(),
        "mean_MAE": fold_scores["MAE"].mean(), "std_MAE": fold_scores["MAE"].std(),
        "n_folds_total": len(fold_scores),
    }


def fit_binomial_glm(X: pd.DataFrame, good_counts, bad_counts, alpha: float = 0.0,
                      L1_wt: float = 0.0):
    """불량비율을 '연속값 회귀 대상'이 아니라 '양품/불량 카운트로부터의 이항분포
    비율'로 취급하는 이항 GLM(Generalized Linear Model, Binomial family, logit link)을
    적합(fit)한다.

    ★ 정규화(alpha)가 왜 사실상 필수인가? (06 노트북에서 실제로 발견한 문제)
    ------------------------------------------------------------------------
    이 프로젝트의 배치별 표본 크기(양품+불량수량)는 배치마다 수십 건에서 6만 건
    이상까지 3자릿수 넘게 차이가 난다. 이항 GLM의 로그가능도(log-likelihood)
    그래디언트는 각 관측치의 표본 크기에 비례해 커지기 때문에, alpha=0(정규화 없음)
    으로 fit()을 호출하면 소수의 초대형 배치가 최적화를 지배하면서 계수가 발산하고
    (실제로 이 프로젝트 데이터에서 재현됨: 특히 피처 수가 24~25개로 표본(108~122건)
    대비 많은 'all'(고-레버리지 포함) 변형에서, 반복 K-Fold 시 예측이 폭발적으로
    벗어나 평균 R2가 -40000대까지 떨어지는 것을 확인했다), 새로운 fold의 검증
    데이터에서 예측이 폭발적으로 벗어난다(수치적으로 overflow/발산). alpha>0으로
    L2(릿지, L1_wt=0) 정규화를 주면 계수 크기가 억제되어 이 문제가 해소된다(본
    프로젝트 데이터로 alpha>=1에서 예측값이 정상 범위(0~1%대)로 안정화됨을 확인).

    Parameters
    ----------
    X : 독립변수 DataFrame (상수항은 이 함수 내부에서 자동으로 추가됨)
    good_counts, bad_counts : X와 같은 길이의 배열/Series. '불량비율 = bad/(bad+good)'
        정의와 일관되도록, endog는 [bad_counts, good_counts] 순서(첫 컬럼이
        '성공'으로 간주되는 사건)로 구성한다.
    alpha : 0.0(기본값)이면 정규화 없는 일반 GLM(.fit())을 적합한다(계수 해석/유의성
        검정이 필요할 때만 사용 권장 — summary()가 표준오차/p-value를 제공하는 것은
        이 경우뿐이다). 0보다 크면 statsmodels의 GLM.fit_regularized(alpha=alpha,
        L1_wt=L1_wt)를 사용해 정규화된 적합을 수행한다(예측 성능 비교/반복 K-Fold에는
        이 경로를 권장).
    L1_wt : 0.0(기본값, 릿지/L2)~1.0(라쏘/L1) 사이의 엘라스틱넷 혼합 비율.
        alpha=0이면 사용되지 않는다.

    자세한 방법론적 설명은 아래(원래 설명) 참고.

    왜 이 방식을 시도하는가?
    --------------------------
    지금까지의 모든 모델(Ridge/Lasso/RandomForest/XGBoost/...)은 '불량비율'이라는
    이미 계산된 연속값 하나만 보고 일반적인 회귀로 접근했다. 하지만 이 값은 사실
    '불량수량 번의 성공(?) / (양품수량+불량수량) 번의 시행' 형태의 이항(Binomial)
    비율이며, 원본 카운트(양품수량, 불량수량)를 그대로 쓰는 이항 GLM은 (1) 비율이
    [0,1] 범위를 벗어나지 않도록 logit 링크로 구조적으로 보장하고, (2) 표본 크기가
    큰 배치의 비율 추정치에 자동으로 더 높은 신뢰도(가중치와 유사한 효과)를 부여하는
    성질이 있어, compute_sample_weights()로 만든 수동 가중치보다 통계적으로 더
    원리에 맞는 방식일 수 있다. statsmodels의 GLM은 endog(종속변수)를
    [성공 횟수, 실패 횟수] 형태의 2열 배열로 주면 이 가중치 효과를 자동으로
    반영한다.

    Parameters
    ----------
    X : 독립변수 DataFrame (상수항은 이 함수 내부에서 자동으로 추가됨)
    good_counts, bad_counts : X와 같은 길이의 배열/Series. '불량비율 = bad/(bad+good)'
        정의와 일관되도록, endog는 [bad_counts, good_counts] 순서(첫 컬럼이
        '성공'으로 간주되는 사건)로 구성한다.

    Returns
    -------
    적합이 끝난 statsmodels GLMResults 객체 (add_constant로 상수항이 추가된 것을
    전제로 하므로, predict_binomial_glm()과 함께 사용할 것)
    """
    import statsmodels.api as sm

    X_const = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    good_arr = np.asarray(good_counts, dtype=float)
    bad_arr = np.asarray(bad_counts, dtype=float)
    endog = np.column_stack([bad_arr, good_arr])  # [성공(=불량), 실패(=양품)]

    glm_model = sm.GLM(endog, X_const, family=sm.families.Binomial())
    if alpha > 0:
        glm_result = glm_model.fit_regularized(alpha=alpha, L1_wt=L1_wt)
    else:
        glm_result = glm_model.fit()
    return glm_result


def predict_binomial_glm(glm_result, X: pd.DataFrame) -> np.ndarray:
    """fit_binomial_glm()으로 적합한 이항 GLM에서, 이 프로젝트의 '불량비율' 스케일
    (make_defect_rate()와 동일하게 bad/(bad+good)*100)에 맞춘 예측값을 반환한다.

    GLM(Binomial, logit link)의 기본 predict()는 [0,1] 범위의 확률(=불량 비율,
    0~1 스케일)을 반환하므로, 이 프로젝트 전체에서 써온 '불량비율(%)' 스케일과
    맞추기 위해 100을 곱해서 반환한다.

    Parameters
    ----------
    glm_result : fit_binomial_glm()이 반환한 GLMResults
    X : 예측할 독립변수 DataFrame (fit에 사용한 것과 같은 컬럼 순서/개수여야 함)

    Returns
    -------
    X와 같은 길이의 numpy 배열 (불량비율(%) 스케일 예측값)
    """
    import statsmodels.api as sm

    X_const = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    proba = glm_result.predict(X_const)  # [0, 1] 범위의 확률(=불량비율/100)
    return proba * 100.0
