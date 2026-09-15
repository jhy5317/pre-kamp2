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


def plot_training_curve(history: pd.DataFrame):
    """SGD 반복학습의 epoch 별 train/val MSE 변화를 그린다 (학습 진행상황 시각화)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_mse"], label="Train MSE")
    ax.plot(history["epoch"], history["val_mse"], label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("학습 곡선 (Training Curve)")
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
