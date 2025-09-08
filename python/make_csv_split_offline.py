from pathlib import Path
import pandas as pd
import numpy as np
import re

# ——— 설정 ———
root = Path("/home/delivery/flexric_oran/dev/data/250905")
scenario_dirs = sorted(root.glob("data_LOS*"))
out_dir = root / "output"
out_dir.mkdir(exist_ok=True)

# 시나리오별 저장을 위한 디렉토리 생성
scenario_out_dir = out_dir / "scenarios"
scenario_out_dir.mkdir(exist_ok=True)

cell_ids = [2,3,4,5,6,7,8]

# 결과를 누적할 리스트
all_results = []

def select_top_neighbors_vectorized(df, n_neighbors=3):
    """벡터화된 방식으로 상위 N개 neighbor 선택 (시간 순서 유지)"""
    
    # 유효한 neighbor 정보를 담을 새로운 컬럼 생성
    for i in range(1, n_neighbors + 1):
        df[f"final_neigh_id_{i}"] = np.nan
        df[f"final_neigh_sinr_{i}"] = np.nan
    
    # 각 행에 대해 처리
    valid_mask = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        neighbors = []
        
        # neighbor 1~7 중 유효한 것들 수집
        for i in range(1, 8):
            cell_id_col = f"L3 neigh Id {i} (cellId)"
            sinr_col = f"L3 neigh SINR 3gpp {i} (convertedSinr)"
            
            if (cell_id_col in df.columns and sinr_col in df.columns and
                pd.notna(row[cell_id_col]) and pd.notna(row[sinr_col]) and
                row[cell_id_col] > 0):
                
                neighbors.append({
                    'cell_id': row[cell_id_col],
                    'sinr': row[sinr_col]
                })
        
        # SINR 기준으로 정렬하고 상위 N개 선택
        if len(neighbors) >= n_neighbors:
            neighbors_sorted = sorted(neighbors, key=lambda x: x['sinr'], reverse=True)
            top_neighbors = neighbors_sorted[:n_neighbors]
            
            # 선택된 neighbor 정보 저장
            for i, neighbor in enumerate(top_neighbors, 1):
                df.iloc[idx, df.columns.get_loc(f"final_neigh_id_{i}")] = neighbor['cell_id']
                df.iloc[idx, df.columns.get_loc(f"final_neigh_sinr_{i}")] = neighbor['sinr']
            
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    
    # 유효한 행들만 필터링 (순서 유지)
    df_filtered = df[valid_mask].copy()
    
    # 기존 neighbor 컬럼을 새로 선택된 것들로 교체
    for i in range(1, n_neighbors + 1):
        if f"L3 neigh Id {i} (cellId)" in df_filtered.columns:
            df_filtered[f"L3 neigh Id {i} (cellId)"] = df_filtered[f"final_neigh_id_{i}"]
        if f"L3 neigh SINR 3gpp {i} (convertedSinr)" in df_filtered.columns:
            df_filtered[f"L3 neigh SINR 3gpp {i} (convertedSinr)"] = df_filtered[f"final_neigh_sinr_{i}"]
    
    # 임시 컬럼 제거
    temp_cols = [f"final_neigh_id_{i}" for i in range(1, n_neighbors + 1)] + \
                [f"final_neigh_sinr_{i}" for i in range(1, n_neighbors + 1)]
    df_filtered = df_filtered.drop(columns=temp_cols)
    
    return df_filtered

for base_dir in scenario_dirs:
    print(f"\n=== Processing {base_dir.name} ===")

    # 1) cu-cp 로그 7개 합치기 & cu-cp의 t0 계산
    dfs = []
    t0_cucp = None
    for cid in cell_ids:
        fp = base_dir / f"cu-cp-cell-{cid}.txt"
        tmp = pd.read_csv(fp)
        dfs.append(tmp)
        mt = tmp["timestamp"].min()
        t0_cucp = mt if t0_cucp is None else min(t0_cucp, mt)
    df = pd.concat(dfs, ignore_index=True)

    # 2) cu-cp의 상대 timestamp 만들기 - 초기에 UE별, 시간별 정렬
    df.sort_values(["timestamp", "UE (imsi)"], inplace=True)
    df = df.reset_index(drop=True)  # 인덱스 리셋
    df["relative_timestamp"] = (df["timestamp"].astype(int) - t0_cucp)

    print(f"Total rows before neighbor selection: {len(df)}")
    
    # 2.5) 시간 순서를 유지하면서 neighbor 선택
    df = select_top_neighbors_vectorized(df, n_neighbors=3)
    
    print(f"Final rows after neighbor selection: {len(df)}")



    # 3) ue_position.txt 처리
    trace = pd.read_csv(base_dir / "ue_position.txt")
    trace = trace[["timestamp", "id", "x", "y"]]
    trace.rename(columns={"id": "UE (imsi)"}, inplace=True)
    
    # ue_position의 t0 계산 및 relative_timestamp 생성
    t0_uepos = trace["timestamp"].min()
    trace["relative_timestamp"] = trace["timestamp"].astype(int) - t0_uepos
    trace = trace[["relative_timestamp", "UE (imsi)", "x", "y"]]

    # 4) 병합 후 다시 정렬
    df = df.merge(
        trace,
        on=["relative_timestamp","UE (imsi)"],
        how="left"
    )
    
    # 병합 후 정렬 (순서 보장)
    df = df.sort_values(["UE (imsi)", "relative_timestamp"]).reset_index(drop=True)

    # 5) 이동평균 윈도우 크기 계산 (5초)
    dt = df["relative_timestamp"].diff().loc[lambda x: x>0].min()
    window_size = max(1, int(5000 / dt))
    print(f" → window_size = {window_size}")


    # 6) SINR 컬럼 이동평균 적용 (serving + 이웃 1~3만)
    sinr_cols = ["L3 serving SINR 3gpp"] + [
        f"L3 neigh SINR 3gpp {i} (convertedSinr)" for i in range(1,4)
    ]

    print(f"Rows before serving cell filter: {len(df)}")

    # serving cell SINR이 유효하지 않은 행 제거
    df = df[df["L3 serving SINR 3gpp"].notna()]

    print(f"Rows after serving cell filter: {len(df)}")

    df = df.sort_values(["UE (imsi)", "relative_timestamp"]).reset_index(drop=True)

    # 이동평균 적용 (이미 정렬되어 있음)
    for col in sinr_cols:
        df[f"{col}_ma"] = (
            df
            .groupby("UE (imsi)")[col]
            .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        )

    # 7) gNB 좌표 하드코딩 매핑
    coords = {
        2: (1500, 1500),
        3: (2500, 1500),
        4: (2000, 2366.03),
        5: (1000, 2366.03),
        6: (500, 1500),
        7: (1000, 633.975),
        8: (2000, 633.975)
    }

    # 8) serving 및 neighbor 좌표 매핑
    # Serving cell 좌표
    df[["serving_x","serving_y"]] = (
        df["L3 serving Id(m_cellId)"]
        .map(lambda i: coords.get(i, (np.nan, np.nan)))
        .apply(pd.Series)
    )

    # Neighbor cells 좌표
    for i in range(1, 4):  # neighbor 1, 2, 3
        col_id = f"L3 neigh Id {i} (cellId)"
        if col_id in df.columns:
            df[[f"neighbor{i}_x", f"neighbor{i}_y"]] = (
                df[col_id]
                .map(lambda cell_id: coords.get(cell_id, (np.nan, np.nan)) if pd.notna(cell_id) else (np.nan, np.nan))
                .apply(pd.Series)
            )
            df[f"neighbor{i}_x"] = df[f"neighbor{i}_x"].apply(lambda x: int(x) if pd.notna(x) else x)
            df[f"neighbor{i}_y"] = df[f"neighbor{i}_y"].apply(lambda x: int(x) if pd.notna(x) else x)

    # 9) UE 좌표 컬럼 이름 변경
    df.rename(columns={"x":"UE_x", "y":"UE_y"}, inplace=True)

    # 10) 최종 컬럼 선택
    final_cols = [
        "relative_timestamp",
        "UE (imsi)",
        "serving_x", "serving_y",
        "L3 serving SINR 3gpp_ma",
        "L3 neigh SINR 3gpp 1 (convertedSinr)_ma",
        "L3 neigh SINR 3gpp 2 (convertedSinr)_ma",
        "L3 neigh SINR 3gpp 3 (convertedSinr)_ma",
        "UE_x", "UE_y"
    ]
    available = [c for c in final_cols if c in df.columns]
    final = df[available].copy()

    # 11) 컬럼명 정리
    final.rename(columns={
        "UE (imsi)": "imsi"
    }, inplace=True)

    # 12) 결측치 처리
    final.dropna(subset=["UE_x","UE_y"], inplace=True)
    final.fillna(0, inplace=True)
    
    # SINR 컬럼들 소수점 3자리로 반올림
    sinr_ma_cols = [f"{col}_ma" for col in sinr_cols]
    for col in sinr_ma_cols:
        if col in final.columns:
            final[col] = final[col].round(3)

    # UE 좌표 소수점 3자리로 반올림
    final["UE_x"] = final["UE_x"].round(3)
    final["UE_y"] = final["UE_y"].round(3)
    final["relative_timestamp"] = final["relative_timestamp"] // 100

    # 시나리오별 개별 저장
    scenario_name = base_dir.name
    scenario_csv_path = scenario_out_dir / f"{scenario_name}.csv"
    final.to_csv(scenario_csv_path, index=False)
    print(f"✅ Saved individual scenario: {scenario_csv_path}")
    print(f"   Shape: {final.shape}")
    

# 시나리오별 파일 목록 출력
print(f"\n📁 Individual scenario files saved in: {scenario_out_dir}")
for scenario_file in sorted(scenario_out_dir.glob("*.csv")):
    df_info = pd.read_csv(scenario_file)
    print(f"   - {scenario_file.name}: {df_info.shape[0]} rows")
