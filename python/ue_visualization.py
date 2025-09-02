#!/usr/bin/env python3
"""
UE Trajectory Comparison Visualization: Actual vs Predicted Trajectories (Simple Version)
- ue_position.txt: Actual positions 
- lstm_trajectory.txt: Predicted positions
- 색상 구분이 잘 되도록 개선
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")

# 🔥 파일 관리 CONFIG 추가
CONFIG = {
    'actual_positions_file': 'ue_position_3gpp1.txt',
    'predicted_positions_file': 'lstm_trajectory_3gpp1.txt',
    'output_plot_file': 'ue_trajectory_comparison.png',
    'plot_dpi': 300,
    'figure_size': (24, 18),
    'ues_per_plot': 7,
    'max_ues': 28
}

def efficient_timestamp_similarity(actual_traj, predicted_traj):
    """효율적인 타임스탬프 매칭 비교"""
    
    if len(actual_traj) < 2 or len(predicted_traj) < 2:
        return None
    
    try:
        # 1. 같은 타임스탬프끼리 매칭 (pandas merge 사용)
        merged = pd.merge(
            actual_traj[['timestamp', 'x', 'y']].rename(columns={'x': 'actual_x', 'y': 'actual_y'}),
            predicted_traj[['timestamp', 'x', 'y']].rename(columns={'x': 'pred_x', 'y': 'pred_y'}),
            on='timestamp',
            how='inner'
        )
        
        if len(merged) < 2:
            return None
        
        # 2. 벡터화된 거리 계산
        distances = np.sqrt((merged['actual_x'] - merged['pred_x'])**2 + 
                           (merged['actual_y'] - merged['pred_y'])**2)
        
        # 🔥 추가: MAE, MSE, RMSE 계산
        mae = distances.mean()  # Mean Absolute Error
        mse = (distances ** 2).mean()  # Mean Squared Error  
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        
        return {
            'mae': mae,                    # 🔥 추가
            'mse': mse,                    # 🔥 추가
            'rmse': rmse,                  # 🔥 추가
            'avg_distance': mae,           # 기존 호환성 유지
            'max_distance': distances.max(),
            'std_distance': distances.std(),
            'median_distance': distances.median(),
            'matched_points': len(merged)
        }
        
    except Exception as e:
        return {'error': str(e), 'avg_distance': float('inf')}

def analyze_trajectory_similarity_optimized(actual_df, predicted_df):
    """최적화된 궤적 유사도 분석"""
    print("\n" + "="*80)
    print("🚀 FAST TRAJECTORY SIMILARITY ANALYSIS")
    print("="*80)

    # 공통 UE 찾기
    common_ues = sorted(set(actual_df['id'].unique()) & set(predicted_df['id'].unique()))
    print(f"📊 Found {len(common_ues)} common UEs")
    
    # 결과 저장
    similarity_results = {}
    
    # 각 UE 처리
    for i, ue_id in enumerate(common_ues):
        print(f"Processing UE {ue_id:2d} ({i+1:2d}/{len(common_ues)})...", end=" ")
        
        # 궤적 데이터 추출
        actual_traj = actual_df[actual_df['id'] == ue_id].copy()
        predicted_traj = predicted_df[predicted_df['id'] == ue_id].copy()
        
        if len(actual_traj) < 2 or len(predicted_traj) < 2:
            print("⚠️  Insufficient points")
            continue
        
        # 🔥 빠른 유사도 계산
        similarity = efficient_timestamp_similarity(actual_traj, predicted_traj)
        
        if similarity and 'error' not in similarity:
            similarity_results[ue_id] = similarity
            print(f"✅ MAE: {similarity['mae']:.1f}m, MSE: {similarity['mse']:.1f}, RMSE: {similarity['rmse']:.1f}m")
        else:
            print(f"❌ Error: {similarity.get('error', 'Unknown')}")
    
    # 전체 통계
    if similarity_results:
        print(f"\n" + "="*80)
        print("📊 OVERALL STATISTICS")
        print("="*80)
        
        all_mae = [r['mae'] for r in similarity_results.values()]
        all_mse = [r['mse'] for r in similarity_results.values()]
        all_rmse = [r['rmse'] for r in similarity_results.values()]
        all_max_dist = [r['max_distance'] for r in similarity_results.values()]
        
        print(f"📊 MAE (Mean Absolute Error):")
        print(f"   Mean: {np.mean(all_mae):.2f}m ± {np.std(all_mae):.2f}m")
        print(f"   Range: {np.min(all_mae):.2f}m - {np.max(all_mae):.2f}m")
        
        print(f"📊 MSE (Mean Squared Error):")
        print(f"   Mean: {np.mean(all_mse):.2f} ± {np.std(all_mse):.2f}")
        print(f"   Range: {np.min(all_mse):.2f} - {np.max(all_mse):.2f}")
        
        print(f"📊 RMSE (Root Mean Squared Error):")
        print(f"   Mean: {np.mean(all_rmse):.2f}m ± {np.std(all_rmse):.2f}m")
        print(f"   Range: {np.min(all_rmse):.2f}m - {np.max(all_rmse):.2f}m")
        
        print(f"📊 Maximum Distance Error:")
        print(f"   Mean: {np.mean(all_max_dist):.2f}m ± {np.std(all_max_dist):.2f}m")
        print(f"   Range: {np.min(all_max_dist):.2f}m - {np.max(all_max_dist):.2f}m")
        
        # 성능 순위 (MAE 기준)
        best_ue = min(similarity_results.keys(), key=lambda x: similarity_results[x]['mae'])
        worst_ue = max(similarity_results.keys(), key=lambda x: similarity_results[x]['mae'])
        
        print(f"\n🏆 Best Performance:  UE {best_ue}")
        print(f"   MAE: {similarity_results[best_ue]['mae']:.2f}m")
        print(f"   MSE: {similarity_results[best_ue]['mse']:.2f}")
        print(f"   RMSE: {similarity_results[best_ue]['rmse']:.2f}m")
        
        print(f"💥 Worst Performance: UE {worst_ue}")
        print(f"   MAE: {similarity_results[worst_ue]['mae']:.2f}m")
        print(f"   MSE: {similarity_results[worst_ue]['mse']:.2f}")
        print(f"   RMSE: {similarity_results[worst_ue]['rmse']:.2f}m")

def calculate_path_length(trajectory):
    """궤적의 총 길이 계산"""
    if len(trajectory) < 2:
        return 0
    
    total_length = 0
    for i in range(1, len(trajectory)):
        dx = trajectory['x'].iloc[i] - trajectory['x'].iloc[i-1]
        dy = trajectory['y'].iloc[i] - trajectory['y'].iloc[i-1]
        total_length += np.sqrt(dx**2 + dy**2)
    
    return total_length

def load_and_process_data():
    """Load data and preprocess for trajectory generation"""
    print("📊 Loading data...")
    
    # Actual positions (ue_position.txt)
    actual_df = pd.read_csv(CONFIG['actual_positions_file'])
    print(f"✅ Actual positions: {actual_df.shape}")
    
    # 🔥 절대 타임스탬프를 상대 타임스탬프로 변환
    # 첫 번째 타임스탬프를 기준점으로 설정
    first_timestamp = actual_df['timestamp'].min()
    print(f"📅 First timestamp: {first_timestamp}")
    
    # 상대 타임스탬프 계산 (100ms 단위로 정규화)
    actual_df['relative_timestamp'] = ((actual_df['timestamp'] - first_timestamp) / 100).astype(int)
    
    # 기존 timestamp를 relative_timestamp로 교체
    actual_df = actual_df.drop(columns=['timestamp']).rename(columns={'relative_timestamp': 'timestamp'})
    
    print(f"📊 Converted timestamps: {actual_df['timestamp'].min()} - {actual_df['timestamp'].max()}")
    
    # Predicted positions (lstm_trajectory.txt)
    predicted_df = pd.read_csv(CONFIG['predicted_positions_file'])
    print(f"✅ Predicted positions: {predicted_df.shape}")
    
    # Unify column names (imsi -> id)
    if 'imsi' in predicted_df.columns:
        predicted_df = predicted_df.rename(columns={'imsi': 'id'})
    
    # Sort by time
    actual_df = actual_df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    predicted_df = predicted_df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    
    print(f"📊 Actual UE IDs: {sorted(actual_df['id'].unique())}")
    print(f"📊 Predicted UE IDs: {sorted(predicted_df['id'].unique())}")
    
    return actual_df, predicted_df

def get_distinct_colors(n):
    """구분이 잘 되는 색상 생성"""
    if n <= 7:
        # 기본 7가지 뚜렷한 색상
        colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', '#FF0080', '#00FFFF']
        return colors[:n]
    elif n <= 12:
        # 12가지 색상 조합
        colors = ['#FF0000', '#0000FF', '#00FF00', '#FF8000', '#8000FF', '#FF0080', 
                 '#FFFF00', '#FF8080', '#8080FF', '#80FF80', '#FF8040', '#4080FF']
        return colors[:n]
    else:
        # 많은 수를 위한 HSV 색상환
        hues = np.linspace(0, 360, n, endpoint=False)
        colors = []
        for i, hue in enumerate(hues):
            # 채도와 명도를 조절해서 구분이 잘 되도록
            saturation = 0.8 if i % 2 == 0 else 1.0
            value = 0.9 if i % 3 == 0 else 0.7
            
            # HSV to RGB 변환
            h = hue / 60.0
            c = value * saturation
            x = c * (1 - abs((h % 2) - 1))
            m = value - c
            
            if 0 <= h < 1:
                r, g, b = c, x, 0
            elif 1 <= h < 2:
                r, g, b = x, c, 0
            elif 2 <= h < 3:
                r, g, b = 0, c, x
            elif 3 <= h < 4:
                r, g, b = 0, x, c
            elif 4 <= h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x
            
            colors.append(f'#{int((r+m)*255):02x}{int((g+m)*255):02x}{int((b+m)*255):02x}')
        
        return colors

def plot_trajectories(actual_df, predicted_df):
    """Plot actual vs predicted trajectory comparison with better colors"""    
    # Select only common UE IDs
    common_ues = sorted(set(actual_df['id'].unique()) & set(predicted_df['id'].unique()))
    total_ues = min(len(common_ues), CONFIG['max_ues'])  # CONFIG 사용
    common_ues = common_ues[:total_ues]
    
    print(f"🎯 Visualizing {total_ues} UE trajectory comparisons...")
    
    # 🔥 개선된 색상 설정
    colors = get_distinct_colors(CONFIG['ues_per_plot'])  # CONFIG 사용
    
    # Create 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=CONFIG['figure_size'])
    fig.suptitle('UE Trajectory Comparison: Actual vs Predicted', 
                 fontsize=20, fontweight='bold')
    
    axes = axes.flatten()
    
    # Divide into 4 groups
    ues_per_plot = CONFIG['ues_per_plot']    

    for plot_idx in range(4):
        ax = axes[plot_idx]
        
        # Select UEs for current plot
        start_idx = plot_idx * ues_per_plot
        end_idx = min(start_idx + ues_per_plot, len(common_ues))
        current_ues = common_ues[start_idx:end_idx]
        
        print(f"  🛣️ Group {plot_idx+1}: UE {current_ues}")
        
        # Plot actual vs predicted trajectories for each UE
        for i, ue_id in enumerate(current_ues):
            color = colors[i % len(colors)]
            
            # Actual trajectory data
            actual_data = actual_df[actual_df['id'] == ue_id].copy()
            if len(actual_data) > 1:
                ax.scatter(actual_data['x'], actual_data['y'], 
                            color='white', s=80, alpha=0.9, 
                            label=f'UE {ue_id} Actual',
                            marker='o', edgecolors=color, linewidths=3)
                
                # Start and end points
                ax.scatter(actual_data['x'].iloc[0], actual_data['y'].iloc[0], 
                          color=color, s=150, marker='s', edgecolors='black', 
                          linewidth=2, alpha=1.0, zorder=10)
                ax.scatter(actual_data['x'].iloc[-1], actual_data['y'].iloc[-1], 
                          color=color, s=200, marker='*', edgecolors='black', 
                          linewidth=2, alpha=1.0, zorder=10)
            
            # Predicted trajectory data
            predicted_data = predicted_df[predicted_df['id'] == ue_id].copy()
            if len(predicted_data) > 1:
                ax.scatter(predicted_data['x'], predicted_data['y'], 
                        color=color, s=80, alpha=0.8, 
                        label=f'UE {ue_id} Predicted',
                        marker='^', edgecolors='black', linewidths=0.5)
                ax.scatter(predicted_data['x'].iloc[0], predicted_data['y'].iloc[0], 
                        color=color, s=150, marker='s', edgecolors='white', 
                        linewidth=2, alpha=1.0, zorder=12)  # 시작점
                ax.scatter(predicted_data['x'].iloc[-1], predicted_data['y'].iloc[-1], 
                        color=color, s=200, marker='*', edgecolors='white', 
                        linewidth=2, alpha=1.0, zorder=12)  # 끝점
        
        # Axis settings
        ax.set_xlabel('X Coordinate (m)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y Coordinate (m)', fontsize=14, fontweight='bold')
        ax.set_title(f'Group {plot_idx+1}: UE {start_idx+1}-{end_idx} Trajectories', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.4, linewidth=1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        ax.set_aspect('equal', adjustable='box')
        
        # Add enhanced symbol legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='black', label='Start Point', 
                      markersize=10, linestyle='None', markerfacecolor='lightgray', 
                      markeredgewidth=2),
            plt.Line2D([0], [0], marker='*', color='black', label='End Point', 
                      markersize=12, linestyle='None', markerfacecolor='lightgray',
                      markeredgewidth=2),
            plt.Line2D([0], [0], marker='o', color='gray', label='Actual Points', 
                       markersize=8, linestyle='None', markerfacecolor='white', markeredgewidth=2),
            plt.Line2D([0], [0], marker='^', color='gray', label='Predicted Points', 
                      markersize=8, linestyle='None', markerfacecolor='gray', markeredgewidth=2)
        ]
        
        # Combine legends
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles + legend_elements
        all_labels = labels + [elem.get_label() for elem in legend_elements]
        ax.legend(all_handles, all_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(CONFIG['output_plot_file'], dpi=CONFIG['plot_dpi'], bbox_inches='tight')
    plt.show()
    print(f"💾 Saved: {CONFIG['output_plot_file']}")

def main():
    """Main execution"""
    print("🚀 Starting UE trajectory visualization!")
    
    try:
        # 데이터 로딩
        actual_df, predicted_df = load_and_process_data()

        # 1. 궤적 유사도 분석 (터미널 출력)
        analyze_trajectory_similarity_optimized(actual_df, predicted_df)
        
        # 2. 궤적 비교 시각화
        plot_trajectories(actual_df, predicted_df)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
