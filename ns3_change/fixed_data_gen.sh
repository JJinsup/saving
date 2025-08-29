#!/usr/bin/env bash

echo "🚀 Starting SINR-based UE Position Tracking Data Generation"

# 5회 반복 (각기 다른 모빌리티 패턴)
for i in $(seq 1 5); do
    echo "=== Mobility Run ${i} Simulation ==="
    
    # 1) 시드/모빌리티 런을 바꿔서 시뮬레이션 실행
    echo "Running with mobility run: ${i}"
    ./ns3 run "scratch/lstm_trajectory_estimation_scenario_train.cc --mobRun=${i}"
    
    # 실행 성공 여부 체크
    if [ $? -ne 0 ]; then
        echo "❌ Simulation ${i} failed!"
        continue
    fi
    
    # 2) txt 파일 압축 (런 번호 포함)
    ARCHIVE_NAME="data_lstm_mobrun_${i}.tar.gz"
    echo "Archiving txt files ⇒ ${ARCHIVE_NAME}"
    find . -maxdepth 1 -type f -name '*.txt' \
         ! -name 'CMakeLists.txt' \
         -print0 \
      | tar --null -czvf "${ARCHIVE_NAME}" --files-from -
    
    # 압축 성공 여부 체크
    if [ $? -eq 0 ]; then
        echo "✅ Archive ${ARCHIVE_NAME} created successfully"
        
        # 3) 원본 txt 파일 삭제
        find . -maxdepth 1 -type f -name '*.txt' \
             ! -name 'CMakeLists.txt' \
             -delete
        echo "🗑️  Original txt files cleaned up"
    else
        echo "❌ Archive creation failed for run ${i}"
    fi
    
    echo "───────────────────────────────────────"
done

echo "✅ All mobility-varied simulations completed!"
