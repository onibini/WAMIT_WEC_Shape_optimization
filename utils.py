import numpy as np
from typing import List
import shutil


# =============================================================================
# 🛠️ 데이터 처리 및 수치 보정 함수
# =============================================================================
def apply_step_size(vector:np.ndarray, step_size:float) -> np.ndarray:
    """
    역할: 연속적인 수치 데이터를 사용자가 정의한 그리드(격자) 간격에 맞게 보정합니다.
    Input:
        - vector: 보정할 수치 데이터 (NumPy 배열)
        - step_size: 그리드 간격 (예: 1.0이면 1단위로 반올림)
    Output: 그리드 간격에 맞춰 보정된 NumPy 배열
    """
    return np.round(vector / step_size) * step_size

def generate_on_grid(n_samples:int, lower:np.ndarray, upper:np.ndarray, step_size:float) -> np.ndarray:
    """
    역할: 설정된 범위(Bounds) 내에서 그리드 간격을 유지하며 무작위 초기 위치들을 생성합니다.
    Input:
        - n_samples: 생성할 샘플(개체)의 개수
        - lower / upper: 각 변수의 최소/최대 범위를 담은 배열
        - step_size: 그리드의 간격
    Output: 그리드 위에 배치된 무작위 초기 위치 행렬 (n_samples x dimensions)
    """
    num_steps = np.rint((upper - lower) / step_size).astype(int) + 1
    random_indices = np.zeros((n_samples, len(lower)), dtype=int)
    for i in range(len(lower)):
        if num_steps[i] > 1:
            random_indices[:, i] = np.random.randint(0, num_steps[i], size=n_samples)
    return lower + random_indices * step_size


# =============================================================================
# 📝 파일 기록 및 결과 관리 함수
# =============================================================================
def write_results(result_vector:List, results_path:str):
    """
    역할: 계산된 결과 데이터 한 줄을 지정된 파일 끝에 추가로 기록합니다.
    Input:
        - result_vector: 기록할 데이터 리스트 (좌표, 성능, 파워 등)
        - results_path: 저장할 파일 경로 (.res 또는 .csv)
    Output: 없음 (파일 쓰기 수행)
    """
    with open(results_path, 'a') as f:
        f.write(', '.join(map(str, result_vector)) + '\n')

def move_results_file(loc_name:str):
    """
    역할: 최적화 완료 후, 임시 결과 파일들을 지역명과 알고리즘명이 포함된 고유 이름으로 변경합니다.
    Input:
        - loc_name: 실험 지역 명칭 (예: 'Incheon')
        - algo_name: 사용된 알고리즘 명칭 (예: 'DEPSO')
    Output: 없음 (파일 이동 및 이름 변경 수행)
    """
    shutil.move('Calculation_results.res', f'{loc_name}_cal.res')
    shutil.move('Iteration_results.res', f'{loc_name}_iter.res') 
    

# =============================================================================
# 📢 터미널 로그 출력 (Logging) 함수
# =============================================================================
def print_start_message(idx:int, vector:np.ndarray):
    """
    역할: 최적화 시작 단계에서 각 개체의 초기 위치 정보를 화면에 출력합니다.
    Input: idx (개체 번호), vector (좌표 배열)
    """
    print("-" * 60 + f'\n 💡 Initial Position {idx + 1}: {np.round(vector, 2)}')

def print_eval_message(elapsed):
    """
    역할: 한 번의 성능 평가(목적함수 계산)가 완료되었을 때 소요 시간을 출력합니다.
    Input: elapsed (소요 시간, 초 단위)
    """
    print(f"      ⏱️ Evaluation finished in {elapsed:.2f} seconds.")

def print_iter_start_message(func_name:str, location:str, Hs:float, Tp:float, h:float):
    """
    역할: 최적화 실험의 기본 정보와 시작을 알리는 헤더를 출력합니다.
    Input: func_name (알고리즘명), location (지역), Hs (파고), Tp (주기), h (수심)
    """
    print("=" * 70)
    print(f"🚀 {func_name} Optimization Start")
    print(f"📍 Location: {location} | Hs: {Hs} m, Tp: {Tp} s, h: {h} m")
    print("=" * 70)

def print_summary_message(iteration: int, gbest: np.ndarray, gbest_fitness: float, total_time: float, cnt_memory: int):
    """
    역할: 매 반복(Iteration) 단계 종료 시 현재까지의 최적 성과와 통계 정보를 요약 출력합니다.
    Input:
        - iteration: 현재 반복 횟수
        - gbest: 현재까지의 전역 최적 위치
        - gbest_fitness: 현재까지의 최고 성능값
        - total_time: 반복에 소요된 총 시간 (분)
        - cnt_memory: 메모리 참조(중복 계산 방지) 횟수
    """
    print(f'\n--- Iteration {iteration} Summary ---')
    print(f'  🌟 Best Position: {np.round(gbest, 2)}')
    print(f'  🏆 Best Fitness : {gbest_fitness / 1000:.2f} kW')
    print(f'  ⏳ Elapsed Time  : {total_time:.2f} min')
    print(f'  🧠 Memory Hits  : {cnt_memory}')
