import numpy as np
import itertools
import textwrap
import os
import subprocess
import psutil
import pandas as pd
from typing import Tuple, Dict
from collections import defaultdict
from scipy.integrate import trapezoid

def normalize_layout_vector(vector:np.ndarray)->np.ndarray:
    '''WEC 배치 벡터를 y-x 좌표 순서로 정규화하여 중복 계산 방지'''
    if (vector[1], vector[0]) < (vector[3], vector[2]):
        vector[[0, 2]] = vector[[2, 0]]
        vector[[1, 3]] = vector[[3, 1]]
    return vector

def distance_check(vector:list, min_distance:float)->bool:
    '''WEC 간 최소 거리 제약 조건 확인'''
    x1, y1, x2, y2, x3, y3, *_ = vector
    point_xy = [(x1, y1), (x2, y2), (x3, y3), (x2, -y2), (x1, -y1)]
    for p1, p2 in itertools.combinations(point_xy, 2):
        dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        if dist < min_distance:
            return False
    return True


def JONSWAP_SPEC(omega:np.ndarray, Hs:float, Tp:float, gamma:float)->np.ndarray:
    wp = 2 * np.pi / Tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    r = np.exp(-((omega - wp)**2) / (2 * sigma**2 * wp**2))

    beta_denominator = 0.23 + 0.0336 * gamma - 0.185 * (1.9 + gamma)**-1
    beta = (0.0624 / beta_denominator) * (1.094 - 0.01915 * np.log(gamma))

    S = np.zeros_like(omega)
    non_zero_indices = omega > 0

    omega_safe = omega[non_zero_indices]
    r_safe = r[non_zero_indices]

    term1 = beta * (Hs ** 2 * wp ** 4) / (omega_safe ** 5)
    term2 = np.exp(-1.25 * (omega_safe / wp) ** -4)
    term3 = gamma ** r_safe
    S[non_zero_indices] = term1 * term2 * term3

    return S

def remove_file():
    remove_list = ['wec.1', 'wec.2', 'wec.3', 'wec.4',
                   'wec.out', 'wec.p2f']
    for file in remove_list:
        if os.path.exists(os.path.join(r'wamit_optimization', file)):
            os.remove(os.path.join(r'wamit_optimization', file))
    print("     Previous WAMIT output files removed.")

def run_wamit():
    remove_file()
    with subprocess.Popen(['wamit.exe', 'fnames.wam'], cwd=r'wamit_optimization',
                        #   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                          ) as proc:
        proc.wait()

def cal_power(rao:Dict[float, np.ndarray], pto_matrix:np.ndarray, Hs:float, Tp:float)->np.ndarray:
    omega_array = np.array(sorted(rao.keys()))
    s_values = JONSWAP_SPEC(omega_array, Hs, Tp, 1.4)
    rao_matrix = np.array([rao[omega] for omega in omega_array])
    cpto_list = np.diag(pto_matrix)

    P_bar_matrix = 0.5 * cpto_list * (omega_array[:, np.newaxis]**2) * (rao_matrix**2)
    response_spec = P_bar_matrix * s_values[:, np.newaxis]

    return trapezoid(response_spec, omega_array, axis=0)


class WamitInputGenerator:
    '''WAMIT 입력 파일 생성기 클래스'''
    def __init__(self, base_dir=r'wamit_optimization', project_title='Shape Optimization'):
        self.base_dir = base_dir
        self.project_title = project_title

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_path(self, filename:str)->str:
        return os.path.join(self.base_dir, filename)
    
    def input_init(self):
        '''WAMIT 입력 파일 초기화'''
        self.fname_file()
        self.cfg_file()
        self.spl_file()
        self.cfgwam_file()
    
    def input_optimize(self, vector:np.ndarray, h:float):
        '''WAMIT 최적화 입력 파일 생성'''
        self.pot_file(vector, h)
        self.local_frc_file(vector)
        self.gdf_file(vector)
        self.global_frc_file()

    def pot_file(self, vector:np.ndarray, h:float):
        """
        역할: WAMIT용 .pot 파일을 생성합니다.
        Input: vector (배치 벡터), h (수심)
        Output: 없음 (.pot 파일 생성)
        """
        x1, y1, x2, y2, x3, y3, *_ = vector

        xb = [f"{x1} {y1} 0 0", f"{x2} {y2} 0 0", f"{x3} {y3} 0 0", 
              f"{x2} {-y2} 0 0", f"{x1} {-y1} 0 0"]

        input = textwrap.dedent(f"""\
            {self.project_title}
            {h:<20}HBOT
            {"0 0":<20}IRAD,IDIFF
            {"-51":<20}NPER
            {"0.5 0.05":<20}PER
            {"1":<20}NBETA
            {"0":<20}BETA
            {"5":<20}NBODY
            {"wec.gdf":<20}BODY1
            {xb[0]:<20}XBODY1
            {"0 0 1 0 0 0":<20}IMODE(1-6)
            {"wec.gdf":<20}BODY2
            {xb[1]:<20}XBODY2
            {"0 0 1 0 0 0":<20}IMODE(1-6)
            {"wec.gdf":<20}BODY3
            {xb[2]:<20}XBODY3
            {"0 0 1 0 0 0":<20}IMODE(1-6)
            {"wec.gdf":<20}BODY4
            {xb[3]:<20}XBODY4
            {"0 0 1 0 0 0":<20}IMODE(1-6)
            {"wec.gdf":<20}BODY5
            {xb[4]:<20}XBODY5
            {"0 0 1 0 0 0":<20}IMODE(1-6)
        """)
        with open(self._get_path('wec.pot'), 'w') as f:
            f.write(input)

    def global_frc_file(self):
        """
        역할: WAMIT용 global.frc 파일을 생성합니다.
        Input: 없음
        Output: 없음 (global.frc 파일 생성)
        """
        input = textwrap.dedent(f"""\
            {self.project_title}
            {"1 1 1 1 0 0 0 0 0":<20}OutputFile
            {"1025":<20}Rho
            {"wec_local.frc":<20}WEC1FRC
            {"wec_local.frc":<20}WEC2FRC
            {"wec_local.frc":<20}WEC3FRC
            {"wec_local.frc":<20}WEC4FRC
            {"wec_local.frc":<20}WEC5FRC
            {"0":<20}NBETAH
            {"0":<20}NFIELD
                                """)
        with open(self._get_path('wec.frc'), 'w') as f:
            f.write(input)

    def local_frc_file(self, vector:np.ndarray):
        """
        역할: WAMIT용 wec_local.frc 파일을 생성합니다.
        Input: vector (배치 벡터)
        Output: 없음 (.frc 파일 생성)
        """
        _, _, _, _, _, _, d, D = vector
        radius = d / 2
        mass = 1025 * np.pi * radius**2 * D
        Izz = 0.5 * mass * radius**2

        input = textwrap.dedent(f"""\
            {self.project_title}
            {"1 1 1 1 0 0 0 0 0":<20}OutputFile
            {"1025":<20}Rho
            {"0 0 0":<20}VCG
            {"1":<20}IMASS
            {mass:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g}
            {0:<8.3g} {mass:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g}
            {0:<8.3g} {0:<8.3g} {mass:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g}
            {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g}
            {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g}
            {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {0:<8.3g} {Izz:<8.3g}
            {"0":<20}IDAMP
            {"0":<20}ISTIFF
            {"0":<20}NBETAH
            {"0":<20}NFIELD
                                """)
        with open(self._get_path('wec_local.frc'), 'w') as f:
            f.write(input)

    def cfg_file(self):
        """
        역할: WAMIT용 wec.cfg 파일을 생성합니다.
        Input: 없음
        Output: 없음 (.cfg 파일 생성)
        """
        input = textwrap.dedent(f"""\
            {self.project_title}
            ipltdat=1
            NUMHDR=1
            IPERIN=2
            IPEROUT=2
            IWALLX0=1
            ISOR=0
            ISOLVE=1
            MAXITT=100
            ISCATT=0
            IALTFRC=3
            IALTFRCN=2 2 2 2 2
            ILOG=1
            ILOWHI=1
            KSPLIN=3
            IQUADO=3
            IQUADI=4
            NOOUT=1 1 1 1 0 0 0 0 0
            USERID_PATH={r"C:/WAMITv7"}
                                """)
        with open(self._get_path('wec.cfg'), 'w') as f:
            f.write(input)

    def gdf_file(self, vector:np.ndarray):
        """
        역할: WAMIT용 wec.gdf 파일을 생성합니다.
        Input: vector (배치 벡터)
        Output: 없음 (.gdf 파일 생성)
        """
        _, _, _, _, _, _, d, D = vector
        radius = d / 2

        input = textwrap.dedent(f"""\
            {self.project_title}
            {"1.0 9.806":<20}ULEN GRAV
            {"1 1":<20}ISX ISY
            {"2 -1":<20}NPATCH IGDEF
            {"2":<20}
            {f"{radius} {D}":<20}RADIUS DRAFT
            {"1":<20}
                                """)
        with open(self._get_path('wec.gdf'), 'w') as f:
            f.write(input)

    def spl_file(self):
        """
        역할: WAMIT용 wec.spl 파일을 생성합니다.
        Input: 없음
        Output: 없음 (.spl 파일 생성)
        """
        input = textwrap.dedent(f"""\
            {self.project_title}
            {"4 4":<20}
            {"4 4":<20}
                                """
                                )
        with open(self._get_path('wec.spl'), 'w') as f:
            f.write(input)

    def fname_file(self):
        """
        역할: WAMIT용 fnames.wam 파일을 생성합니다.
        Input: 없음
        Output: 없음 (.wam 파일 생성)
        """
        input = textwrap.dedent(f"""\
            {"wec.cfg"}
            {"wec.pot"}
            {"wec.frc"}
                                """)
        with open(self._get_path('fnames.wam'), 'w') as f:
            f.write(input)
    
    def cfgwam_file(self):
        """
        역할: WAMIT용 cfgwam.wam 파일을 생성합니다.
        Input: 없음
        Output: 없음 (cfgwam.wam 파일 생성)
        """
        num_cpu = os.cpu_count()
        memory_info = psutil.virtual_memory()
        total_ram = memory_info.total / (1024**3)

        input = textwrap.dedent(f"""\
            {self.project_title}
            RANGBMAX={round(total_ram/2)}
            NCPU={num_cpu}
                                """)
        with open(self._get_path('config.wam'), 'w') as f:
            f.write(input)

class WamitOutputParser:
    '''WAMIT 출력 파일 파서 클래스'''
    def __init__(self, base_dir=r'wamit_optimization'):
        self.base_dir = base_dir
        self.matrix_map = {idx: i for i, idx in enumerate(self.indices_wec())}

    def _get_path(self, filename:str)->str:
        return os.path.join(self.base_dir, filename)
    
    def num_wec(self)->int:
        """
        역할: WAMIT 출력 파일에서 WEC 수를 읽어옵니다.
        Input: 없음
        Output: WEC 수
        """
        with open(self._get_path('wec.pot'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'NBODY' in line:
                num_wec = int(line.split()[0])
                return num_wec
            
    def indices_wec(self)->Tuple[int, ...]:
        """
        역할: WAMIT 출력 파일에서 WEC 인덱스를 읽어옵니다.
        Input: 없음
        Output: WEC 인덱스 튜플
        """
        num_wec = self.num_wec()
        return tuple(range(3, num_wec*6+1, 6))

    def wec_mass(self)->float:
        """
        역할: WAMIT frc 파일에서 WEC 질량을 가져옵니다.
        Input: 없음
        Output: 질량
        """
        with open(self._get_path('wec_local.frc'), 'r') as f:
            lines = f.readlines()
        for idx, line in enumerate(lines):
            if idx == 5:
                mass = float(line.split()[0])
                return mass

    def read_hydro(self):
        """
        역할: WAMIT 출력 파일에서 수력학적 계수를 읽어옵니다.
        Input: 없음
        Output: 수력학적 계수 딕셔너리
        """
        hydro_path = self._get_path('wec.1')
        hydro_df = pd.read_csv(hydro_path, sep=r'\s+', skiprows=1, names = ['freq', 'I', 'J', 'AddedMass', 'Damping'])
        hydro_df['AddedMass'] *= 1025
        hydro_df['Damping'] *= 1025 * hydro_df['freq']

        unique_freqs = sorted(hydro_df['freq'].unique())
        added_mass, damping = {}, {}
        for freq in unique_freqs:
            A, B = np.zeros((self.num_wec(), self.num_wec())), np.zeros((self.num_wec(), self.num_wec()))
            for _, row in hydro_df[hydro_df['freq'] == freq].iterrows():
                i, j = row['I'], row['J']
                if i in self.matrix_map and j in self.matrix_map:
                    idx_i, idx_j = self.matrix_map[i], self.matrix_map[j]
                    A[idx_i, idx_j] = row['AddedMass']
                    B[idx_i, idx_j] = row['Damping']
            added_mass[freq] = A
            damping[freq] = B
        return added_mass, damping
    
    def read_force(self):
        """
        역할: WAMIT 출력 파일에서 파력 데이터를 읽어옵니다.
        Input: 없음
        Output: 파력 데이터 배열
        """
        force_path = self._get_path('wec.2')
        force_df = pd.read_csv(force_path, sep=r'\s+', skiprows=1, names=['freq', 'beta', 'I', 'Mod', 'Phase', 'Re', 'Im'])
        force_df['Complex'] = (force_df['Re'] + 1j * force_df['Im']) * 1025 * 9.806

        unique_freqs = sorted(force_df['freq'].unique())
        force = {}
        for freq in unique_freqs:
            F = np.zeros(self.num_wec(), dtype=complex)
            for _, row in force_df[force_df['freq'] == freq].iterrows():
                i = row['I']
                if i in self.matrix_map:
                    F[self.matrix_map[i]] = row['Complex']
            force[freq] = F
        return force

    def read_stiff(self):
        """
        역할: WAMIT 출력 파일에서 강성 계수를 읽어옵니다.
        Input: 없음
        Output: 강성 계수 배열
        """
        stiff_path = self._get_path('wec.hst')
        stiff_df = pd.read_csv(stiff_path, sep=r'\s+', skiprows=1, names=['I', 'J', 'Stiffness'])
        stiff_df['Stiffness'] *= 1025 * 9.806
        
        num_wecs = self.num_wec()
        stiffness_matrix = np.zeros((num_wecs, num_wecs))

        for _, row in stiff_df.iterrows():
            i, j = int(row['I']), int(row['J'])
            if i in self.matrix_map and j in self.matrix_map:
                idx_i, idx_j = self.matrix_map[i], self.matrix_map[j]
                stiffness_matrix[idx_i, idx_j] = row['Stiffness']
        
        return stiffness_matrix
    
    def read_wn(self)->float:
        """
        역할: WAMIT 출력 파일에서 고유 진동수를 읽어옵니다.
        Input: 없음
        Output: 고유 진동수
        """
        rao = defaultdict(list)
        wn_path = self._get_path('wec.4')
        with open(wn_path, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:
            freq = float(line.split()[0])
            rao[freq].append(np.mean(float(line.split()[3])))
        max_wn = max(rao, key=lambda k: max(rao[k]))
        return max_wn
            
    def pto_matrix(self)->np.ndarray:
        """
        역할: WAMIT 출력 파일에서 PTO 감쇠 행렬을 읽어옵니다.
        Input: 없음
        Output: PTO 감쇠 행렬
        """
        _, damping = self.read_hydro()
        wn = self.read_wn()
        b_pto_mat = damping[wn]
        return b_pto_mat

    def cal_rao(self)->Dict[float, np.ndarray]:
        """
        역할: RAO (Response Amplitude Operator)를 계산합니다.
        Input: addedmass (추가 질량 딕셔너리), damping (감쇠 딕셔너리), force (파력 딕셔너리), stiffness (강성 행렬)
        Output: RAO 딕셔너리
        """
        addedmass, damping = self.read_hydro()
        force = self.read_force()
        stiffness = self.read_stiff()

        rao_vector = {}
        mass_mat = np.diag([self.wec_mass()] * self.num_wec())

        c_mat = stiffness
        b_pto_mat = self.pto_matrix()

        for omega, A in addedmass.items():
            B = damping[omega]
            F = force[omega]
            total_damping_mat = B + b_pto_mat
            term1 = -omega**2 * (mass_mat + A)
            term2 = 1j * omega * total_damping_mat
            impedance_matrix = term1 + term2 + c_mat

            rao = np.linalg.solve(impedance_matrix, F)
            rao_vector[omega] = np.abs(rao)

        return rao_vector


if __name__ == "__main__":
    remove_file()
    
    
