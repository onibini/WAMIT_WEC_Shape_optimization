# 🌊 WEC Shape and Layout Optimization using WAMIT Automation

This project is a Python-based framework designed for the **hydrodynamic shape and layout optimization** of Point-absorber Wave Energy Converters (WECs). It automates the complex workflow of **WAMIT** (Potential Flow Solver) and utilizes metaheuristic algorithms to maximize energy capture efficiency while strictly adhering to physical and spatial constraints.

---

## 🚀 Key Features

* **Hybrid Optimization Algorithms**: Provides implementations of Differential Evolution (**DE**), Particle Swarm Optimization (**PSO**), and a hybrid **DEPSO** algorithm to navigate complex design spaces.
* **Full WAMIT Automation**: Automatically generates all necessary WAMIT input files (`.gdf`, `.pot`, `.frc`, `.cfg`, etc.), executes the solver via subprocess, and parses numerical outputs (`.1`, `.2`, `.hst`, `.4`).
* **Spectral Energy Analysis**: Evaluates performance in irregular wave environments using the **JONSWAP Spectrum** and Response Amplitude Operators (RAO).
* **Efficiency Engine (Memory Caching)**: Includes a "Memory Hit" system that stores previously evaluated fitness values for specific coordinates and shapes to avoid redundant, time-consuming WAMIT simulations.
* **Spatial Constraint Enforcement**: Automatically enforces minimum distance requirements between multiple WECs and maintains design parameters within a defined grid (`STEP_SIZE`).

---

## 📂 Project Structure

The project is organized into modular components to separate optimization logic from hydrodynamic calculations:

```text
.
├── main.py                # Entry point for starting the optimization process
├── config.py              # Environmental data, physical bounds, and hyperparameters
├── objective_functions.py # Interface between the optimizers and the WAMIT solver
├── wamit_utils.py         # WAMIT I/O management, JONSWAP spectrum, and power logic
├── utils.py               # Logging, grid generation, and result management
└── optimizers/            # Metaheuristic algorithm implementations
    ├── DE.py              # Differential Evolution
    ├── PSO.py              # Particle Swarm Optimization
    └── DEPSO.py            # Hybrid DE-PSO Algorithm

```

---

## 📐 Methodology

### 1. Design Variables (8-Dimensions)

The framework optimizes an 8-dimensional vector representing both the arrangement and the physical dimensions of the WECs:

* **Layout**: Planar coordinates for three WECs (assuming symmetry): .
* **Geometry**: Buoy diameter () and draft ().

### 2. Performance Evaluation

The objective function (`shape_opt_func`) follows an automated loop:

1. **Input Generation**: Design variables are used to create WAMIT geometry and force files.
2. **Simulation**: `wamit.exe` is executed in a dedicated directory.
3. **Parsing**: Hydrodynamic coefficients (added mass, damping, wave excitation force) are extracted from output files.
4. **Power Calculation**: Total power is calculated by integrating the response spectrum (RAO + JONSWAP) across a defined frequency range.

---

## 💻 Getting Started

### Prerequisites

* **WAMIT v7.x** or higher (must be installed, and `wamit.exe` must be accessible).
* **Python 3.8+** with the following dependencies:
```bash
pip install numpy pandas scipy psutil

```



### Configuration

You can select pre-defined environmental data (Buan, Chilbaldo, Yeongilmanhang) and set physical constraints in `config.py`:

```python
# config.py
CURRENT_LOC = "Buan"  # Selected location for Hs, Tp, and depth
BOUNDS = {'x1': [3, 15], ... , 'D': [1, 5]}  # Search space boundaries
MAX_ITER = 100  # Maximum optimization iterations

```

### Execution

Run the optimization process using:

```bash
python main.py

```

---

## 📊 Result Management

Upon completion, the system generates two distinct result files categorized by location:

* **`{Location}_cal.res`**: A detailed log of every unique WAMIT evaluation, including design parameters and specific power output for each WEC.
* **`{Location}_iter.res`**: A history of the best fitness (global best) achieved at each iteration to track algorithm convergence.

---

## 📝 License

This project is developed for research purposes in Wave Energy Converter design and optimization.
