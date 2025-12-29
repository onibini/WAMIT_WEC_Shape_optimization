🌊 WEC Shape & Layout Optimization with WAMIT Automation
This project provides a comprehensive Python framework for the hydrodynamic shape and layout optimization of Point-absorber Wave Energy Converters (WECs). It automates the execution of WAMIT (Potential Flow Solver) and employs metaheuristic algorithms to maximize power capture efficiency while adhering to physical constraints.

🚀 Key Features
Hybrid Optimization Algorithms: Includes Differential Evolution (DE), Particle Swarm Optimization (PSO), and a hybrid DEPSO to find global optima in a complex design space.

Automated WAMIT Workflow: Automatically generates .pot, .frc, .gdf, and .cfg files, executes WAMIT via subprocess, and parses the output .1, .2, .hst files.

Efficiency Engine (Memory Hit): Implements a memory-based caching system that stores previously calculated fitness values to avoid redundant, time-consuming WAMIT simulations.

Constraint Management: Enforces minimum distance checks between WECs and ensures the buoy remains within a predefined grid (STEP_SIZE).

Spectral Analysis: Calculates annual energy production using the JONSWAP Spectrum and RAO (Response Amplitude Operator).

🛠 Project Structure
.
├── main.py                # Entry point for starting the optimization
├── config.py              # Environment (Hs, Tp, h), Bounds, and Hyperparameters
├── objective_functions.py # Interface between optimizers and WAMIT
├── wamit_utils.py         # WAMIT I/O, JONSWAP, and Power calculation logic
├── utils.py               # Logging, grid generation, and file management
└── optimizers/            # Metaheuristic algorithm implementations
    ├── __init__.py        # Package level exports
    ├── DE.py              # Differential Evolution
    ├── PSO.py              # Particle Swarm Optimization
    └── DEPSO.py            # Hybrid DE-PSO Algorithm

📐 Methodology
1. Design Variables (8-Dimensions)
  The optimization vector includes the layout coordinates of three WECs (assuming symmetry) and the buoy dimensions:
    Position: $(x_1, y_1), (x_2, y_2), (x_3, y_3)$
    Geometry: Diameter ($d$) and Draft ($D$)
2.Hydrodynamic Power Calculation
  The objective function maximizes the total power absorbed across a frequency range:
    $$P_{avg} = \int_{0}^{\infty} S_{\eta}(\omega) \cdot \bar{P}(\omega) d\omega$$
  Where $\bar{P}(\omega)$ is derived from the RAO and the optimized PTO damping matrix.

💻 Getting Started
Prerequisites
  WAMIT v7.x or higher (must be installed and wamit.exe accessible).

  Python 3.8+ with following packages:

Configuration
Edit config.py to set your target location (e.g., Incheon, Buan) and physical bounds:

  # config.py
  CURRENT_LOC = "Buan"
  BOUNDS = {'x1': [3, 15], ... , 'D': [1, 5]}
  MAX_ITER = 100

Running the Optimization
Execute the main script to start the process:

  python main.py

📊 Result Management
The system generates two primary result files for each run:

  1. {Location}_cal.res: Detailed log of every unique WAMIT evaluation (parameters + power).

  2. {Location}_iter.res: Best fitness achieved at each iteration to track convergence.
