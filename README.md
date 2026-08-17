The Generalized Adaptive Bridge Regression (GABR) framework based on a universal Coordinate Descent solver are implemented in Julia (v1.10) to leverage high-performance Just-In-Time (JIT) compilation. For details see:  https://doi.org/10.1007/s11222-026-10938-1
Note that some improvements have been done on the CV implementation for better stability, the GABR-L0 penalty allows for tuning of the variable selection threshold in the 'dense' bridge domain (1 < q < 2) and binary responses can now be analyzed. In addition, some printing errors in the paper are outlined in the errata document published here.

1. Prerequisites
You only need to have Julia installed. The script will automatically detect and install all required Julia packages (e.g., DataFrames, TreeParzen, CSV) on its first run.

2. Prepare Your Data
Ensure your dataset is a clean .csv file where all features are numeric columns with headings.

For Gaussian (regression): Target variable should be continuous.

For Binomial (classification): Target variable must be binary (0 or 1).

3. Configure the Script
Open the script and locate the 1. USER CONFIGURATION block. Update the PENALTY_SELECTION and the CONFIG struct to match your dataset:

Julia
# SELECTION: Choose your penalty
const PENALTY_SELECTION = "GABR-L0" 

const CONFIG = Config(
    "QTLMAS2010bin.csv",    # <-- 1. Path to your CSV
    :Y,                     # <-- 2. Target column name
    "binomial",             # <-- 3. "gaussian" or "binomial"
    5,                      # 4. Number of Outer CV folds
    2024,                   # 5. Random seed
    250,                    # 6. TPE optimization rounds (budget)
    2000,                   # 7. Max Coordinate Descent iterations
    1e-7,                   # 8. Coordinate Descent tolerance
    100,                    # 9. Max Outer KKT active-set loops
    15,                     # 10. Max Newton-Raphson iterations (for GABR)
    nothing                 # 11. Fixed Alpha (for ElasticNet only)
)

4. Run the Pipeline
You can run the script directly from your terminal:
Bash
julia GABR.jl

Or from inside the Julia REPL:
Julia
julia> include("GABR.jl")

5. Once the distributed Cross-Validation finishes, the script prints summary statistics (Loss, AUC/dCor, optimal parameters, and selected feature counts) to the console and generates several artifacts in your working directory:
[PENALTY]_Coefficients_Average_TPE.csv
Contains the final bagged coefficients (averaged across all outer folds).
Includes a Stability Score (0.0 to 1.0) indicating the percentage of folds where the feature was selected (non-zero), which is highly useful for robust feature selection.

[PENALTY]_Coefficients_PerFold_TPE.csv
The raw coefficient vectors produced by each individual CV fold.

[PENALTY]_convergence_Loss.csv
The TPE optimization trajectory (Mean Loss vs. Iteration).

[PENALTY]_timing.csv
Wall-clock execution time metrics.
    15,                     # 10. Max Newton-Raphson iterations (for GABR)
    nothing                 # 11. Fixed Alpha (for ElasticNet only)
)
