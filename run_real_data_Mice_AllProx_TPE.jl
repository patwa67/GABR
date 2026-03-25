# ==============================================================================
# ROBUST STOCHASTIC TPE OPTIMIZATION FOR HIGH-DIMENSIONAL REGRESSION
# Penalties: ELASTICNET, BRIDGE, CAPPEDBRIDGE, MCP, SCAD, CAPPEDL1, LOGSUM
# ==============================================================================

using Distributed
using Pkg

# --- 0. PACKAGE SETUP ---
const REQUIRED_PKGS = ["DataFrames", "CSV", "SpecialFunctions", "Distributions", 
                       "TreeParzen", "Printf", "LinearAlgebra", "Logging"]

for pkg in REQUIRED_PKGS
    if Base.find_package(pkg) === nothing
        println("Installing missing package: $pkg...")
        Pkg.add(pkg)
    end
end

# Initialize multiprocessing workers to parallelize the outer Cross-Validation folds.
const DESIRED_WORKERS = 15
if nprocs() < (DESIRED_WORKERS + 1)
    addprocs((DESIRED_WORKERS + 1) - nprocs())
end
println("Active Workers: $(nworkers())")
flush(stdout)

@everywhere begin
    using LinearAlgebra, Statistics, Random, DataFrames, CSV, SpecialFunctions, Printf, Logging
    using TreeParzen
    const HP = TreeParzen.HP

    # --- 1. GLOBAL CONFIGURATION ---
    # SELECTION: Choose one of ["ELASTICNET", "BRIDGE", "CAPPEDBRIDGE", "MCP", "SCAD", "CAPPEDL1", "LOGSUM"] (CAPPEDBRIDGE is an experimental penalty)
    const PENALTY_SELECTION = "BRIDGE" 
    
    # Unified empirical sparsity threshold (τ = 10^-8). 
    # Coefficients smaller than this are truncated to exactly zero.
    const SPARSITY_THRESHOLD = 1e-8

    """
    Configuration for the solver and TPE hyperparameter optimization.
    """
    struct Config
        data_file::String                       # Name of data file, should contain both reponse and predictors. It should also have named columns
        target_col::Symbol                      # Name of response
        n_folds::Int                            # Number of outer CV folds, ideally number of cores/threads
        seed::Int                               # Global random seed
        tpe_rounds::Int                         # Iterations of Bayesian optimization
        cd_max_iter::Int                        # Max iterations for inner coordinate descent
        cd_tol::Float64                         # Convergence tolerance for coordinate descent
        fixed_alpha::Union{Float64, Nothing}    # Alpha value (if fixed). 'nothing' means tune it
    end

    const CONFIG = Config(
        "Mice_BodyLength.csv",  
        :Y,                     
        15,                     # n_folds
        2024,                   # seed
        500,                    # tpe_rounds
        2000,                   # cd_max_iter (Increase for dense models)
        1e-7,                   # cd_tol
        nothing                 # Alpha Setting: Set to 'nothing' to tune dynamically. Set to 1.0 for Lasso, 0.0 for Ridge, or 0.5 for fixed mix.
    )
    
    """
    Stores the testing results and tuned parameters of a single outer CV fold.
    """
    struct FoldResult
        fold::Int
        mse::Float64
        dcor::Float64
        best_p2::Float64
        best_p3::Float64        # Gamma for CAPPEDBRIDGE
        best_lam::Float64
        beta::Vector{Float64}
    end
end

@everywhere begin
    # ==========================================================================
    # 2. PROXIMAL OPERATORS & METRICS
    # ==========================================================================
    
    """
    Computes the distance correlation between two vectors. 
    Uses biased double centering, matching the R 'energy' package implementation.
    """
    function distance_correlation(x::AbstractVector{Float64}, y::AbstractVector{Float64})
        n = length(x)
        if n < 2 return 0.0 end

        A = abs.(x .- x')
        B = abs.(y .- y')

        row_mean_A = mean(A, dims=2)
        col_mean_A = mean(A, dims=1)
        grand_mean_A = mean(A)
        A_cent = A .- row_mean_A .- col_mean_A .+ grand_mean_A

        row_mean_B = mean(B, dims=2)
        col_mean_B = mean(B, dims=1)
        grand_mean_B = mean(B)
        B_cent = B .- row_mean_B .- col_mean_B .+ grand_mean_B

        dcov2_xy = sum(A_cent .* B_cent) / n^2
        dcov2_xx = sum(A_cent .* A_cent) / n^2
        dcov2_yy = sum(B_cent .* B_cent) / n^2

        if dcov2_xx > 1e-15 && dcov2_yy > 1e-15
            return sqrt(max(0.0, dcov2_xy / sqrt(dcov2_xx * dcov2_yy)))
        else
            return 0.0
        end
    end

    # --- Elastic Net ---
    @inline function prox_elasticnet(z::Float64, lam_scaled::Float64, alpha::Float64)
        if alpha >= 1.0 - 1e-9 return sign(z) * max(0.0, abs(z) - lam_scaled)
        elseif alpha <= 1e-9 return z / (1.0 + lam_scaled)
        else
            val = sign(z) * max(0.0, abs(z) - lam_scaled * alpha)
            return val / (1.0 + lam_scaled * (1.0 - alpha))
        end
    end

    # --- Generalized Bridge (L_q) ---
    @inline function prox_bridge(v::Float64, lambda::Float64, q::Float64)
        v_abs = abs(v)
        if v_abs < 1e-15 return 0.0 end
        
        # Exact solutions for standard geometries
        if abs(q - 1.0) < 1e-9 return sign(v) * max(0.0, v_abs - lambda) end
        if abs(q - 2.0) < 1e-9 return v / (1.0 + 2.0 * lambda) end
        
        # Newton-Raphson solver for the non-linear gradient root
        x_curr = v_abs
        for k in 1:15
            if x_curr <= 0.0
                x_curr = 0.0
                break
            end
            term    = lambda * q * (x_curr^(q - 1.0))
            f_val   = x_curr - v_abs + term
            f_prime = 1.0 + lambda * q * (q - 1.0) * (x_curr^(q - 2.0))
            x_next = max(x_curr - f_val / f_prime, 1e-10)
            
            if abs(x_next - x_curr) < 1e-9
                x_curr = x_next
                break
            end
            x_curr = x_next
        end
        
        if x_curr < 0.0 x_curr = 0.0 end
        
        # Explicit evaluation against the origin for the non-convex domain (q < 1)
        if q < 1.0
            obj_nz = 0.5 * (x_curr - v_abs)^2 + lambda * (x_curr^q)
            obj_z  = 0.5 * v_abs^2
            if obj_z <= obj_nz return 0.0 end
        end
        return sign(v) * x_curr
    end

    # --- Capped Bridge (L0 + Lq) ---
    @inline function prox_capped_bridge(v::Float64, lambda::Float64, q::Float64, gamma::Float64)
        v_abs = abs(v)
        if v_abs < 1e-15 return 0.0 end
        
        x_nz = 0.0
        if abs(q - 1.0) < 1e-9 
            x_nz = max(0.0, v_abs - lambda) 
        elseif abs(q - 2.0) < 1e-9 
            x_nz = v_abs / (1.0 + 2.0 * lambda) 
        else
            x_curr = v_abs / (1.0 + lambda * q)
            for k in 1:50
                if x_curr <= 0.0
                    x_curr = 0.0
                    break
                end
                f_val   = x_curr - v_abs + lambda * q * (x_curr^(q - 1.0))
                f_prime = 1.0 + lambda * q * (q - 1.0) * (x_curr^(q - 2.0))
                x_next  = max(x_curr - f_val / f_prime, 1e-12)
                if abs(x_next - x_curr) < 1e-12
                    x_curr = x_next
                    break
                end
                x_curr = x_next
            end
            if x_curr < 0.0 x_curr = 0.0 end
            x_nz = x_curr
        end
        
        cost_zero = 0.5 * v_abs^2
        cost_nz   = 0.5 * (x_nz - v_abs)^2 + lambda * (x_nz^q) + gamma
        
        if cost_zero <= cost_nz
            return 0.0
        else
            return sign(v) * x_nz
        end
    end

    # --- MCP ---
    @inline function prox_mcp(z::Float64, lam::Float64, a::Float64)
        abs_z = abs(z)
        if abs_z > a * lam return z
        elseif abs_z <= lam return 0.0
        else return sign(z) * (abs_z - lam) / (1.0 - 1.0/a)
        end
    end

    # --- SCAD ---
    @inline function prox_scad(z::Float64, lam::Float64, a::Float64)
        abs_z = abs(z)
        if abs_z > a * lam return z
        elseif abs_z <= 2.0 * lam return sign(z) * max(0.0, abs_z - lam)
        else return ( (a - 1.0) * z - sign(z) * a * lam ) / (a - 2.0)
        end
    end

    # --- Capped L1 ---
    @inline function prox_capped_l1(z::Float64, lam::Float64, a::Float64)
        abs_z = abs(z)
        theta = a * lam
        min_cost = 0.5 * abs_z^2
        best_x = 0.0
        
        if abs_z > lam
            val = abs_z - lam
            if val < theta
                cost_soft = lam * abs_z - 0.5 * lam^2
                if cost_soft < min_cost
                    min_cost = cost_soft
                    best_x = sign(z) * val
                end
            end
        end
        
        unconstrained_cost = lam * min(abs_z, theta)
        if unconstrained_cost < min_cost
            best_x = z
        end
        return best_x
    end

    # --- Log-Sum ---
    @inline function prox_logsum(z::Float64, lam::Float64, eps::Float64)
        abs_z = abs(z)
        best_x = 0.0
        min_cost = 0.5 * abs_z^2
        
        b = eps - abs_z
        c = lam - abs_z * eps
        D = b^2 - 4.0 * c
        
        if D >= 0
            sqrt_D = sqrt(D)
            for r in [(-b + sqrt_D) / 2.0, (-b - sqrt_D) / 2.0]
                if r > 0
                    val = 0.5 * (r - abs_z)^2 + lam * log(1.0 + r/eps)
                    if val < min_cost
                        min_cost = val
                        best_x = sign(z) * r
                    end
                end
            end
        end
        return best_x
    end

    # ==========================================================================
    # 3. UNIVERSAL SOLVER (Coordinate Descent)
    # ==========================================================================
    """
    Solves penalized linear regression using Coordinate Descent with a dynamic 
    active-set strategy and empirical KKT checks.
    """
    function solve_universal_cd(y, X, lambda_val, param2_val, param3_val, beta_init)
        n, p = size(X)
        x_sq_norms = vec(sum(X.^2, dims=1))
        null_deviance = sum(y.^2) + 1e-10
        beta = copy(beta_init)
        resid = y - X * beta

        # Initialize the active set. Features below threshold are empirically inactive.
        active_set = BitSet(findall(x -> abs(x) > SPARSITY_THRESHOLD, beta))
        if isempty(active_set) push!(active_set, 1) end

        for outer_iter in 1:100
            # --- INNER LOOP: Iterate only over currently active features ---
            for iter in 1:CONFIG.cd_max_iter
                max_weighted_change = 0.0
                for j in collect(active_set)
                    old_bj = beta[j]
                    grad_term = dot(view(X, :, j), resid)
                    rho_j = grad_term + x_sq_norms[j] * old_bj
                    z_j   = rho_j / x_sq_norms[j]
                    lam_scaled = lambda_val / x_sq_norms[j]
                    gamma_scaled = param3_val / x_sq_norms[j]

                    new_bj = 0.0
                    if PENALTY_SELECTION == "ELASTICNET" new_bj = prox_elasticnet(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "BRIDGE" new_bj = prox_bridge(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "CAPPEDBRIDGE" new_bj = prox_capped_bridge(z_j, lam_scaled, param2_val, gamma_scaled)
                    elseif PENALTY_SELECTION == "MCP" new_bj = prox_mcp(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "SCAD" new_bj = prox_scad(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "CAPPEDL1" new_bj = prox_capped_l1(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "LOGSUM" new_bj = prox_logsum(z_j, lam_scaled, param2_val)
                    end

                    diff = new_bj - old_bj
                    # Update residuals only if the feature's change is numerically significant
                    if abs(diff) > SPARSITY_THRESHOLD
                        @views resid .-= X[:, j] .* diff
                        beta[j] = new_bj
                        max_weighted_change = max(max_weighted_change, (diff^2) * x_sq_norms[j])
                    end
                end

                if max_weighted_change < (CONFIG.cd_tol * null_deviance)
                    break
                end
            end

            # --- OUTER LOOP: Empirical KKT Checks ---
            # Check if any inactive features should be added to the active set
            violations = 0
            for j in 1:p
                if !(j in active_set)
                    grad_term = dot(view(X, :, j), resid)
                    z_j = grad_term / x_sq_norms[j]
                    lam_scaled = lambda_val / x_sq_norms[j]
                    gamma_scaled = param3_val / x_sq_norms[j]

                    # Proximal Trial Step
                    trial_val = 0.0
                    if PENALTY_SELECTION == "ELASTICNET" trial_val = prox_elasticnet(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "BRIDGE" trial_val = prox_bridge(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "CAPPEDBRIDGE" trial_val = prox_capped_bridge(z_j, lam_scaled, param2_val, gamma_scaled)
                    elseif PENALTY_SELECTION == "MCP" trial_val = prox_mcp(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "SCAD" trial_val = prox_scad(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "CAPPEDL1" trial_val = prox_capped_l1(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "LOGSUM" trial_val = prox_logsum(z_j, lam_scaled, param2_val)
                    end

                    # If the trial step exceeds empirical sparsity threshold, KKT is violated
                    if abs(trial_val) > SPARSITY_THRESHOLD
                        push!(active_set, j)
                        violations += 1
                    end
                end
            end

            # Convergence guaranteed when no KKT violations exist
            if violations == 0 break end
        end
        return beta
    end

    # ==========================================================================
    # 4. WORKER FUNCTION (STOCHASTIC TPE)
    # ==========================================================================
    function run_tpe_fold(fold_id, train_idx, test_idx, X, y, conf::Config)
        # Unique local seed based on fold_id to guarantee reproducible asynchronous TPE iterations
        Random.seed!(conf.seed + fold_id)

        p = size(X, 2)
        y_tr_raw = y[train_idx]
        y_te_raw = y[test_idx]
        X_tr_raw = X[train_idx, :]
        X_te_raw = X[test_idx, :]
        
        # Standardize features based strictly on the training fold (prevent data leakage)
        my = mean(y_tr_raw)
        mx = mean(X_tr_raw, dims=1)
        sx = std(X_tr_raw, dims=1) .+ 1e-6
        y_tr = y_tr_raw .- my
        X_tr = (X_tr_raw .- mx) ./ sx
        X_te = (X_te_raw .- mx) ./ sx

        # Stochastic Splits Setup: Pre-generate 5 internal folds.
        n_tr = length(y_tr)
        stochastic_folds = []
        perm = shuffle(1:n_tr)
        k_inner = 5
        fold_sz = floor(Int, n_tr / k_inner)
        
        for k in 1:k_inner
            s = (k-1)*fold_sz + 1
            e = (k == k_inner) ? n_tr : k*fold_sz
            val_idx = perm[s:e]
            tr_idx  = setdiff(perm, val_idx)
            push!(stochastic_folds, (tr_idx, val_idx))
        end

        # Dynamic Bounds: Define lambda max analytically based on data scale
        lam_max = maximum(abs.(X_tr' * y_tr))
        
        upper_lam_mult = 1.05 
        if PENALTY_SELECTION == "ELASTICNET"
            if conf.fixed_alpha !== nothing && conf.fixed_alpha >= 0.99
                upper_lam_mult = 1.05   
            else
                upper_lam_mult = 100.0  
            end
        elseif PENALTY_SELECTION in ["BRIDGE", "CAPPEDBRIDGE"]
            upper_lam_mult = 100.0      
        elseif PENALTY_SELECTION in ["MCP", "SCAD", "CAPPEDL1", "LOGSUM"]
            upper_lam_mult = 1.05       
        end
        
        bound_log_lam = (log(lam_max * 1e-4), log(lam_max * upper_lam_mult))
        
        # Set bounds for the secondary shape parameter (q, alpha, a, etc.)
        bounds_p2 = (0.0, 0.0)
        if PENALTY_SELECTION == "ELASTICNET" bounds_p2 = (0.0, 1.0)
        elseif PENALTY_SELECTION == "BRIDGE" bounds_p2 = (0.1, 2.0)
        elseif PENALTY_SELECTION == "CAPPEDBRIDGE" bounds_p2 = (0.1, 2.0) 
        elseif PENALTY_SELECTION == "MCP" bounds_p2 = (1.5, 10.0)
        elseif PENALTY_SELECTION == "SCAD" bounds_p2 = (2.5, 10.0)
        elseif PENALTY_SELECTION == "CAPPEDL1" bounds_p2 = (0.5, 10.0)
        elseif PENALTY_SELECTION == "LOGSUM" bounds_p2 = (0.01, 1.0)
        end

        bound_log_gamma = (log(1e-6), log(max(lam_max^2, 10.0)))

        best_stochastic_mse = Inf
        best_params = (p2=0.0, p3=0.0, lam=1.0) 

        # Sub-function to evaluate a candidate on one inner split
        function eval_stochastic(p2_val, p3_val, log_lam, fold_k_idx)
            (tr_i, val_i) = stochastic_folds[fold_k_idx]
            p2_safe = clamp(p2_val, bounds_p2[1], bounds_p2[2])
            lam_val = exp(log_lam)
            
            beta_local = solve_universal_cd(y_tr[tr_i], X_tr[tr_i, :], lam_val, p2_safe, p3_val, zeros(p))
            preds = X_tr[val_i, :] * beta_local
            mse = mean((y_tr[val_i] .- preds).^2)
            return mse
        end

        if fold_id == 1
            println("Fold 1: [$PENALTY_SELECTION] Running TPE ($(conf.tpe_rounds) rounds)...")
        end
        
        space = Dict(:log_lam => HP.Uniform(:log_lam, bound_log_lam[1], bound_log_lam[2]))
        
        should_tune_p2 = true
        if PENALTY_SELECTION == "ELASTICNET" && conf.fixed_alpha !== nothing
            should_tune_p2 = false
        end
        
        if should_tune_p2
            space[:p2] = HP.Uniform(:p2, bounds_p2[1], bounds_p2[2])
        end

        if PENALTY_SELECTION == "CAPPEDBRIDGE" 
            space[:log_gamma] = HP.Uniform(:log_gamma, bound_log_gamma[1], bound_log_gamma[2]) 
        end
        
        # The objective function sampled by TreeParzen
        function objective_tpe(params)
            p2_val = should_tune_p2 ? params[:p2] : conf.fixed_alpha
            p3_val = (PENALTY_SELECTION == "CAPPEDBRIDGE") ? exp(params[:log_gamma]) : 0.0
            
            # Stochasticity: Evaluate candidate on just 1 of the 5 inner folds
            k = rand(1:k_inner) 
            mse = eval_stochastic(p2_val, p3_val, params[:log_lam], k)
            
            if mse < best_stochastic_mse
                best_stochastic_mse = mse
                best_params = (p2=p2_val, p3=p3_val, lam=exp(params[:log_lam]))
            end
            
            return mse
        end
        
        # Suppress TreeParzen standard text outputs
        with_logger(NullLogger()) do
            TreeParzen.fmin(objective_tpe, space, conf.tpe_rounds)
        end
        
        # FINAL REFIT: Train on the FULL outer training set using the best discovered parameters
        final_beta = solve_universal_cd(y_tr, X_tr, best_params.lam, best_params.p2, best_params.p3, zeros(p))
        
        # Test on the strictly held-out Outer Test set
        final_preds = (X_te * final_beta) .+ my
        
        test_mse = mean((y_te_raw .- final_preds).^2)
        test_dcor = distance_correlation(y_te_raw, final_preds)
        
        println("Fold $fold_id Finished. Test MSE=$(round(test_mse,digits=4)) | dCor=$(round(test_dcor,digits=4))")
        
        return FoldResult(fold_id, test_mse, test_dcor, best_params.p2, best_params.p3, best_params.lam, final_beta)
    end
end

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

function load_data(file, target)
    if !isfile(file)
        println("File $file not found. Generating dummy fallback data...")
        n, p = 200, 500; X = randn(n, p); y = X * [2.0; zeros(p-1)] + randn(n)
        return X, y
    end
    println("Loading data from $file...")
    df = CSV.read(file, DataFrame)
    tgt = string(target) in names(df) ? target : ("y" in names(df) ? "y" : "Y")
    y = Vector{Float64}(df[!, tgt])
    X = Matrix{Float64}(select(df, Not(tgt)))
    return X, y
end

println("\n=== STOCHASTIC TPE OPTIMIZATION: $PENALTY_SELECTION ===")
if CONFIG.fixed_alpha !== nothing
    println("Status: Alpha is FIXED to $(CONFIG.fixed_alpha)")
else
    println("Status: Alpha is TUNED (0.0 - 1.0)")
end
println(CONFIG)

X_raw, y_raw = load_data(CONFIG.data_file, CONFIG.target_col)
n_total, p_features = size(X_raw)
println("Data Loaded: $n_total samples, $p_features features.") 

Random.seed!(CONFIG.seed)
indices = shuffle(1:n_total)
fold_size = floor(Int, n_total / CONFIG.n_folds)
folds = []
for k in 1:CONFIG.n_folds
    s = (k-1)*fold_size + 1
    e = (k == CONFIG.n_folds) ? n_total : k*fold_size
    tst = indices[s:e]
    trn = setdiff(indices, tst)
    push!(folds, (k, trn, tst))
end

println("Starting $(CONFIG.n_folds)-Fold CV on $(nworkers()) workers...")
flush(stdout)
start_time = time()

# Execute outer cross-validation folds in parallel
results = pmap(f -> run_tpe_fold(f[1], f[2], f[3], X_raw, y_raw, CONFIG), folds)

elapsed = time() - start_time
println("\n--- COMPLETED in $(round(elapsed, digits=2)) seconds ---")

# ==============================================================================
# 6. AGGREGATION & SUMMARY
# ==============================================================================
println("\n=== PER-FOLD RESULTS: $PENALTY_SELECTION ===")
if PENALTY_SELECTION == "CAPPEDBRIDGE"
    println(rpad("Fold", 6) * rpad("MSE", 12) * rpad("dCor", 12) * rpad("Best Param2", 15) * rpad("Best Gamma", 15) * rpad("Best Lambda", 15))
else
    println(rpad("Fold", 6) * rpad("MSE", 12) * rpad("dCor", 12) * rpad("Best Param2", 15) * rpad("Best Lambda", 15))
end
println("-"^75)

sort!(results, by = x -> x.fold)
for r in results
    if PENALTY_SELECTION == "CAPPEDBRIDGE"
        @printf("%-6d %-12.4f %-12.4f %-15.4f %-15.4f %-15.4f\n", r.fold, r.mse, r.dcor, r.best_p2, r.best_p3, r.best_lam)
    else
        @printf("%-6d %-12.4f %-12.4f %-15.4f %-15.4f\n", r.fold, r.mse, r.dcor, r.best_p2, r.best_lam)
    end
end
println("-"^75)

# Extract and save all individual fold coefficients
beta_matrix = hcat([r.beta for r in results]...) 
df_per_fold = DataFrame(beta_matrix, :auto)
rename!(df_per_fold, [Symbol("Fold_$i") for i in 1:CONFIG.n_folds])
insertcols!(df_per_fold, 1, :Feature_Index => 1:p_features)
CSV.write("$(PENALTY_SELECTION)_Coefficients_PerFold_TPE.csv", df_per_fold)

# Calculate Summary Statistics
mse_scores  = [r.mse for r in results]
dcor_scores = [r.dcor for r in results]
p2_scores   = [r.best_p2 for r in results]
lam_scores  = [r.best_lam for r in results]

# Determine sparsity utilizing the exact same threshold used by the Universal Solver
sparsity_scores = [count(x -> abs(x) > SPARSITY_THRESHOLD, r.beta) for r in results]

# Stability Selection logic: Average the coefficients across outer folds
avg_beta = mean(beta_matrix, dims=2)[:]
stability_score = mean(abs.(beta_matrix) .> SPARSITY_THRESHOLD, dims=2)[:]

println("\n=== SUMMARY STATISTICS ===")
println("Test MSE:             $(round(mean(mse_scores), digits=4)) ± $(round(std(mse_scores), digits=4))")
println("Test dCor:            $(round(mean(dcor_scores), digits=4)) ± $(round(std(dcor_scores), digits=4))")
println("Best Param2 (Shape):  $(round(mean(p2_scores), digits=4)) ± $(round(std(p2_scores), digits=4))")
if PENALTY_SELECTION == "CAPPEDBRIDGE"
    p3_scores = [r.best_p3 for r in results]
    println("Best Param3 (Gamma):  $(round(mean(p3_scores), digits=4)) ± $(round(std(p3_scores), digits=4))")
end
println("Best Lambda:          $(round(mean(lam_scores), digits=4)) ± $(round(std(lam_scores), digits=4))")
println("Selected Features:    $(round(mean(sparsity_scores), digits=2)) ± $(round(std(sparsity_scores), digits=2))")

# Save Averaged/Bagged Coefficients
df_agg = DataFrame(
    Feature_Index = 1:p_features,
    Bagged_Beta = avg_beta,
    Stability_Score = stability_score
)
sort!(df_agg, [:Stability_Score, :Bagged_Beta], rev=true)
CSV.write("$(PENALTY_SELECTION)_Coefficients_Average_TPE.csv", df_agg)
println("Saved aggregated coefficients.")