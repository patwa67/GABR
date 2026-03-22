# ==============================================================================
# ROBUST STOCHASTIC TPE OPTIMIZATION: MONTE CARLO SIMULATION
# Penalties: ELASTICNET (incl. Lasso and Ridge), BRIDGE, MCP, SCAD, CAPPEDL1, LOGSUM, (CAPPEDBRIDGE)
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

# Initialize multiprocessing workers to parallelize the Cross-Validation folds.
const DESIRED_WORKERS = 15
if nprocs() < (DESIRED_WORKERS + 1)
    addprocs((DESIRED_WORKERS + 1) - nprocs())
end
println("Active Workers: $(nworkers())")
flush(stdout)


@everywhere begin
    using LinearAlgebra, Statistics, Random, DataFrames, CSV, SpecialFunctions, Printf, Distributions
    using Logging
    using TreeParzen
    const HP = TreeParzen.HP

    # --- 1. GLOBAL CONFIGURATION ---
    # SELECTION: Choose one of ["ELASTICNET", "BRIDGE", "CAPPEDBRIDGE", "MCP", "SCAD", "CAPPEDL1", "LOGSUM"]
    const PENALTY_SELECTION = "BRIDGE"
    
    # Coefficients smaller than this threshold are empirically truncated to zero.
    const SPARSITY_THRESHOLD = 1e-8

    """
    Configuration for the solver and TPE hyperparameter optimization.
    """
    struct Config
        n_folds::Int
        tpe_rounds::Int
        cd_max_iter::Int
        cd_tol::Float64
        fixed_alpha::Union{Float64, Nothing}
    end

    const CONFIG = Config(
        15,                      # n_folds
        500,                     # tpe_rounds
        2000,                    # cd_max_iter
        1e-7,                    # cd_tol
        nothing                  # fixed_alpha (tuned dynamically)
    )

    """
    Stores the results of a single outer cross-validation fold.
    """
    struct FoldResult
        fold::Int
        mse::Float64
        dcor::Float64
        best_p2::Float64         # Shape parameter (q, alpha, etc.)
        best_p3::Float64         # Gamma (for Capped Bridge only)
        best_lam::Float64
        beta::Vector{Float64}
    end
    
    """
    Stores aggregated evaluation metrics for a single Monte Carlo simulation run.
    """
    struct SimMetrics
        mse::Float64
        dcor::Float64
        p2::Float64              
        p3::Float64              # Gamma parameter
        lam::Float64             
        tp::Int                  
        fp::Int                  
        tn::Int                  
        fn::Int                  
        tpr::Float64             
        fpr::Float64             
        fdr::Float64             
    end

    # ==========================================================================
    # --- PROXIMAL OPERATORS ---
    # ==========================================================================

    function distance_correlation(x::AbstractVector{Float64}, y::AbstractVector{Float64})
        n = length(x)
        if n < 2 return 0.0 end
        A = abs.(x .- x')
        B = abs.(y .- y')
        A_cent = A .- mean(A, dims=2) .- mean(A, dims=1) .+ mean(A)
        B_cent = B .- mean(B, dims=2) .- mean(B, dims=1) .+ mean(B)
        dcov2_xy = sum(A_cent .* B_cent) / n^2
        dcov2_xx = sum(A_cent .* A_cent) / n^2
        dcov2_yy = sum(B_cent .* B_cent) / n^2
        if dcov2_xx > 1e-15 && dcov2_yy > 1e-15
            return sqrt(max(0.0, dcov2_xy / sqrt(dcov2_xx * dcov2_yy)))
        else
            return 0.0
        end
    end

    @inline function prox_elasticnet(z::Float64, lam_scaled::Float64, alpha::Float64)
        if alpha >= 1.0 - 1e-9 return sign(z) * max(0.0, abs(z) - lam_scaled)
        elseif alpha <= 1e-9 return z / (1.0 + lam_scaled)
        else return (sign(z) * max(0.0, abs(z) - lam_scaled * alpha)) / (1.0 + lam_scaled * (1.0 - alpha))
        end
    end

    @inline function prox_bridge(v::Float64, lambda::Float64, q::Float64)
        v_abs = abs(v)
        if v_abs < 1e-15 return 0.0 end
        
        if abs(q - 1.0) < 1e-9 return sign(v) * max(0.0, v_abs - lambda) end
        if abs(q - 2.0) < 1e-9 return v / (1.0 + 2.0 * lambda) end

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
        
        # Explicit zero-check
        obj_nz = 0.5 * (x_curr - v_abs)^2 + lambda * (x_curr^q)
        obj_z  = 0.5 * v_abs^2
        if obj_z <= obj_nz return 0.0 end
        return sign(v) * x_curr
    end

    # NEW: Capped Bridge (L0-Penalized Convex Bridge)
    @inline function prox_capped_bridge(v::Float64, lambda::Float64, q::Float64, gamma::Float64)
        v_abs = abs(v)
        if v_abs < 1e-15 return 0.0 end
        
        # 1. Compute standard bridge update (Active State)
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
        
        # 2. Evaluate competing global costs (Inactive vs Active)
        cost_zero = 0.5 * v_abs^2
        cost_nz   = 0.5 * (x_nz - v_abs)^2 + lambda * (x_nz^q) + gamma
        
        # 3. L0 Hard Thresholding Jump
        if cost_zero <= cost_nz
            return 0.0
        else
            return sign(v) * x_nz
        end
    end

    @inline function prox_mcp(z::Float64, lam::Float64, a::Float64)
        abs_z = abs(z)
        if abs_z > a * lam return z
        elseif abs_z <= lam return 0.0
        else return sign(z) * (abs_z - lam) / (1.0 - 1.0/a)
        end
    end

    @inline function prox_scad(z::Float64, lam::Float64, a::Float64)
        abs_z = abs(z)
        if abs_z > a * lam return z
        elseif abs_z <= 2.0 * lam return sign(z) * max(0.0, abs_z - lam)
        else return ( (a - 1.0) * z - sign(z) * a * lam ) / (a - 2.0)
        end
    end

    @inline function prox_capped_l1(z::Float64, lam::Float64, a::Float64)
        abs_z = abs(z)
        theta = a * lam
        cost_zero = 0.5 * abs_z^2
        best_x, min_cost = 0.0, cost_zero

        if abs_z > lam
            x_soft = abs_z - lam
            if x_soft < theta
                cost_soft = 0.5 * (x_soft - abs_z)^2 + lam * x_soft
                if cost_soft < min_cost
                    min_cost, best_x = cost_soft, sign(z) * x_soft
                end
            end
        end

        if abs_z >= theta
            cost_unpen = lam * theta
            if cost_unpen < min_cost
                best_x = z
            end
        end
        return best_x
    end

    @inline function prox_logsum(z::Float64, lam::Float64, eps::Float64)
        abs_z = abs(z)
        best_x, min_cost = 0.0, 0.5 * abs_z^2
        b, c = eps - abs_z, lam - abs_z * eps
        D = b^2 - 4.0 * c

        if D >= 0
            sqrt_D = sqrt(D)
            for r in [(-b + sqrt_D) / 2.0, (-b - sqrt_D) / 2.0]
                if r > 0
                    val = 0.5 * (r - abs_z)^2 + lam * log(1.0 + r/eps)
                    if val < min_cost
                        min_cost, best_x = val, sign(z) * r
                    end
                end
            end
        end
        return best_x
    end

    # ==========================================================================
    # --- UNIVERSAL SOLVER ---
    # ==========================================================================

    function solve_universal_cd(y, X, lambda_val, param2_val, param3_val, beta_init)
        n, p = size(X)
        x_sq_norms = max.(vec(sum(X.^2, dims=1)), 1e-10)
        null_deviance = sum(y.^2) + 1e-10
        beta  = copy(beta_init)
        resid = y - X * beta

        active_set = BitSet(findall(x -> abs(x) > SPARSITY_THRESHOLD, beta))
        if isempty(active_set) push!(active_set, 1) end 

        for outer_iter in 1:100
            for iter in 1:CONFIG.cd_max_iter
                max_weighted_change = 0.0
                for j in collect(active_set)
                    old_bj = beta[j]
                    grad_term = dot(view(X, :, j), resid)
                    rho_j = grad_term + x_sq_norms[j] * old_bj
                    z_j = rho_j / x_sq_norms[j]
                    lam_scaled = lambda_val / x_sq_norms[j]
                    
                    # Scale gamma by z_j exactly like lambda
                    gamma_scaled = param3_val / x_sq_norms[j]

                    new_bj = if PENALTY_SELECTION == "ELASTICNET" prox_elasticnet(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "BRIDGE" prox_bridge(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "CAPPEDBRIDGE" prox_capped_bridge(z_j, lam_scaled, param2_val, gamma_scaled)
                        elseif PENALTY_SELECTION == "MCP" prox_mcp(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "SCAD" prox_scad(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "CAPPEDL1" prox_capped_l1(z_j, lam_scaled, param2_val)
                        else prox_logsum(z_j, lam_scaled, param2_val)
                    end

                    diff = new_bj - old_bj
                    
                    if abs(diff) > SPARSITY_THRESHOLD
                        @views resid .-= X[:, j] .* diff
                        beta[j] = new_bj
                        max_weighted_change = max(max_weighted_change, (diff^2) * x_sq_norms[j])
                    end
                end
                
                if max_weighted_change < (CONFIG.cd_tol * null_deviance) break end
            end

            violations = 0
            for j in 1:p
                if !(j in active_set)
                    grad_term  = dot(view(X, :, j), resid)
                    z_j        = grad_term / x_sq_norms[j]
                    lam_scaled = lambda_val / x_sq_norms[j]
                    gamma_scaled = param3_val / x_sq_norms[j]

                    trial_val = if PENALTY_SELECTION == "ELASTICNET" prox_elasticnet(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "BRIDGE" prox_bridge(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "CAPPEDBRIDGE" prox_capped_bridge(z_j, lam_scaled, param2_val, gamma_scaled)
                        elseif PENALTY_SELECTION == "MCP" prox_mcp(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "SCAD" prox_scad(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "CAPPEDL1" prox_capped_l1(z_j, lam_scaled, param2_val)
                        else prox_logsum(z_j, lam_scaled, param2_val)
                    end

                    if abs(trial_val) > SPARSITY_THRESHOLD
                        push!(active_set, j)
                        violations += 1
                    end
                end
            end
            if violations == 0 break end
        end
        return beta
    end

    # ==========================================================================
    # --- WORKER FUNCTION FOR HYPERPARAMETER TUNING ---
    # ==========================================================================

    function run_tpe_fold(fold_id, train_idx, test_idx, X, y, conf::Config, seed_offset)
        Random.seed!(seed_offset + fold_id)

        p = size(X, 2)
        y_tr_raw, y_te_raw = y[train_idx], y[test_idx]
        X_tr_raw, X_te_raw = X[train_idx, :], X[test_idx, :]

        my = mean(y_tr_raw)
        mx = mean(X_tr_raw, dims=1)
        sx = std(X_tr_raw,  dims=1) .+ 1e-6
        y_tr = y_tr_raw .- my
        X_tr = (X_tr_raw .- mx) ./ sx
        X_te = (X_te_raw .- mx) ./ sx

        n_tr = length(y_tr)
        k_inner = 5
        stochastic_folds = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, k_inner)
        perm = shuffle(1:n_tr)
        fold_sz = floor(Int, n_tr / k_inner)
        
        for k in 1:k_inner
            s = (k-1)*fold_sz + 1
            e = (k == k_inner) ? n_tr : k*fold_sz
            val_idx = perm[s:e]
            tr_idx  = setdiff(perm, val_idx)
            stochastic_folds[k] = (tr_idx, val_idx)
        end

        lam_max = maximum(abs.(X_tr' * y_tr))
        upper_lam_mult = 1.05
        if PENALTY_SELECTION == "ELASTICNET"
            upper_lam_mult = (conf.fixed_alpha !== nothing && conf.fixed_alpha >= 0.99) ? 1.05 : 100.0
        elseif PENALTY_SELECTION in ["BRIDGE", "CAPPEDBRIDGE"]
            upper_lam_mult = 100.0
        end

        bound_log_lam = (log(lam_max * 1e-4), log(lam_max * upper_lam_mult))
        
        bounds_p2 = (0.0, 0.0)
        if PENALTY_SELECTION == "ELASTICNET" bounds_p2 = (0.0, 1.0)
        elseif PENALTY_SELECTION == "BRIDGE" bounds_p2 = (0.1, 2.0)
        elseif PENALTY_SELECTION == "CAPPEDBRIDGE" bounds_p2 = (1.0, 2.0) # Restrict to convex grouping domain
        elseif PENALTY_SELECTION == "MCP" bounds_p2 = (1.5, 10.0)
        elseif PENALTY_SELECTION == "SCAD" bounds_p2 = (2.5, 10.0)
        elseif PENALTY_SELECTION == "CAPPEDL1" bounds_p2 = (0.5, 10.0)
        elseif PENALTY_SELECTION == "LOGSUM" bounds_p2 = (0.01, 1.0)
        end

        # Bounds for Gamma (L0 penalty weight)
        bound_log_gamma = (log(1e-6), log(max(lam_max^2, 10.0)))

        best_stochastic_mse = Ref(Inf)
        best_params = Ref((p2 = bounds_p2[1], p3 = 0.0, lam = exp(bound_log_lam[1])))

        function eval_stochastic(p2_val, p3_val, log_lam, fold_k_idx)
            (tr_i, val_i) = stochastic_folds[fold_k_idx]
            p2_safe  = clamp(p2_val, bounds_p2[1], bounds_p2[2])
            lam_val  = exp(log_lam)
            beta_local = solve_universal_cd(y_tr[tr_i], X_tr[tr_i, :], lam_val, p2_safe, p3_val, zeros(p))
            preds = X_tr[val_i, :] * beta_local
            return mean((y_tr[val_i] .- preds).^2)
        end

        space = Dict(:log_lam => HP.Uniform(:log_lam, bound_log_lam[1], bound_log_lam[2]))
        should_tune_p2 = !(PENALTY_SELECTION == "ELASTICNET" && conf.fixed_alpha !== nothing)
        if should_tune_p2 space[:p2] = HP.Uniform(:p2, bounds_p2[1], bounds_p2[2]) end
        if PENALTY_SELECTION == "CAPPEDBRIDGE" space[:log_gamma] = HP.Uniform(:log_gamma, bound_log_gamma[1], bound_log_gamma[2]) end

        function objective_tpe(params)
            p2_val = should_tune_p2 ? params[:p2] : conf.fixed_alpha
            p3_val = (PENALTY_SELECTION == "CAPPEDBRIDGE") ? exp(params[:log_gamma]) : 0.0
            k = rand(1:k_inner) 
            
            mse = eval_stochastic(p2_val, p3_val, params[:log_lam], k)
            
            if mse < best_stochastic_mse[]
                best_stochastic_mse[] = mse
                best_params[] = (p2 = p2_val, p3 = p3_val, lam = exp(params[:log_lam]))
            end
            return mse
        end

        # Execute Bayesian Optimization (TPE) silently
        with_logger(NullLogger()) do
            TreeParzen.fmin(objective_tpe, space, conf.tpe_rounds)
        end

        final_beta = solve_universal_cd(y_tr, X_tr, best_params[].lam, best_params[].p2, best_params[].p3, zeros(p))
        final_preds = (X_te * final_beta) .+ my
        
        test_mse = mean((y_te_raw .- final_preds).^2)
        test_dcor = distance_correlation(y_te_raw, final_preds)

        return FoldResult(fold_id, test_mse, test_dcor, best_params[].p2, best_params[].p3, best_params[].lam, final_beta)
    end
end

# ==============================================================================
# DATA SIMULATION & EVALUATION FUNCTIONS
# ==============================================================================

function simulate_regression(n::Int, p::Int, n_active::Int, snr::Float64, rho::Float64; seed::Int=123)
    Random.seed!(seed)
    
    Sigma = zeros(Float64, p, p)
    for i in 1:p, j in 1:p
        Sigma[i, j] = rho^(abs(i - j))
    end
    
    Z = randn(n, p)
    C = cholesky(Sigma).L
    X = Z * C'
    
    beta_true = zeros(Float64, p)
    if n_active > 0
        active_indices = shuffle(1:p)[1:n_active]
        for idx in active_indices
            sign = rand([-1.0, 1.0])
            magnitude = rand(Uniform(0.5, 2.0)) 
            beta_true[idx] = sign * magnitude
        end
    end
    
    y_true = X * beta_true
    var_signal = var(y_true)
    var_noise = var_signal > 0 ? var_signal / snr : 1.0
    
    epsilon = rand(Normal(0.0, sqrt(var_noise)), n)
    y = y_true .+ epsilon
    
    return X, y, beta_true
end

function evaluate_metrics(beta_est::Vector{Float64}, beta_true::Vector{Float64}, 
                          mse::Float64, dcor::Float64, p2::Float64, p3::Float64, lam::Float64)
    active_est  = abs.(beta_est) .> SPARSITY_THRESHOLD
    active_true = abs.(beta_true) .> SPARSITY_THRESHOLD
    
    tp = sum(active_est .& active_true)
    fp = sum(active_est .& .!active_true)
    tn = sum(.!active_est .& .!active_true)
    fn = sum(.!active_est .& active_true)
    
    tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
    fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0.0
    fdr = (tp + fp) > 0 ? fp / (tp + fp) : 0.0
    
    return SimMetrics(mse, dcor, p2, p3, lam, tp, fp, tn, fn, tpr, fpr, fdr)
end

# ==============================================================================
# MAIN SIMULATION LOOP
# ==============================================================================

# Simulation configuration
const N_SIMULATIONS = 50
const SIM_N_SAMPLES = 500
const SIM_P_FEATURES = 1000
const SIM_N_ACTIVE = 50     # Mid scenario
const SIM_SNR = 5.0         # High clarity signal
const SIM_RHO = 0.8         # Highly Correlated features

println("\n=== STARTING MONTE CARLO SIMULATION ===")
println("Penalty: $PENALTY_SELECTION")
println("Simulations: $N_SIMULATIONS | Samples: $SIM_N_SAMPLES | Features: $SIM_P_FEATURES | Active: $SIM_N_ACTIVE")

all_metrics = SimMetrics[]
start_total_time = time()

for sim in 1:N_SIMULATIONS
    println("\n--- Running Simulation $sim / $N_SIMULATIONS ---")
    
    sim_seed = 2024 + sim * 100
    X_raw, y_raw, beta_true = simulate_regression(
        SIM_N_SAMPLES, SIM_P_FEATURES, SIM_N_ACTIVE, SIM_SNR, SIM_RHO, seed=sim_seed
    )
    
    indices = shuffle(MersenneTwister(sim_seed), 1:SIM_N_SAMPLES)
    fold_size = floor(Int, SIM_N_SAMPLES / CONFIG.n_folds)
    folds = Vector{Tuple{Int, Vector{Int}, Vector{Int}}}(undef, CONFIG.n_folds)
    
    for k in 1:CONFIG.n_folds
        s = (k-1)*fold_size + 1
        e = (k == CONFIG.n_folds) ? SIM_N_SAMPLES : k*fold_size
        tst = indices[s:e]
        trn = setdiff(indices, tst)
        folds[k] = (k, trn, tst)
    end
    
    results = pmap(f -> run_tpe_fold(f[1], f[2], f[3], X_raw, y_raw, CONFIG, sim_seed), folds)
    
    avg_beta = mean(hcat([r.beta for r in results]...), dims=2)[:]
    avg_mse  = mean([r.mse for r in results])
    avg_dcor = mean([r.dcor for r in results])
    avg_p2   = mean([r.best_p2 for r in results])
    avg_p3   = mean([r.best_p3 for r in results])
    avg_lam  = mean([r.best_lam for r in results])
    
    metrics = evaluate_metrics(avg_beta, beta_true, avg_mse, avg_dcor, avg_p2, avg_p3, avg_lam)
    push!(all_metrics, metrics)
    
    @printf("Sim %d Complete | MSE: %.4f | TP: %d/%d | FP: %d | FDR: %.2f%%\n", 
            sim, metrics.mse, metrics.tp, SIM_N_ACTIVE, metrics.fp, metrics.fdr * 100)
end

elapsed = time() - start_total_time

# ==============================================================================
# SUMMARY STATISTICS AGGREGATION
# ==============================================================================

println("\n" * "="^50)
println("FINAL AGGREGATED RESULTS ($N_SIMULATIONS Runs)")
println("="^50)
@printf("Time Elapsed:       %.2f seconds\n\n", elapsed)

mean_p2  = mean([m.p2 for m in all_metrics])
std_p2   = std([m.p2 for m in all_metrics])

mean_p3  = mean([m.p3 for m in all_metrics])
std_p3   = std([m.p3 for m in all_metrics])

mean_lam = mean([m.lam for m in all_metrics])
std_lam  = std([m.lam for m in all_metrics])

mean_mse = mean([m.mse for m in all_metrics])
std_mse  = std([m.mse for m in all_metrics])
mean_dcor = mean([m.dcor for m in all_metrics])
std_dcor  = std([m.dcor for m in all_metrics])

mean_tpr = mean([m.tpr for m in all_metrics])
std_tpr  = std([m.tpr for m in all_metrics])
mean_fpr = mean([m.fpr for m in all_metrics])
std_fpr  = std([m.fpr for m in all_metrics])
mean_fdr = mean([m.fdr for m in all_metrics])
std_fdr  = std([m.fdr for m in all_metrics])

mean_tp = mean([m.tp for m in all_metrics])
std_tp  = std([m.tp for m in all_metrics])
mean_fp = mean([m.fp for m in all_metrics])
std_fp  = std([m.fp for m in all_metrics])

println("Predictive Performance:")
@printf("  Test MSE:         %.4f ± %.4f\n", mean_mse, std_mse)
@printf("  Test dCor:        %.4f ± %.4f\n\n", mean_dcor, std_dcor)

println("Tuned Hyperparameters:")
@printf("  Param2 (Shape q): %.4f ± %.4f\n", mean_p2, std_p2)
if PENALTY_SELECTION == "CAPPEDBRIDGE"
    @printf("  Param3 (Gamma):   %.4f ± %.4f\n", mean_p3, std_p3)
end
@printf("  Lambda (Penalty): %.4f ± %.4f\n\n", mean_lam, std_lam)

println("Feature Selection Performance:")
@printf("  True Positives:   %.1f ± %.1f (out of %d)\n", mean_tp, std_tp, SIM_N_ACTIVE)
@printf("  False Positives:  %.1f ± %.1f\n", mean_fp, std_fp)
@printf("  Sensitivity/TPR:  %.2f%% ± %.2f%%\n", mean_tpr * 100, std_tpr * 100)
@printf("  FPR:              %.2f%% ± %.2f%%\n", mean_fpr * 100, std_fpr * 100)
@printf("  False Discovery:  %.2f%% ± %.2f%%\n", mean_fdr * 100, std_fdr * 100)
println("="^50)