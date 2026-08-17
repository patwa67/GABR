# ==============================================================================
# GENERALIZED ADAPTIVE BRIDGE REGRESSION (GABR) & PENALIZED REGRESSION SOLVER
# Penalties: ELASTICNET, GABR, GABR-L0, MCP, SCAD, CAPPEDL1, LOGSUM
# Families: "gaussian" (Continuous Y), "binomial" (Binary Y = 0, 1)
# Features: Distributed CV, Deterministic Inner CV, Dual-TPE, Orthogonal L0 Truncation
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

const DESIRED_WORKERS = 16
if nprocs() < (DESIRED_WORKERS + 1)
    addprocs((DESIRED_WORKERS + 1) - nprocs())
end
println("Active Workers: $(nworkers())")
flush(stdout)

@everywhere begin
    using LinearAlgebra, Statistics, Random, DataFrames, CSV, SpecialFunctions, Printf, Logging
    using TreeParzen
    const HP = TreeParzen.HP

    # ==========================================================================
    # 1. USER CONFIGURATION (Edit this block for new datasets)
    # ==========================================================================
    
    # SELECTION: Choose one of ["ELASTICNET", "GABR", "GABR-L0", "MCP", "SCAD", "CAPPEDL1", "LOGSUM"]
    const PENALTY_SELECTION = "GABR-L0" 
    
    # Mathematical zero bound (coefficients smaller than this are set to exactly zero)
    const SPARSITY_THRESHOLD = 1e-8

    """
    Configuration struct for dataset I/O and TPE hyperparameter optimization.
    """
    struct Config
        data_file::String                       # Path to the CSV dataset
        target_col::Symbol                      # Target variable column name
        family::String                          # "gaussian" (regression) or "binomial" (classification)
        n_folds::Int                            # Number of outer CV folds (e.g., 5 or 10)
        seed::Int                               # Global random seed for reproducibility
        tpe_rounds::Int                         # Budget for Bayesian optimization evaluations
        cd_max_iter::Int                        # Max iterations for inner coordinate descent loop
        cd_tol::Float64                         # Convergence tolerance for coordinate descent
        kkt_max_iter::Int                       # Max iterations for outer empirical KKT active-set updates
        newt_max_iter::Int                      # Max iterations for Newton-Raphson root solver (GABR)
        fixed_alpha::Union{Float64, Nothing}    # Alpha value for Elastic Net (1.0 = Lasso, 0.0 = Ridge, nothing = Tune)
    end

    # Define your specific run parameters here:
    const CONFIG = Config(
        "Mice_BodyLength.csv",  # <-- CHANGE THIS to your actual CSV file path
        :Y,                     # <-- CHANGE THIS to your target column name
        "gaussian",             # <-- "gaussian" or "binomial"
        5,                      # n_folds
        2024,                   # seed
        500,                    # tpe_rounds
        2000,                   # cd_max_iter 
        1e-7,                   # cd_tol
        100,                    # kkt_max_iter
        15,                     # newt_max_iter
        nothing                 # Alpha Setting 
    )
    
    struct FoldResult
        fold::Int
        loss::Float64                      # Test MSE (Gaussian) or Log-Loss (Binomial)
        metric::Float64                    # Test dCor (Gaussian) or AUC (Binomial)
        best_p2::Float64                   # Shape parameter
        best_lam::Float64                  # Regularization strength
        best_tau::Float64                  # Tau Ratio (Scale-Invariant Truncation proportion)
        beta::Vector{Float64}              # Final coefficient vector (Index 1 is Intercept!)
        tpe_time::Float64                  
        convergence_curve::Vector{Float64} 
    end
end

@everywhere begin
    # ==========================================================================
    # 2. PROXIMAL OPERATORS & METRICS
    # ==========================================================================
    
    function distance_correlation(x::AbstractVector{Float64}, y::AbstractVector{Float64})
        n = length(x)
        if n < 2 return 0.0 end
        A, B = abs.(x .- x'), abs.(y .- y')
        A_cent = A .- mean(A, dims=2) .- mean(A, dims=1) .+ mean(A)
        B_cent = B .- mean(B, dims=2) .- mean(B, dims=1) .+ mean(B)
        dcov2_xy, dcov2_xx, dcov2_yy = sum(A_cent .* B_cent)/n^2, sum(A_cent .* A_cent)/n^2, sum(B_cent .* B_cent)/n^2
        return (dcov2_xx > 1e-15 && dcov2_yy > 1e-15) ? sqrt(max(0.0, dcov2_xy / sqrt(dcov2_xx * dcov2_yy))) : 0.0
    end

    function roc_auc(y_true::AbstractVector{Float64}, y_pred::AbstractVector{Float64})
        n = length(y_true)
        ord = sortperm(y_pred)
        y_sorted = y_true[ord]
        
        rank_sum, n_pos = 0.0, 0
        for i in 1:n
            if y_sorted[i] > 0.5
                rank_sum += i
                n_pos += 1
            end
        end
        n_neg = n - n_pos
        if n_pos == 0 || n_neg == 0 return 0.5 end
        return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
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

    # --- Generalized Adaptive Bridge Regression (GABR & GABR-L0) ---
    # ADDED: newt_max argument passed here
    @inline function prox_gabr(v::Float64, lambda::Float64, q::Float64, tau_abs::Float64, newt_max::Int)
        v_abs = abs(v)
        if v_abs < 1e-15 return 0.0 end
        if abs(q - 1.0) < 1e-9 return sign(v) * max(0.0, v_abs - lambda) end
        if abs(q - 2.0) < 1e-9 return v / (1.0 + 2.0 * lambda) end
        
        x_curr = v_abs
        # UPDATED: Loop uses dynamic newt_max
        for k in 1:newt_max
            if x_curr <= 0.0 x_curr = 0.0; break end
            term    = lambda * q * (x_curr^(q - 1.0))
            f_val   = x_curr - v_abs + term
            f_prime = 1.0 + lambda * q * (q - 1.0) * (x_curr^(q - 2.0))
            x_next = max(x_curr - f_val / f_prime, 1e-10) 
            if abs(x_next - x_curr) < 1e-9 x_curr = x_next; break end
            x_curr = x_next
        end
        if x_curr < 0.0 x_curr = 0.0 end
        
        if q < 1.0
            if 0.5 * v_abs^2 <= 0.5 * (x_curr - v_abs)^2 + lambda * (x_curr^q) return 0.0 end
        elseif 1.0 < q < 2.0
            if tau_abs > 0.0 && x_curr < tau_abs return 0.0 end
        end
        return sign(v) * x_curr
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
        best_x, min_cost = 0.0, 0.5 * abs_z^2
        
        if abs_z > lam
            x_soft = abs_z - lam
            if x_soft < theta
                 cost_soft = 0.5 * (x_soft - abs_z)^2 + lam * x_soft
                if cost_soft < min_cost min_cost, best_x = cost_soft, sign(z) * x_soft end
            end
        end
        if abs_z >= theta && lam * theta < min_cost best_x = z end
        return best_x
    end

    # --- Log-Sum ---
    @inline function prox_logsum(z::Float64, lam::Float64, eps::Float64)
        abs_z, best_x, min_cost = abs(z), 0.0, 0.5 * abs(z)^2
        b, c = eps - abs_z, lam - abs_z * eps
        D = b^2 - 4.0 * c
        if D >= 0
            sqrt_D = sqrt(D)
            for r in [(-b + sqrt_D) / 2.0, (-b - sqrt_D) / 2.0]
                if r > 0
                    val = 0.5 * (r - abs_z)^2 + lam * log(1.0 + r/eps)
                    if val < min_cost min_cost, best_x = val, sign(z) * r end
                end
            end
        end
        return best_x
    end

    # ==========================================================================
    # 3. UNIVERSAL SOLVER (Coordinate Descent)
    # ==========================================================================
    """
    Solves penalized regression using Coordinate Descent.
    For Binomial, uses Majorization-Minimization with Bohning's 0.25 Hessian bound.
    """
    # ADDED: newt_max_iter argument passed to the solver
    function solve_universal_cd(y, X, lambda_val, param2_val, tau_ratio, beta_init, family, newt_max_iter)
        n, p = size(X)
        beta = copy(beta_init)
        eta = X * beta
        
        if family == "gaussian"
            denom = max.(vec(sum(abs2, X, dims=1)), 1e-10)
            resid = y .- eta
            null_deviance = sum(y.^2) + 1e-10 
        elseif family == "binomial"
            denom = max.(vec(sum(abs2, X, dims=1)) .* 0.25, 1e-10) # 0.25 is Böhning bound
            p_prob = 1.0 ./ (1.0 .+ exp.(.-eta))
            resid = y .- p_prob
            p_null = clamp(mean(y), 1e-5, 1.0 - 1e-5)
            null_deviance = -2.0 * sum(y .* log.(p_null) .+ (1.0 .- y) .* log.(1.0 .- p_null))
        end

        active_set = BitSet(findall(x -> abs(x) > SPARSITY_THRESHOLD, beta))
        push!(active_set, 1) # Intercept (Column 1) is ALWAYS active

        # Dynamic Outer KKT Loop
        for outer_iter in 1:CONFIG.kkt_max_iter
            for iter in 1:CONFIG.cd_max_iter
                max_weighted_change = 0.0
                
                for j in collect(active_set)
                    old_bj = beta[j]
                    grad_term = dot(view(X, :, j), resid)
                    z_j = old_bj + grad_term / denom[j] 
                    
                    new_bj = 0.0
                    if j == 1 
                        new_bj = z_j # Intercept is strictly unpenalized
                    else
                        lam_scaled = lambda_val / denom[j]
                        local_tau = tau_ratio * lam_scaled # SCALE-INVARIANT TRUNCATION
                        
                        if PENALTY_SELECTION == "ELASTICNET" new_bj = prox_elasticnet(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "GABR" new_bj = prox_gabr(z_j, lam_scaled, param2_val, 0.0, newt_max_iter) # PASS DOWN
                        elseif PENALTY_SELECTION == "GABR-L0" new_bj = prox_gabr(z_j, lam_scaled, param2_val, local_tau, newt_max_iter) # PASS DOWN
                        elseif PENALTY_SELECTION == "MCP" new_bj = prox_mcp(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "SCAD" new_bj = prox_scad(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "CAPPEDL1" new_bj = prox_capped_l1(z_j, lam_scaled, param2_val)
                        elseif PENALTY_SELECTION == "LOGSUM" new_bj = prox_logsum(z_j, lam_scaled, param2_val)
                        end
                    end

                    diff = new_bj - old_bj
                    if abs(diff) > SPARSITY_THRESHOLD
                        beta[j] = new_bj
                        max_weighted_change = max(max_weighted_change, (diff^2) * denom[j])
                        
                        # Dynamically update vectors without full recalculation
                        @views eta .+= X[:, j] .* diff
                        if family == "gaussian"
                            @views resid .-= X[:, j] .* diff
                        elseif family == "binomial"
                            p_prob .= 1.0 ./ (1.0 .+ exp.(.-eta))
                            resid .= y .- p_prob
                        end
                    end
                end

                if max_weighted_change < (CONFIG.cd_tol * null_deviance)
                    break
                end
            end

            violations = 0
            for j in 1:p
                if !(j in active_set)
                    grad_term = dot(view(X, :, j), resid)
                    z_j = grad_term / denom[j]
                    lam_scaled = lambda_val / denom[j]
                    local_tau = tau_ratio * lam_scaled # APPLY ORTHOGONAL TAU TO KKT
                    
                    trial_val = 0.0
                    if PENALTY_SELECTION == "ELASTICNET" trial_val = prox_elasticnet(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "GABR" trial_val = prox_gabr(z_j, lam_scaled, param2_val, 0.0, newt_max_iter) # PASS DOWN
                    elseif PENALTY_SELECTION == "GABR-L0" trial_val = prox_gabr(z_j, lam_scaled, param2_val, local_tau, newt_max_iter) # PASS DOWN
                    elseif PENALTY_SELECTION == "MCP" trial_val = prox_mcp(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "SCAD" trial_val = prox_scad(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "CAPPEDL1" trial_val = prox_capped_l1(z_j, lam_scaled, param2_val)
                    elseif PENALTY_SELECTION == "LOGSUM" trial_val = prox_logsum(z_j, lam_scaled, param2_val)
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
    # 4. WORKER FUNCTION (INNER CV TPE)
    # ==========================================================================
    function run_tpe_fold(fold_id, train_idx, test_idx, X, y, conf::Config)
        Random.seed!(conf.seed + fold_id)

        y_tr_raw = y[train_idx]
        y_te_raw = y[test_idx]
        X_tr_raw = X[train_idx, :]
        X_te_raw = X[test_idx, :]
        
        my = mean(y_tr_raw)
        mx = mean(X_tr_raw, dims=1)
        sx = std(X_tr_raw, dims=1) .+ 1e-6
        X_tr_norm = (X_tr_raw .- mx) ./ sx
        X_te_norm = (X_te_raw .- mx) ./ sx
        
        if conf.family == "gaussian"
            y_tr = y_tr_raw .- my
        else
            y_tr = copy(y_tr_raw) # Do not center binary labels
        end
        
        # Explicit intercept column (Column 1)
        X_tr = hcat(ones(length(y_tr)), X_tr_norm)
        X_te = hcat(ones(length(y_te_raw)), X_te_norm)
        p_aug = size(X_tr, 2)
        n_tr = length(y_tr)

        # ------------------------------------------------------------------
        # Setup Inner CV (3-Fold) for Robust TPE Evaluation
        # ------------------------------------------------------------------
        perm = shuffle(1:n_tr)
        k_inner = 3 
        fold_sz = floor(Int, n_tr / k_inner)
        stochastic_folds = []
        for k in 1:k_inner
            s = (k-1)*fold_sz + 1
            e = (k == k_inner) ? n_tr : k*fold_sz
            val_idx = perm[s:e]
            tr_idx  = setdiff(perm, val_idx)
            push!(stochastic_folds, (tr_idx, val_idx))
        end
        
        max_tr_len = maximum(length(f[1]) for f in stochastic_folds)
        max_val_len = maximum(length(f[2]) for f in stochastic_folds)
        
        X_tr_buf = Matrix{Float64}(undef, max_tr_len, p_aug)
        y_tr_buf = Vector{Float64}(undef, max_tr_len)
        X_val_buf = Matrix{Float64}(undef, max_val_len, p_aug)
        y_val_buf = Vector{Float64}(undef, max_val_len)

        # ------------------------------------------------------------------
        # GLMNET Heuristic bounds for Lambda
        # ------------------------------------------------------------------
        lam_max = maximum(abs.(X_tr_norm' * (y_tr .- mean(y_tr))))
        lam_min_ratio = (n_tr < p_aug) ? 1e-2 : 1e-4 
        
        upper_lam_mult = 1.05 
        if PENALTY_SELECTION == "ELASTICNET"
            upper_lam_mult = (conf.fixed_alpha !== nothing && conf.fixed_alpha >= 0.99) ? 1.05 : 100.0  
        elseif PENALTY_SELECTION in ["GABR", "GABR-L0"]
            upper_lam_mult = 100.0      
        end
        bound_log_lam = (log(lam_max * lam_min_ratio), log(lam_max * upper_lam_mult))
        
        # ------------------------------------------------------------------
        # Adjusted Bounds for Stability
        # ------------------------------------------------------------------
        bounds_p2 = (0.0, 0.0)
        if PENALTY_SELECTION == "ELASTICNET" bounds_p2 = (0.0, 1.0)
        elseif PENALTY_SELECTION in ["GABR", "GABR-L0"] bounds_p2 = (0.5, 2.0) # Numerical stability floor = 0.5
        elseif PENALTY_SELECTION in ["MCP", "SCAD"] bounds_p2 = (1.5, 10.0)
        elseif PENALTY_SELECTION == "CAPPEDL1" bounds_p2 = (0.5, 10.0)
        elseif PENALTY_SELECTION == "LOGSUM" bounds_p2 = (0.01, 1.0)
        end

        best_cv_loss = Ref(Inf)
        best_params = Ref((p2=0.0, lam=1.0, tau=0.0))
        convergence_curve = Float64[]

        # ------------------------------------------------------------------
        # FULL INNER CV EVALUATION
        # ------------------------------------------------------------------
        function eval_inner_cv(p2_val, lam_val, tau_ratio_val)
            p2_safe = clamp(p2_val, bounds_p2[1], bounds_p2[2])
            total_loss = 0.0
            
            for (tr_i, val_i) in stochastic_folds
                X_curr_tr = view(X_tr_buf, 1:length(tr_i), :); y_curr_tr = view(y_tr_buf, 1:length(tr_i))
                X_curr_val = view(X_val_buf, 1:length(val_i), :); y_curr_val = view(y_val_buf, 1:length(val_i))
                
                X_curr_tr .= view(X_tr, tr_i, :); y_curr_tr .= view(y_tr, tr_i)
                X_curr_val .= view(X_tr, val_i, :); y_curr_val .= view(y_tr, val_i)
                
                # EXTRACT conf.newt_max_iter and pass to solver
                beta_local = solve_universal_cd(y_curr_tr, X_curr_tr, lam_val, p2_safe, tau_ratio_val, zeros(p_aug), conf.family, conf.newt_max_iter)
                preds_linear = X_curr_val * beta_local
                
                if conf.family == "gaussian"
                    total_loss += mean((y_curr_val .- preds_linear).^2)
                elseif conf.family == "binomial"
                    p_val = 1.0 ./ (1.0 .+ exp.(.-preds_linear))
                    total_loss += -mean(y_curr_val .* log.(p_val .+ 1e-15) .+ (1.0 .- y_curr_val) .* log.(1.0 .- p_val .+ 1e-15))
                end
            end
            
            return total_loss / k_inner # Return the AVERAGE loss across folds
        end

        if fold_id == 1
            println("Fold 1: [$PENALTY_SELECTION] Running TPE ($(conf.tpe_rounds) rounds)...")
        end
        
        tpe_time = @elapsed begin
            with_logger(NullLogger()) do
                
                if PENALTY_SELECTION == "GABR-L0"
                    rounds_nc = conf.tpe_rounds ÷ 2
                    rounds_c  = conf.tpe_rounds - rounds_nc
                    
                    # 1. Non-Convex Search
                    space_nc = Dict{Symbol, Any}(
                        :q       => HP.Uniform(:q_nc, 0.5, 1.0),
                        :log_lam => HP.Uniform(:log_lam_nc, bound_log_lam[1], bound_log_lam[2])
                    )
                    function obj_nc(params)
                        lam_val = exp(params[:log_lam])
                        loss = eval_inner_cv(params[:q], lam_val, 0.0)
                        
                        if loss < best_cv_loss[]
                            best_cv_loss[] = loss
                            best_params[] = (p2=params[:q], lam=lam_val, tau=0.0)
                        end
                        push!(convergence_curve, best_cv_loss[])
                        return loss
                    end
                    TreeParzen.fmin(obj_nc, space_nc, rounds_nc)
                    
                    # 2. Convex Search + Scale-Invariant Tau Ratio
                    space_c = Dict{Symbol, Any}(
                        :q             => HP.Uniform(:q_c, 1.0, 2.0),
                        :log_tau_ratio => HP.Uniform(:log_tau_ratio, log(1e-4), log(1.0)), # Ratio from 0.01% to 100% of lambda
                        :log_lam       => HP.Uniform(:log_lam_c, bound_log_lam[1], bound_log_lam[2])
                    )
                    function obj_c(params)
                        lam_val = exp(params[:log_lam])
                        tau_ratio_val = exp(params[:log_tau_ratio])
                        loss = eval_inner_cv(params[:q], lam_val, tau_ratio_val)
                        
                        if loss < best_cv_loss[]
                            best_cv_loss[] = loss
                            best_params[] = (p2=params[:q], lam=lam_val, tau=tau_ratio_val)
                        end
                        push!(convergence_curve, best_cv_loss[])
                        return loss
                    end
                    TreeParzen.fmin(obj_c, space_c, rounds_c)
                    
                else
                    # Standard TPE (No Dual-Regime logic required)
                    space_std = Dict{Symbol, Any}(:log_lam => HP.Uniform(:log_lam, bound_log_lam[1], bound_log_lam[2]))
                    should_tune_p2 = (PENALTY_SELECTION != "ELASTICNET" || conf.fixed_alpha === nothing)
                    
                    if should_tune_p2
                        space_std[:p2] = HP.Uniform(:p2, bounds_p2[1], bounds_p2[2])
                    end
                    
                    function obj_std(params)
                        lam_val = exp(params[:log_lam])
                        p2_val = should_tune_p2 ? params[:p2] : conf.fixed_alpha
                        loss = eval_inner_cv(p2_val, lam_val, 0.0)
                        
                        if loss < best_cv_loss[]
                            best_cv_loss[] = loss
                            best_params[] = (p2=p2_val, lam=lam_val, tau=0.0)
                        end
                        push!(convergence_curve, best_cv_loss[])
                        return loss
                    end
                    TreeParzen.fmin(obj_std, space_std, conf.tpe_rounds)
                end
            end 
        end 

        # FINAL REFIT: Extract conf.newt_max_iter and pass to solver
        final_beta = solve_universal_cd(y_tr, X_tr, best_params[].lam, best_params[].p2, best_params[].tau, zeros(p_aug), conf.family, conf.newt_max_iter)
        
        preds_linear = X_te * final_beta
        if conf.family == "gaussian"
            final_preds = preds_linear .+ my
            test_loss = mean((y_te_raw .- final_preds).^2)
            test_metric = distance_correlation(y_te_raw, final_preds)
            metric_name = "dCor"
        elseif conf.family == "binomial"
            final_preds = 1.0 ./ (1.0 .+ exp.(.-preds_linear))
            test_loss = -mean(y_te_raw .* log.(final_preds .+ 1e-15) .+ (1.0 .- y_te_raw) .* log.(1.0 .- final_preds .+ 1e-15))
            test_metric = roc_auc(y_te_raw, final_preds)
            metric_name = "AUC"
        end
        
        println("Fold $fold_id Finished. Test Loss=$(round(test_loss,digits=4)) | $metric_name=$(round(test_metric,digits=4)) | Time=$(round(tpe_time,digits=2))s")
        
        return FoldResult(fold_id, test_loss, test_metric, best_params[].p2, best_params[].lam, best_params[].tau, final_beta, tpe_time, convergence_curve)
    end
end

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================

function load_data(file, target)
    if !isfile(file)
        println("File $file not found. Generating dummy fallback data...")
        n, p = 200, 500; X = randn(n, p); y = X * [2.0; zeros(p-1)] + randn(n)
        if CONFIG.family == "binomial" y = (y .> median(y)) .* 1.0 end
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
if PENALTY_SELECTION == "ELASTICNET"
    println(CONFIG.fixed_alpha !== nothing ? "Status: Alpha (α) is FIXED to $(CONFIG.fixed_alpha)" : "Status: Alpha (α) is TUNED")
elseif PENALTY_SELECTION in ["GABR", "GABR-L0"]
    println("Status: Shape parameter (q) is TUNED")
    if PENALTY_SELECTION == "GABR-L0" println("Status: Truncation threshold (τ) is TUNED as an orthogonal ratio") end
elseif PENALTY_SELECTION in ["MCP", "SCAD", "CAPPEDL1"]
    println("Status: Shape parameter (a) is TUNED")
elseif PENALTY_SELECTION == "LOGSUM"
    println("Status: Shape parameter (eps) is TUNED")
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

results = pmap(f -> run_tpe_fold(f[1], f[2], f[3], X_raw, y_raw, CONFIG), folds)

elapsed = time() - start_time
println("\n--- COMPLETED in $(round(elapsed, digits=2)) seconds ---")

# ==============================================================================
# 6. AGGREGATION & SUMMARY
# ==============================================================================
println("\n=== PER-FOLD RESULTS: $PENALTY_SELECTION ===")
loss_label = CONFIG.family == "binomial" ? "Log-Loss" : "MSE"
metr_label = CONFIG.family == "binomial" ? "AUC" : "dCor"
println(rpad("Fold", 6) * rpad(loss_label, 12) * rpad(metr_label, 12) * rpad("Best Param2", 15) * rpad("Best Lambda", 15) * rpad("Best Tau Ratio", 15))
println("-"^85)

sort!(results, by = x -> x.fold)
for r in results
    @printf("%-6d %-12.4f %-12.4f %-15.4f %-15.4f %-15.4e\n", r.fold, r.loss, r.metric, r.best_p2, r.best_lam, r.best_tau)
end
println("-"^85)

# Note: The output beta matrix now has (p + 1) rows, where row 1 is the Intercept.
beta_matrix = hcat([r.beta for r in results]...) 
df_per_fold = DataFrame(beta_matrix, :auto)
rename!(df_per_fold, [Symbol("Fold_$i") for i in 1:CONFIG.n_folds])
feature_labels = ["Intercept"; string.(1:p_features)]
insertcols!(df_per_fold, 1, :Feature_Index => feature_labels)
CSV.write("$(PENALTY_SELECTION)_Coefficients_PerFold_TPE.csv", df_per_fold)

avg_tpe_time = mean([r.tpe_time for r in results])
curves_matrix = hcat([r.convergence_curve for r in results]...)
avg_curve = vec(mean(curves_matrix, dims=2))

df_conv = DataFrame(Iteration = 1:CONFIG.tpe_rounds, Mean_Loss = avg_curve)
CSV.write("$(PENALTY_SELECTION)_convergence_Loss.csv", df_conv)

df_time = DataFrame(Penalty = [PENALTY_SELECTION], Mean_TPE_Time_Per_Fold_Sec = [avg_tpe_time])
CSV.write("$(PENALTY_SELECTION)_timing.csv", df_time)

loss_scores   = [r.loss for r in results]
metric_scores  = [r.metric for r in results]
p2_scores    = [r.best_p2 for r in results]
lam_scores   = [r.best_lam for r in results]
tau_scores   = [r.best_tau for r in results]

# Sparsity score strictly checks non-intercept features (rows 2:end)
sparsity_scores = [count(x -> abs(x) > SPARSITY_THRESHOLD, r.beta[2:end]) for r in results]

avg_beta = mean(beta_matrix, dims=2)[:]
stability_score = mean(abs.(beta_matrix) .> SPARSITY_THRESHOLD, dims=2)[:]
stability_score[1] = 1.0 # Intercept is always 100% stable

println("\n=== SUMMARY STATISTICS ===")
println("Test $loss_label:         $(round(mean(loss_scores), digits=4)) ± $(round(std(loss_scores), digits=4))")
println("Test $metr_label:              $(round(mean(metric_scores), digits=4)) ± $(round(std(metric_scores), digits=4))")
println("Best Param2 (Shape):  $(round(mean(p2_scores), digits=4)) ± $(round(std(p2_scores), digits=4))")
println("Best Lambda:          $(round(mean(lam_scores), digits=4)) ± $(round(std(lam_scores), digits=4))")
if PENALTY_SELECTION == "GABR-L0"
    println("Best Tau Ratio (L0):  $(round(mean(tau_scores), digits=6)) ± $(round(std(tau_scores), digits=6))")
end
println("Selected Features:    $(round(mean(sparsity_scores), digits=2)) ± $(round(std(sparsity_scores), digits=2))")
println("Average TPE Fold Time: $(round(avg_tpe_time, digits=2)) seconds")

df_agg = DataFrame(Feature_Index = feature_labels, Bagged_Beta = avg_beta, Stability_Score = stability_score)
sort!(df_agg, [:Stability_Score, :Bagged_Beta], rev=true)
CSV.write("$(PENALTY_SELECTION)_Coefficients_Average_TPE.csv", df_agg)
println("Saved aggregated coefficients.")
