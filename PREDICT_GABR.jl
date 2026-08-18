# ==============================================================================
# PREDICT_GABR.JL
# Loads a model saved by gabr_solver.jl and scores new data with it.
# Only needs Serialization (stdlib) — no need to reload the full solver.
# ==============================================================================

using Serialization, CSV, DataFrames

# Struct definition must match the one used when saving, so Julia can
# reconstruct the object from the .jls file.
struct SavedModel
    penalty::String
    family::String
    feature_names::Vector{String}
    beta::Vector{Float64}
    x_mean::Vector{Float64}
    x_std::Vector{Float64}
    y_mean::Float64
    p2::Float64
    lambda::Float64
    tau::Float64
end

"""
    predict_gabr(model, X_new) -> Vector{Float64}

X_new must have the same columns, in the same order, as the training data
(excluding the target column, excluding any intercept — that's added here).
Returns predicted Y for "gaussian", or predicted probabilities for "binomial".
"""
function predict_gabr(model::SavedModel, X_new::AbstractMatrix{Float64})
    if size(X_new, 2) != length(model.feature_names)
        error("Expected $(length(model.feature_names)) feature columns, got $(size(X_new, 2))")
    end
    X_norm = (X_new .- model.x_mean') ./ model.x_std'
    X_aug  = hcat(ones(size(X_norm, 1)), X_norm)
    eta    = X_aug * model.beta

    if model.family == "gaussian"
        return eta .+ model.y_mean
    else # binomial
        return 1.0 ./ (1.0 .+ exp.(.-eta))
    end
end

"""
    predict_gabr(model, x_new::AbstractVector{Float64}) -> Float64

Convenience overload for a SINGLE new observation on the original (raw) scale,
e.g. x_new = [5.2, 130.0, 0.0, 27.5, ...] with one value per feature, in the
same order as model.feature_names. Reshapes to a 1×p matrix internally and
returns a plain scalar (predicted Y for "gaussian", predicted probability for
"binomial") instead of a length-1 vector.
"""
function predict_gabr(model::SavedModel, x_new::AbstractVector{Float64})
    return predict_gabr(model, reshape(x_new, 1, :))[1]
end

# ------------------------------------------------------------------------------
# EXAMPLE USAGE: load the saved model, read the original data file, predict
# ------------------------------------------------------------------------------

# --- 1. Point these at what you used in gabr_solver.jl's CONFIG ---
const MODEL_FILE = "GABR-L0_final_model.jls"   # <-- matches your PENALTY_SELECTION
const DATA_FILE  = "Mice_BodyLength.csv"       # <-- same CSV path used for training
const TARGET_COL = :Y                          # <-- same target column name

# --- 2. Load the model and the data ---
model = deserialize(MODEL_FILE)
df = CSV.read(DATA_FILE, DataFrame)

# --- 3. Split into X / y exactly like gabr_solver.jl's load_data() did,
#         so column order matches what the model's beta was fit on.
#         The target column may be ABSENT entirely (genuinely new, unlabeled
#         data — the normal case for real predictions) or PRESENT with some
#         missing (NA) values mixed in. Both are handled via Julia's `missing`. ---
tgt_candidates = [TARGET_COL, :y, :Y]
tgt_pos = findfirst(c -> string(c) in names(df), tgt_candidates)
has_target = tgt_pos !== nothing

if has_target
    tgt = tgt_candidates[tgt_pos]
    y_actual = Vector{Union{Float64,Missing}}(df[!, tgt])  # allows NA cells even though the column exists
    X_new    = Matrix{Float64}(select(df, Not(tgt)))
else
    y_actual = Union{Float64,Missing}[missing for _ in 1:nrow(df)]  # no target column at all
    X_new    = Matrix{Float64}(df)
    println("No target column found in $DATA_FILE — treating all $(nrow(df)) rows as unlabeled.")
end

# --- 4. Predict ---
preds = predict_gabr(model, X_new)

# --- 5. Sanity-check against known outcomes, where available ---
# NOTE: rows with a real Actual value here are still IN-SAMPLE if this is the
# training file — not a generalization estimate. Trust the CV Test MSE/AUC
# from gabr_solver.jl for that, or score a genuinely held-out file.
valid_idx = findall(!ismissing, y_actual)
if isempty(valid_idx)
    println("No true values available in this file — skipping the accuracy check.")
else
    if length(valid_idx) < length(y_actual)
        println("$(length(valid_idx)) / $(length(y_actual)) rows have a true value — checking those only.")
    end
    y_known     = Float64.(y_actual[valid_idx])
    preds_known = preds[valid_idx]
    if model.family == "gaussian"
        mse = sum((y_known .- preds_known) .^ 2) / length(y_known)
        println("MSE (on rows with known values): ", round(mse, digits=4))
        for i in 1:min(10, length(preds_known))
            println("  actual=$(y_known[i])  predicted=$(round(preds_known[i], digits=3))")
        end
    else # binomial
        pred_labels = preds_known .> 0.5
        accuracy = sum(pred_labels .== (y_known .> 0.5)) / length(y_known)
        println("Accuracy (on rows with known values): ", round(accuracy, digits=4))
        for i in 1:min(10, length(preds_known))
            println("  actual=$(y_known[i])  predicted_prob=$(round(preds_known[i], digits=3))")
        end
    end
end

# --- 6. Save predictions to file ---
# If Actual (true value) is `missing` for any row without a known true value (or every row, if
# there was no target column at all) — Residual then propagates `missing` for
# those rows automatically via Julia's missing-arithmetic rules.
dfpreds = DataFrame(Actual = y_actual, Predicted = preds)
if model.family == "gaussian"
    dfpreds.Residual = y_actual .- preds
end
pred_path = "$(model.penalty)_Predictions.csv"   # model.penalty, not PENALTY_SELECTION (undefined here)
CSV.write(pred_path, dfpreds)
println("Saved predictions -> $pred_path")
