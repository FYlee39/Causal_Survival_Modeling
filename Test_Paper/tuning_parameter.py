from helper_functions import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from tqdm import tqdm

model_name = "DCM Model"  # name of model to be tuning
random_seed = 42

# Generate synthetic data
num_samples = 2000
num_groups = 3
num_covariates = (2, 5, 2)
df, save_dict, lambda0, lambda1, Z, actual_censor_rate, diagnostics = generate_simulated_data(num_sample=num_samples,
                                                                                              num_group=num_groups,
                                                                                              num_covariates=num_covariates,
                                                                                              random_seed=random_seed)

# Identify categorical and numerical features
categorical_features_list = [col for col in df.columns if col.startswith("X_c")] + ["treatment"]

# Process data for DCM (using the same processing function)
data_splits = processing_data_2_DCM(df,
                                    categorical_features_list=categorical_features_list,
                                    train_test_val_size=(0.7, 0.2, 0.1),
                                    random_seed=42)

(X_train, X_val, X_test), (t_train, t_val, t_test), (e_train, e_val, e_test), cat_feats, num_feats = data_splits

train_set = (X_train, t_train, e_train)
val_set = (X_val, t_val, e_val)

# Define hyperparameter grid for DCM
params_grid = list(ParameterGrid({
    'k': [[2], [3], [4], [5], [6]],
    'layers': [[[50]], [[50, 50]], [[100]], [[100, 100]], [[100, 100, 100]], [[200, 200]], [[200, 200, 200]]],
    "gamma": [[8], [9], [10], [11], [12]],
    "smoothing_factor": [[1e-3], [1e-4], [1e-5], [1e-6]],
    "use_activation": [[True], [False]],
    'iters': [[50], [100], [150], [200]],
    'learning_rate': [[1e-2], [1e-3], [1e-4], [1e-5], [1e-6]],
    "batch_size": [[32], [64], [128], [256]],
    "optimizer": [['Adam']],
    "use_treatment": [[True], [False]],
}))

# To extract best params: Track during manual grid for clarity
best_loss = float('inf')
best_params = None
models = []

def train_and_eval(param, train_set, val_set):
    # Note: Assuming the wrapper has been modified to accept a single param dict in params_grid=[param]
    # If not, adjust accordingly (e.g., extract values like k=param['k'], etc., and pass directly to DeepCoxMixtures)
    model_wrapper = model_dict[model_name](params_grid=[param], use_treatment=param["use_treatment"])
    model_wrapper.fit(train_set, val_set)
    loss = model_wrapper.get_loss()
    return loss, param  # Return loss and param; avoid returning model_wrapper to prevent pickling issues

for param in params_grid:

    model_wrapper = model_dict[model_name](params_grid=param,
                                           use_treatment=param["use_treatment"])

    model_wrapper.fit((X_train, t_train, e_train), (X_val, t_val, e_val))

    loss = model_wrapper.get_loss()

    models.append((loss, model_wrapper, param))
    if loss < best_loss:
        best_loss = loss
        best_params = param

print("Best parameters for DCM on synthetic data:", best_params)