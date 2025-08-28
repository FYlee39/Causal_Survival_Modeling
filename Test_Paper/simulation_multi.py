from helper_functions import *
from sklearn.model_selection import ParameterGrid
import multiprocessing
from functools import partial
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
models_list=['DCM Model',
             "DCM IPTW Model",
             "DSM IPTW Model",
             'DSM Model',
             'Cox Regression Model',
             'Cox IPTW Model',
             'RSF Model',
             'RSF IPTW Model',
             'AFT Model',
             'AFT IPTW Model']

params_grids = {
        "DCM Model": {"k": [4], "layers": [[100, 100, 100]], "iters": [200], "learning_rate": [1e-3]},
        "DCM IPTW Model": {"k": [4], "layers": [[100, 100, 100]], "iters": [200], "learning_rate": [1e-3]},
        "DSM Model": {"k": [3], "distribution": ["Weibull"], "layers": [[100, 100, 100]], "iters": [200], "learning_rate": [1e-3]},
        "DSM IPTW Model": {"k": [3], "distribution": ["Weibull"], "layers": [[100, 100, 100]], "iters": [200], "learning_rate": [1e-3]},
        "Cox Regression Model": {"penalizer": [0.1]},
        "Cox IPTW Model": {"penalizer": [0.1]},
        "RSF Model": {"n_estimators": [100], "max_depth": [5]},
        "RSF IPTW Model": {"n_estimators": [100], "max_depth": [5]},
        "AFT Model": {"penalizer": [0.1]},
        "AFT IPTW Model": {"penalizer": [0.1]}
    }

def _run_replicate(rep: int,
                   params_grids,
                   num_group: int=3,
                   num_sample: int=1000,
                   censor_rate: float=0.3,
                   random_seed_base=RANDOM_SEED,
                   models_list=models_list,
                   train_test_val_size=None,
                   save_dict=None
                   ):
    """
    Worker function for a single replicate: Runs data generation, processing, model fitting, and metrics.
    :param params_grids: list of model parameters
    :param rep: number of replicates
    :param num_group: number of latent groups
    :param censor_rate: censoring rate
    :param random_seed_base: random seed
    :param models_list: list of models to use
    :param train_test_val_size: size of test set for each model
    :param save_dict: dictionary to data
    :return: None dict of {model: metrics} for this replicate.
    """
    seed = random_seed_base + rep
    df, save_dict, lambda0, lambda1, Z, actual_censor_rate, diagnostics = generate_simulated_data(
        num_sample=num_sample,
        num_group=num_group,
        censor_rate=censor_rate,
        random_seed=seed,
        **save_dict
    )

    # Identify features
    categorical_features_list = [col for col in df.columns if col.startswith("X_c")] + ["treatment"]

    # Process data
    processed = processing_data_2_DCM(df,
                                      categorical_features_list,
                                      train_test_val_size=train_test_val_size)

    (X_train, X_val, _), (t_train, t_val, _), (e_train, e_val, _), cat_feats, num_feats = processed

    covariates = [col for col in X_train.columns if col not in ["time", "event", "treatment"]]

    rep_results = {}
    for model_name in models_list:
        print(model_name)
        rep_results[model_name] = {}
        params_grid = params_grids.get(model_name)

        use_treatment = False if "DCM" in model_name else True

        if 'IPTW' in model_name:
            model_wrapper = model_dict[model_name](params_grid=params_grid,
                                                   covariates=covariates,
                                                   use_treatment=use_treatment)
        else:
            model_wrapper = model_dict[model_name](params_grid=params_grid,
                                                   use_treatment=use_treatment)

        try:
            model_wrapper.fit((X_train, t_train, e_train), (X_val, t_val, e_val))


            df, _, lambda0, lambda1, Z, actual_censor_rate, diagnostics = generate_simulated_data(
                num_sample=int(num_sample / 3),  # one third of the sample size
                num_group=num_group,
                censor_rate=censor_rate,
                random_seed=seed,
                **save_dict  # use same parameters to generate test data
            )

            # Process data
            processed = processing_data_2_DCM(df,
                                              categorical_features_list,
                                              train_test_val_size=None)

            (_, _, X_test), (_, _, t_test), (_, _, e_test), _, _ = processed

            # Time grid for predictions: Use test set range, excluding exact max to satisfy strict < max condition
            min_time = np.min(t_test)
            max_time = np.max(t_test)
            times = np.linspace(min_time, max_time, 100, endpoint=False)

            metrics = compute_metrics(model_wrapper,
                                      X_test, t_test, e_test,
                                      times,
                                      Z,
                                      lambda0,
                                      lambda1)

            rep_results[model_name] = metrics
        except Exception as ex:
            print(f"Error in {model_name}: {ex}")
            rep_results[model_name] = {'c_index': np.nan,
                                       'Integrated Brier Score': np.nan,
                                       'Adjusted Random Index': np.nan,
                                       "abs_bias": np.nan,
                                       "rel_bias": np.nan}

            for g in range(num_group):
                rep_results[model_name][f'abs_bias_group_{g}'] = np.nan
                rep_results[model_name][f'rel_bias_group_{g}'] = np.nan

    return rep_results


def run_simulation(num_replicates: int=100,
                   num_groups_list: list=[1, 3, 5],
                   num_sample: int=1000,
                   censor_rate: float=0.3,
                   models_list=models_list,
                   params_grids={},
                   random_seed_base=RANDOM_SEED,
                   num_worker=1,
                   train_test_val_size=None):
    """
    Run simulations
    :param num_replicates: number of replicates
    :param num_groups_list: list of numbers of latent groups
    :param num_sample: number of samples
    :param censor_rate: censoring rate
    :param models_list: list of models to use
    :param params_grids: list of model parameters
    :param random_seed_base: random seed
    :param num_worker: number of workers
    :param train_test_val_size: size of test set for each model
    :return:
    """

    results = {}
    for num_group in num_groups_list:

        _, save_dict, _, _, _, _, _ = generate_simulated_data(
            num_sample=num_sample,
            num_group=num_group,
            censor_rate=censor_rate,
            random_seed=random_seed_base
        )

        results[num_group] = {model: [] for model in models_list}

        # Partial function for worker (fixes non-rep args, including save_dict)
        worker = partial(
            _run_replicate,
            num_group=num_group,
            num_sample=num_sample,
            censor_rate=censor_rate,
            random_seed_base=random_seed_base + num_group,
            models_list=models_list,
            params_grids=params_grids,
            train_test_val_size=train_test_val_size,
            save_dict=save_dict
        )

        # Parallelize over replicates
        with multiprocessing.Pool(processes=num_worker) as pool:
            rep_outputs = list(
                tqdm(
                    pool.imap(
                        worker, range(num_replicates)
                    ), total=num_replicates, desc=f"Processing replicates for num_group={num_group}"
                )
            )

        # Collect results
        for rep_result in rep_outputs:
            for model, metrics in rep_result.items():
                results[num_group][model].append(metrics)

    # Aggregate results
    agg_results = {}
    for num_group in num_groups_list:
        agg_results[num_group] = {model: {k: np.nanmean([m[k] for m in results[num_group][model]]) for k in results[num_group][model][0]} for model in models_list}

    return agg_results, results

if __name__ == '__main__':

    experiment_params = {
        "num_replicates": 200,
        "num_groups_list": [4],
        "num_sample": 5000,
        "censor_rate": 0.3,
        "models_list": models_list,
        "params_grids": params_grids,
        "random_seed_base": 0,
        "fixed_param_seed": 0,
        "num_worker": 20,
        "train_test_val_size": None
    }

    # Run the simulation (adjust replicates/groups for your compute; use fixed_param_seed for consistent params across reps)
    agg_results, results = run_simulation(
        num_replicates=experiment_params["num_replicates"],  # Balance speed and precision; 100+ for tight SEs
        num_groups_list=experiment_params["num_groups_list"],  # Test sensitivity to heterogeneity
        num_sample=experiment_params["num_sample"],  # Sample size; increase for better estimation
        censor_rate=experiment_params["censor_rate"],  # Target censoring; administrative method ensures ~this value
        models_list=experiment_params["models_list"],
        params_grids=experiment_params["params_grids"],
        random_seed_base=experiment_params["random_seed_base"],  # For reproducibility
        num_worker=experiment_params["num_worker"],
        train_test_val_size=experiment_params["train_test_val_size"]
    )

    # Convert to DataFrame and display (statistically, add SEs for inference)
    df_results = results_to_dataframe(agg_results)
    print(df_results)

    # Save everything
    save_simulation_results(agg_results, experiment_params)