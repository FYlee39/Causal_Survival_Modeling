import numpy as np
import pandas as pd
from scipy.stats import expon, uniform

def generate_simulated_data(num_sample: int=1000,
                            num_group: int=1,
                            num_covariates=None,
                            **kwargs
                            ):
    """
    Generate simulated data
    :param num_sample: number of samples
    :param num_group: number of latent groups
    :param num_covariates: number of covariates in list [linear, non-linear, categorical]
    :return: simulated data
    """
    pi = np.full(num_group, 1.0 / num_group)  # groups probability

    # Latent groups Z (0 to num_group - 1)
    Z = np.random.choice(range(num_group), num_sample, p=pi)

    # Number of covariates
    if num_covariates is None:
        num_covariates = (3, 3, 3)
    else:
        pass

    n_linear, n_non_linear, n_categorical = num_covariates
    N = n_linear + n_non_linear + n_categorical

    covariates = {}

    # dictionary to save parameters
    save_dict = {}

    # Group-specific parameters for covariates (for num_group=1, single value)
    # Mean of gaussian distribution for linear covariates
    if "means_k" not in kwargs:
        means_k = np.random.uniform(-1, 1, num_group) if num_group > 1 else np.array([0.0])
    else:
        means_k = kwargs["means_k"]
    save_dict["means_k"] = means_k

    # Parameter of binomial distribution for categorical covariates
    if "p_k" not in kwargs:
        p_k = np.random.uniform(0.3, 0.7, num_group) if num_group > 1 else np.array([0.5])
    else:
        p_k = kwargs["p_k"]
    save_dict["p_k"] = p_k

    # Group-specific betas (shape (num_group, N); for num_group=1, (1, N))
    if "prop_betas" not in kwargs:
        prop_betas = np.random.uniform(-1, 1, (num_group, N))
    else:
        prop_betas = kwargs["prop_betas"]
    save_dict["prop_betas"] = prop_betas

    if "haz_betas" not in kwargs:
        haz_betas = np.random.uniform(-1, 1, (num_group, N))
    else:
        haz_betas = kwargs["haz_betas"]
    save_dict["haz_betas"] = haz_betas

    # Group-specific baseline, treatment effect, interaction beta
    if "beta0_k" not in kwargs:
        beta0_k = np.random.uniform(-3, -1, num_group)
    else:
        beta0_k = kwargs["beta0_k"]
    save_dict["beta0_k"] = beta0_k

    if "beta_A_k" not in kwargs:
        beta_A_k = np.random.uniform(-1, 0, num_group)  # Negative for protective
    else:
        beta_A_k = kwargs["beta_A_k"]
    save_dict["beta_A_k"] = beta_A_k

    if "interact_beta_k" not in kwargs:
        interact_beta_k = np.random.uniform(0, 0.5, num_group)
    else:
        interact_beta_k = kwargs["interact_beta_k"]
    save_dict["interact_beta_k"] = interact_beta_k

    # Generate covariates
    for i in range(1, n_linear + 1):
        # Continuous: Normal distribution
        covariates[f"X{i}"] = np.random.normal(means_k[Z], 1, num_sample)

    for i in range(1 + n_linear, n_non_linear + n_linear + 1):
        # Continuous: Uniform for trig functions
        covariates[f"X{i}"] = np.random.uniform(0, 2 * np.pi, num_sample)

    for i in range(1, n_categorical + 1):
        # Categorical: Binomial
        covariates[f"X_c{i}"] = np.random.binomial(1, p_k[Z], num_sample)

    # Generate confounded treatment (binary)
    # Logistic propensity score with non-linear effects on covariates
    logit = -1.0  # Intercept

    for i in range(1, n_linear + 1):
        # Continuous: Normal distribution
        if i % 2 == 0:
            func = lambda x: x
        else:
            func = lambda x: x ** 2
        logit += prop_betas[Z, i - 1] * func(covariates[f"X{i}"])

    for i in range(1 + n_linear, n_non_linear + n_linear + 1):
        # Continuous: Uniform for trig functions
        func = np.sin if i % 2 == 0 else np.cos
        logit += prop_betas[Z, i - 1] * func(covariates[f"X{i}"])

    for i in range(1, n_categorical + 1):
        # Categorical: Binomial
        func = lambda x: x
        logit += prop_betas[Z, i + n_non_linear + n_linear - 1] * func(covariates[f"X_c{i}"])

    prob = 1 / (1 + np.exp(-logit))
    # Generate treatment
    treatment = np.random.binomial(1, prob, num_sample)

    # Generate true survival times from exponential distribution
    # Rate lambda includes non-linear terms on covariates, treatment, and one interaction if N>=2
    log_lambda = beta0_k[Z]

    for i in range(1, n_linear + 1):
        # Continuous: Normal distribution
        if i % 2 == 0:
            func = lambda x: x
        else:
            func = lambda x: x ** 2
        log_lambda += haz_betas[Z, i - 1] * prop_betas[Z, i - 1] * func(covariates[f'X{i}'])

    for i in range(1 + n_linear, n_non_linear + n_linear + 1):
        # Continuous: Uniform for trig functions
        func = np.sin if i % 2 == 0 else np.cos
        log_lambda += haz_betas[Z, i - 1] * prop_betas[Z, i - 1] * func(covariates[f'X{i}'])

    for i in range(1, n_categorical + 1):
        # Categorical: Binomial
        func = lambda x: x
        log_lambda += haz_betas[Z, i + n_non_linear + n_linear - 1] * prop_betas[Z, i + n_non_linear + n_linear - 1] * func(
            covariates[f'X_c{i}'])

    log_lambda += beta_A_k[Z] * treatment

    # Add an interaction term using first two covariates
    if N >= 2:
        log_lambda += interact_beta_k[Z] * covariates["X1"] * covariates["X2"]
    lambda_ = np.exp(log_lambda)

    # True survival time
    ture_time = expon.rvs(scale=1 / lambda_)

    # Censoring
    censor_time = uniform.rvs(0, np.max(ture_time), num_sample)
    obs_time = np.minimum(ture_time, censor_time)
    event = (ture_time <= censor_time).astype(int)
    # DataFrame
    df_data = {**covariates, "treatment": treatment, "time": obs_time, "event": event}
    df = pd.DataFrame(df_data)

    return df, save_dict,