import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from auton_survival.preprocessing import Preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.integrate import trapz

def processing_data(path: str="C:\\Users\\lee39\\OneDrive\\Desktop\\final_merged_dataset.csv") -> pd.DataFrame:
    """
    processed eICU data
    :param path: final_merged_dataset.csv path
    :return: processed eICU data frame
    """
    eICU_data = pd.read_csv("C:\\Users\\lee39\\OneDrive\\Desktop\\final_merged_dataset.csv")
    cleaned_df = eICU_data.copy()

    # drop ids
    cleaned_df.drop(["patientunitstayid"], inplace=True, axis=1)
    cleaned_df.drop(["hospitalid"], inplace=True, axis=1)

    # change the name of time, event and treatment variables

    cleaned_df.rename(columns={"unitdischargeoffset": "time",
                               "unitdischargestatus": "event",
                               "has_Vasopressor": "treatment",},
                      inplace=True)


    cleaned_df["event"] = (cleaned_df['event'] == 'Expired').astype(int)

    return cleaned_df


def processing_data_2_DCM(df: pd.DataFrame,
                          categorical_features_list: list,
                          train_test_val_size: tuple=(0.7, 0.2, 0.1),
                          random_seed=42):
    """
    further processed the data to meet the requirements of DCM
    :param df: processed eICU data frame
    :param categorical_features_list: categorical features list
    :param train_test_val_size: default train, test, validation proportion
    :param random_seed: default random seed
    :return:
    """
    treatment = df["treatment"]
    time = df["time"]
    event = df["event"]
    outcomes = pd.concat([time, event], axis=1)
    features = df.drop(["time", "event"], axis=1)  # include the treatment
    numerical_features_list = list(features.columns.drop(categorical_features_list + ["treatment"]))

    # process data for DCM
    processed_features = Preprocessor().fit_transform(features,
                                            cat_feats=categorical_features_list + ["treatment"],
                                            num_feats=numerical_features_list)


    # train test and val split
    original_indices = np.arange(len(features))
    X_train, X_val_test, y_train, y_val_test, idx_train, idx_val_test = train_test_split(
        processed_features, outcomes, original_indices,
        test_size=1 - train_test_val_size[0],
        random_state=random_seed
    )

    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_val_test, y_val_test, idx_val_test,
        test_size=train_test_val_size[-2] / (train_test_val_size[-1] + train_test_val_size[-2]),
        random_state=random_seed
    )

    t_train, e_train = y_train["time"], y_train["event"]
    t_val, e_val = y_val["time"], y_val["event"]
    t_test, e_test = y_test["time"], y_test["event"]

    return [(X_train, X_val, X_test),
            (t_train, t_val, t_test),
            (e_train, e_val, e_test),
            categorical_features_list,
            numerical_features_list]


def compute_PS_and_IPTW(df: pd.DataFrame,
                        covariates: list,
                        treatment: str,
                        max_iter: int=1000) -> pd.DataFrame:
    """
    Compute the propensity score and IPTW
    :param df: processed eICU data frame
    :param covariates: list of covariates exclude the treatment
    :param treatment: name of the treatment column
    :param max_iter: maximum number of iterations for the logistic regression
    :return: the new data frame with PS and IPTW
    """
    df_ps = df.copy()
    X = df[covariates]
    y = df[treatment]

    # using logistic regression to estimate PS
    log_reg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
    log_reg.fit(X, y)

    # Predict propensity scores (probability of being treated)
    df_ps["propensity_score"] = log_reg.predict_proba(X)[:, 1]

    # compute IPTW weights
    t = df_ps[treatment]
    ps = df_ps["propensity_score"]
    df_ps["iptw_weight"] = t / ps + (1 - t) / (1 - ps)

    return df_ps

def plot_DCM_avg_survival_curve(df_ps: pd.DataFrame,
                                group_index: np.ndarray,
                                model,
                                covariates: list,
                                treatment: str,
                                figsize: tuple=(10, 10)):
    """
    plot the average survival curve for the DCM based on the group
    :param df: dataframe with PS and IPTW
    :param group_index: index of the group
    :param model: the DCM model
    :param covariates: list of covariates exclude the treatment
    :param treatment: name of the treatment column
    :param figsize: default size of the figure
    :return: the estimated causal effect for three groups
    """
    # get the number of groups
    num_group = len(np.unique(group_index))

    # store the causal effect
    causal_effects = []

    plt.figure(figsize=figsize)

    for i in range(0, num_group):
        # find all data in selected group
        df_group = df_ps[group_index == i]
        df_treated = df_group[df_group[treatment] == 1]
        df_untreated = df_group[df_group[treatment] == 0]

        # Compute AUC via trapezoidal rule
        time_grid = list(np.linspace(0, np.max(df_group["time"]), 100))

        X_treated = df_treated[covariates + [treatment]]
        X_control = df_untreated[covariates + [treatment]]

        S1 = model.predict_survival(X_treated, time_grid)
        S0 = model.predict_survival(X_control, time_grid)

        S1_avg = np.mean(S1, axis=0)
        S0_avg = np.mean(S0, axis=0)

        plt.subplot(num_group, 1, i + 1)

        plt.plot(time_grid, S1_avg, label="Treated (model-adjusted)", color='blue')
        plt.plot(time_grid, S0_avg, label="Untreated (model-adjusted)", color='red')
        plt.xlabel("Time")
        plt.ylabel("Average Survival Probability")

        # Final plot settings
        plt.title(f"IPTW-Adjusted Survival Curves (Treated vs Untreated) for group {i}")
        plt.xlabel("Time Since ICU Admission (minutes)")
        plt.ylabel("Survival Probability")

        plt.legend()
        plt.grid(True)

        rmst1 = trapz(S1, time_grid)
        rmst0 = trapz(S0, time_grid)

        # Causal effect estimate using AUC
        causal_effects.append(np.mean(rmst1) - np.mean(rmst0))

    plt.subplots_adjust(wspace=0.5, hspace=2)
    plt.tight_layout()
    plt.legend()
    plt.show()

    return causal_effects
