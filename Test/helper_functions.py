import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.integrate import trapz

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

from lifelines import CoxPHFitter

from auton_survival.preprocessing import Preprocessor
from auton_survival.models.dsm import DeepSurvivalMachines
from auton_survival.models.dcm import DeepCoxMixtures
from auton_survival.models.dcm.dcm_utilities import test_step


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


def clustering_data(df: pd.DataFrame,
                    n_clusters,
                    features: list,
                    random_seed=42) -> pd.DataFrame:
    """
    Clustering data according to the designated features
    :param df: data frame
    :param n_clusters: number of clusters
    :param features: features used to cluster
    :return: new data frame with cluster number
    """
    X = df[features]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    df["cluster index"] = kmeans.fit_predict(X)

    return df


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


class DSM_Wrapper(object):
    """
    A wrapper for DSM model
    """
    def __init__(self, params_grid):
        """
        Initialization
        :param params_grid: parameters grid for DSM
        """
        self.params_grid = params_grid
        self.model = None

    def fit(self, train_set, val_set):
        """
        Fit the model
        :param train_set: training set
        :param val_set: validation set
        :return:
        """
        models = []
        x_train, t_train, e_train = train_set
        x_val, t_val, e_val = val_set
        for param in self.params_grid:
            model = DeepSurvivalMachines(k=param["k"],
                                 distribution=param["distribution"],
                                 layers=param["layers"],)

            model.fit(x_train, t_train, e_train,
                      iters=param["iters"],
                      learning_rate=param["learning_rate"])
            models.append([model.compute_nll(x_val, t_val, e_val), model])

        best_model_entry = min(models, key=lambda x: x[0])

        # Extract the model
        self.model = best_model_entry[1]

    def predict(self, test_set, times):
        """
        Predict survival probability and risk
        :param test_set: test set
        :param times: times to predict
        :return: survival probability, risk
        """
        x_test, t_test, e_test = test_set

        out_survival = self.model.predict_survival(x_test, times)
        out_risk = 1 - out_survival

        return out_survival, out_risk


class DCM_Wrapper(object):
    """
    A wrapper for DCM model
    """
    def __init__(self, params_grid):
        """
        Initialization
        :param params_grid: parameters grid for DCM
        """
        self.params_grid = params_grid
        self.model = None

    def fit(self, train_set, val_set):
        """
        Fit the model
        :param train_set: training set
        :param val_set: validation set
        :return:
        """

        def dataframe_to_tensor(data):
            """Function that converts a pandas dataframe into a tensor"""
            if isinstance(data, (pd.Series, pd.DataFrame)):
                data =  data.to_numpy()
                return torch.from_numpy(data).float()
            else:
                return torch.from_numpy(data).float()

        x_train, t_train, e_train = train_set
        x_val, t_val, e_val = val_set
        x_val_tensor = dataframe_to_tensor(x_val)
        t_val_tensor = dataframe_to_tensor(t_val)
        e_val_tensor = dataframe_to_tensor(e_val)

        models = []
        for param in self.params_grid:
            model = DeepCoxMixtures(k=param["k"],
                                    layers=param["layers"])
            # The fit method is called to train the model
            model.fit(x_train, t_train, e_train,
                      iters=param["iters"],
                      learning_rate=param["learning_rate"]
                      )

            # store the performance on the validation set
            breslow_splines = model.torch_model[1]
            val_result = test_step(model.torch_model[0], x_val_tensor, t_val_tensor, e_val_tensor, breslow_splines)
            models.append([[val_result, model]])

        best_model = min(models)
        self.model = best_model[0][1]

    def predict(self, test_set, times):
        """
        Predict survival probability and risk
        :param test_set: test set
        :param times: times to predict
        :return: survival probability, risk
        """
        x_test, t_test, e_test = test_set

        out_survival = self.model.predict_survival(x_test, times)
        out_risk = 1 - out_survival

        return out_survival, out_risk


class Cox_Regression_Wrapper(object):
    """A wrapper for Cox regression model"""
    def __init__(self):
        """
        Initialization
        """
        self.model = None

    def fit(self, train_set, val_set):
        """
        Fit the model
        :param train_set: training set
        :param val_set: validation set
        :return:
        """
        models = []
        train_df = pd.concat(train_set, axis=1)
        test_df = pd.concat(val_set, axis=1)
        whole_df = pd.concat([train_df, test_df], axis=0)

        model = CoxPHFitter()

        model.fit(whole_df,
                  duration_col="time",
                  event_col="event")


        # Extract the model
        self.model = model

    def predict(self, test_set, times):
        """
        Predict survival probability and risk
        :param test_set: test set
        :param times: times to predict
        :return: survival probability, risk
        """
        x_test, t_test, e_test = test_set
        survival_curves = self.model.predict_survival_function(x_test)
        out_survival = survival_curves.loc[times]

        out_risk = 1 - out_survival

        return out_survival, out_risk


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


def plot_avg_survival_curve(df: pd.DataFrame,
                                group_index: np.ndarray,
                                model,
                                covariates: list,
                                treatment: str,
                                figsize: tuple=(10, 10)):
    """
    plot the average survival curve for the selected model based on the group
    :param df: dataframe of all data
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
        df_group = df[group_index == i]
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
        plt.title(f"Survival Curves (Treated vs Untreated) for group {i}")
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

