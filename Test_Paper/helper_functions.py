import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
import json
import os
import pickle


from scipy.integrate import trapz
from scipy.stats import expon, uniform
from scipy.optimize import root_scalar

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
from lifelines import KaplanMeierFitter

from auton_survival.preprocessing import Preprocessor
from auton_survival.models.dsm import DeepSurvivalMachines
from auton_survival.models.dcm import DeepCoxMixtures
from auton_survival.models.dcm.dcm_utilities import test_step



class Model_Wrapper(object):
    """
    Base class for model wrappers
    Must rewrite fit method
    """
    def __init__(self,
                 params_grid=None,
                 use_treatment=True):
        """
        Initialization
        :param params_grid: parameters grid for model
        :param use_treatment: whether to use treatment during fitting
        """
        self.params_grid = params_grid
        self.use_treatment = use_treatment

        self.model = None

        self.fitted = False

        self.model_name = "Base Model"

    def predict(self, x_test, times):
        """
        Predict survival probability and risk
        :param x_test: test set
        :param times: times to predict
        :return: survival probability, risk
        """
        if self.fitted:
            times = list(times)

            x_test = remove_treatment(self.use_treatment, x_test)

            out_survival = self.model.predict_survival(x_test, times)
            out_risk = 1 - out_survival

            return out_survival, out_risk

        else:

            print("Model not fitted")
            return


class DSM_Wrapper(Model_Wrapper):
    """
    A wrapper for DSM model
    """
    def __init__(self, params_grid,
                 use_treatment):
        """
        Initialization
        :param params_grid: parameters grid for DSM
        """
        super(DSM_Wrapper, self).__init__(params_grid, use_treatment)

        self.model_name = "DSM Model"

    def fit(self, train_set,
            val_set=None):
        """
        Fit the model
        :param train_set: training set
        :param val_set: validation set
        :return:
        """
        models = []
        x_train, t_train, e_train = train_set
        if val_set is not None:
            pass
        else:
            val_set = train_set
        x_val, t_val, e_val = val_set
        for param in ParameterGrid(self.params_grid):
            model = DeepSurvivalMachines(k=param["k"], distribution=param["distribution"], layers=param["layers"])
            model.fit(x_train, t_train, e_train, iters=param["iters"], learning_rate=param["learning_rate"])
            nll = model.compute_nll(x_val, t_val, e_val)
            models.append((nll, model))

        # Extract the model
        self.model = min(models, key=lambda x: x[0])[1]
        self.fitted = True


class DCM_Wrapper(Model_Wrapper):
    """
    A wrapper for DCM model
    """
    def __init__(self,
                 params_grid,
                 use_treatment):
        """
        Initialization
        :param params_grid: parameters grid for DCM
        :param use_treatment: whether to use treatment during fitting
        """
        super(DCM_Wrapper, self).__init__(params_grid,
                                          use_treatment)
        self.model_name = "DCM Model"

    def fit(self, train_set, val_set=None):
        """
        Fit the model
        :param train_set: training set
        :param val_set: validation set
        :return:
        """
        def dataframe_to_tensor(data):
            """Function that converts a pandas dataframe into a tensor"""
            if isinstance(data, (pd.Series, pd.DataFrame)):
                return torch.from_numpy(data.to_numpy()).float()
            return torch.from_numpy(data.to_numpy()).float()

        x_train, t_train, e_train = train_set

        x_train = remove_treatment(self.use_treatment, x_train)

        if val_set is None:
            # use train set as the validation set
            x_val, t_val, e_val = x_train, t_train, e_train
        else:
            x_val, t_val, e_val = val_set

            x_val = remove_treatment(self.use_treatment, x_val)

        x_val_tensor = dataframe_to_tensor(x_val)
        t_val_tensor = dataframe_to_tensor(t_val)
        e_val_tensor = dataframe_to_tensor(e_val)

        models = []
        for param in ParameterGrid(self.params_grid):
            model = DeepCoxMixtures(k=param["k"],
                                    layers=param["layers"],
                                    gamma=param["gamma"],
                                    smoothing_factor=param["smoothing_factor"],
                                    use_activation=param["use_activation"])
            model.fit(x_train, t_train, e_train,
                      iters=param["iters"],
                      learning_rate=param["learning_rate"],
                      batch_size=param["batch_size"],
                      optimizer=param["optimizer"],
                      )
            breslow_splines = model.torch_model[1]
            val_result = test_step(model.torch_model[0], x_val_tensor, t_val_tensor, e_val_tensor, breslow_splines)
            models.append([[val_result, model]])

        best_model = min(models)
        self.model = best_model[0][1]
        self.loss = best_model[0][0]
        self.fitted = True

    def get_loss(self):
        """
        Return the validation loss of the model
        """
        return self.loss

class Cox_Regression_Wrapper(Model_Wrapper):
    """A wrapper for Cox regression model"""
    def __init__(self, params_grid,
                 use_treatment):
        """
        Initialization
        :param params_grid: parameters grid for Cox Regression
        """
        super(Cox_Regression_Wrapper, self).__init__(params_grid, use_treatment)
        self.model_name = "Cox Regression Model"

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

        for param in ParameterGrid(self.params_grid):
            model = CoxPHFitter(penalizer=param["penalizer"])

            model.fit(train_df,
                      duration_col="time",
                      event_col="event")

            models.append(model)

        # Extract the model
        self.model = models[0]
        self.fitted = True

    def predict(self, x_test, times):
        """
        Rewrite predict function
        :param x_test: test set
        :param times: times to predict
        :return: survival probability, risk
        """

        surv_curves = self.model.predict_survival_function(x_test)

        # Transpose to align time along columns for interpolation
        interpolated = surv_curves.T.interpolate(method="index", axis=1)

        # Interpolate to custom times
        survival_at_times = interpolated.reindex(columns=times, method=None).interpolate(method="values", axis=1)

        out_risk = 1 - survival_at_times

        return survival_at_times, out_risk


class RSF_Wrapper(Model_Wrapper):
    """A wrapper for Random Survival Forest model"""
    def __init__(self, params_grid, use_treatment):
        super(RSF_Wrapper, self).__init__(params_grid, use_treatment)
        self.model_name = "RSF Model"

    def fit(self, train_set, val_set=None):
        x_train, t_train, e_train = train_set
        e_train = np.asarray(e_train, dtype=bool)  # Convert to boolean
        y_train = Surv.from_arrays(event=e_train, time=t_train)
        models = []
        for param in ParameterGrid(self.params_grid):
            model = RandomSurvivalForest(n_estimators=param["n_estimators"],
                                         max_depth=param["max_depth"],
                                         random_state=42)
            model.fit(x_train, y_train)
            models.append(model)  # No validation NLL; use first or add CV
        self.model = models[0]
        self.fitted = True

    def predict(self, x_test, times):
        surv_func = self.model.predict_survival_function(x_test)
        survival_at_times = np.array([fn(times) for fn in surv_func])
        out_risk = 1 - survival_at_times
        return survival_at_times, out_risk


class AFT_Wrapper(Model_Wrapper):
    """A wrapper for Accelerated Failure Time model"""
    def __init__(self, params_grid, use_treatment):
        super(AFT_Wrapper, self).__init__(params_grid, use_treatment)
        self.model_name = "AFT Model"

    def fit(self, train_set, val_set=None):
        train_df = pd.DataFrame({"time": train_set[1], "event": train_set[2]})
        train_df = pd.concat([train_df, pd.DataFrame(train_set[0])], axis=1)
        models = []
        for param in ParameterGrid(self.params_grid):
            model = WeibullAFTFitter(penalizer=param["penalizer"])
            model.fit(train_df,
                      duration_col="time",
                      event_col="event",
                      robust=True)
            models.append(model)
        self.model = models[0]
        self.fitted = True

    def predict(self, x_test, times):
        surv_curves = self.model.predict_survival_function(pd.DataFrame(x_test))
        interpolated = surv_curves.T.interpolate(method="index", axis=1)
        survival_at_times = interpolated.reindex(columns=times, method=None).interpolate(method="values", axis=1)
        out_risk = 1 - survival_at_times
        return survival_at_times, out_risk


class AFT_IPTW_Wrapper(AFT_Wrapper):
    """A wrapper for Accelerated Failure Time with IPTW model"""
    def __init__(self, params_grid, covariates, use_treatment, treatment="treatment"):
        super(AFT_IPTW_Wrapper, self).__init__(params_grid, use_treatment)
        self.model_name = "AFT IPTW Model"
        self.covariates = covariates
        self.treatment = treatment

    def fit(self, train_set, val_set=None):
        x_train, t_train, e_train = train_set
        df_train = pd.DataFrame(x_train, columns=self.covariates + [self.treatment])
        df_train["time"] = t_train
        df_train["event"] = e_train
        df_ps = compute_PS_and_IPTW(df_train, self.covariates, self.treatment)
        train_df_weighted = df_ps[self.covariates + [self.treatment, "time", "event", "iptw_weight"]]
        models = []
        for param in ParameterGrid(self.params_grid):
            model = WeibullAFTFitter(penalizer=param["penalizer"])
            model.fit(train_df_weighted,
                      duration_col="time",
                      event_col="event",
                      weights_col="iptw_weight",
                      robust=True)
            models.append(model)
        self.model = models[0]
        self.fitted = True


class RSF_IPTW_Wrapper(RSF_Wrapper):
    """A wrapper for Random Survival Forest with IPTW model"""
    def __init__(self, params_grid, covariates, use_treatment, treatment="treatment"):
        super(RSF_IPTW_Wrapper, self).__init__(params_grid, use_treatment)
        self.model_name = "RSF IPTW Model"
        self.covariates = covariates
        self.treatment = treatment

    def fit(self, train_set, val_set=None):
        x_train, t_train, e_train = train_set
        e_train = np.asarray(e_train, dtype=bool)
        df_train = pd.DataFrame(x_train, columns=self.covariates + [self.treatment])
        df_train["time"] = t_train
        df_train["event"] = e_train
        df_ps = compute_PS_and_IPTW(df_train, self.covariates, self.treatment)
        weights = df_ps["iptw_weight"]

        y_train = Surv.from_arrays(event=e_train, time=t_train)
        models = []
        for param in ParameterGrid(self.params_grid):
            model = RandomSurvivalForest(n_estimators=param["n_estimators"], max_depth=param["max_depth"], random_state=42)
            model.fit(x_train, y_train, sample_weight=weights.values)
            models.append(model)
        self.model = models[0]
        self.fitted = True


class Cox_IPTW_Wrapper(Cox_Regression_Wrapper):
    """
    A wrapper for COX with IPTW model
    """
    def __init__(self,
                 params_grid,
                 covariates,
                 use_treatment,
                 treatment="treatment"):
        """
        Initialization
        :param params_grid: parameters grid for Cox IPTW
        :param covariates: covariates for computing IPTW
        :param treatment: name of the treatment variable
        """
        super(Cox_IPTW_Wrapper, self).__init__(params_grid, use_treatment)
        self.model_name = "Cox IPTW Model"
        self.covariates = covariates
        self.treatment = treatment

    def fit(self, train_set, val_set=None):
        x_train, t_train, e_train = train_set
        df_train = pd.DataFrame(x_train, columns=self.covariates + [self.treatment])
        df_train["time"] = t_train
        df_train["event"] = e_train
        df_ps = compute_PS_and_IPTW(df_train, self.covariates, self.treatment)
        train_df_weighted = df_ps[self.covariates + [self.treatment, "time", "event", "iptw_weight"]]
        models = []
        for param in ParameterGrid(self.params_grid):
            model = CoxPHFitter(penalizer=param["penalizer"])
            model.fit(train_df_weighted, duration_col="time", event_col="event", weights_col="iptw_weight", robust=True)
            models.append(model)
        self.model = models[0]
        self.fitted = True


class DSM_IPTW_Wrapper(DSM_Wrapper):
    """
    A wrapper for DSM IPTW model
    """
    def __init__(self,
                 covariates,
                 params_grid,
                 use_treatment,
                 treatment="treatment"):
        """
        Initialization
        :param params_grid: parameters grid for DSM IPTW
        :param covariates: covariates for computing IPTW
        :param treatment: name of the treatment variable
        """
        super(DSM_IPTW_Wrapper, self).__init__(params_grid, use_treatment)
        self.covariates = covariates
        self.model_name = "DSM IPTW Model"
        self.treatment = treatment

    def fit(self,
            train_set,
            val_set=None):
        """
        Fit the model
        :param train_set: training set
        :param val_set: validation set
        :return:
        """
        models = []
        x_train, t_train, e_train = train_set
        df_train = pd.DataFrame(x_train, columns=self.covariates + [self.treatment])
        df_train["time"] = t_train
        df_train["event"] = e_train
        df_ps = compute_PS_and_IPTW(df_train, self.covariates, self.treatment)

        # Resample with replacement proportional to normalized weights
        weights = df_ps["iptw_weight"]
        probs = weights / weights.sum()
        resampled_idx = np.random.choice(len(df_ps), size=len(df_ps), replace=True, p=probs)
        resampled_df = df_ps.iloc[resampled_idx]

        # Extract resampled inputs for DSM (drop extra columns)
        new_x = resampled_df.drop(["time", "event", "propensity_score", "iptw_weight"], axis=1).values
        new_t = resampled_df["time"].values
        new_e = resampled_df["event"].values

        # Use validation set as-is or resampled if desired; here keep original for simplicity
        if val_set is None:
            val_set = (new_x, new_t, new_e)
        super().fit((new_x, new_t, new_e), val_set)


class DCM_IPTW_Wrapper(DCM_Wrapper):
    """
    A wrapper for DCM IPTW model
    """
    def __init__(self,
                 covariates,
                 params_grid,
                 use_treatment,
                 treatment="treatment"):
        """
        Initialization
        :param params_grid: parameters grid for DCM IPTW
        :param covariates: covariates for computing IPTW
        :param treatment: name of the treatment variable
        """
        super(DCM_IPTW_Wrapper, self).__init__(params_grid, use_treatment)
        self.covariates = covariates
        self.model_name = "DCM IPTW Model"
        self.treatment = treatment

    def fit(self, train_set, val_set=None):
        """
        Fit the model
        :param train_set: training set
        :param val_set: validation set
        :return:
        """
        x_train, t_train, e_train = train_set

        df_train = pd.DataFrame(x_train, columns=self.covariates + [self.treatment])
        df_train["time"] = t_train
        df_train["event"] = e_train
        df_ps = compute_PS_and_IPTW(df_train, self.covariates, self.treatment)

        # Resample with replacement proportional to normalized weights
        weights = df_ps["iptw_weight"]
        probs = weights / weights.sum()
        resampled_idx = np.random.choice(len(df_ps), size=len(df_ps), replace=True, p=probs)
        resampled_df = df_ps.iloc[resampled_idx]

        # Extract resampled inputs for DSM (drop extra columns)
        new_x = resampled_df.drop(["time", "event", "propensity_score", "iptw_weight"], axis=1).values
        new_t = resampled_df["time"].values
        new_e = resampled_df["event"].values

        new_x = pd.DataFrame(new_x, columns=self.covariates + [self.treatment])
        new_t = pd.DataFrame(new_t, columns=["time"]).squeeze()
        new_e = pd.DataFrame(new_e, columns=["event"]).squeeze()

        # Use validation set as-is or resampled if desired; here keep original for simplicity
        if val_set is None:
            val_set = (new_x, new_t, new_e)
        super().fit((new_x, new_t, new_e), val_set)

model_dict = {
    "DSM Model": DSM_Wrapper,
    "DCM Model": DCM_Wrapper,
    "Cox Regression Model": Cox_Regression_Wrapper,
    "RSF Model": RSF_Wrapper,
    "AFT Model": AFT_Wrapper,
    "Cox IPTW Model": Cox_IPTW_Wrapper,
    "RSF IPTW Model": RSF_IPTW_Wrapper,
    "AFT IPTW Model": AFT_IPTW_Wrapper,
    "DSM IPTW Model": DSM_IPTW_Wrapper,
    "DCM IPTW Model": DCM_IPTW_Wrapper,
}

def remove_treatment(include_treatment, x_df):
    """
    Removes treatment variable according to requirement
    :param include_treatment: whether include treatment variable
    :param x_df: the df may include treatment variable
    :return:
    """
    if include_treatment:
        assert "treatment" in x_df.columns
    elif "treatment" in x_df.columns:
        try:
            x_df = x_df.drop("treatment", axis=1, inplace=False)
        except:
            pass

    return x_df


def processing_data(path: str="C:\\Users\\lee39\\OneDrive\\Desktop\\final_merged_dataset.csv",
                    time_from_admission=True) -> pd.DataFrame:
    """
    processed eICU data
    :param path: final_merged_dataset.csv path
    :param time_from_admission: use time start from admission
    :return: processed eICU data frame
    """
    eICU_data = pd.read_csv(path)
    cleaned_df = eICU_data.copy()

    # drop ids
    cleaned_df.drop(["patientunitstayid", "hospitalid"], inplace=True, axis=1)

    # change the name of time, event and treatment variables

    if time_from_admission:
        time = cleaned_df["unitdischargeoffset"] - cleaned_df["hospitaladmitoffset"]

    else:
        time = cleaned_df["unitdischargeoffset"]

    cleaned_df["time"] = time

    cleaned_df = cleaned_df[cleaned_df["time"] > 0]

    cleaned_df.drop(["unitdischargeoffset", "hospitaladmitoffset", "hospitaldischargeoffset"],
                    inplace=True, axis=1)

    cleaned_df.rename(columns={"unitdischargestatus": "event",
                               "has_Vasopressor": "treatment"},
                      inplace=True)

    cleaned_df["event"] = (cleaned_df["event"] == "Expired").astype(int)

    return cleaned_df.reset_index(drop=True)


def clustering_data(df: pd.DataFrame,
                    n_clusters,
                    features: list=None,
                    random_seed=42) -> pd.DataFrame:
    """
    Clustering data according to the designated features
    :param df: data frame
    :param n_clusters: number of clusters
    :param features: features used to cluster
    :return: new data frame with cluster number
    """
    if features is None:
        """Using all features except time and event"""

        features = list(df.columns)
        if "time" in features:
            features.remove("time")
        if "event" in features:
            features.remove("event")

    X = df[features]

    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_seed,
                    n_init="auto")
    df["cluster index"] = kmeans.fit_predict(X)

    return df


def processing_data_2_DCM(df: pd.DataFrame,
                          categorical_features_list: list,
                          train_test_val_size=(0.7, 0.2, 0.1),
                          random_seed=42,
                          clustering=False,
                          n_clusters: int=1,
                          clustering_features: list=None
                          ):
    """
    further processed the data to meet the requirements of DCM
    :param df: processed eICU data frame
    :param categorical_features_list: categorical features list
    :param train_test_val_size: default train, test, validation proportion
    :param random_seed: default random seed
    :param clustering: clustering index, default not to clustering
    :param: n_clusters: number of clusters
    :param clustering_features: features used to cluster
    :return: preprocessor_provided: provided preprocessor
    :return:
    """
    time = df["time"]
    event = df["event"]
    outcomes = pd.concat([time, event], axis=1)
    features = df.drop(["time", "event"], axis=1)  # include the treatment

    numerical_features_list = [col for col in features.columns if
                               col not in categorical_features_list and col != "treatment"]

    # process data for DCM
    processed_features = Preprocessor().fit_transform(features,
                                                      cat_feats=categorical_features_list,
                                                      num_feats=numerical_features_list)
    processed_features.rename(columns={"treatment_1": "treatment"}, inplace=True)

    # train test and val split
    if train_test_val_size is None:
        X_train = processed_features
        y_train = outcomes

        X_val, y_val = X_train, y_train
        X_test, y_test = X_train, y_train

    else:
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            processed_features, outcomes,
            test_size=1 - train_test_val_size[0],
            random_state=random_seed
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test,
            test_size=train_test_val_size[1] / (train_test_val_size[2] + train_test_val_size[1]),
            random_state=random_seed
        )

    t_train, e_train = y_train["time"], y_train["event"]
    t_val, e_val = y_val["time"], y_val["event"]
    t_test, e_test = y_test["time"], y_test["event"]

    if clustering:
        if n_clusters <= 1:
            print("WARNING: n_clusters should be greater than 1")
        else:

            clustering_df = pd.concat([X_train, X_val, X_test], axis=0)

            clustering_df = clustering_data(df=clustering_df,
                                 n_clusters=n_clusters,
                                 features=clustering_features,
                                 random_seed=random_seed)

            cluster_index = clustering_df["cluster index"]

            # add clustering index to the data
            idx_train = X_train.index
            idx_val = X_val.index
            idx_test = X_test.index
            X_train = pd.concat([X_train, cluster_index[idx_train]], axis=1)
            X_val = pd.concat([X_val, cluster_index[idx_val]], axis=1)
            X_test = pd.concat([X_test, cluster_index[idx_test]], axis=1)

            t_train = pd.concat([t_train, cluster_index[idx_train]], axis=1)
            t_val = pd.concat([t_val, cluster_index[idx_val]], axis=1)
            t_test = pd.concat([t_test, cluster_index[idx_test]], axis=1)

            e_train = pd.concat([e_train, cluster_index[idx_train]], axis=1)
            e_val = pd.concat([e_val, cluster_index[idx_val]], axis=1)
            e_test = pd.concat([e_test, cluster_index[idx_test]], axis=1)

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
    :param covariates: list of covariates used to compute IPTW
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
                            model_wrapper,
                            covariates: list,
                            treatment: str,
                            figsize: tuple=(10, 10),
                            num_time: int = 100,
                            save_figure: bool = False,
                            save_name: str = None,
                            given_title: str = None,
                            show_figure: bool = True,
                            ):
    """
    plot the average survival curve for the selected model based on the group
    :param df: dataframe of all data
    :param group_index: index of the group
    :param model_wrapper: the DCM model
    :param covariates: list of covariates exclude the treatment
    :param treatment: name of the treatment column
    :param figsize: default size of the figure
    :param num_time: number of the time points in the plot
    :param save_figure: whether to save the figure
    :param save_name: name of the saved figure
    :param given_title: title of the figure
    :param show_figure: whether to show the figure
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
        time_grid = list(np.linspace(0, np.max(df_group["time"]), num_time))

        X_treated = df_treated[covariates]
        X_control = df_untreated[covariates]

        S1, _  = model_wrapper.predict(X_treated, time_grid)
        S0, _ = model_wrapper.predict(X_control, time_grid)

        S1_avg = np.nanmean(S1, axis=0)
        S0_avg = np.nanmean(S0, axis=0)

        S1_avg[S1_avg >= 1] = 1
        S0_avg[S0_avg >= 1] = 1

        # remove nan
        s1_mask = ~np.isnan(S1_avg)
        s0_mask = ~np.isnan(S0_avg)

        # Use the mask to select only the non-NaN values from the array
        S1_avg = S1_avg[s1_mask]
        S0_avg = S0_avg[s0_mask]

        time_grid_array = np.array(time_grid)

        ax = plt.subplot(num_group, 1, i + 1)

        plt.plot(time_grid_array[s1_mask], S1_avg, label="Treated (model-adjusted)", color="blue")
        plt.plot(time_grid_array[s0_mask], S0_avg, label="Untreated (model-adjusted)", color="red")
        plt.xlabel("Time")
        plt.ylabel("Average Survival Probability")

        # Final plot settings
        if given_title is not None:
            title = given_title
        else:
            title = f"Average Survival Probability for group {i}"
        ax.title.set_text(title)
        plt.xlabel("Time Since ICU Admission (minutes)")
        plt.ylabel("Survival Probability")

        plt.legend()
        plt.grid(True)

        rmst1 = trapz(S1[:, s1_mask], time_grid_array[s1_mask])
        rmst0 = trapz(S0[:, s0_mask], time_grid_array[s0_mask])

        # Causal effect estimate using AUC
        causal_effects.append(np.nanmean(rmst1) - np.nanmean(rmst0))

    plt.subplots_adjust(wspace=0.5, hspace=2)
    plt.tight_layout()
    plt.legend()

    if save_figure:
        if save_name is None:
            warnings.warn("No save_name provided")
            save_name = f"survival_curve_{num_time}.png"
        plt.savefig(save_name)

    if show_figure:
        plt.show()
    else:
        plt.close("all")

    return causal_effects


def clustering_fit_model(model_name: str,
                         params_grid,
                         train_set: list,
                         val_set: list,
                         ):
    """
    Fit the models based one different clusters
    :param model_name: the name of the model
    :param params_grid: parameters grid for model
    :param train_set: list of training data
    :param val_set: list of validation data
    :return: list of model wrappers
    """
    if model_name not in model_dict.keys():
        raise KeyError(f"Model {model_name} is not available")

    model_wrapper = model_dict[model_name]

    if "cluster index" not in train_set[0] or "cluster index" not in val_set[0]:
        raise KeyError(f"Training or validation data frame has no clusters, please run 'clustering_data' first")

    # extract the number of clusters
    n_clusters = np.unique([train_set[0]["cluster index"].unique(),
                            val_set[0]["cluster index"].unique()])

    model_wrapper_list = []

    for i in range(0, len(n_clusters)):

        new_train_set = [data[train_set[0]["cluster index"] == i].drop("cluster index", axis=1).squeeze() for data in train_set]
        new_val_set = [data[val_set[0]["cluster index"] == i].drop("cluster index", axis=1).squeeze() for data in val_set]

        model = model_wrapper(params_grid=params_grid)
        model.fit(train_set=new_train_set,
                  val_set=new_val_set)
        model_wrapper_list.append(model)

    return model_wrapper_list


def generate_simulated_data(num_sample: int=1000,
                            num_group: int=1,
                            num_covariates=None,
                            censor_rate: float=0.3,
                            random_seed: int=None,
                            **kwargs
                            ):
    """
    Generate simulated survival data with latent groups, confounding, non-linear effects, and treatment.
    :param num_sample: Number of samples.
    :param num_group: Number of latent groups.
    :param num_covariates: Tuple (n_linear, n_non_linear, n_categorical).
    :param censor_rate: Target censoring rate (0-1).
    :param random_seed: Seed for reproducibility.
    :param kwargs: Overrides for group-specific parameters (means_k, p_k, prop_betas, haz_betas, beta0_k, beta_A_k, interact_beta_k).
    :return: df (DataFrame), save_dict (params), lambda0, lambda1, Z (true groups), actual_censor_rate, diagnostics.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if num_group < 1:
        raise ValueError("num_group must be at least 1")

    pi = np.full(num_group, 1.0 / num_group)  # Equal group probabilities
    Z = np.random.choice(range(num_group), num_sample, p=pi)

    # Check group sizes
    group_sizes = np.bincount(Z)

    if num_covariates is None:
        num_covariates = (2, 5, 2)
    n_linear, n_non_linear, n_categorical = num_covariates
    N = n_linear + n_non_linear + n_categorical

    covariates = {}

    # Group-specific parameters
    save_dict = {}

    start_lower_bound = num_group - 1

    means_k = kwargs.get("means_k",
                         np.array([np.random.uniform(-start_lower_bound + g, g, 1) for g in range(num_group)]).squeeze()
                         )
    save_dict["means_k"] = means_k

    p_k = kwargs.get("p_k",
                     np.array([ 1 / (num_group + 1) * (g + 1) for g in range(num_group)]).squeeze()
                     )
    save_dict["p_k"] = p_k

    prop_betas = kwargs.get("prop_betas",
                            np.array([np.random.uniform(-start_lower_bound + g, g, N) for g in range(num_group)])
                            )
    save_dict["prop_betas"] = prop_betas

    haz_betas = kwargs.get("haz_betas",
                           np.array([np.random.uniform(-start_lower_bound + g, g, N) for g in range(num_group)])
                           )
    save_dict["haz_betas"] = haz_betas

    beta0_k = kwargs.get("beta0_k",
                         np.array([np.random.uniform(-start_lower_bound + g, g, 1) for g in range(num_group)]).squeeze()
                         )
    save_dict["beta0_k"] = beta0_k

    beta_A_k = kwargs.get("beta_A_k",
                          np.linspace(-5, 5, num_group)
                          )
    save_dict["beta_A_k"] = beta_A_k

    interact_beta_k = kwargs.get("interact_beta_k",
                                 np.array([np.random.uniform(start_lower_bound + g, g, 1) for g in range(num_group)]).squeeze()
                                 )
    save_dict["interact_beta_k"] = interact_beta_k

    # Generate covariates (vectorized)
    for i in range(1, n_linear + 1):
        covariates[f"X{i}"] = np.random.normal(means_k[Z], 1, num_sample)

    for i in range(1 + n_linear, n_non_linear + n_linear + 1):
        covariates[f"X{i}"] = np.random.uniform(0, 2 * np.pi, num_sample)

    for i in range(1, n_categorical + 1):
        covariates[f"X_c{i}"] = np.random.binomial(1, p_k[Z], num_sample)

    # Propensity score logit (non-linear effects)
    logit = -1.0 * np.ones(num_sample)  # Intercept

    for i in range(1, n_linear + 1):
        if i % 2 == 0:
            func = lambda x: x
        else:
            func = lambda x: x ** 2
        logit += prop_betas[Z, i - 1] * func(covariates[f"X{i}"])

    for i in range(1 + n_linear, n_non_linear + n_linear + 1):
        func = np.sin if i % 2 == 0 else np.cos
        logit += prop_betas[Z, i - 1] * func(covariates[f"X{i}"])

    for i in range(1, n_categorical + 1):
        logit += prop_betas[Z, i + n_non_linear + n_linear - 1] * covariates[f"X_c{i}"]

    prob = 1 / (1 + np.exp(-logit))
    treatment = np.random.binomial(1, prob, num_sample)

    # Hazard rate log_lambda (decoupled from prop_betas)
    log_lambda = beta0_k[Z].copy()

    for i in range(1, n_linear + 1):
        if i % 2 == 0:
            func = lambda x: x
        else:
            func = lambda x: x ** 2
        log_lambda += haz_betas[Z, i - 1] * func(covariates[f"X{i}"])

    for i in range(1 + n_linear, n_non_linear + n_linear + 1):
        func = np.sin if i % 2 == 0 else np.cos
        log_lambda += haz_betas[Z, i - 1] * func(covariates[f"X{i}"])

    for i in range(1, n_categorical + 1):
        log_lambda += haz_betas[Z, i + n_non_linear + n_linear - 1] * covariates[f"X_c{i}"]

    log_lambda += beta_A_k[Z] * treatment

    if N >= 2:
        log_lambda += interact_beta_k[Z] * covariates["X1"] * covariates["X2"]

    log_lambda = np.clip(log_lambda, -5, 5)

    lambda_ = np.exp(log_lambda)

    # Baselines without treatment
    log_lambda_base = log_lambda - beta_A_k[Z] * treatment
    lambda0 = np.exp(log_lambda_base)
    lambda1 = np.exp(log_lambda_base + beta_A_k[Z])

    true_time = expon.rvs(scale=1 / lambda_)

    # add noisy
    true_time = true_time + np.random.uniform(0, 1, size=true_time.size)

    n = len(true_time)
    k = int(np.round(censor_rate * n))
    # Randomly select k indices to censor
    censor_indices = np.random.choice(n, size=k, replace=False)
    observed_time = np.array(true_time)
    status = np.ones(n, dtype=int)  # Start as uncensored
    for idx in censor_indices:
        # Censor at uniform(0, t_i)
        observed_time[idx] = np.random.uniform(0, true_time[idx])
        status[idx] = 0
    actual_censor_rate = 1 - np.mean(status)

    event = status.astype(int)

    # DataFrame
    df_data = {**covariates, "treatment": treatment, "time": observed_time, "event": event}
    df = pd.DataFrame(df_data)

    # Diagnostics
    diagnostics = {
        "group_sizes": group_sizes,
        "treatment_balance_per_group": [np.mean(treatment[Z == k]) for k in range(num_group)],
        "avg_lambda_per_group": [np.mean(lambda_[Z == k]) for k in range(num_group)],
        "actual_censor_rate": actual_censor_rate
    }

    return df, save_dict, lambda0, lambda1, Z, actual_censor_rate, diagnostics


def compute_metrics(model_wrapper,
                    x_test,
                    t_test,
                    e_test,
                    times,
                    Z_test,
                    lambda0_test,
                    lambda1_test,
                    ):
    """
    Function to compute the metrics
    :param model_wrapper: wrapper contains the model
    :param x_test: test features
    :param t_test: test times
    :param e_test: test events
    :param times: time grid
    :param Z_test: latent group for test data
    :param lambda0_test: counterfactual hazards for untreated
    :param lambda1_test: counterfactual hazards for treated
    :return: dict of metrics
    """
    treated_idx = x_test["treatment"] == 1

    surv, risk = model_wrapper.predict(x_test, times)

    # replace nan with 0 or 1
    surv = np.nan_to_num(surv, nan=0.0)
    risk = np.nan_to_num(risk, nan=1.0)

    # C-index
    c_index = concordance_index_censored(e_test.astype(bool), t_test, np.mean(risk, axis=1))[0]

    # Integrated Brier Score (requires sksurv format for y)
    y_test_sksurv = np.array([(e, t) for e, t in zip(e_test, t_test)], dtype=[("event", bool), ("time", float)])
    ibs = integrated_brier_score(y_test_sksurv, y_test_sksurv, surv, times)

    # ARI (if model supports clustering, e.g., DCM; else 0)
    ari = 0.0
    if model_wrapper.model_name in ["DCM Model", "DCM IPTW Model"]:  # Adapt for actual latent group extraction
        # Pseudo-code: assume DCM has a method to get latent Z

        x_test = remove_treatment(model_wrapper.use_treatment, x_test)
        est_Z = np.argmax(model_wrapper.model.predict_latent_z(x_test), axis=1)  # Adjust based on auton_survival API
        ari = adjusted_rand_score(Z_test, est_Z)

    # ATE Bias
    true_S0 = expon.sf(times[:, np.newaxis], scale=1 / lambda0_test).mean(axis=1)
    true_S1 = expon.sf(times[:, np.newaxis], scale=1 / lambda1_test).mean(axis=1)
    true_ate_test = trapz(true_S1, times) - trapz(true_S0, times)

    # Model-estimated ATE: Split test set by treatment (assuming "treatment" in x_test columns)
    S1_est = np.mean(surv[treated_idx], axis=0) if np.any(treated_idx) else np.zeros_like(times)
    S0_est = np.mean(surv[~treated_idx], axis=0) if np.any(~treated_idx) else np.zeros_like(times)
    est_ate = trapz(S1_est, times) - trapz(S0_est, times)
    abs_bias = np.abs(est_ate - true_ate_test)
    rel_bias = abs_bias / np.abs(true_ate_test)

    metrics = {"c_index": c_index, "Integrated Brier Score": ibs, "Adjusted Random Index": ari, "abs_bias": abs_bias, "rel_bias": rel_bias}

    # Add per-group ATE bias
    unique_groups = np.unique(Z_test)
    for g in unique_groups:
        group_idx = Z_test == g
        if np.sum(group_idx) == 0:
            metrics[f'abs_bias_group_{g}'] = np.nan
            metrics[f'rel_bias_group_{g}'] = np.nan
            continue

        # True per-group
        true_S0_g = expon.sf(times[:, np.newaxis], scale=1 / lambda0_test[group_idx]).mean(axis=1)
        true_S1_g = expon.sf(times[:, np.newaxis], scale=1 / lambda1_test[group_idx]).mean(axis=1)
        true_ate_g = trapz(true_S1_g, times) - trapz(true_S0_g, times)

        # Estimated per-group (further split by treatment within group)
        treated_g = treated_idx & group_idx
        untreated_g = (~treated_idx) & group_idx
        S1_est_g = np.mean(surv[treated_g], axis=0) if np.any(treated_g) else np.zeros_like(times)
        S0_est_g = np.mean(surv[untreated_g], axis=0) if np.any(untreated_g) else np.zeros_like(times)
        est_ate_g = trapz(S1_est_g, times) - trapz(S0_est_g, times)

        abs_bias_g = np.abs(est_ate_g - true_ate_g)
        metrics[f'abs_bias_group_{g}'] = abs_bias_g
        metrics[f'rel_bias_group_{g}'] = abs_bias_g / np.abs(true_ate_g)

    return metrics


def results_to_dataframe(agg_results):
    """
    Convert aggregated simulation results to a nicely formatted pandas DataFrame.
    @param agg_results: aggregated simulation results
    @return: pandas: A formatted DataFrame with MultiIndex and rounded metric values.
    """
    data = []
    for num_group, models in agg_results.items():
        for model, metrics in models.items():
            row = {'num_group': num_group, 'model': model}
            row.update(metrics)
            data.append(row)

    df = pd.DataFrame(data)
    df = df.set_index(['num_group', 'model'])

    # Sort: by num_group ascending, then by abs_bias descending within groups
    df = df.sort_values(by=['num_group'], ascending=[True])

    return df


def get_next_run_index(base_dir="simulations"):
    """
    Finds the next available run index by scanning existing files in base_dir.
    Assumes files named like 'run_XXX_results.pkl'.
    Returns: int, the next index (starting from 1).
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    existing_runs = [f for f in os.listdir(base_dir) if f.startswith("run_") and f.endswith("_results.pkl")]
    if not existing_runs:
        return 1
    indices = [int(f.split("_")[1]) for f in existing_runs]
    return max(indices) + 1


def save_simulation_results(agg_results, params, base_dir="simulations"):
    """
    Saves aggregated results, SEs, and params with incremental index.
    :param agg_results: dict of means
    :param se_results: dict of standard errors
    :param params: dict of experiment settings
    :param base_dir: str, directory to save files
    """
    run_index = get_next_run_index(base_dir)
    run_prefix = f"run_{run_index:03d}"  # e.g., run_001

    # Save results as pickle
    results_path = os.path.join(base_dir, f"{run_prefix}_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(agg_results, f)

    # Save settings as JSON
    settings_path = os.path.join(base_dir, f"{run_prefix}_settings.json")
    with open(settings_path, 'w') as f:
        json.dump(params, f, indent=4)

    print(f"Simulation saved: Results at {results_path}, Settings at {settings_path}")


def plot_km_curve(df, Z):
    unique_groups = np.unique(Z)
    num_groups = len(unique_groups)

    fig, axs = plt.subplots(num_groups, 1, figsize=(8, 6 * num_groups), sharex=True)

    if num_groups == 1:
        axs = [axs]  # Ensure axs is iterable for single group

    for i, g in enumerate(unique_groups):
        group_idx = Z == g
        df_g = df[group_idx]

        ax = axs[i]

        times = df_g['time']
        max_time = np.max(times)

        for treatment_group, group_df in df_g.groupby('treatment'):
            kmf = KaplanMeierFitter()
            kmf.fit(durations=group_df['time'], event_observed=group_df['event'], label=f'Treatment={treatment_group}')
            kmf.plot_survival_function(ax=ax)

        ax.set_xlim([0, max_time * 1.05])

        ax.set_title(f'Kaplan-Meier Survival Curves by Treatment for Latent Group {g}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df, save_dict, lambda0, lambda1, Z, actual_censor_rate, diagnostics = generate_simulated_data(
        num_sample=5000,
        num_group=3,
        censor_rate=0.3,
        random_seed=0
    )

    plot_km_curve(df, Z)

"""    # Identify features
    categorical_features_list = [col for col in df.columns if col.startswith("X_c")] + ["treatment"]

    # Process data
    processed = processing_data_2_DCM(df,
                                      categorical_features_list,
                                      train_test_val_size=None)

    (X_train, X_val, _), (t_train, t_val, _), (e_train, e_val, _), cat_feats, num_feats = processed

    covariates = [col for col in X_train.columns if col not in ["time", "event", "treatment"]]

    model_wrapper = RSF_Wrapper({"n_estimators": [100], "max_depth": [5]},
                            use_treatment=True)

    model_wrapper.fit((X_train, t_train, e_train), (X_val, t_val, e_val))

    min_time = np.min(t_train)
    max_time = np.max(t_train)
    times = np.linspace(min_time, max_time, 100, endpoint=False)

    metrics = compute_metrics(model_wrapper,
                              X_train, t_train, e_train,
                              times,
                              Z,
                              lambda0,
                              lambda1)

    print(metrics)"""