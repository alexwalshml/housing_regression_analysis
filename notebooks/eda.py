import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import correlate
from statsmodels.graphics.gofplots import qqplot


def mode(arr, dtype):
    # return most common non-null values
    unique, counts = np.unique(arr.astype(str), return_counts=True)

    max_count = np.max(counts)

    modes = [
        (np.array([u]).astype(dtype)[0], c)
        for u, c in zip(unique, counts)
        if c == max_count
    ]

    if len(modes) == 1:
        return modes[0]
    else:
        return modes


def descriptive_statistics(df):
    descriptive_df = pd.DataFrame(
        columns=[
            "count",  # count of non null values in the column
            "null_count",  # count of null values
            "mean",  # mean of data
            "median",  # meadian of data
            "mode",  # mode of data, does not included nan
            "unique_count",  # number of unique values in the data, does not include nan
            "std",  # standard deviation of data
            "variance",  # variance of data
            "minimum",  # minimum value in data
            "z_min",  # z score of minimum value
            "maximum",  # maximum value
            "z_max",  # z score of maximum
            "5%",  # 5th percentile
            "25%",  # 25th percentile
            "50%",  # 50th percentile
            "75%",  # 75th percentile
            "95%",  # 95th percentile
            "standard_error",  # standard error of the mean
            "variation",  # variation of the data
            "skew",  # skew of the data
            "kurtosis",  # kurtosis of the data
            "autocorrelation_mean",  # mean of the column autocorrelation
            "autocorrelation_time",  # autocorrelation time of the data
            "outlier_count",  # number of entries with an absolute value z score greater than 3
            "entropy",  # entropy of the data assuming an underlying normal distribution
        ]
    )

    for col, values in df.items():
        dtype = df[col].dtype
        null_count = values.isna().sum()
        count = len(values) - null_count
        nu = values.nunique()
        values = np.array(values.dropna())
        mo = mode(values, dtype)
        if dtype not in [object, pd.Categorical]:
            sigma = np.std(values)
            var = np.var(values)
            mu = np.mean(values)
            me = np.median(values)
            minimum = np.min(values)
            maximum = np.max(values)
            z = (values - mu) / sigma
            min_z = np.min(z)
            max_z = np.max(z)
            percentile_5 = np.quantile(values, 0.05)
            percentile_25 = np.quantile(values, 0.25)
            percentile_50 = np.quantile(values, 0.50)
            percentile_75 = np.quantile(values, 0.75)
            percentile_95 = np.quantile(values, 0.95)
            skew = stats.skew(values, nan_policy="omit")
            kurtosis = stats.kurtosis(values, nan_policy="omit")
            autocorrelation = correlate(z, z, mode="full")
            autocorrelation_mean = np.mean(autocorrelation)
            autocorrelation_time = 1 + np.sum(autocorrelation)
            outlier_count = np.sum(np.abs(z) > 3)
            standard_error = sigma / np.sqrt(count)
            variation = stats.variation(values, nan_policy="omit")
            p = (1.0 / (np.sqrt(2.0 * np.pi * sigma**2))) * np.exp(
                -((values - mu) ** 2) / (2.0 * sigma**2)
            )
            entropy = -np.sum(p * np.log(p))
            row = [
                count,
                null_count,
                mu,
                me,
                mo,
                nu,
                sigma,
                var,
                minimum,
                min_z,
                maximum,
                max_z,
                percentile_5,
                percentile_25,
                percentile_50,
                percentile_75,
                percentile_95,
                standard_error,
                variation,
                skew,
                kurtosis,
                autocorrelation_mean,
                autocorrelation_time,
                outlier_count,
                entropy,
            ]
        else:
            row = [
                count,
                null_count,
                np.nan,
                np.nan,
                mo,
                nu,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]

        descriptive_df.loc[col] = row

    return descriptive_df


def name_sanitizer(name):
    return name.replace(" ", "_").replace("/", "_").lower()


def numeric_data_visualizer(numerics, target_values, path, show=True, cols="all"):
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    n_total = len(numerics)
    if cols != "all":
        numerics = pd.DataFrame(numerics[cols])
    for col, values in numerics.items():
        n_null = values.isna().sum()
        n_exist = n_total - n_null

        n_str = f"N: {n_exist}\nNull Count: {n_null}"

        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(27, 8)
        
        # plot distribution
        sns.histplot(values, kde=True, ax=axs[0])
        axs[0].set_title(f"{col} Distribution")
        axs[0].set_xlabel(col)
        props = dict(boxstyle="square", facecolor="white", alpha=0.5)
        axs[0].text(0.05, 0.95, n_str, transform=axs[0].transAxes, bbox=props)
        # plot target scatterplot
        sns.scatterplot(x=values, y=target_values, ax=axs[1])
        axs[1].set_title(f"{col} Spread")
        axs[1].set_xlabel(col)
        
        # plot qunatile-quantile distribution

        values_std = (values - values.mean()) / values.std()

        qqplot(values_std, line="q", ax=axs[2])
        xlim, ylim = axs[2].get_xlim(), axs[2].get_ylim()
        identity = [-100, 100]
        axs[2].plot(identity, identity, color="black", label="Identity")
        axs[2].set_xlim(xlim)
        axs[2].set_ylim(ylim)
        axs[2].set_title(f"{col} Quantile-Quantile Plot")
        axs[2].legend()

        name = name_sanitizer(f"{col}.png")
        plt.savefig(f"{path}/{name}")
        
        if not show:
            plt.close()
        
def object_data_visualizer(objects, target_values, path, show=True, null_label="Null", cols="all"):
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    n_total = len(objects)
    n_str = f"N: {n_total}"
    if cols != "all":
        objects = pd.DataFrame(objects[cols])
    for col, values in objects.items():
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18, 12)
        # barplot of category counts
        values = values.fillna(null_label)
        counts = values.groupby(values).count().sort_index()
        x_order = counts.index
        reorderer = np.array([n for n in range(len(x_order))])
        if null_label in x_order:
            null_index = np.where(x_order == null_label)[0][0]
            reorderer = np.concatenate(
                [reorderer[:null_index], reorderer[null_index + 1 :], [null_index]]
            )
        x_order = x_order[reorderer]
        sns.barplot(x=counts.index, y=counts.values, ax=axs[0], order=x_order)
        axs[0].set_title(f"{col} - Counts")
        axs[0].set_xlabel(col)
        axs[0].set_ylabel("Count")
        props = dict(boxstyle="square", facecolor="white", alpha=0.5)
        axs[0].text(0.05, 0.95, n_str, transform=axs[0].transAxes, bbox=props)

        # target boxplots by category
        sns.boxplot(x=values.sort_values(), y=target_values, ax=axs[1], order=x_order)
        axs[1].set_title(f"{col} - Target Spreads")
        name = name_sanitizer(f"{col}.png")
        plt.savefig(f"{path}/{name}")
        
        if not show:
            plt.close()

def correlation_visualizer(X, y, path, show=True, filename="correlation.png"):
    # calculate correlations and eigenvalues
    corr = np.corrcoef(X, rowvar=False)
    y_corr = np.array([])
    for feature in range(X.shape[1]):
        y_corr = np.append(y_corr, np.corrcoef(X[:, feature], y)[0, 1])
    e_val, e_vec = np.linalg.eigh(corr)

    # get number of entries and sorter for eigenvalue magnitudes
    dim = e_val.shape[0]
    sorter = np.argsort(np.abs(e_val))[::-1]

    # create figure
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(24, 12)

    # feature bivariate correlation heatmap
    sns.heatmap(corr, ax=axs[0, 0], xticklabels=False, yticklabels=False)
    axs[0, 0].set_title("Correlation Heatmap")

    # feature target correlations
    sns.barplot(x=list(range(dim)), y=y_corr, ax=axs[0, 1], color="green")
    axs[0, 1].set_title("Correlation Coefficient with Target")
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_ylabel("Pearson Correlation Coefficient")
    axs[0, 1].set_xlabel("Feature Index")
    axs[0, 1].set_xticks([])

    # feature target correlation distribution
    sns.histplot(y_corr, ax=axs[0, 2], color="purple")
    axs[0, 2].set_xlabel("Pearson Correlation Coefficient")
    axs[0, 2].set_title("Target Correlation Distribution")

    # calculate percentages for eigenvalue axes
    # much more readable label
    x_percentages = np.array([100.0 * n / dim for n in range(dim)])
    tick_indices = [np.argmin(np.abs(x_percentages - n)) for n in range(0, 101, 10)]
    x_labels = []
    p = 0
    for i in range(dim):
        if i in tick_indices:
            x_labels.append(str(p))
            p += 10
        else:
            x_labels.append("")
        if p > 100:
            break

    # barplot of sorted eigenvalues
    sns.barplot(
        x=list(range(0, dim)), y=e_val[sorter], ax=axs[1, 0], color="blue", width=1.0
    )
    axs[1, 0].set_title("Sorted Correlation Matrix Eigenvalues")
    # reset xticklabels to a more readable format
    axs[1, 0].set_xticklabels(x_labels)
    axs[1, 0].set_xlabel("Percentage of Total Features")
    axs[1, 0].set_ylabel("Eigenvalue Magnitude")

    # plot cumulative variance of dataset vs feature index
    cumvar = np.cumsum(e_val[sorter]) / np.sum(e_val)
    sns.lineplot(x=x_percentages, y=cumvar, ax=axs[1, 1], color="orange")
    axs[1, 1].set_title("Sorted Cumulative Variance")
    axs[1, 1].set_xlabel("Percentage of Total Features")
    axs[1, 1].set_ylabel("Fraction of Total Dataset Variance")

    # plot distribution of eigenvalues
    sns.histplot(e_val, ax=axs[1, 2], color="pink")
    axs[1, 2].set_title("Correlation Matrix Eigenvalue Distribution")
    axs[1, 2].set_xlabel("Correlation Matrix Eigenvalue")
    
    plt.savefig(f"{path}/{filename}")
    
    if not show:
        plt.close()