# Helper page for Instagram Usage Analytics

## Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import chi2_contingency
from IPython.display import display

sns.set_theme(style="whitegrid")

# SMALL HELPERS 

def _prep(df, filters=None, cols=None, dropna=True):
    """Apply filters, keep only needed columns, drop NaNs in them."""
    df = apply_filters(df, filters=filters)
    if cols:
        cols = [c for c in dict.fromkeys(cols) if c is not None]
        df = df[cols]
        if dropna:
            df = df.dropna()
    print("Shape:", df.shape)
    return df.copy()


def _title(txt):
    return txt.replace("_", " ")


def _num_cols(df):
    return df.select_dtypes(include="number").select_dtypes(exclude=["bool", "boolean"]).columns.tolist()


def _cat_cols(df):
    return df.select_dtypes(include=["category", "object", "bool", "boolean", "string"]).columns.tolist()


def _log_prep(d, col, log):
    """Optionally add a log1p version of `col`. Returns (dataframe, column_name_to_plot)."""
    s = d[col]
    if log == "auto":
        log = abs(s.skew()) > 1
    if log and (s < 0).any():
        print(f"{col} has negative values, so log1p is skipped.")
        log = False
    if log:
        d = d.copy()
        new = f"log1p({_title(col)})"
        d[new] = np.log1p(s)
        return d, new
    return d, col

# Functions for applying filters to the dataframes based on the quality flags and other conditions.

def apply_filters(df, filters=None):
    if not filters:
        return df
    
    mask = pd.Series(True, index=df.index)
    
    for col, condition in filters.items():
        
        # Simple equality
        if not isinstance(condition, tuple):
            mask &= df[col] == condition
        
        else:
            op, val = condition
            
            if op == '==':
                mask &= df[col] == val
            
            elif op == '!=':
                mask &= df[col] != val
            
            elif op == '>':
                mask &= df[col] > val
            
            elif op == '<':
                mask &= df[col] < val
            
            elif op == '>=':
                mask &= df[col] >= val
            
            elif op == '<=':
                mask &= df[col] <= val
            
            elif op == 'in':
                mask &= df[col].isin(val)
            
            elif op == 'between':
                low, high = val
                mask &= df[col].between(low, high)
            
            elif op == 'isnull':
                mask &= df[col].isna()
            
            elif op == 'notnull':
                mask &= df[col].notna()
            
            else:
                raise ValueError(f"Unsupported operation: {op}")
    
    return df[mask]

# This function takes a dataframe and a dictionary of filters where the keys are column names and the values 
# are either a single value for equality or a tuple specifying an operation and a value. 
# The function returns a filtered dataframe based on the specified conditions.

def info(df, filters=None, cols=None, dtype=None):
    
    df = apply_filters(df, filters=filters)

    if dtype == 'num':
        df = df.select_dtypes(include=np.number)

    elif dtype == 'cat':
        df = df.select_dtypes(include=['category', 'object', 'bool', 'boolean'])

    # Select specific columns if provided
    if cols is not None:
        df = df[cols]
        
    return df.info()

## Structure for univariate analysis

### Functions for Univariate Analysis


def corr_matrix(df, filters=None, methods=("pearson", "spearman", "kendall")):

    df = apply_filters(df, filters=filters)


    print("Shape after filtering:", df.shape)

    if df.shape[0] == 0:
        print("No data available after filtering.")
        return

    # Only int + float (exclude bool)
    num_df = df.select_dtypes(include='number').select_dtypes(exclude='bool')

    # Remove constant columns
    num_df = num_df.loc[:, num_df.nunique() > 1]

    if num_df.shape[1] == 0:
        print("No usable numeric columns.")
        return

    for method in methods:

        corr = num_df.corr(method=method).round(2)
    
        # Remove useless columns
        mask = (corr.abs().sum(axis=1) - 1) != 0
        corr = corr.loc[mask, mask]

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"{method.capitalize()} Correlation Matrix")
        plt.show()


#### Structure for Group-wise descriptive Analysis (level 1)


def association_matrix(df, filters=None, cols=None):
    # 1. Apply your existing filters, if provided
    data = apply_filters(df, filters) if filters is not None else df.copy()

    # 2. Validate the input
    if data.empty:
        print("DataFrame is empty.")
        return None

    # 3. Select categorical columns
    cat_cols = data.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Keep only selected cols if provided
    if cols is not None:
        cat_cols = [
            col for col in cols
            if col in cat_cols
        ]

    if len(cat_cols) < 2:
        print("At least two categorical columns are required.")
        return None

    # 4. Remove constant columns
    cat_cols = [
        col for col in cat_cols
        if data[col].nunique(dropna=False) > 1
    ]

    if len(cat_cols) < 2:
        print("Fewer than two usable categorical columns remain.")
        return None

    data = data[cat_cols].copy()

    # 5. Calculate bias-corrected Cramér's V
    def cramers_v(x, y):
        # Retain missing values as an explicit category
        x = x.astype("object").where(x.notna(), "__MISSING__")
        y = y.astype("object").where(y.notna(), "__MISSING__")

        table = pd.crosstab(x, y)
        n = table.to_numpy().sum()

        if n <= 1 or min(table.shape) < 2:
            return np.nan

        chi2 = chi2_contingency(table, correction=False)[0]

        r, k = table.shape
        phi2 = chi2 / n

        # Bias correction
        phi2_corr = max(
            0,
            phi2 - ((k - 1) * (r - 1)) / (n - 1)
        )

        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)

        denominator = min(r_corr - 1, k_corr - 1)

        if denominator <= 0:
            return np.nan

        return np.sqrt(phi2_corr / denominator)

    # 6. Build the pairwise association matrix
    cols = data.columns
    matrix = pd.DataFrame(
        np.eye(len(cols)),
        index=cols,
        columns=cols,
        dtype=float
    )

    for i, col1 in enumerate(cols):
        for col2 in cols[i + 1:]:
            value = cramers_v(data[col1], data[col2])
            matrix.loc[col1, col2] = value
            matrix.loc[col2, col1] = value

    # 7. Visualize
    plt.figure(figsize=(max(8, len(cols)), max(6, len(cols) * 0.7)))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        square=True
    )

    plt.title("Categorical Association Matrix (Cramér's V)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return matrix



# Created featured columns based on numerical cols to further understand and analyze them.

def cat_col_chk1(df, outcome, filters=None):

    df = apply_filters(df, filters=filters)

    for col in df.select_dtypes(include=["category", "boolean"]).select_dtypes(exclude=["bool"]).columns:
        print("\n", col.upper() +' & '+ outcome)

        # basic describe()
        stats = df.groupby(col, observed=False)[outcome].describe().round(2)

        # Computing quartiles
        q1 = df.groupby(col, observed=False)[outcome].quantile(0.25)
        q3 = df.groupby(col, observed=False)[outcome].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        # Adding new columns
        stats['IQR'] = iqr.round(2)
        stats['Lower_Fence'] = lower_fence.round(2)
        stats['Upper_Fence'] = upper_fence.round(2)
        stats["CV"] = np.where(
        stats["mean"] != 0,
        (stats["std"] / stats["mean"]) * 100,
        np.nan
        ).round(2)

        display(stats)

# In this list, we will check the categorical columns in the mtj dataframe against the outcome variable.
# outcome variables are = (daily_active_minutes_instagram, user_engagement_score, avg_session_length_minutes,
# perceived_stress_score, age, self_reported_happiness)

#### Structure for Group-wise descriptive Analysis (level 2)


# function for group-wise descriptive analysis of the categorical columns against the outcome variable with the option 
# to apply filters and show the IQR and fences to check variability and outliers in the data.

def cat_col_chk2(df, driver, filters=None):

    df = apply_filters(df, filters=filters)

    print("Shape:", df.shape)

    for col in df.select_dtypes(include='number').select_dtypes(exclude=['bool', 'boolean']).columns:
        print("\n", col.upper() +' & '+ driver)
        stats = df.groupby(driver, observed=False)[col].describe().round(2)

        # Computing quartiles
        q1 = df.groupby(driver, observed=False)[col].quantile(0.25)
        q3 = df.groupby(driver, observed=False)[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - (1.5 * iqr)
        upper_fence = q3 + (1.5 * iqr)

        # Adding new columns
        stats['IQR'] = iqr.round(2)
        stats['Lower_Fence'] = lower_fence.round(2)
        stats['Upper_Fence'] = upper_fence.round(2)
        stats["CV"] = np.where(
        stats["mean"] != 0,
        (stats["std"] / stats["mean"]) * 100,
        np.nan
        ).round(2)

        display(stats)


def trend_analysis(
    df,
    date_col,
    value_cols=None,
    freq="ME",
    filters=None
):
    # Apply filters
    df = apply_filters(df, filters)

    print("Shape after filtering:", df.shape)

    # Validate date column
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Remove rows with invalid dates
    df = df.dropna(subset=[date_col])

    # Select numeric columns
    num_df = df.select_dtypes(include=np.number)

    if value_cols is None:
        value_cols = num_df.columns.tolist()
    else:
        value_cols = [
            col for col in value_cols
            if col in num_df.columns
        ]

    if not value_cols:
        raise ValueError("No valid numeric columns selected.")

    # Group data by time period
    trend_df = (
        df.set_index(date_col)[value_cols]
        .resample(freq)
        .agg(["count", "sum", "mean", "median", "min", "max"])
    )

    # Flatten multi-level column names
    trend_df.columns = [
        f"{col}_{stat}" for col, stat in trend_df.columns
    ]

    # Period-over-period percentage change in mean
    for col in value_cols:
        mean_col = f"{col}_mean"
        trend_df[f"{col}_change"] = trend_df[mean_col].diff()
        trend_df[f"{col}_change_%"] = (
            trend_df[mean_col].pct_change(fill_method=None) * 100
        )

    return trend_df.round(2)


def describe_with_fences_b(df, x_var, y_var):
    # Standard describe
    stats = df.groupby([x_var], observed=True)[y_var].describe().round(2)

    # Compute IQR, fences
    q1 = df.groupby([x_var], observed=True)[y_var].quantile(0.25)
    q3 = df.groupby([x_var], observed=True)[y_var].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)

    # Add new columns
    stats['IQR'] = iqr.round(2)
    stats['Lower_Fence'] = lower_fence.round(2)
    stats['Upper_Fence'] = upper_fence.round(2)
    stats["CV"] = np.where(
    stats["mean"] != 0,
    (stats["std"] / stats["mean"]) * 100,
    np.nan
    ).round(2)

    return stats

def bivar_bar_c(df, x_var, y_var, view_type='both', filters=None):
    print()
    print(f'Statistical representation of {x_var} across {y_var}')
    print()

# this apply filters function is used to format the dataframe by one's needs with most arguments having none as their default value and only valid having a True value so this function will only access the valid data fro the given dataframe

    df = apply_filters(df, filters=filters)

    print("Shape:", df.shape)

# the if clause is used to trigger only what is asked by the code writer to show in the cell output and the default value is both

    if view_type in ['both', 'stats']:
        display(df.groupby([x_var], observed=False)[y_var].mean().round(2))

    if view_type in ['both', 'plot']:

        print()
        print(f'Barchart of {x_var} across {y_var}')
        print()

        df.groupby([x_var], observed=False)[y_var].mean().round(2).plot(kind='bar', figsize=(10,5))
        plt.title(f"Mean across {x_var}")
        plt.ylabel(f'Mean')
        plt.xlabel(f'{x_var}')

        plt.show()

# Created a function to deploy bar chart for bivariate analysis with statisical distribution.

def bivar_violin_c(df, x_var, y_var, view_type='both', filters=None):
    print()
    print(f'Statistical representation of {x_var} across {y_var}')
    print()

# this apply filters function is used to format the dataframe by one's needs with most arguments having none as their default value and only valid having a True value so this function will only access the valid data fro the given dataframe

    df = apply_filters(df, filters=filters)

    print("Shape:", df.shape)
    
# the if clause is used to trigger only what is asked by the code writer to show in the cell output and the default value is both

    if view_type in ['both', 'stats']:
        display(describe_with_fences_b(df, x_var, y_var))

    if view_type in ['both', 'plot']:

        print()
        print(f'Violinplot of {x_var} across {y_var}')
        print()
        
        # Respect categorical order if column is categorical, else fallback to unique
        if isinstance(df[x_var].dtype, pd.CategoricalDtype):
            category_order = df[x_var].cat.categories[::-1]   # reverse defined category order
        else:
            category_order = df[x_var].unique()[::-1]         # fallback for non-categorical

        plt.figure(figsize=(10,5))
        sns.violinplot(x=y_var,
                    y=x_var,
                    hue=x_var, 
                    data=df,
                    palette='bright',
                    inner='box',
                    order=category_order)
        plt.title(f"{y_var.replace('_', ' ')} across {x_var.replace('_', ' ')}")
        plt.show()

# Created a function to deploy violinplot for bivariate analysis with statisical distribution.
# Here x in violinplot is identified by outcome variable and y by categorical variable to stimulate a landscape (horizontal) violinplot 
# to better understand the distribution.

def bivar_box_c(df, x_var, y_var, view_type='both', filters=None):
    print()
    print(f'Statistical representation of {x_var} across {y_var}')
    print()

# this apply filters function is used to format the dataframe by one's needs with most arguments having none as their default value and only valid having a True value so this function will only access the valid data fro the given dataframe

    df = apply_filters(df, filters=filters)

    print("Shape:", df.shape)
    
# the if clause is used to trigger only what is asked by the code writer to show in the cell output and the default value is both

    if view_type in ['both', 'stats']:
        display(describe_with_fences_b(df, x_var, y_var))

    if view_type in ['both', 'plot']:

        print()
        print(f'Boxplot of {x_var} across {y_var}')
        print()
        
        plt.figure(figsize=(10,5))
        sns.boxplot(x=x_var,
                    y=y_var,
                    hue=x_var, 
                    data=df,
                    showmeans=True,
                    meanprops={
                        "marker": "o",    
                        "markerfacecolor": "white",
                        "markeredgecolor": "black",
                        "markersize": 8
                        },
                    palette='muted')
        plt.title(f"{y_var.replace('_', ' ')} across {x_var.replace('_', ' ')}")
        plt.show()

# Created a function to deploy boxplot for bivariate analysis with statisical distribution.


def bivar_boxen_c(df, x_var, y_var, view_type='both', filters=None):
    print()
    print(f'Statistical representation of {x_var} across {y_var}')
    print()

# this apply filters function is used to format the dataframe by one's needs with most arguments having none as their default value and only valid having a True value so this function will only access the valid data fro the given dataframe

    df = apply_filters(df, filters=filters)

    print("Shape:", df.shape)
    
# the if clause is used to trigger only what is asked by the code writer to show in the cell output and the default value is both

    if view_type in ['both', 'stats']:
        display(describe_with_fences_b(df, x_var, y_var))

    if view_type in ['both', 'plot']:
        
        print()
        print(f'Boxenplot of {x_var} across {y_var}')
        print()
        
        plt.figure(figsize=(10,5))
        sns.boxenplot(x=x_var,
                    y=y_var,
                    hue=x_var, 
                    data=df,
                    palette='dark')
        plt.yscale('log')
        plt.title(f"{y_var.replace('_', ' ')} across {x_var.replace('_', ' ')}")
        plt.show()

# Created a function to deploy boxenplot for bivariate analysis with statisical distribution.


def describe_with_fences(df, x_var, x1_var, y_var):
    # Standard describe
    stats = df.groupby([x_var, x1_var], observed=True)[y_var].describe().round(2)

    # Compute IQR, fences
    q1 = df.groupby([x_var, x1_var], observed=True)[y_var].quantile(0.25)
    q3 = df.groupby([x_var, x1_var], observed=True)[y_var].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - (1.5 * iqr)
    upper_fence = q3 + (1.5 * iqr)

    # Add new columns
    stats['IQR'] = iqr.round(2)
    stats['Lower_Fence'] = lower_fence.round(2)
    stats['Upper_Fence'] = upper_fence.round(2)
    stats["CV"] = np.where(
    stats["mean"] != 0,
    (stats["std"] / stats["mean"]) * 100,
    np.nan
    ).round(2)

    return stats

# Created a function to reuse code for creating statistics.

def m_eda_violin_c(df, x_var, x1_var, y_var, legend_position='upper right', view_type='both', filters=None):
    print()
    print(f'Statistical representation btw {x_var} & {x1_var} across {y_var}')
    print()
    
# this apply filters function is used to format the dataframe by one's needs with most arguments having none as their default value and only valid having a True value so this function will only access the valid data fro the given dataframe

    df = apply_filters(df, filters=filters)

    print("Shape:", df.shape)

# the if clause is used to trigger only what is asked by the code writer to show in the cell output and the default value is both

    if view_type in ['both', 'stats']:
        display(describe_with_fences(df, x_var, x1_var, y_var))

    if view_type in ['both', 'plot']:
    
        print()
        print(f'Violinplot btw {x_var} & {x1_var} across {y_var}')
        print()
        
        # Respect categorical order if column is categorical, else fallback to unique
        if isinstance(df[x_var].dtype, pd.CategoricalDtype):
            category_order = df[x_var].cat.categories[::-1]   # reverse defined category order
        else:
            category_order = df[x_var].unique()[::-1]         # fallback for non-categorical

        # Respect categorical order if column is categorical, else fallback to unique
        if isinstance(df[x1_var].dtype, pd.CategoricalDtype):
            hue_order = df[x1_var].cat.categories[::-1]   # reverse defined category order
        else:
            hue_order = df[x1_var].unique()[::-1]         # fallback for non-categorical

        plt.figure(figsize=(12,8))
        sns.violinplot(y=x_var,
                    x=y_var, 
                    hue=x1_var,
                    data=df,
                    inner='box',
                    order=category_order,
                    hue_order=hue_order
                    )
        plt.title(f"{x1_var.replace('_', ' ')} across {x_var.replace('_', ' ')} and {y_var.replace('_', ' ')}")
        plt.legend(loc=legend_position)
        plt.show()

# Created a function for building visuals with the help of 2 other functions.

def m_eda_box_c(df, x_var, x1_var, y_var, legend_position='upper right', view_type='both', filters=None):
    print()
    print(f'Statistical representation btw {x_var} & {x1_var} across {y_var}')
    print()

# this apply filters function is used to format the dataframe by one's needs with most arguments having none as their default value and only valid having a True value so this function will only access the valid data fro the given dataframe

    df = apply_filters(df, filters=filters)

    print("Shape:", df.shape)

# the if clause is used to trigger only what is asked by the code writer to show in the cell output and the default value is both

    if view_type in ['both', 'stats']:
        display(describe_with_fences(df, x_var, x1_var, y_var))

    if view_type in ['both', 'plot']:
        print()
        print(f'Boxplot btw {x_var} & {x1_var} across {y_var}')
        print()
        
        plt.figure(figsize=(12,8))
        sns.boxplot(x=x_var,
                    y=y_var, 
                    hue=x1_var,
                    data=df,
                    meanprops={
                    "marker": "o",
                    "markerfacecolor": "white",        
                    "markeredgecolor": "black", 
                    "markersize": 8
                    },
                    showmeans=True
                    )
        plt.title(f"{x1_var.replace('_', ' ')} across {x_var.replace('_', ' ')} and {y_var.replace('_', ' ')}")
        plt.legend(loc=legend_position)
        plt.show()    

# Created a function for building visuals with the help of 2 other functions.

def m_eda_boxen_c(df, x_var, x1_var, y_var, legend_position='upper right', view_type='both', filters=None):
    print()
    print(f'Statistical representation btw {x_var} & {x1_var} across {y_var}')
    print()

    df = apply_filters(df, filters=filters)

    print("Shape:", df.shape)

# this apply filters function is used to format the dataframe by one's needs with most arguments having none as their default value and only valid having a True value so this function will only access the valid data fro the given dataframe
# the if clause is used to trigger only what is asked by the code writer to show in the cell output and the default value is both

    if view_type in ['both', 'stats']:
        display(describe_with_fences(df, x_var, x1_var, y_var))

    if view_type in ['both', 'plot']:
        print()
        print(f'Boxenplot btw {x_var} & {x1_var} across {y_var}')
        print()
        
        plt.figure(figsize=(12,8))
        sns.boxenplot(x=x_var,
                    y=y_var, 
                    hue=x1_var,
                    data=df
                    )
        plt.yscale('log')
        plt.title(f"{x1_var.replace('_', ' ')} across {x_var.replace('_', ' ')} and {y_var.replace('_', ' ')}")
        plt.legend(loc=legend_position)
        plt.show()

# Created a function for building visuals with the help of 2 other functions.






# =============================================================================
# ADDITIONS FOR "Helper page for Instagram Usage Analytics"
# Paste the IMPORTS block over your existing imports at the top of the file,
# then paste everything from "UNIVARIATE" downward at the bottom of the file.
# Everything here reuses your existing apply_filters().
# =============================================================================



# =============================================================================
# STEP 1: UNIVARIATE ANALYSIS
# =============================================================================

# function for numeric summary

def num_col_chk(df, filters=None, cols=None):

    # Apply filters
    df = apply_filters(df, filters)

    print("Shape after filtering:", df.shape)

    # Numeric columns only
    num_df = df.select_dtypes(include=np.number).copy()

    # Select specific columns if provided
    if cols is not None:
        num_df = num_df[cols]

    personal_num_summary = num_df.describe(
    percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    ).round(2).T

    personal_num_summary["missing_%"] = (
        num_df.isna().mean() * 100
    )

    q1 = personal_num_summary["25%"]
    q3 = personal_num_summary["75%"]

    iqr = q3 - q1

    personal_num_summary["range"] = (personal_num_summary['max'] - personal_num_summary['min']).round(2)

    personal_num_summary["IQR"] = iqr.round(2)

    personal_num_summary["Lower_Fence"] = (
        q1 - 1.5 * iqr
    ).round(2)

    personal_num_summary["Upper_Fence"] = (
        q3 + 1.5 * iqr
    ).round(2)

    personal_num_summary["CV"] = np.where(
        personal_num_summary["mean"] != 0,
        (personal_num_summary["std"] /
         personal_num_summary["mean"]) * 100,
        np.nan
    )

    # Skewness
    personal_num_summary["skewness"] = num_df.skew().round(2)

    # Kurtosis (excess kurtosis)
    personal_num_summary["kurtosis"] = num_df.kurt().round(2)

    personal_num_summary["unique_count"] = (
        num_df.nunique()
    )

    mode_df = num_df.mode()

    if not mode_df.empty:
        personal_num_summary["most_frequent"] = (
            mode_df.iloc[0].round(2)
        )
    else:
        personal_num_summary["most_frequent"] = np.nan

    return personal_num_summary


def uni_hist_c(df, col, bins=30, kde=True, log=False, filters=None):
    """Histogram + KDE with mean / median markers.
    log: False | True (log1p only) | 'both' (original + log1p) | 'auto' ('both' only if |skew| > 1)"""
    d = _prep(df, filters, [col])
    if log == "auto":
        log = "both" if abs(d[col].skew()) > 1 else False
    d, lx = _log_prep(d, col, bool(log))
    xs = list(dict.fromkeys([col, lx])) if log == "both" else [lx]

    fig, axes = plt.subplots(1, len(xs), figsize=(7 * len(xs), 5), squeeze=False)
    for ax, x in zip(axes[0], xs):
        sns.histplot(d[x], bins=bins, kde=kde, color="steelblue", ax=ax)
        ax.axvline(d[x].mean(), color="red", ls="--", label=f"Mean: {d[x].mean():.2f}")
        ax.axvline(d[x].median(), color="green", ls="-.", label=f"Median: {d[x].median():.2f}")
        ax.set(title=f"Distribution of {_title(x)}", xlabel=_title(x))
        ax.legend()
        print(f"Skewness ({x}): {d[x].skew():.2f}")
    plt.tight_layout()
    plt.show()


def uni_hist_px(df, col, nbins=40, marginal="box", log=False, filters=None):
    """Interactive histogram with marginal box/violin/rug (plotly).
    log: False | True (log1p) | 'auto' (log1p only if |skew| > 1)"""
    d = _prep(df, filters, [col])
    d, x = _log_prep(d, col, log)
    px.histogram(d, x=x, nbins=nbins, marginal=marginal,
                 title=f"Distribution of {_title(x)}").show()


def uni_ecdf_px(df, col, log=False, filters=None):
    d = _prep(df, filters, [col])
    d, x = _log_prep(d, col, log)
    px.ecdf(d, x=x, title=f"ECDF of {_title(x)}").show()


def uni_num_grid(df, cols=None, kind="hist", ncols=3, log=False, filters=None):
    """Grid of univariate plots for many numeric columns.
    kind: 'hist' | 'box' | 'violin' | 'ecdf'
    log : False | True (log1p on every column) | 'auto' (log1p only where |skew| > 1)"""
    plots = {"hist":   lambda s, ax: sns.histplot(s, kde=True, ax=ax),
             "box":    lambda s, ax: sns.boxplot(x=s, ax=ax, showmeans=True),
             "violin": lambda s, ax: sns.violinplot(x=s, ax=ax, inner="box"),
             "ecdf":   lambda s, ax: sns.ecdfplot(s, ax=ax)}
    if kind not in plots:
        raise ValueError("kind must be hist, box, violin or ecdf")

    d = apply_filters(df, filters)
    cols = cols or _num_cols(d)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, c in zip(axes, cols):
        dc, x = _log_prep(d[[c]].dropna(), c, log)
        plots[kind](dc[x], ax)
        ax.set_title(_title(x))
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()


def uni_box_violin_boxen_c(df, col, log=False, filters=None):
    """Boxplot, violin and boxen side by side for one numeric column.
    log: False | True (log1p) | 'auto' (log1p only if |skew| > 1)"""
    d = _prep(df, filters, [col])
    d, x = _log_prep(d, col, log)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    sns.boxplot(x=d[x], ax=ax[0], showmeans=True, color="lightblue")
    sns.violinplot(x=d[x], ax=ax[1], inner="quartile", color="lightgreen")
    sns.boxenplot(x=d[x], ax=ax[2], color="salmon")
    ax[0].set_title(f"Boxplot: {_title(x)}")
    ax[1].set_title(f"Violin: {_title(x)}")
    ax[2].set_title(f"Boxen: {_title(x)}")
    plt.tight_layout()
    plt.show()


def uni_qq_c(df, col, filters=None):
    """QQ plot + Shapiro-Wilk normality test (sampled to 5000 rows for the test)."""
    d = _prep(df, filters, [col])
    plt.figure(figsize=(6, 6))
    stats.probplot(d[col], dist="norm", plot=plt)
    plt.title(f"QQ plot: {_title(col)}")
    plt.show()
    sample = d[col].sample(min(5000, len(d)), random_state=42)
    w, p = stats.shapiro(sample)
    print(f"Shapiro-Wilk: W={w:.4f}, p={p:.4g} ->", "looks normal" if p > 0.05 else "not normal")


# Categorical summary

def cat_col_chk(df, filters=None, cols=None, proportions=True):

    # Apply filters
    df = apply_filters(df, filters)

    print("Shape after filtering:", df.shape)

    # Select categorical-like columns
    cat_cols = df.select_dtypes(include=["category", "object", "bool", "boolean"]).columns

    # Keep only selected cols if provided
    if cols is not None:
        cat_cols = [
            col for col in cols
            if col in cat_cols
        ]

    for col in cat_cols:
        print("\n", col.upper())

        counts = df[col].value_counts(dropna=False)

        if proportions:
            result = (counts / counts.sum()).round(2)
        else:
            result = counts

        display(result)

# creating a function to generate categoric summary of the dataframe with the option to apply filters and show proportions or counts.


def uni_count_c(df, col, top_n=15, normalize=False, horizontal=True, filters=None):
    """Countplot / proportion bar for a categorical column."""
    d = _prep(df, filters, [col], dropna=False)
    counts = d[col].value_counts(normalize=normalize, dropna=False).head(top_n)
    plt.figure(figsize=(10, 5))
    if horizontal:
        sns.barplot(x=counts.values, y=counts.index.astype(str), hue=counts.index.astype(str),
                    palette="viridis", legend=False)
        plt.xlabel("Proportion" if normalize else "Count")
    else:
        sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str),
                    palette="viridis", legend=False)
        plt.ylabel("Proportion" if normalize else "Count")
        plt.xticks(rotation=45, ha="right")
    plt.title(f"{_title(col)} (top {top_n})")
    plt.tight_layout()
    plt.show()


def uni_pie_px(df, col, donut=True, top_n=10, filters=None):
    """Interactive pie / donut (plotly). Small categories are grouped into 'Other'."""
    d = _prep(df, filters, [col], dropna=False)
    vc = d[col].astype(str).value_counts()
    if len(vc) > top_n:
        vc = pd.concat([vc.head(top_n), pd.Series({"Other": vc.iloc[top_n:].sum()})])
    fig = px.pie(names=vc.index, values=vc.values, hole=0.45 if donut else 0,
                 title=f"{_title(col)} share")
    fig.update_traces(textinfo="percent+label")
    fig.show()


# =============================================================================
# STEP 2: BIVARIATE ANALYSIS
# =============================================================================

# ---- numeric vs numeric -----------------------------------------------------

def bivar_scatter_c(df, x_var, y_var, hue=None, reg=True, sample=None, alpha=0.5, filters=None):
    """Scatterplot (+ optional regression line) with Pearson / Spearman in the title."""
    d = _prep(df, filters, [x_var, y_var, hue])
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    r = d[x_var].corr(d[y_var])
    rho = d[x_var].corr(d[y_var], method="spearman")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=d, x=x_var, y=y_var, hue=hue, alpha=alpha)
    if reg:
        sns.regplot(data=d, x=x_var, y=y_var, scatter=False, color="black", line_kws={"lw": 2})
    plt.title(f"{_title(y_var)} vs {_title(x_var)}  (Pearson={r:.2f}, Spearman={rho:.2f})")
    plt.show()


def bivar_scatter_px(df, x_var, y_var, color=None, size=None, trendline=None, hover=None, filters=None):
    """Interactive scatter (plotly). trendline='ols' needs statsmodels; 'lowess' too."""
    d = _prep(df, filters, [x_var, y_var, color, size] + (hover or []))
    px.scatter(d, x=x_var, y=y_var, color=color, size=size, hover_data=hover,
               trendline=trendline, opacity=0.65,
               title=f"{_title(y_var)} vs {_title(x_var)}").show()


def bivar_joint_c(df, x_var, y_var, kind="hex", filters=None):
    """Jointplot for big data. kind: 'scatter' | 'hex' | 'kde' | 'reg' | 'hist'."""
    d = _prep(df, filters, [x_var, y_var])
    g = sns.jointplot(data=d, x=x_var, y=y_var, kind=kind, height=7)
    g.fig.suptitle(f"{_title(y_var)} vs {_title(x_var)}", y=1.02)
    plt.show()


def bivar_density_px(df, x_var, y_var, filters=None):
    """2D density contour with marginal histograms (plotly)."""
    d = _prep(df, filters, [x_var, y_var])
    px.density_contour(d, x=x_var, y=y_var, marginal_x="histogram", marginal_y="histogram").show()


def bivar_binned_trend_c(df, x_var, y_var, bins=10, filters=None):
    """Mean of y within quantile-bins of x, with error bars. Great for messy scatter."""
    d = _prep(df, filters, [x_var, y_var])
    d["_bin"] = pd.qcut(d[x_var], q=bins, duplicates="drop")
    g = d.groupby("_bin", observed=True)[y_var].agg(["mean", "sem"]).reset_index()
    g["_mid"] = g["_bin"].apply(lambda i: i.mid)
    plt.figure(figsize=(10, 5))
    plt.errorbar(g["_mid"], g["mean"], yerr=1.96 * g["sem"], marker="o", capsize=4)
    plt.xlabel(_title(x_var))
    plt.ylabel(f"Mean {_title(y_var)}")
    plt.title(f"Binned trend of {_title(y_var)} across {_title(x_var)} (95% CI)")
    plt.show()


# ---- categorical vs numeric -------------------------------------------------

def bivar_strip_c(df, cat, num, kind="strip", filters=None):
    """kind: 'strip' | 'swarm'. Swarm is slow above ~2000 rows, so it is auto-sampled."""
    d = _prep(df, filters, [cat, num])
    plt.figure(figsize=(10, 5))
    if kind == "swarm":
        d = d.sample(min(2000, len(d)), random_state=42)
        sns.swarmplot(data=d, x=cat, y=num, hue=cat, legend=False)
    else:
        sns.stripplot(data=d, x=cat, y=num, hue=cat, alpha=0.4, jitter=True, legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{_title(num)} across {_title(cat)}")
    plt.show()


def bivar_point_c(df, cat, num, filters=None):
    """Mean with 95% CI per category (seaborn)."""
    d = _prep(df, filters, [cat, num])
    plt.figure(figsize=(10, 5))
    sns.pointplot(data=d, x=cat, y=num, errorbar=("ci", 95), capsize=0.15)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Mean {_title(num)} by {_title(cat)} (95% CI)")
    plt.show()


def bivar_ecdf_c(df, cat, num, filters=None):
    """ECDF of a numeric column split by a category, good for comparing distributions."""
    d = _prep(df, filters, [cat, num])
    plt.figure(figsize=(10, 5))
    sns.ecdfplot(data=d, x=num, hue=cat)
    plt.title(f"ECDF of {_title(num)} by {_title(cat)}")
    plt.show()


def bivar_kde_c(df, cat, num, filters=None):
    """Overlapping KDEs per category."""
    d = _prep(df, filters, [cat, num])
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=d, x=num, hue=cat, fill=True, common_norm=False, alpha=0.35)
    plt.title(f"Density of {_title(num)} by {_title(cat)}")
    plt.show()


def bivar_box_px(df, cat, num, kind="box", filters=None):
    """Interactive box / violin / strip (plotly). kind: 'box' | 'violin' | 'strip'."""
    d = _prep(df, filters, [cat, num])
    fn = {"box": px.box, "violin": px.violin, "strip": px.strip}[kind]
    kw = {"points": "outliers"} if kind in ("box",) else {}
    fn(d, x=cat, y=num, color=cat, title=f"{_title(num)} across {_title(cat)}", **kw).show()


# ---- categorical vs categorical --------------------------------------------

def bivar_crosstab_heatmap(df, x_var, y_var, normalize="index", filters=None):
    """Crosstab heatmap. normalize: 'index' (row %), 'columns' (col %), 'all', or None (counts)."""
    d = _prep(df, filters, [x_var, y_var])
    ct = pd.crosstab(d[x_var], d[y_var], normalize=normalize)
    plt.figure(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt=".2f" if normalize else "d", cmap="YlGnBu")
    plt.title(f"{_title(x_var)} vs {_title(y_var)}")
    plt.show()


def bivar_stacked_bar_c(df, x_var, y_var, percent=True, filters=None):
    """Stacked bar of y_var composition within each x_var."""
    d = _prep(df, filters, [x_var, y_var])
    ct = pd.crosstab(d[x_var], d[y_var], normalize="index" if percent else False)
    ct.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20")
    plt.ylabel("Proportion" if percent else "Count")
    plt.title(f"{_title(y_var)} within {_title(x_var)}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def bivar_grouped_bar_px(df, x_var, y_var, filters=None):
    """Interactive grouped bar of counts (plotly)."""
    d = _prep(df, filters, [x_var, y_var])
    c = d.groupby([x_var, y_var], observed=True).size().reset_index(name="count")
    px.bar(c, x=x_var, y="count", color=y_var, barmode="group",
           title=f"{_title(x_var)} by {_title(y_var)}").show()


# ---- time -------------------------------------------------------------------

def bivar_line_c(df, date_col, y_var, freq="ME", agg="mean", hue=None, filters=None):
    """Time series of an aggregated numeric column, optionally split by a category."""
    d = _prep(df, filters, [date_col, y_var, hue])
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if hue:
        t = (d.set_index(date_col).groupby(hue, observed=True)[y_var]
               .resample(freq).agg(agg).reset_index())
    else:
        t = d.set_index(date_col)[y_var].resample(freq).agg(agg).reset_index()
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=t, x=date_col, y=y_var, hue=hue, marker="o")
    plt.title(f"{agg.capitalize()} {_title(y_var)} over time")
    plt.xticks(rotation=45)
    plt.show()


def bivar_line_px(df, date_col, y_var, freq="ME", agg="mean", color=None, filters=None):
    d = _prep(df, filters, [date_col, y_var, color])
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if color:
        t = (d.set_index(date_col).groupby(color, observed=True)[y_var]
               .resample(freq).agg(agg).reset_index())
    else:
        t = d.set_index(date_col)[y_var].resample(freq).agg(agg).reset_index()
    px.line(t, x=date_col, y=y_var, color=color, markers=True,
            title=f"{agg.capitalize()} {_title(y_var)} over time").show()


# ---- automatic statistical test --------------------------------------------

def bivar_test(df, x_var, y_var, filters=None):
    """Picks a test from the dtypes of x_var / y_var and prints the result.
       num-num : Pearson + Spearman + Kendall
       cat-num : ANOVA + Kruskal-Wallis (+ Mann-Whitney if 2 groups) + eta-squared
       cat-cat : Chi-square + bias-corrected Cramer's V"""
    d = _prep(df, filters, [x_var, y_var])
    x_num = pd.api.types.is_numeric_dtype(d[x_var]) and not pd.api.types.is_bool_dtype(d[x_var])
    y_num = pd.api.types.is_numeric_dtype(d[y_var]) and not pd.api.types.is_bool_dtype(d[y_var])

    if x_num and y_num:
        for name, fn in [("Pearson", stats.pearsonr), ("Spearman", stats.spearmanr),
                         ("Kendall", stats.kendalltau)]:
            r, p = fn(d[x_var], d[y_var])
            print(f"{name:9s} r={r:.3f}  p={p:.4g}")

    elif x_num != y_num:
        cat, num = (y_var, x_var) if x_num else (x_var, y_var)
        groups = [g[num].values for _, g in d.groupby(cat, observed=True) if len(g) > 1]
        if len(groups) < 2:
            print("Need at least two groups.")
            return
        f, p = stats.f_oneway(*groups)
        h, pk = stats.kruskal(*groups)
        grand = d[num].mean()
        ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        sst = ((d[num] - grand) ** 2).sum()
        print(f"ANOVA          F={f:.3f}  p={p:.4g}")
        print(f"Kruskal-Wallis H={h:.3f}  p={pk:.4g}")
        print(f"Eta-squared    {ssb / sst:.3f}  (0.01 small, 0.06 medium, 0.14 large)")
        if len(groups) == 2:
            u, pm = stats.mannwhitneyu(groups[0], groups[1])
            print(f"Mann-Whitney   U={u:.1f}  p={pm:.4g}")

    else:
        ct = pd.crosstab(d[x_var], d[y_var])
        chi2, p, dof, _ = chi2_contingency(ct, correction=False)
        n = ct.to_numpy().sum()
        r, k = ct.shape
        phi2c = max(0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
        rc = r - ((r - 1) ** 2) / (n - 1)
        kc = k - ((k - 1) ** 2) / (n - 1)
        v = np.sqrt(phi2c / max(min(rc - 1, kc - 1), 1e-9))
        print(f"Chi-square={chi2:.3f}  dof={dof}  p={p:.4g}")
        print(f"Cramer's V (bias-corrected) = {v:.3f}")


# =============================================================================
# STEP 3: MULTIVARIATE ANALYSIS
# =============================================================================

# ---- many numeric variables -------------------------------------------------

def multi_pairplot_c(df, cols, hue=None, kind="scatter", sample=2000, filters=None):
    """Seaborn pairplot. kind: 'scatter' | 'reg' | 'kde' | 'hist'."""
    d = _prep(df, filters, list(cols) + [hue])
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    g = sns.pairplot(d, vars=cols, hue=hue, kind=kind, corner=True,
                     plot_kws={"alpha": 0.4} if kind == "scatter" else {})
    g.fig.suptitle("Pairplot", y=1.02)
    plt.show()


def multi_scatter_matrix_px(df, cols, color=None, sample=2000, filters=None):
    """Interactive scatter matrix (plotly)."""
    d = _prep(df, filters, list(cols) + [color])
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    px.scatter_matrix(d, dimensions=cols, color=color, opacity=0.5).update_traces(diagonal_visible=False).show()


def multi_corr_px(df, cols=None, method="pearson", filters=None):
    """Interactive correlation heatmap (plotly)."""
    d = apply_filters(df, filters)
    cols = cols or _num_cols(d)
    c = d[cols].corr(method=method).round(2)
    px.imshow(c, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
              title=f"{method.capitalize()} correlation").show()


def multi_clustermap_c(df, cols=None, method="pearson", filters=None):
    """Hierarchically clustered correlation heatmap (groups similar variables)."""
    d = apply_filters(df, filters)
    cols = cols or _num_cols(d)
    c = d[cols].loc[:, d[cols].nunique() > 1].corr(method=method)
    g = sns.clustermap(c, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, figsize=(10, 10))
    g.fig.suptitle(f"Clustered {method} correlation", y=1.02)
    plt.show()


def multi_scatter3d_px(df, x_var, y_var, z_var, color=None, sample=3000, filters=None):
    """Interactive 3D scatter (plotly)."""
    d = _prep(df, filters, [x_var, y_var, z_var, color])
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    px.scatter_3d(d, x=x_var, y=y_var, z=z_var, color=color, opacity=0.7).show()


def multi_bubble_px(df, x_var, y_var, size, color=None, facet=None, sample=3000, filters=None):
    """Bubble chart: 4+ variables in one plot (x, y, size, color, facet)."""
    d = _prep(df, filters, [x_var, y_var, size, color, facet])
    d = d[d[size] > 0]
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    px.scatter(d, x=x_var, y=y_var, size=size, color=color, facet_col=facet,
               size_max=30, opacity=0.6).show()


def multi_parallel_px(df, cols, color, sample=2000, filters=None):
    """Parallel coordinates for numeric columns, coloured by one numeric column."""
    d = _prep(df, filters, list(cols) + [color])
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    px.parallel_coordinates(d, dimensions=cols, color=color,
                            color_continuous_scale=px.colors.diverging.Tealrose).show()


def multi_pca_biplot(df, cols=None, color=None, filters=None):
    """PCA scores + loadings arrows + explained-variance bar, using numpy only (no sklearn)."""
    d = apply_filters(df, filters)
    cols = cols or _num_cols(d)
    d = d[cols + ([color] if color else [])].dropna()
    X = d[cols]
    X = X.loc[:, X.std() > 0]
    Z = ((X - X.mean()) / X.std()).to_numpy()
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    scores = U * S
    var = (S ** 2) / (S ** 2).sum()
    loadings = Vt.T * (S / np.sqrt(len(Z) - 1))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].bar(range(1, len(var) + 1), var * 100)
    ax[0].plot(range(1, len(var) + 1), np.cumsum(var) * 100, "r-o")
    ax[0].set_xlabel("Component")
    ax[0].set_ylabel("Explained variance (%)")
    ax[0].set_title("Scree plot (red = cumulative)")

    hue = d[color].astype(str).values if color else None
    sns.scatterplot(x=scores[:, 0], y=scores[:, 1], hue=hue, alpha=0.4, ax=ax[1], legend="brief")
    scale = np.abs(scores[:, :2]).max() / np.abs(loadings[:, :2]).max() * 0.8
    for i, name in enumerate(X.columns):
        ax[1].arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale, color="red", head_width=0.05 * scale)
        ax[1].text(loadings[i, 0] * scale * 1.1, loadings[i, 1] * scale * 1.1, _title(name), color="red")
    ax[1].set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax[1].set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    ax[1].set_title("PCA biplot")
    plt.tight_layout()
    plt.show()
    display(pd.DataFrame(loadings[:, :3], index=X.columns, columns=["PC1", "PC2", "PC3"]).round(2))


# ---- numeric + several categoricals ----------------------------------------

def multi_facet_scatter_c(df, x_var, y_var, row=None, col=None, hue=None, sample=3000, filters=None):
    """Small multiples of scatterplots split by up to two categories."""
    d = _prep(df, filters, [x_var, y_var, row, col, hue])
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    g = sns.relplot(data=d, x=x_var, y=y_var, row=row, col=col, hue=hue, alpha=0.5, height=3.5, aspect=1.2)
    g.fig.suptitle(f"{_title(y_var)} vs {_title(x_var)}", y=1.02)
    plt.show()


def multi_facet_dist_c(df, num, row=None, col=None, hue=None, kind="hist", filters=None):
    """Small multiples of distributions. kind: 'hist' | 'kde' | 'ecdf' | 'box' | 'violin'."""
    d = _prep(df, filters, [num, row, col, hue])
    if kind in ("box", "violin"):
        g = sns.catplot(data=d, x=col or hue, y=num, row=row if col else None, hue=hue, kind=kind,
                        height=3.8, aspect=1.3)
    else:
        g = sns.displot(data=d, x=num, row=row, col=col, hue=hue, kind=kind, height=3.5, aspect=1.2)
    plt.show()


def multi_heatmap_pivot(df, index, columns, values, agg="mean", annot=True, filters=None):
    """Pivot heatmap: aggregated numeric `values` across two categoricals."""
    d = _prep(df, filters, [index, columns, values])
    p = d.pivot_table(index=index, columns=columns, values=values, aggfunc=agg, observed=True)
    plt.figure(figsize=(max(8, 1.1 * p.shape[1]), max(5, 0.6 * p.shape[0])))
    sns.heatmap(p, annot=annot, fmt=".1f", cmap="YlOrRd")
    plt.title(f"{agg.capitalize()} {_title(values)} by {_title(index)} x {_title(columns)}")
    plt.show()


def multi_grouped_bar_c(df, x_var, hue, y_var, agg="mean", filters=None):
    """Grouped bars of an aggregated numeric by two categoricals (with 95% CI when agg='mean')."""
    d = _prep(df, filters, [x_var, hue, y_var])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=d, x=x_var, y=y_var, hue=hue, estimator=agg, errorbar=("ci", 95) if agg == "mean" else None)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{agg.capitalize()} {_title(y_var)} by {_title(x_var)} and {_title(hue)}")
    plt.show()


def multi_line_c(df, date_col, y_var, hue, freq="ME", agg="mean", filters=None):
    """Trend lines per category (thin wrapper around bivar_line_c)."""
    bivar_line_c(df, date_col, y_var, freq=freq, agg=agg, hue=hue, filters=filters)


def multi_radar_px(df, group_col, cols, filters=None):
    """Radar / spider chart comparing group profiles on min-max scaled numeric columns."""
    d = _prep(df, filters, [group_col] + list(cols))
    scaled = (d[cols] - d[cols].min()) / (d[cols].max() - d[cols].min()).replace(0, 1)
    scaled[group_col] = d[group_col].values
    prof = scaled.groupby(group_col, observed=True)[cols].mean()
    fig = go.Figure()
    for g, row in prof.iterrows():
        fig.add_trace(go.Scatterpolar(r=row.tolist() + [row.iloc[0]],
                                      theta=[_title(c) for c in cols] + [_title(cols[0])],
                                      fill="toself", name=str(g), opacity=0.5))
    fig.update_layout(title=f"Profile of {_title(group_col)} (min-max scaled means)",
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    fig.show()


# ---- hierarchical / flow charts: treemap, sunburst, sankey, icicle ---------

def _hier(df, path, values=None, color=None, agg="mean", filters=None):
    """Aggregate to one row per path combination: size = count (or sum of `values`),
    color = agg of `color` column."""
    d = _prep(df, filters, list(path) + [values, color])
    for c in path:
        d[c] = d[c].astype(str)
    g = d.groupby(list(path), observed=True)
    out = g.size().reset_index(name="count")
    if values:
        out["size"] = g[values].sum().values
    else:
        out["size"] = out["count"]
    if color:
        out[color] = g[color].agg(agg).values
    return out


def multi_treemap_px(df, path, values=None, color=None, agg="mean", filters=None):
    """Treemap. path = list of categorical columns (outer -> inner).
    Tile size = row count (or sum of `values`); tile color = `agg` of `color` column.
    e.g. multi_treemap_px(df, ['country','gender'], color='daily_active_minutes_instagram')"""
    h = _hier(df, path, values, color, agg, filters)
    fig = px.treemap(h, path=[px.Constant("All")] + list(path), values="size", color=color,
                     color_continuous_scale="RdYlGn_r" if color else None,
                     title=f"Treemap: {' > '.join(map(_title, path))}")
    fig.update_traces(textinfo="label+percent root")
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    fig.show()


def multi_sunburst_px(df, path, values=None, color=None, agg="mean", filters=None):
    """Sunburst. Same arguments as multi_treemap_px."""
    h = _hier(df, path, values, color, agg, filters)
    fig = px.sunburst(h, path=list(path), values="size", color=color,
                      color_continuous_scale="RdYlGn_r" if color else None,
                      title=f"Sunburst: {' > '.join(map(_title, path))}")
    fig.update_traces(textinfo="label+percent parent")
    fig.show()


def multi_icicle_px(df, path, values=None, color=None, agg="mean", filters=None):
    """Icicle chart (rectangular alternative to sunburst)."""
    h = _hier(df, path, values, color, agg, filters)
    px.icicle(h, path=[px.Constant("All")] + list(path), values="size", color=color,
              color_continuous_scale="Viridis" if color else None).show()


def multi_sankey_px(df, cols, filters=None, top_n=12):
    """Sankey of flows between 2+ categorical columns, left to right in the given order.
    Each column keeps its `top_n` most common levels; the rest become 'Other'."""
    d = _prep(df, filters, list(cols))
    for c in cols:
        keep = d[c].astype(str).value_counts().head(top_n).index
        d[c] = d[c].astype(str).where(d[c].astype(str).isin(keep), "Other")

    labels, idx = [], {}
    for c in cols:
        for v in d[c].unique():
            key = (c, v)
            idx[key] = len(labels)
            labels.append(f"{v}")

    src, tgt, val = [], [], []
    for a, b in zip(cols[:-1], cols[1:]):
        flow = d.groupby([a, b], observed=True).size().reset_index(name="n")
        for _, r in flow.iterrows():
            src.append(idx[(a, r[a])])
            tgt.append(idx[(b, r[b])])
            val.append(r["n"])

    fig = go.Figure(go.Sankey(node=dict(label=labels, pad=15, thickness=18),
                              link=dict(source=src, target=tgt, value=val)))
    fig.update_layout(title=" -> ".join(map(_title, cols)))
    fig.show()


def multi_parcats_px(df, cols, color=None, sample=5000, filters=None):
    """Parallel categories plot (interactive alternative to Sankey)."""
    d = _prep(df, filters, list(cols) + [color])
    if sample and len(d) > sample:
        d = d.sample(sample, random_state=42)
    px.parallel_categories(d, dimensions=cols, color=color,
                           color_continuous_scale=px.colors.sequential.Inferno if color else None).show()


# =============================================================================
# QUICK-START (copy into notebook cells)
# =============================================================================
# --- Step 1: Univariate ---
# uni_hist_c(df, 'daily_active_minutes_instagram')
# uni_num_grid(df, kind='box')
# uni_qq_c(df, 'perceived_stress_score')
# uni_count_c(df, 'gender', normalize=True)
# uni_pie_px(df, 'country')
#
# --- Step 2: Bivariate ---
# bivar_scatter_c(df, 'daily_active_minutes_instagram', 'perceived_stress_score', hue='gender')
# bivar_binned_trend_c(df, 'daily_active_minutes_instagram', 'self_reported_happiness')
# bivar_kde_c(df, 'gender', 'user_engagement_score')
# bivar_crosstab_heatmap(df, 'gender', 'country')
# bivar_test(df, 'gender', 'self_reported_happiness')
#
# --- Step 3: Multivariate ---
# multi_pairplot_c(df, ['daily_active_minutes_instagram','perceived_stress_score','self_reported_happiness'], hue='gender')
# multi_treemap_px(df, ['country','gender'], color='daily_active_minutes_instagram')
# multi_sunburst_px(df, ['country','gender','account_type'], color='perceived_stress_score')
# multi_sankey_px(df, ['age_group','gender','country'])
# multi_heatmap_pivot(df, 'country', 'gender', 'self_reported_happiness')
# multi_pca_biplot(df, color='gender')
# multi_radar_px(df, 'gender', ['daily_active_minutes_instagram','user_engagement_score','self_reported_happiness'])