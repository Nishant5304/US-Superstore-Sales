# Helper page for Instagram Usage Analytics

# Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from IPython.display import display


# Data Loading

orders_fact = pd.read_parquet('/app/data/processed/orders_fact.parquet')

date_dim = pd.read_parquet('/app/data/processed/date_dim.parquet')
products_dim = pd.read_parquet('/app/data/processed/products_dim.parquet')
customers_dim = pd.read_parquet('/app/data/processed/customers_dim.parquet')
location_dim = pd.read_parquet('/app/data/processed/location_dim.parquet')
ship_mode_dim = pd.read_parquet('/app/data/processed/ship_mode_dim.parquet')

# Functions to check the dataframes and their shapes

def dataframe():

    print('Customer Dimension Data:', customers_dim.columns.to_list())
    print('Orders Fact Data:', orders_fact.columns.to_list())
    print('Product Dimension Data:', products_dim.columns.to_list())
    print('Date Dimension Data:', date_dim.columns.to_list())
    print('Location Dimension Data:', location_dim.columns.to_list())
    print('Ship Mode Dimension Data:', ship_mode_dim.columns.to_list())

def dataframe_shape():

    print('Customer Dimension Data:', customers_dim.columns.shape)
    print('Orders Fact Data:', orders_fact.columns.shape)
    print('Product Dimension Data:', products_dim.columns.shape)
    print('Date Dimension Data:', date_dim.columns.shape)
    print('Location Dimension Data:', location_dim.columns.shape)
    print('Ship Mode Dimension Data:', ship_mode_dim.columns.shape)


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

# This function takes a dataframe and a dictionary of filters where the keys are column names and the values are either a single value for equality or a tuple specifying an operation and a value. The function returns a filtered dataframe based on the specified conditions.

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

# Structure for univariate analysis



### Functions for Univariate Analysis

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

    personal_num_summary = num_df.describe().round(2).T

    personal_num_summary["missing_%"] = (
        num_df.isna().mean() * 100
    )

    q1 = personal_num_summary["25%"]
    q3 = personal_num_summary["75%"]

    iqr = q3 - q1

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


# creating a function to generate categoric summary of the dataframe with the option to apply filters and show proportions or counts.


# Created merged tables to use for checking correlations btw numerical columns that comprised of int and float datatype.


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


# Created merged tables for group-wise descriptive analysis, where we will find if the categorical columns in our dataframe have any effect on the already high correlationed daily & weekly usage variables.

# featured columns


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



# featured columns


# featured columns



# Created a subset of the main merged table to test which numerical variables(outcomes) show the strongest relation to our 3 drivers.

# function for group-wise descriptive analysis of the categorical columns against the outcome variable with the option to apply filters and show the IQR and fences to check variability and outliers in the data.

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

# Once we found the significant categorical columns i.e.. the Drivers, we will check the numerical columns against the driver variable to check maximum variability.
# The driver variables are = (age_group, stress_cat, & happiness_cat)


# Created a subset of the main merged table to test which numerical variables(outcomes) show the strongest relation to our 3 drivers.


time_related = [ 'time_on_feed_per_day',
                 'time_on_reels_per_day',
                 'time_on_explore_per_day',
                 'time_on_messages_per_day'
                 ]

message_related= ['dms_sent_per_week',
                  'dms_received_per_week',
                  'posts_created_per_week'
                  ]

interaction_related = ['reels_watched_per_day',
                       'likes_given_per_day',
                       'stories_viewed_per_day',
                       'comments_written_per_day'
                       ]

ads = [ 'ads_viewed_per_day',
       'ads_clicked_per_day'
       ]


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
# Here x in violinplot is identified by outcome variable and y by categorical variable to stimulate a landscape (horizontal) violinplot to better understand the distribution.

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