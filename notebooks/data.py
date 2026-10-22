
# Helper page for Instagram Usage Analytics

# Libraries

import pandas as pd


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
