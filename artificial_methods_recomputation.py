# This scrpit is used to recompute the imputation results for the iris, stamp_type and wola datasets. It was designed for a singular usage, but might be helpful for future reference.
import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from helpers.streamlit.callbacks import update_point_color
from helpers.streamlit.utils import init_session_state, read_settings_from_file, validate, init_new_annotation_task, calculate_y_pred_uncertain
from helpers.utils.dataset import add_projection_dimensions_to_df_incomplete, save_results, generate_incomplete_dataset
from helpers.utils.imputation import impute_global_mean, impute_cluster_mean, impute_knn_mean, impute_cluster_knn_mean
from helpers.utils.projections import projections_dict, plot_scatter
from helpers.streamlit.callbacks import update_session_state

def init_annotation_task(dataset_settings, seed, fraction):
    """
    Initializes a new annotation task by generating an incomplete dataset, calculating uncertainty predictions,
    and setting session state variables.

    Args:
        dataset_settings (dict): A dictionary containing the settings for the dataset.

    Returns:
        pandas.DataFrame: The generated incomplete dataset.
    """

    st.session_state['dataset_generation_seed'] = seed

    incomplete_column = dataset_settings['incomplete_column']
    df_incomplete = generate_incomplete_dataset(st.session_state['dataset_generation_seed'],
                                                dataset_settings['name'],
                                                incomplete_column,
                                                fraction)
    update_session_state(*calculate_y_pred_uncertain(df_incomplete, dataset_settings))
    df_incomplete['y_pred'] = st.session_state['y_pred']
    st.session_state['imputed_values'] = {}
    st.session_state['annotated_points'] = 0
    st.session_state['min_value'] = float(df_incomplete[incomplete_column].min())
    st.session_state['max_value'] = float(df_incomplete[incomplete_column].max())


    if not len(st.session_state["border_points"]):
        st.text('Sorry, there are no border points to annotate. Choose other dataset or na_fraction.')
    else:
        st.session_state['started'] = True
    return df_incomplete


iris_result = pd.read_csv('.streamlit/results - iris.csv')
stamp_type_result = pd.read_csv('.streamlit/results - stamp_type.csv')
wola_result = pd.read_csv('.streamlit/results - wola.csv')
print(iris_result.iloc[:,[2,3]])
datasets_settings = read_settings_from_file()
with st.sidebar:
        st.markdown('## Settings')
        dataset_name_selectbox = st.selectbox('Select dataset', [dataset['name'] for dataset in datasets_settings])
        na_fraction_selectbox = st.selectbox('Select na_fraction', [0.1, 0.2, 0.3], key="na_fraction_selectbox")

iris_fixed_results = []
for i in range(iris_result.shape[0]):
    seed = iris_result.iloc[i,2]
    fraction = iris_result.iloc[i,3]
    print('Seed:', seed, 'Fraction:', fraction)
    init_session_state(datasets_settings)
    print('dataset_name_selectbox', dataset_name_selectbox)
    dataset_name_selectbox = 'iris'
    dataset_settings = next(dataset for dataset in datasets_settings if dataset['name'] == dataset_name_selectbox)
    incomplete_column = dataset_settings['incomplete_column']
    reference_columns = dataset_settings['reference_columns']

    df_incomplete = init_annotation_task(dataset_settings, seed = seed, fraction = fraction)
    st.session_state.df_incomplete = df_incomplete
    incomplete_column = dataset_settings['incomplete_column']

    na_indexes    = st.session_state['border_points']
    df_incomplete = add_projection_dimensions_to_df_incomplete(str('tsne'), 
                                                            st.session_state.X, 
                                                            st.session_state.df_incomplete)
    real_values   = st.session_state[f'ds_{dataset_settings["name"]}'][incomplete_column][na_indexes].tolist()
    global_means  = impute_global_mean(df_incomplete, incomplete_column, na_indexes)
    cluster_means = impute_cluster_mean(df_incomplete, incomplete_column, na_indexes)
    knn           = impute_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
    cluster_knn   = impute_cluster_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
    rows          = ["Mean", "Cluster mean", "knn", "cluster knn"]
    imputations   = [global_means, cluster_means, knn, cluster_knn]
    iris_fixed_results.append([seed, fraction, global_means, cluster_means, knn, cluster_knn])

iris_fixed_results_df = pd.DataFrame(iris_fixed_results, columns=['Seed', 'Fraction', 'Global Mean', 'Cluster Mean', 'KNN', 'Cluster KNN'])
iris_fixed_results_df.to_csv('.streamlit/results_fixed_iris.csv', index=False)

stamp_type_fixed_result = []
for i in range(stamp_type_result.shape[0]):
    seed = stamp_type_result.iloc[i,2]
    fraction = stamp_type_result.iloc[i,3]
    print('Seed:', seed, 'Fraction:', fraction)
    init_session_state(datasets_settings)
    dataset_name_selectbox = 'stamp_type'
    dataset_settings = next(dataset for dataset in datasets_settings if dataset['name'] == dataset_name_selectbox)
    incomplete_column = dataset_settings['incomplete_column']
    reference_columns = dataset_settings['reference_columns']

    df_incomplete = init_annotation_task(dataset_settings, seed = seed, fraction = fraction)
    st.session_state.df_incomplete = df_incomplete
    incomplete_column = dataset_settings['incomplete_column']

    na_indexes    = st.session_state['border_points']
    df_incomplete = add_projection_dimensions_to_df_incomplete(str('tsne'), 
                                                            st.session_state.X, 
                                                            st.session_state.df_incomplete)
    real_values   = st.session_state[f'ds_{dataset_settings["name"]}'][incomplete_column][na_indexes].tolist()
    global_means  = impute_global_mean(df_incomplete, incomplete_column, na_indexes)
    cluster_means = impute_cluster_mean(df_incomplete, incomplete_column, na_indexes)
    knn           = impute_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
    cluster_knn   = impute_cluster_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
    rows          = ["Mean", "Cluster mean", "knn", "cluster knn"]
    imputations   = [global_means, cluster_means, knn, cluster_knn]
    stamp_type_fixed_result.append([seed, fraction, global_means, cluster_means, knn, cluster_knn])

stamp_type_fixed_result = pd.DataFrame(stamp_type_fixed_result, columns=['Seed', 'Fraction', 'Global Mean', 'Cluster Mean', 'KNN', 'Cluster KNN'])
stamp_type_fixed_result.to_csv('.streamlit/results_fixed_stamp_type.csv', index=False)

wola_fixed_result = []
for i in range(wola_result.shape[0]):
    seed = wola_result.iloc[i,2]
    fraction = wola_result.iloc[i,3]
    print('Seed:', seed, 'Fraction:', fraction)
    init_session_state(datasets_settings)
    dataset_name_selectbox = 'wola'
    dataset_settings = next(dataset for dataset in datasets_settings if dataset['name'] == dataset_name_selectbox)
    incomplete_column = dataset_settings['incomplete_column']
    reference_columns = dataset_settings['reference_columns']
    df_incomplete = init_annotation_task(dataset_settings, seed = seed, fraction = fraction)
    st.session_state.df_incomplete = df_incomplete
    incomplete_column = dataset_settings['incomplete_column']

    na_indexes    = st.session_state['border_points']
    df_incomplete = add_projection_dimensions_to_df_incomplete(str('tsne'), 
                                                            st.session_state.X, 
                                                            st.session_state.df_incomplete)
    real_values   = st.session_state[f'ds_{dataset_settings["name"]}'][incomplete_column][na_indexes].tolist()
    global_means  = impute_global_mean(df_incomplete, incomplete_column, na_indexes)
    cluster_means = impute_cluster_mean(df_incomplete, incomplete_column, na_indexes)
    knn           = impute_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
    cluster_knn   = impute_cluster_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
    rows          = ["Mean", "Cluster mean", "knn", "cluster knn"]
    imputations   = [global_means, cluster_means, knn, cluster_knn]
    wola_fixed_result.append([seed, fraction, global_means, cluster_means, knn, cluster_knn])

wola_fixed_result = pd.DataFrame(wola_fixed_result, columns=['Seed', 'Fraction', 'Global Mean', 'Cluster Mean', 'KNN', 'Cluster KNN'])
wola_fixed_result.to_csv('.streamlit/results_fixed_wola.csv', index=False)