import json
import time

import streamlit as st
from helpers.drive.api import GSAuthentication, download_datasets
from helpers.streamlit.callbacks import update_session_state
from helpers.utils.dataset import generate_incomplete_dataset, find_uncertain_y_indexes
from sklearn.cluster import KMeans


def validate(user_id):
    validated = user_id.isdigit()
    if validated:
        st.session_state['id'] = user_id
    return validated


def read_settings_from_file(filename="settings.json"):
    with open(filename, "r") as f:
        settings = json.load(f)
    datasets_settings = settings['data']
    return datasets_settings


def init_session_state(datasets_settings):
    if 'gsheet' not in st.session_state:
        st.session_state["gsheet"] = GSAuthentication()
    if 'data' not in st.session_state:
        with st.spinner(text="In progress..."):
            download_datasets(st.session_state["gsheet"], [dataset['name'] for dataset in datasets_settings])
            st.session_state['data'] = True
    if 'started' not in st.session_state:
        st.session_state['started'] = False
    if 'annotated_points' not in st.session_state:
        st.session_state['annotated_points'] = 0


def calculate_y_pred_uncertain(df_incomplete, dataset_settings):
    """
    Calculate the predicted y values and border points for incomplete data.

    Args:
        df_incomplete (pandas.DataFrame): The DataFrame containing incomplete data.
        dataset_settings (dict): A dictionary containing dataset settings.

    Returns:
        tuple: A tuple containing the following:
            - X (numpy.ndarray): The normalized feature matrix.
            - y_pred (numpy.ndarray): The predicted cluster labels.
            - border_points (list): The indices of the border points.
    """
    X = df_incomplete[dataset_settings['reference_columns']]
    X = X.apply(lambda x: x / x.max(), axis=0).values

    kmeans = KMeans(n_clusters=dataset_settings['number_of_clusters'], random_state=0)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    incomplete_indexes  = df_incomplete[df_incomplete[dataset_settings['incomplete_column']].isnull()].index.tolist()
    uncertain_y_indexes = find_uncertain_y_indexes(df_incomplete, dataset_settings, incomplete_indexes)

    border_points = list(set(uncertain_y_indexes).intersection(incomplete_indexes))

    return X, y_pred, border_points


def init_new_annotation_task(dataset_settings):
    """
    Initializes a new annotation task by generating an incomplete dataset, calculating uncertainty predictions,
    and setting session state variables.

    Args:
        dataset_settings (dict): A dictionary containing the settings for the dataset.

    Returns:
        pandas.DataFrame: The generated incomplete dataset.
    """

    st.session_state['dataset_generation_seed'] = int(time.time())

    incomplete_column = dataset_settings['incomplete_column']
    df_incomplete = generate_incomplete_dataset(st.session_state['dataset_generation_seed'],
                                                dataset_settings['name'],
                                                incomplete_column,
                                                st.session_state.na_fraction_selectbox)
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



