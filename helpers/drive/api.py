import time
import pandas as pd
import gspread
import streamlit as st
from google.oauth2 import service_account
# from gsheetsdb import connect

class GSAuthentication:
    """
    Singleton class for Google Sheets authentication.

    This class provides a singleton instance for authenticating with Google Sheets API
    using a service account and managing the credentials.

    Attributes:
        _instance: The singleton instance of the class.
        gc: The gspread.Client object for interacting with Google Sheets API.
        url: The URL of the private Google Sheets document.

    Methods:
        __new__: Creates a new instance of the class if it doesn't exist, otherwise returns the existing instance.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GSAuthentication, cls).__new__(cls)
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive.readonly",
                ],
            )
            print("created")
            # TODO: refactor to use SQL not custom code
            # conn = connect(credentials=credentials)
            cls.gc  = gspread.Client(credentials)
            cls.url = st.secrets['private_gsheets_url']

        return cls._instance


def download_datasets(g, names):
    """
    Download datasets from Google Sheets and store them in session state.

    Args:
        g (GoogleClient): An instance of the GoogleClient class.
        names (list): A list of names of the Google Sheets to download.

    Returns:
        None
    """
    for name in names:
        gsheet = g.gc.open(name, st.secrets['private_data_folder_id'])
        data   = gsheet.sheet1.get_all_values()
        df     = pd.DataFrame(data[1:], columns=data[0])
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
        st.session_state[f'ds_{name}'] = df


def save_results(g, human_imputation, mean_imputation, cluster_mean_imputation, knn_imputation, cluster_knn_imputation, annotator_id, seed, dataset_name, na_fraction, projection_key):
    """
    Saves the imputation results to a Google Sheets worksheet.

    Args:
        g (object): The Google Sheets client object.
        human_imputation (list): Human imputation results.
        mean_imputation (list): Mean imputation results.
        cluster_mean_imputation (list): Cluster mean imputation results.
        knn_imputation (list): k-nearest neighbors imputation results.
        cluster_knn_imputation (list): Cluster k-nearest neighbors imputation results.
        annotator_id (str): The ID of the annotator.
        seed (int): The random seed used for imputation.
        dataset_name (str): The name of the dataset.
        na_fraction (float): The fraction of missing values in the dataset.
        projection_key (str): The key of the projection used for the imputation.

    Returns:
        None
    """

    gsheet   = g.gc.open_by_url(g.url)
    datasets = {"iris": 0, "wola": 1, "stamp_type": 2}
    worksheet_number = datasets[dataset_name]
    wsheet   = gsheet.get_worksheet(worksheet_number)
    data     = pd.DataFrame(wsheet.get_all_values())
    now      = time.localtime(time.time())
    row_data = [
        time.strftime("%m/%d/%Y, %H:%M:%S", now),
        annotator_id,
        seed,
        na_fraction,
        projection_key,
        str(human_imputation),
        str(mean_imputation),
        str(cluster_mean_imputation),
        str(knn_imputation),
        str(cluster_knn_imputation)
    ]
    wsheet.insert_row(row_data, len(data) + 1)
