import streamlit as st


def update_point_color(df_incomplete, incomplete_column):
    """
    Update the color of a point in a DataFrame based on a slider value.
    
    Args:
        df_incomplete (pandas.DataFrame): The DataFrame containing the point to be updated.
        incomplete_column (str): The name of the column in which the point is located.
    """
    df_incomplete.at[st.session_state['current_null_index'], incomplete_column] = st.session_state['my_slider']

def update_session_state(X, y_pred, border_points):
    """
    Update the session state variables with the given values.

    Parameters:
    X (numpy.ndarray): The input data.
    y_pred (numpy.ndarray): The predicted values.
    border_points (list): The list of border points.

    Returns:
    None
    """
    st.session_state.X = X
    st.session_state['border_points'] = border_points
    st.session_state['y_pred'] = y_pred
