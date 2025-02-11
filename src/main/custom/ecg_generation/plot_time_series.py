import pandas as pd
import plotly.graph_objects as go
import argparse

def create_ecg_interactive_plot_with_dropdown(csv_path: str) -> None:
    """
    Reads ECG data from a CSV file and creates an interactive ECG time series plot
    with a dropdown to select patients.

    Parameters:
      csv_path (str): The path to the ECG CSV file.
      
    Assumptions:
      - The CSV file must contain at least the following columns:
          "patient_id", "time_in_block_sec", "ecg_amplitude".
    """
    # Read CSV data into an immutable DataFrame
    df = pd.read_csv(csv_path)
    
    # Get the unique patient IDs
    patient_ids = df["patient_id"].unique()
    
    # Create a figure and add one trace per patient.
    # Only the first patient's data is visible by default.
    fig = go.Figure()
    for patient_id in patient_ids:
        patient_data = df[df["patient_id"] == patient_id]
        fig.add_trace(go.Scatter(
            x=patient_data["time_in_block_sec"],
            y=patient_data["ecg_amplitude"],
            mode="lines+markers",
            name=patient_id,
            visible=(patient_id == patient_ids[0])
        ))
    
    # Create dropdown buttons for patient selection.
    # Each button updates the "visible" array so that only one patient's trace is shown.
    buttons = []
    for i, patient_id in enumerate(patient_ids):
        # Create a visibility list with only the i-th trace visible.
        visibility = [False] * len(patient_ids)
        visibility[i] = True
        buttons.append(dict(
            label=patient_id,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"ECG Signal for Patient {patient_id}"}
            ]
        ))
    
    # Update layout with the dropdown menu and axes labels.
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )],
        title=f"ECG Signal for Patient {patient_ids[0]}",
        xaxis_title="Time in Block (sec)",
        yaxis_title="ECG Amplitude"
    )
    
    # Display the interactive plot
    fig.show()

def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.

    Returns:
      argparse.Namespace: Parsed command line arguments having the CSV file path.
    """
    parser = argparse.ArgumentParser(
        description="Create an interactive ECG time series plot with a patient dropdown."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the ECG CSV file (e.g., 'ecg_data.csv')."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    create_ecg_interactive_plot_with_dropdown(args.csv_path)