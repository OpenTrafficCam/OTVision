from tkinter import filedialog
import os

# Define relative path to test data (using os.path.dirname repeatedly)
TEST_DATA_FOLDER = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + r"\tests\data"
)


def select_refpts_files():
    """
    Select files containing reference points in pixel and world coordinates via file
    browser.
    """

    refpts_pixel_path = filedialog.askopenfilename(
        initialdir=TEST_DATA_FOLDER,
        title="Select reference points in pixel coordinates (.txt or .npy)",
        filetypes=(
            ("Numpy files", "*.npy"),
            ("Text files", "*.txt"),
            ("all files", "*.*"),
        ),
    )

    refpts_world_path = filedialog.askopenfilename(
        initialdir=TEST_DATA_FOLDER,
        title="Select reference points in World coordinates (.txt or .npy)",
        filetypes=(
            ("Numpy files", "*.npy"),
            ("Text files", "*.txt"),
            ("all files", "*.*"),
        ),
    )

    return refpts_pixel_path, refpts_world_path


def select_traj_files():
    """
    Select one or multiple files containing trajectories in world coordinates via file
    browser
    """

    traj_pixel_paths = filedialog.askopenfilenames(
        initialdir=TEST_DATA_FOLDER,
        title="Select trajectories in pixel coordinates (.pkl. or .csv)",
        filetypes=(
            ("Python pickle files", "*.pkl"),
            ("CSV files", "*.csv"),
            ("All files", "*.*"),
        ),
    )

    return traj_pixel_paths


if __name__ == "__main__":
    print(select_refpts_files())
    print(select_traj_files())
