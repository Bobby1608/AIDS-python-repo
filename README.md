# AI-Based Network Intrusion Detection System

This project implements a robust Intrusion Detection System (IDS) using a hybrid machine learning approach. It leverages models like Random Forest, XGBoost, and a Multi-Layer Perceptron (MLP) to accurately classify network traffic and identify potential threats. The system is served through a user-friendly web interface built with Flask.

## Features

- **Data Preprocessing:** Efficiently cleans and preprocesses raw network data for feature extraction.
- **Multiple ML Models:** Implements and evaluates several high-performance classifiers (Random Forest, XGBoost, MLP).
- **Model Evaluation:** Generates confusion matrices, ROC curves, and classification reports to assess model performance.
- **Real-time Web Dashboard:** A Flask-based web application for interacting with the trained models and classifying new traffic data.
- **Large File Handling:** Uses Git LFS to manage large trained model files within the repository.

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- [Python 3.8+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- **Git LFS (Large File Storage)** - This is crucial for downloading the trained models.
  - Install it from [here](https://git-lfs.github.com/).
  - After installing, run `git lfs install` once in your terminal to initialize it.

### Installation & Setup

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/Bobby1608/AIDS-python-repo.git](https://github.com/Bobby1608/AIDS-python-repo.git)
    cd AIDS-python-repo
    ```

2.  **Download Model Files**
    The trained models are stored using Git LFS. Run the following command to download them.

    ```bash
    git lfs pull
    ```

    You should see the `.pkl` files in the `models/` directory change from being a few bytes to their full size (e.g., 405 MB).

3.  **Download the Datasets**
    Due to their large size, the raw datasets (`.csv`) and processed NumPy arrays (`.npy`) are not stored in this Git repository. Please download them from the link(s) below and place them in the correct folders.

    - **Required Datasets:**
      - Download the dataset files here: `[LINK_TO_YOUR_DATA_ON_GOOGLE_DRIVE_OR_RELEASE]` - **Placement:**
      - Place all `.csv` files inside the `dataset/` folder.
      - Place all `.npy` files inside the `processed/` folder.

4.  **Create a Virtual Environment**
    It is highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    # Create the virtual environment
    python -m venv .venv

    # Activate it (on Windows)
    .venv\Scripts\activate

    # On macOS/Linux, you would use:
    # source .venv/bin/activate
    ```

5.  **Install Dependencies**
    Install all required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

After completing the setup, you can run the main web application.

1.  **Start the Flask Server**
    Navigate to the project's root directory and run the main application script:

    ```bash
    python src/webapp/app_combined2.py
    ```

    _(Note: If your main Flask app file has a different name, please update the command accordingly.)_

2.  **Access the Web Interface**
    Open your web browser and navigate to the following address:
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Project Team

This section can be filled out by the project contributors.

-Bhuban Wakode

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
