# 🛡️ AI-Based Network Intrusion Detection System (NIDS)

This project implements a robust, real-time Intrusion Detection System (IDS) utilizing a **hybrid machine learning approach**. It accurately classifies network traffic into benign or malicious categories by leveraging high-performance ensemble models and neural networks.

The system is deployed via a modern web interface built with **Flask**, allowing for interactive demonstration and real-time classification of new network flow data.

## 🚀 Key Features

* **Hybrid ML Core:** Uses an optimized stack featuring **Random Forest**, **XGBoost**, and a **Multi-Layer Perceptron (MLP)** for maximum classification accuracy.
* **Data Pipeline:** Handles large datasets with efficient **preprocessing** and **feature engineering** tailored for network flow analysis.
* **Model Validation:** Comprehensive evaluation generating **Confusion Matrices**, **ROC Curves**, and full classification reports.
* **Web Dashboard:** A user-friendly, responsive Flask application for live interaction with trained models.
* **Large File Management:** Utilizes Git LFS (Large File Storage) for managing and versioning multi-hundred-megabyte trained model files (`.pkl`).

---

## ⚙️ Getting Started

Follow these instructions to set up and run the NIDS project locally.

### Prerequisites

Ensure the following tools are installed on your system:

* [**Python 3.8+**](https://www.python.org/downloads/)
* [**Git**](https://git-scm.com/)
* **Git LFS (Large File Storage):** Crucial for downloading the trained model files.
    * Install it from [the official site](https://git-lfs.github.com/).
    * Initialize it once in your terminal: `git lfs install`

### Installation Steps

1.  **Clone the Repository**

    ```bash
    git clone [https://github.com/Bobby1608/AIDS-python-repo](https://github.com/Bobby1608/AIDS-python-repo)
    cd AIDS-python-repo
    ```

2.  **Download Model Files (via Git LFS)**

    The pre-trained models are stored using LFS. Run this command to download the full files:

    ```bash
    git lfs pull
    ```

    > **Verification:** Check the `models/` directory; the `.pkl` files should now show their full size (e.g., 400+ MB).

3.  **Acquire Dataset Files**

    The large raw and processed datasets are hosted externally. The project uses the well-known **NSL-KDD Dataset** (KDDTrain+ and KDDTest+ files). Please search for and download these files, and place them in the correct directory structure:

    * **Required Datasets:** **UNSW-NB15 Dataset** (The canonical dataset for this type of IDS)
    * **Placement:**
        * Place all **raw CSV files** (`.csv`) inside the `./dataset/` folder.
        * Place all **processed NumPy arrays** (`.npy`) inside the `./processed/` folder.

4.  **Create & Activate Virtual Environment**

    It is mandatory to use a virtual environment to isolate project dependencies.

    ```bash
    # Create the virtual environment
    python -m venv .venv

    # Activate on Windows (Command Prompt/PowerShell)
    .venv\Scripts\activate

    # Activate on macOS/Linux (Bash/Zsh)
    # source .venv/bin/activate
    ```

5.  **Install Python Dependencies**

    Install all necessary packages listed in the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run the Web Application

1.  **Start the Flask Server**

    Ensure your virtual environment is active, navigate to the project root, and run the main application file:

    ```bash
    python src/webapp/app_combined2.py
    ```

    > **Note:** If you renamed the main file, update the command above accordingly.

2.  **Access the Dashboard**

    Open your web browser and navigate to the local server address:

    [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🤝 Project Contributor

* Bhuban Wakode - *Lead Developer*
    * *GitHub:* `Bobby1608`
    * *LinkedIn:* (https://www.linkedin.com/in/bhuban-wakode/)

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file in the repository root for full details.

