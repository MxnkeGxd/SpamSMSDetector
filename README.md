# Spam SMS Detection App

This repository contains a machine learning project that detects whether an SMS message is "spam" or "ham" (not spam). The project includes a Jupyter Notebook for the model training process, a dataset, a trained model, a TF-IDF vectorizer, and a Streamlit web application to easily test the model.

### 📁 **Project Files**

* **`spamsmsdetection.ipynb`**: A Jupyter Notebook detailing the data cleaning, exploratory data analysis, model training, and evaluation steps. It uses the `spam.csv` dataset to train a spam detection model.
* **`spam.csv`**: The dataset used for training the model. It contains SMS messages labeled as either 'spam' or 'ham'.
* **`spam_model.pkl`**: The trained machine learning model, saved as a pickle file. This model is a `LinearSVC` from `scikit-learn`, which has been optimized for this classification task.
* **`vectorizer.pkl`**: The `TfidfVectorizer` used to convert text data into numerical features for the model. This vectorizer must be used on new input messages before making a prediction.
* **`app.py`**: A Streamlit web application that provides a user-friendly interface for the spam detection model. Users can input a message and get an instant prediction.

### 🚀 **How to Run the App**

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

The app will open in your web browser, allowing you to enter an SMS message and check if it's spam.

### 🛠️ **Dependencies**

The project dependencies are listed in the `requirements.txt` file. They include `pandas`, `scikit-learn`, `joblib`, and `streamlit`.
