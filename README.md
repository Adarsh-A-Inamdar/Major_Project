# Major_Project
# Leukemia Prediction Web Application

This project is a single-file web application that uses a multi-task deep learning model to predict the type and grade of leukemia from a blood cell image. The application is built with Flask on the backend and a simple, user-friendly interface on the frontend, allowing for quick and accurate predictions directly from your browser.

## ✨ Features

* **Dual Prediction:** Predicts both the **type** of leukemia (ALL, AML, CLL, CML) and the **grade** (Chronic, Accelerated, Blast).

* **Simple Web Interface:** Upload an image via a clean, modern web page.

* **Backend & Frontend in One:** The entire application runs from a single Python file, simplifying development and deployment.

* **Production-Ready Server:** Utilizes Gunicorn for robust and efficient deployment.

## 🛠️ Technologies Used

* **Python:** The core language for the application.

* **Flask:** The web framework used to build the backend server.

* **PyTorch:** The deep learning framework for the prediction model.

* **Gunicorn:** A production-grade web server used for deployment.

* **HTML, CSS, & JavaScript:** For the single-page frontend interface.

* **Tailwind CSS:** A utility-first CSS framework for rapid styling.

## 🚀 Getting Started

### Prerequisites

To run this application, you will need to have Python and the following libraries installed:

* `Flask`

* `torch`

* `torchvision`

* `Pillow`

* `gunicorn`

You can install all required dependencies using pip:

Project Readme

Markdown
# Leukemia Prediction Web Application

This project is a single-file web application that uses a multi-task deep learning model to predict the type and grade of leukemia from a blood cell image. The application is built with Flask on the backend and a simple, user-friendly interface on the frontend, allowing for quick and accurate predictions directly from your browser.

## ✨ Features

* **Dual Prediction:** Predicts both the **type** of leukemia (ALL, AML, CLL, CML) and the **grade** (Chronic, Accelerated, Blast).

* **Simple Web Interface:** Upload an image via a clean, modern web page.

* **Backend & Frontend in One:** The entire application runs from a single Python file, simplifying development and deployment.

* **Production-Ready Server:** Utilizes Gunicorn for robust and efficient deployment.

## 🛠️ Technologies Used

* **Python:** The core language for the application.

* **Flask:** The web framework used to build the backend server.

* **PyTorch:** The deep learning framework for the prediction model.

* **Gunicorn:** A production-grade web server used for deployment.

* **HTML, CSS, & JavaScript:** For the single-page frontend interface.

* **Tailwind CSS:** A utility-first CSS framework for rapid styling.

## 🚀 Getting Started

### Prerequisites

To run this application, you will need to have Python and the following libraries installed:

* `Flask`

* `torch`

* `torchvision`

* `Pillow`

* `gunicorn`

You can install all required dependencies using pip:

pip install Flask torch torchvision Pillow gunicorn

### Installation

1.  **Clone the repository** from GitHub.

2.  **Place the model file:** Ensure your trained PyTorch model file, named `multitask_model.pt`, is located in the root directory of your project.

3.  **Verify the file structure:** Your project should have a structure similar to this:

    ```
    /your-project/
    ├── src/
    │   └── main.py       # Your single-file application
    ├── requirements.txt  # Project dependencies
    ├── Procfile          # For deployment on platforms like Koyeb
    └── multitask_model.pt  # Your PyTorch model file

    ```

## 🖥️ Usage

### Running Locally

To run the application locally, navigate to the `src` directory and execute the main file.

python main.py


This will start the Flask development server. Open your web browser and navigate to `http://127.0.0.1:8080` to access the application.

### Running with Gunicorn

For a production environment, you should use Gunicorn. Make sure you are in the root directory of your project and run:

gunicorn --bind 0.0.0.0:8000 --workers 4 src.main:app


Then, access the application at `http://127.0.0.1:8000`.

## ⚙️ Deployment

This application is configured for easy deployment on platforms like Koyeb using the provided `requirements.txt` and `Procfile`. Simply connect your GitHub repository to your chosen platform, and it should automatically build and deploy the application.

