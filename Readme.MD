TSL Letter Prediction API
Project Overview
This project implements a Thai Sign Language (TSL) letter prediction service that processes hand landmarks and predicts corresponding letters using a machine learning model. The system uses a pre-trained MLP (Multilayer Perceptron) model to interpret hand gestures and convert them to letters.
Repository Structure
PRODUCTION MODEL/
├── main.py               # FastAPI application and endpoint definitions
├── model_loader.py       # Model loading and prediction functionality
├── utils.py              # Utility functions for data preprocessing
├── requirements.txt      # Project dependencies
└── model/
    └── mlp_tsl_static.pkl  # Pre-trained TSL prediction model

How It Works
The application follows this workflow:

Client sends hand landmark data (21 points with x,y,z coordinates) to the API
The API normalizes the landmarks using min-max scaling
The normalized data is fed into the pre-trained model
The model predicts the corresponding letter
The API returns the predicted letter to the client
Dependencies
The project requires the following dependencies:

FastAPI - Web framework
Uvicorn - ASGI server
Joblib - Model serialization
Scikit-learn - ML framework
NumPy - Numerical processing
Mediapipe - Hand landmark detection
OpenCV - Computer vision
Pillow - Image processing
Python-dotenv - Environment configuration
Azure Cognitive Services Speech - Speech capabilities

Deployment
The application can be deployed using:

Docker containers
Cloud platforms like Azure App Service or AWS Elastic Beanstalk
Traditional server deployment with Gunicorn/Uvicorn