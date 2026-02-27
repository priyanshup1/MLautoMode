# Mini AutoML Platform

A unified, full-stack Machine Learning automation platform that enables users to easily upload datasets, preprocess data, train models, and generate actionable business insights—all without writing a single line of code.

## 🚀 Features

### 1. Dataset Upload & Detection
- **Smart Data Upload:** Seamlessly accept CSV uploads.
- **Problem Type Detection:** Automatically analyze the dataset to detect if it relates to a Classification or Regression problem.
- **Dataset Summary:** Extract detailed summaries including missing values, data types, and distribution.

### 2. Comprehensive Data Preprocessing
- **Missing Value Imputation:** Address missing data using mean, median, or mode.
- **Data Cleansing:** Remove duplicates safely.
- **Encoding:** Transform categorical variables into machine-readable formats.
- **Feature Scaling:** Apply appropriate scaling techniques.
- **Outlier Detection:** (Optional) Interquartile Range (IQR) detection to clean skewed data.

### 3. Powerful Model Training Engine
- **Automated Data Splitting:** Intelligent train/test splits.
- **Classification Models:** Logistic Regression, Random Forest, SVM, KNN, XGBoost.
- **Regression Models:** Linear Regression, Random Forest, Decision Tree, SVR, XGBoost.
- **Metrics Calculation:** Robust model evaluation using metrics suited for the problem type.

### 4. Interactive Visualizations & Insights
- **Core Visuals:** Target distribution arrays, correlation heatmaps, feature importance graphs.
- **Classification Graphs:** Class imbalance charts, ROC curves.
- **Regression Graphs:** Actual vs. Predicted plots, Residual analysis.
- **Business Insights:** Generate natural language "Why this matters" summaries to translate complex model outputs into actionable business intelligence.

---

## 🛠️ Technology Stack

**Backend**
- [FastAPI](https://fastapi.tiangolo.com/) - High-performance web framework for APIs.
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) - Data manipulation and numerical operations.
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library.
- [Uvicorn](https://www.uvicorn.org/) - ASGI web server implementation for Python.

**Frontend**
- [React (Vite)](https://react.dev/) - Fast, modern UI development.
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework for rapid UI design.
- [Recharts](https://recharts.org/) - Composable charting library built on React components.
- [Axios](https://axios-http.com/) - Promise-based HTTP client.
- [Lucide React](https://lucide.dev/) - Beautiful and consistent iconography.

---

## ⚙️ Local Development Setup

### Prerequisites
- [Node.js](https://nodejs.org/) (v16 or higher)
- [Python](https://www.python.org/) (v3.8 or higher)
- [Git](https://git-scm.com/)

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd MLautoMode
```

### 2. Backend Setup
Navigate to the backend directory, set up your virtual environment, install dependencies, and run the server.

```bash
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload
```
The backend API will be available at `http://localhost:8000`. You can test endpoints via the interactive Swagger UI at `http://localhost:8000/docs`.

### 3. Frontend Setup
Open a new terminal session, navigate to the frontend directory, install npm packages, and start the Vite dev server.

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
The web application will be accessible at `http://localhost:5173`.

---

## 📂 Project Structure

```
MLautoMode/
├── backend/
│   ├── main.py              # Main FastAPI application entry point
│   ├── preprocessor.py      # Data preprocessing logic
│   ├── requirements.txt     # Python dependencies
│   └── ...
├── frontend/
│   ├── src/                 # React components and pages
│   ├── package.json         # Node.js dependencies and scripts
│   ├── tailwind.config.js   # Tailwind CSS configuration
│   └── ...
├── Task                     # Internal to-do breakdown
└── README.md                # Project documentation
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to check [issues page](<your-issues-url>) for any ongoing bugs or feature requests.

## 📝 License
This project is licensed under the [MIT License](LICENSE).
