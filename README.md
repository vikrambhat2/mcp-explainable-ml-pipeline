# MCP Explainable ML Pipeline

This repository implements a diabetes risk prediction and explainability chatbot using a FastMCP server, LangGraph with a ReAct agent, Ollama for local LLM inference, and Streamlit for an interactive interface.

---

## ✨ Features

* **Diabetes Risk Prediction**: Predicts diabetes risk based on input features (e.g., age, BMI, pedigree).
* **Prediction Explanation (SHAP)**: Explains model predictions with SHAP values, showing feature contributions.
* **FastMCP Integration**: Tools exposed using FastMCP’s server-client protocol.
* **LangGraph ReAct Agent**: Dynamically invokes prediction and explanation tools via Ollama (Qwen3:32b).
* **Streamlit Interface**: Enables user interaction with predictions, explanations, and visual insights.
* **Claude/LLM Interop**: Tools can also be tested with Claude or other MCP-compatible LLMs.

---

## 🧰 Prerequisites

* Python 3.8+
* Ollama (with `qwen3:32b` or `llama-3.3-70b-versatile` model pulled)
* Streamlit
* LangGraph, LangChain, Ollama, FastMCP (in `requirements.txt`)

---

## ⚙️ Installation

```bash
git clone https://github.com/vikrambhat2/mcp-explainable-ml-pipeline.git
cd mcp-explainable-ml-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧠 Ollama Setup

```bash
ollama pull qwen3:32b
ollama serve
```

---

## 📁 Project Structure

```plaintext
mcp-explainable-ml-pipeline/
├── client.ipynb          # LLM-driven prediction+explanation testing via MCP
├── client.py             # Streamlit chatbot UI
├── data/
│   └── pima_diabetes.csv # Source data
├── models/
│   └── model.pkl         # Trained model (RandomForest)
├── tools/
│   ├── predict.py        # Diabetes risk prediction tool
│   ├── explain.py        # SHAP-based explanation tool
│   └── train_model.py    # Model training logic
├── server.py             # FastMCP server exposing tools
├── requirements.txt      # Dependency list
└── README.md             # This file
```

---

## 🚀 Usage

### 1. Start the FastMCP Server

```bash
python server.py
```

> Ensure it's available at `http://localhost:8000/mcp`.

### 2. Launch the Streamlit Chatbot

```bash
streamlit run client.py
```

> Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Example Queries

* `Predict diabetes risk for age 52, BMI 30.1, pedigree 0.52`
* `Explain prediction for age 52, BMI 30.1, pedigree 0.52`

The LLM will dynamically select between prediction and explanation tools.

---

## ⚙️ Configuration

Update `client.py` or LangGraph config to:

* Change MCP server URL
* Switch LLM (e.g., Claude vs Ollama)

---

## 🔍 Tool Demonstrations

### ✅ Claude Desktop

Tool calls can be tested with Claude or any MCP-compatible client:

> `explain_diabetes_risk(age=52, bmi=30.1, diabetes_pedigree_function=0.52)`

<img width="1992" height="1596" alt="Claude demo" src="https://github.com/user-attachments/assets/cd367d43-bd3f-4565-b0d7-b82789fa49ee" />

---

### ✅ Streamlit Chat App

Live interaction and explanations via LLM agent:

<img width="3414" height="1874" alt="Streamlit chat demo" src="https://github.com/user-attachments/assets/738314fb-3433-479d-aa08-a128eade8878" />

---

## 🧪 Explanation Tool Overview

The `explain_diabetes_risk` tool (in `tools/explain.py`) uses SHAP to return per-feature contributions to the model's prediction. This enables interpretability for each user-specific input.

**Example output:**

```json
{
  "age": 0.018,
  "bmi": 0.246,
  "diabetes_pedigree_function": 0.035
}
```

---

## 🤝 Contributing

Fork the repo → Create a branch → Make changes → Submit PR.

