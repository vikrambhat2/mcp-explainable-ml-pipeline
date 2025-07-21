# MCP Explainable ML Pipeline

This repository showcases a **Machine Learning Control Plane (MCP)** with an integrated prediction model.  
You can host the MCP server, call prediction endpoints using a LangGraph tool, and interact with the system via a modern Streamlit UI.

---

## Features

- **MCP Server Hosting:** Easily launch the MCP server to serve model predictions via HTTP.
- **Prediction Model:** Pre-trained and customizable model files for demonstrations and real-world scenarios.
- **LangGraph Integration:** Use the LangGraph tool for orchestrated, explainable calls to the server.
- **Streamlit UI:** User-friendly Streamlit dashboard for interactive predictions and analysis.
- **Explainability:** Tools for generating and visualizing model explanations.

---

## Directory Structure
```
├── data/ # Sample and test datasets
├── models/ # Serialized model artifacts (e.g., .pkl, .joblib)
├── tools/ # Utility scripts, including LangGraph integration
├── client.ipynb # Jupyter notebook client for experimentation
├── client.py # Python client script to call MCP server
├── http_server.py # Main script to launch the MCP server
├── server.py # (Optional) Additional/alternative server implementation
├── requirements.txt # Python dependencies
├── README.md # Project documentation

```

---

## Getting Started

1. **Clone the repository:**
    ```
    git clone https://github.com/vikrambhat2/mcp-explainable-ml-pipeline.git
    cd mcp-explainable-ml-pipeline
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Prepare your data and models:**
    - Place datasets in the `data/` directory (update paths in code as needed).
    - Store your trained models in the `models/` folder.

---

## Usage

### 1. Host the MCP Server

Launch the main server to serve predictions:
python server.py



### 2. Call the MCP Server Using LangGraph

- Use scripts in `tools/` to invoke server endpoints with the LangGraph tool for explainable inference workflows.
- Example usage:
    ```
    python tools/langgraph_client.py
    ```

### 3. Interact via Streamlit UI

Start the Streamlit dashboard for easy interaction:
streamlit run tools/streamlit_ui.py

text
Adjust the path if your Streamlit UI file is named differently.

### 4. Experiment and Extend

- Use `client.ipynb` or `client.py` for API calls, batch predictions, or custom analysis.
- Extend utilities in `tools/` for more explainability features or data processing.

---

## Requirements

- Python 3.7+
- See `requirements.txt` for packages such as:  
  - scikit-learn, pandas, streamlit, langgraph, etc.

---

## License

This project is licensed under the MIT License.

---

## Contact

For issues, questions, or feature requests, please [open an issue](https://github.com/vikrambhat2/mcp-explainable-ml-pipeline/issues) on GitHub.

---

*Showcase, understand, and share your machine learning predictions the explainable way!*
