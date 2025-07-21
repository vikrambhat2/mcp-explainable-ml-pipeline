# MCP Explainable ML Pipeline

This repository implements a diabetes risk prediction chatbot using a FastMCP server, LangGraph with a ReAct agent, Ollama for local LLM inference, and Streamlit for an interactive interface.

## Features

- **Diabetes Risk Prediction**: Predicts diabetes risk based on user inputs (e.g., age, BMI, pedigree) using an ML model.
- **FastMCP Integration**: Utilizes MultiServerMCPClient for tool-based model interaction.
- **LangGraph**: Employs a ReAct agent for dynamic tool calling with Ollama's Qwen3:32b model.
- **Streamlit Interface**: Offers a real-time chatbot for user queries and detailed response insights.
- **Local Deployment**: Runs locally with privacy-focused setup.

## Prerequisites

- Python 3.8+
- Ollama (with Qwen3:32b model pulled)
- Streamlit
- LangGraph, LangChain Ollama, and FastMCP dependencies (listed in `requirements.txt`)
- Git

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/vikrambhat2/mcp-explainable-ml-pipeline.git
   cd mcp-explainable-ml-pipeline
   ```

2. **Set Up Python Environment**

   Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  
   pip install -r requirements.txt
   ```

3. **Configure Ollama**

   Pull the Qwen3:32b model and start the Ollama service:

   ```bash
   ollama pull qwen3:32b
   ollama serve
   ```


## Project Structure

```plaintext
mcp-explainable-ml-pipeline/
├── client.ipynb          # Jupyter notebook for client exploration
├── client.py             # Streamlit-based chatbot with FastMCP integration
├── data/                 # Data directory
│   └── pima_diabetes.csv # Dataset for training
├── mlenv/                # Environment or model-related files
├── models/               # Models directory
│   └── model.pkl         # Trained model file
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── server.py             # FastMCP server implementation
├── tools/                # Tools directory
│   ├── predict.py        # Prediction script
│   └── train_model.py    # Model training script
```

## Usage

1. **Start the FastMCP Server**

   Run the server:

   ```bash
   python server.py
   ```

   Ensure it runs on `http://localhost:8000/mcp`.

2. **Run the Chatbot**

   Launch the Streamlit app:

   ```bash
   streamlit run client.py
   ```

   Open `http://localhost:8501` in your browser.

3. **Interact with the Chatbot**

   - Enter queries like "Predict diabetes risk for age 45, BMI 28, pedigree 0.5".
   - View predictions and expand "Response Details" for additional insights.

## Configuration

- Edit `client.py` to update the MCP server URL or model if needed.
- Ensure the `.env` file contains the correct API key.

## Contributing

Fork the repo, create a branch, commit changes, and submit a pull request.

## License

MIT License. See `LICENSE` for details.