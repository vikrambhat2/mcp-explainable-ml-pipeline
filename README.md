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

## Tool Calling Demonstrations

#### Claude Desktop: 
The MCP tool can be invoked via the Claude desktop application, showcasing seamless integration with the FastMCP server. This allows for testing tool calls in a controlled environment, with responses reflecting the diabetes risk prediction model.


<br>
<img width="1992" height="1596" alt="image" src="https://github.com/user-attachments/assets/cd367d43-bd3f-4565-b0d7-b82789fa49ee" />



<br>

#### Streamlit Chat App: 
The Streamlit-based chatbot (client.py) demonstrates real-time MCP tool calling. Users can input queries (e.g., "Predict diabetes risk for age 45, BMI 28, pedigree 0.5") and receive predictions with detailed insights, accessible via the app at http://localhost:8501.


<img width="3414" height="1874" alt="image" src="https://github.com/user-attachments/assets/738314fb-3433-479d-aa08-a128eade8878" />





## Contributing

Fork the repo, create a branch, commit changes, and submit a pull request.


