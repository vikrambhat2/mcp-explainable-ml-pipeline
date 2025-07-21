import streamlit as st
import asyncio
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from dotenv import load_dotenv
load_dotenv()
# Streamlit app title
st.title("Diabetes Risk Prediction Chat")

# Initialize session state for chat history and graph
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "graph" not in st.session_state:
    async def setup_graph():
        # Set up the MCP client
        client = MultiServerMCPClient({
            "diabetes_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http"
            }
        })
        
        # Await the tool loading (since it's async)
        try:
            mcp_tools = await client.get_tools()
            st.session_state["mcp_tools"] = mcp_tools
        except Exception as e:
            st.error(f"Error loading tools: {e}")
            return None

        # Load domain resources (optional)
        resources = client.get_resources("diabetes_server", uris=["diabetes://guidelines/risk-factors"])
        #st.write("Loaded resources:", resources)

        # Load model and bind tools
        
        llm = ChatOllama(model="qwen3:32b")
        model_with_tools = llm.bind_tools(mcp_tools)

        # LangGraph step functions
        def should_continue(state):
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            return END

        def call_model(state):
            messages = state["messages"]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        # Build the LangGraph
        builder = StateGraph(MessagesState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", ToolNode(mcp_tools))
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges("call_model", should_continue, ["tools", END])
        builder.add_edge("tools", "call_model")
        graph = builder.compile()
        return graph

    # Run async setup
    st.session_state["graph"] = asyncio.run(setup_graph())

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "details" in message:
            with st.expander("Response Details"):
                st.json(message["details"])

# Chat input for user query
query = st.chat_input("Enter your query (e.g., 'Predict diabetes risk for age 45, BMI 28, pedigree 0.5'):")

# Process query
if query:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    async def run_query():
        try:
            graph = st.session_state.get("graph")
            if graph is None:
                st.error("Graph not initialized. Check tool loading.")
                return

            # Run the query
            result = await graph.ainvoke({
                "messages": [{"role": "user", "content": query}]
            })

            # Extract response and add to chat history
            response = result["messages"][-1].content
            st.session_state["messages"].append({
                "role": "assistant",
                "content": response,
                "details": result
            })

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
                with st.expander("Response Details"):
                    st.json(result)

        except Exception as e:
            st.error(f"Error processing query: {e}")

    # Run async query
    asyncio.run(run_query())