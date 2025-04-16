from agent.flow import build_flow_graph

if __name__ == "__main__":
    app = build_flow_graph()

    # Test questions
    question = "Do I have to pay estimated taxes as a NebulaForge contractor?"
    result = app.invoke({"input_text": question})
    print("\nResponse:\n", result["response"])
