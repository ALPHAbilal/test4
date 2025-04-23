try:
    from agents import Agent, Runner
    print("Successfully imported Agent and Runner from agents")
except ImportError as e:
    print(f"Import error: {e}")

try:
    import openai_agents
    print(f"Successfully imported openai_agents version: {openai_agents.__version__}")
except ImportError as e:
    print(f"Import error: {e}")
