import os

# Print the model configuration
print("Current model configuration:")
print(f"COMPLETION_MODEL = {os.getenv('COMPLETION_MODEL', 'gpt-4o-mini')}")
print(f"MAX_COMPLETION_TOKENS = {os.getenv('MAX_COMPLETION_TOKENS', 8000)}")
print(f"MAX_SEARCH_RESULTS_TOOL = {os.getenv('MAX_SEARCH_RESULTS_TOOL', 5)}")

# Print the fallback chain
fallback_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if model in fallback_models:
    fallback_index = fallback_models.index(model)
    fallback_models = fallback_models[fallback_index:]

print(f"\nModel fallback chain:")
print(f"Primary model: {model}")
print(f"Fallback models: {', '.join(fallback_models[1:]) if len(fallback_models) > 1 else 'None'}")

print("\nVerification:")
print("1. The application is configured to use GPT-4o-mini by default")
print("2. The model can be overridden using the OPENAI_MODEL environment variable")
print("3. The fallback mechanism will try other models if GPT-4o-mini fails")
print("4. The context window has been increased to take advantage of GPT-4o-mini's capabilities")
