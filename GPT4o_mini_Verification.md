# Verifying GPT-4o-mini Implementation

This document provides methods to verify that the application is correctly using GPT-4o-mini.

## Configuration Verification

The application is configured to use GPT-4o-mini by default. You can verify this by checking:

1. **Code Configuration**: In `app.py`, the model is set to GPT-4o-mini:
   ```python
   COMPLETION_MODEL = os.getenv('COMPLETION_MODEL', 'gpt-4o-mini')
   ```

2. **Environment Variables**: You can override the model using environment variables:
   ```bash
   # Set a different model
   export OPENAI_MODEL=gpt-4o
   
   # Or set it back to GPT-4o-mini
   export OPENAI_MODEL=gpt-4o-mini
   ```

3. **Run the Verification Script**: Execute `verify_model.py` to confirm the configuration:
   ```bash
   python verify_model.py
   ```

## Runtime Verification

To verify that GPT-4o-mini is being used at runtime:

1. **Check Application Logs**: When the application makes API calls, it logs the model being used:
   ```
   INFO: Using model: gpt-4o-mini for data extraction
   ```

2. **Monitor API Requests**: Use a tool like Postman or the OpenAI dashboard to monitor API requests and confirm they're using GPT-4o-mini.

3. **Observe Response Times**: GPT-4o-mini typically has faster response times than larger models. If responses are noticeably quicker, this is a good indication the model switch is working.

4. **Check Billing**: In your OpenAI billing dashboard, you should see usage of GPT-4o-mini, which costs 15¢/M input tokens and 60¢/M output tokens.

## Fallback Mechanism Verification

The application includes a fallback mechanism that tries other models if GPT-4o-mini fails:

1. **Induce a Model Error**: You can test this by temporarily setting an invalid model name:
   ```bash
   export OPENAI_MODEL=invalid-model-name
   ```

2. **Check Logs for Fallback**: When the application runs, you should see log entries like:
   ```
   WARNING: Error with model invalid-model-name: Model not found. Trying fallbacks.
   INFO: Trying fallback model: gpt-4o
   ```

3. **Verify Successful Fallback**: The application should continue to function, using the fallback model instead.

## Context Window Verification

To verify that the application is taking advantage of GPT-4o-mini's larger context window:

1. **Check Context Size Parameters**: The application has been configured with increased context sizes:
   ```python
   MAX_COMPLETION_TOKENS = int(os.getenv('MAX_COMPLETION_TOKENS', 8000))
   MAX_SEARCH_RESULTS_TOOL = int(os.getenv('MAX_SEARCH_RESULTS_TOOL', 5))
   ```

2. **Test with Large Documents**: Upload and process larger documents than were previously possible. The application should handle them without truncation issues.

3. **Check Extraction Context**: In the data extraction function, the context limit has been increased:
   ```python
   {combined_context[:16000]}
   ```

## Conclusion

If all these verification steps pass, the application is correctly configured to use GPT-4o-mini with appropriate fallback mechanisms and context window optimizations.
