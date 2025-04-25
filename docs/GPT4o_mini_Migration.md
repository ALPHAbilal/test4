# Migration to GPT-4o-mini

This document outlines the changes made to migrate the application from using the previous model to GPT-4o-mini.

## Changes Implemented

1. **Updated Model Name**
   - Changed the default model from `gpt-4o` to `gpt-4o-mini`
   - Added environment variable override capability

2. **Increased Context Window**
   - Increased `MAX_COMPLETION_TOKENS` from 2500 to 8000
   - Increased context limit in data extraction from 8000 to 16000 characters
   - Increased `MAX_SEARCH_RESULTS_TOOL` from 3 to 5 chunks per tool call

3. **Added Model Fallback Mechanism**
   - Created `get_model_with_fallback()` function to handle model fallbacks
   - Implemented fallback chain: gpt-4o-mini → gpt-4o → gpt-3.5-turbo
   - Added error handling to try fallback models when the primary model fails

## Benefits of GPT-4o-mini

GPT-4o-mini offers several advantages over previous models:

- **Larger Context Window**: 128K tokens (vs. smaller windows in previous models)
- **Cost Efficiency**: 15¢/M input tokens, 60¢/M output tokens (60% cheaper than GPT-3.5 Turbo)
- **Performance**: Scores 82% on MMLU, outperforming GPT-4 on chat preferences in the LMSYS leaderboard
- **Multimodal Capabilities**: Supports text and vision processing

## Usage

The application now uses GPT-4o-mini by default, but you can override this by setting the `OPENAI_MODEL` environment variable:

```
# Use GPT-4o instead
export OPENAI_MODEL=gpt-4o

# Use GPT-3.5 Turbo instead
export OPENAI_MODEL=gpt-3.5-turbo
```

## Monitoring and Optimization

To get the most out of GPT-4o-mini:

1. **Monitor Performance**: Watch for any changes in response quality or latency
2. **Adjust Context Size**: Experiment with different context sizes to find the optimal balance
3. **Optimize Prompts**: GPT-4o-mini may respond differently to prompts than previous models
4. **Track Costs**: Monitor usage to quantify cost savings

## Future Improvements

Potential future improvements include:

1. **Batch Processing**: Implement batch processing for cost efficiency
2. **Prompt Optimization**: Further refine prompts for GPT-4o-mini's capabilities
3. **Context Window Utilization**: Better utilize the expanded context window
4. **Performance Metrics**: Add telemetry to measure performance differences
