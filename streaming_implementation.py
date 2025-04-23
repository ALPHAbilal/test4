"""
Streaming Implementation for Knowledge Base UX Optimization

This module contains the implementation of streaming responses
to improve perceived performance.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, List, Optional, Any
from flask import Response, stream_with_context
import html

# Configure logging
logger = logging.getLogger(__name__)

# --- Streaming Response Generator ---

async def generate_streaming_response(
    user_query: str,
    vs_id: str,
    history: List[Dict[str, str]],
    workflow_context: Dict,
    chat_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming response for the user query.
    
    Args:
        user_query: The user's query
        vs_id: Vector store ID
        history: Conversation history
        workflow_context: Workflow context
        chat_id: Optional chat ID
        
    Yields:
        Chunks of the response as they are generated
    """
    try:
        # First, yield a processing message
        yield json.dumps({"type": "status", "content": "Processing your query..."})
        
        # Retrieve KB content (non-streaming part)
        kb_content = ""
        if vs_id:
            yield json.dumps({"type": "status", "content": "Searching knowledge base..."})
            
            kb_context = workflow_context.copy()
            kb_query = user_query
            document_type = "general"
            
            # Get KB content using the minimal data gathering agent
            # Note: This part is not streamed yet, but could be enhanced in the future
            kb_data_raw = await Runner.run(
                data_gathering_agent_minimal,
                input=f"Get KB content about '{document_type}' related to: {kb_query}",
                context=kb_context
            )
            kb_data = kb_data_raw.final_output
            
            if isinstance(kb_data, RetrievalSuccess):
                kb_content = kb_data.content
                # Yield information about the sources
                source_info = f"Found relevant information in: {kb_data.source_filename}"
                yield json.dumps({"type": "status", "content": source_info})
            elif isinstance(kb_data, RetrievalError):
                # Try a more general search
                yield json.dumps({"type": "status", "content": "Trying alternative search approach..."})
                
                kb_data_raw = await Runner.run(
                    data_gathering_agent_minimal,
                    input=f"Get KB content about 'general' related to: {kb_query}",
                    context=kb_context
                )
                kb_data = kb_data_raw.final_output
                
                if isinstance(kb_data, RetrievalSuccess):
                    kb_content = kb_data.content
                    source_info = f"Found relevant information in: {kb_data.source_filename}"
                    yield json.dumps({"type": "status", "content": source_info})
        
        # Create a prompt that includes the query and KB content
        prompt = f"""Answer the following question using ONLY the knowledge base content provided below.
        
        Question: {user_query}
        
        IMPORTANT: If the knowledge base content does not contain information to answer this question, 
        clearly state this limitation. DO NOT fabricate or make up information that is not in the 
        provided content. Accuracy is more important than helpfulness."""
        
        if kb_content:
            prompt += f"\n\nRelevant Knowledge Base Content:\n{kb_content}"
        
        synthesis_messages = history + [{
            "role": "user",
            "content": prompt
        }]
        
        # Yield a message that we're generating the response
        yield json.dumps({"type": "status", "content": "Generating response..."})
        
        # Run the final synthesizer agent with streaming
        async for chunk in stream_final_synthesizer(synthesis_messages, workflow_context):
            yield json.dumps({"type": "content", "content": chunk})
            
        # Final message indicating completion
        yield json.dumps({"type": "status", "content": "Response complete"})
        
    except Exception as e:
        logger.error(f"Streaming response generation failed: {e}", exc_info=True)
        error_message = f"Sorry, an error occurred during processing: {html.escape(str(e))}"
        yield json.dumps({"type": "error", "content": error_message})

async def stream_final_synthesizer(messages: List[Dict[str, str]], context: Dict) -> AsyncGenerator[str, None]:
    """
    Stream the output from the final synthesizer agent.
    
    Args:
        messages: List of conversation messages
        context: Workflow context
        
    Yields:
        Chunks of the generated response
    """
    # This is a simplified implementation that would need to be adapted based on
    # the actual streaming capabilities of the Runner and OpenAI client
    
    # In a real implementation, you would use the OpenAI streaming API
    # Here's a simplified example of what it might look like:
    
    client = context.get("client")
    if not client:
        yield "Error: OpenAI client not available"
        return
    
    try:
        # Create a streaming completion
        stream = await asyncio.to_thread(
            client.chat.completions.create,
            model=context.get("model", "gpt-4o-mini"),
            messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
            stream=True
        )
        
        # Buffer to accumulate partial tokens into words/sentences
        buffer = ""
        
        # Process the streaming response
        async for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                buffer += content
                
                # Yield complete sentences or when buffer gets too large
                if "." in buffer or "?" in buffer or "!" in buffer or len(buffer) > 100:
                    yield buffer
                    buffer = ""
        
        # Yield any remaining content in the buffer
        if buffer:
            yield buffer
            
    except Exception as e:
        logger.error(f"Error in streaming synthesis: {e}", exc_info=True)
        yield f"\nError during response generation: {str(e)}"

# --- Flask Route Implementation ---

"""
@app.route('/chat/<chat_id>/stream', methods=['POST'])
async def chat_stream_api(chat_id):
    """Streaming version of the chat API endpoint.
    
    This is a template for how to implement a streaming endpoint.
    The actual implementation would need to be integrated with the existing routes.
    """
    if not chat_db:
        return jsonify({"error": "Database service not available."}), 500
    
    # Handle both JSON and form data (similar to the original endpoint)
    user_message = ""
    # ... (extract user_message from request)
    
    # Validate message
    if not user_message:
        return jsonify({"error": "Message is empty."}), 400
    
    # Get chat details and prepare history
    try:
        chat_details = await asyncio.to_thread(chat_db.get_chat_details, chat_id)
        if not chat_details:
            return jsonify({"error": "Chat not found."}), 404
            
        vector_store_id = chat_details.get('vector_store_id')
        if not vector_store_id:
            return jsonify({"error": "Chat KB link missing."}), 400
        
        # Add user message to database
        await asyncio.to_thread(chat_db.add_message, chat_id, 'user', user_message)
        
        # Get message history for context
        message_history_db = await asyncio.to_thread(chat_db.get_messages, chat_id, limit=10)
        history_for_workflow = [{"role": msg["role"], "content": msg["content"]} 
                               for msg in message_history_db if msg["role"] != 'user'][-6:]
        
        # Prepare workflow context
        workflow_context = {
            "vector_store_id": vector_store_id,
            "client": get_openai_client(),
            "chat_id": chat_id,
            "model": get_model_with_fallback()[0]
        }
        
        # Create a streaming response
        return Response(
            stream_with_context(generate_streaming_response(
                user_query=user_message,
                vs_id=vector_store_id,
                history=history_for_workflow,
                workflow_context=workflow_context,
                chat_id=chat_id
            )),
            content_type='text/event-stream'
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat API: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
"""

# --- Frontend JavaScript Implementation ---

"""
// JavaScript for handling streaming responses

// Function to initialize streaming for a chat
function initChatStreaming() {
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const chatMessages = document.getElementById('chat-messages');
    const chatId = chatForm.dataset.chatId;
    
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;
        
        // Add user message to UI
        appendMessage('user', userMessage);
        messageInput.value = '';
        
        // Create a placeholder for the assistant's response
        const assistantMsgElement = document.createElement('div');
        assistantMsgElement.className = 'message assistant-message';
        assistantMsgElement.innerHTML = '<div class="message-content"><p>Thinking...</p></div>';
        chatMessages.appendChild(assistantMsgElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Start the streaming request
        try {
            const response = await fetch(`/chat/${chatId}/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage }),
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let responseContent = '';
            
            // Process the stream
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                // Decode and parse the chunk
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim());
                
                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.type === 'status') {
                            // Update status message
                            assistantMsgElement.querySelector('.message-content').innerHTML = 
                                `<p><em>${data.content}</em></p>`;
                        } 
                        else if (data.type === 'content') {
                            // Append content to the response
                            responseContent += data.content;
                            assistantMsgElement.querySelector('.message-content').innerHTML = 
                                marked.parse(responseContent);
                        }
                        else if (data.type === 'error') {
                            // Display error
                            assistantMsgElement.querySelector('.message-content').innerHTML = 
                                `<p class="error">${data.content}</p>`;
                        }
                        
                        // Scroll to bottom
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    } catch (e) {
                        console.error('Error parsing stream chunk:', e, line);
                    }
                }
            }
            
            // Final rendering of the complete response
            if (responseContent) {
                assistantMsgElement.querySelector('.message-content').innerHTML = 
                    marked.parse(responseContent);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
        } catch (error) {
            console.error('Error in streaming chat:', error);
            assistantMsgElement.querySelector('.message-content').innerHTML = 
                `<p class="error">Error: ${error.message}</p>`;
        }
    });
    
    function appendMessage(role, content) {
        const msgElement = document.createElement('div');
        msgElement.className = `message ${role}-message`;
        
        if (role === 'user') {
            msgElement.innerHTML = `<div class="message-content"><p>${content}</p></div>`;
        } else {
            // For assistant messages, we'll use the placeholder approach above
        }
        
        chatMessages.appendChild(msgElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// Initialize when the DOM is loaded
document.addEventListener('DOMContentLoaded', initChatStreaming);
"""
