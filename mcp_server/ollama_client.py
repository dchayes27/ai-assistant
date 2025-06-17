"""
Ollama API client for LLM interactions
"""

import asyncio
import json
from typing import Dict, Any, Optional, AsyncGenerator, List
import httpx
from loguru import logger

from .models import QueryRequest, QueryResponse, StreamChunk


class OllamaClient:
    """Async client for Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
        self.available_models = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize client and load available models"""
        try:
            await self.refresh_models()
            logger.info(f"Ollama client initialized with {len(self.available_models)} models")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def refresh_models(self) -> List[str]:
        """Refresh the list of available models"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            
            data = response.json()
            self.available_models = [model["name"] for model in data.get("models", [])]
            logger.debug(f"Available models: {self.available_models}")
            return self.available_models
            
        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        return model_name in self.available_models
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"Pulling model: {model_name}")
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=1800.0  # 30 minute timeout for model downloads
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "status" in data:
                                logger.debug(f"Pull status: {data['status']}")
                        except json.JSONDecodeError:
                            continue
            
            # Refresh models after successful pull
            await self.refresh_models()
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def generate(
        self,
        request: QueryRequest,
        context_messages: Optional[List[Dict[str, Any]]] = None
    ) -> QueryResponse:
        """Generate a response using Ollama"""
        try:
            # Ensure model is available
            if not self.is_model_available(request.model):
                logger.warning(f"Model {request.model} not available, attempting to pull...")
                success = await self.pull_model(request.model)
                if not success:
                    raise ValueError(f"Model {request.model} is not available and could not be pulled")
            
            # Prepare the prompt with context
            messages = []
            
            # Add context messages if provided
            if context_messages:
                for msg in context_messages:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": request.message
            })
            
            # Prepare request payload
            payload = {
                "model": request.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                }
            }
            
            # Add metadata to options if provided
            if request.metadata:
                payload["options"].update(request.metadata)
            
            logger.debug(f"Generating response with model: {request.model}")
            start_time = asyncio.get_event_loop().time()
            
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300.0
            )
            response.raise_for_status()
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            data = response.json()
            
            return QueryResponse(
                response=data["message"]["content"],
                conversation_id=request.conversation_id or "default",
                message_id=f"msg_{int(asyncio.get_event_loop().time())}",
                model_used=request.model,
                tokens_used=data.get("eval_count"),
                response_time=response_time,
                context_used=context_messages or []
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Ollama API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_stream(
        self,
        request: QueryRequest,
        context_messages: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate a streaming response using Ollama"""
        try:
            # Ensure model is available
            if not self.is_model_available(request.model):
                logger.warning(f"Model {request.model} not available, attempting to pull...")
                success = await self.pull_model(request.model)
                if not success:
                    raise ValueError(f"Model {request.model} is not available and could not be pulled")
            
            # Prepare messages
            messages = []
            
            if context_messages:
                for msg in context_messages:
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })
            
            messages.append({
                "role": "user",
                "content": request.message
            })
            
            # Prepare request payload
            payload = {
                "model": request.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                }
            }
            
            if request.metadata:
                payload["options"].update(request.metadata)
            
            logger.debug(f"Starting stream generation with model: {request.model}")
            conversation_id = request.conversation_id or "default"
            message_id = f"msg_{int(asyncio.get_event_loop().time())}"
            
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300.0
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            
                            if "message" in data and "content" in data["message"]:
                                chunk_content = data["message"]["content"]
                                is_final = data.get("done", False)
                                
                                yield StreamChunk(
                                    chunk=chunk_content,
                                    is_final=is_final,
                                    conversation_id=conversation_id,
                                    message_id=message_id
                                )
                                
                                if is_final:
                                    break
                                    
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing stream chunk: {e}")
                            continue
            
            logger.debug(f"Stream generation completed for message: {message_id}")
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama stream: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Ollama API stream error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error in stream generation: {e}")
            raise
    
    async def embed(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """Generate embeddings for text"""
        try:
            # Ensure embedding model is available
            if not self.is_model_available(model):
                logger.warning(f"Embedding model {model} not available, attempting to pull...")
                success = await self.pull_model(model)
                if not success:
                    raise ValueError(f"Embedding model {model} is not available and could not be pulled")
            
            payload = {
                "model": model,
                "prompt": text
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("embedding", [])
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama embed: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Ollama embed API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=30.0
            )
            response.raise_for_status()
            
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Model {model_name} not found")
            logger.error(f"HTTP error getting model info: {e.response.status_code} - {e.response.text}")
            raise ValueError(f"Ollama API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise
    
    async def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        try:
            response = await self.client.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name},
                timeout=30.0
            )
            response.raise_for_status()
            
            # Refresh models after deletion
            await self.refresh_models()
            logger.info(f"Successfully deleted model: {model_name}")
            return True
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error deleting model: {e.response.status_code} - {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False


# Global Ollama client instance
ollama_client = OllamaClient()