"""LLM provider interface and implementations."""
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
import json
import requests
from pathlib import Path
import logging
import time
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt for the model.
            **kwargs: Additional model-specific parameters.
            
        Returns:
            The generated response as a string.
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the LLM provider.
        
        Returns:
            The name of the provider.
        """
        pass


class ClaudeProvider(LLMProvider):
    """Claude API provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-sonnet-20240229"):
        """Initialize the Claude provider.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
            model: The Claude model to use.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
        
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.cache_dir = Path("data/cache/claude")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_name(self) -> str:
        """Return the name of the provider."""
        return f"Claude ({self.model})"
    
    @lru_cache(maxsize=100)
    def _cached_response(self, prompt_hash: str) -> Optional[str]:
        """Get cached response if available.
        
        Args:
            prompt_hash: Hash of the prompt.
            
        Returns:
            Cached response or None if not found.
        """
        cache_file = self.cache_dir / f"{prompt_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)["response"]
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
        return None
    
    def _save_to_cache(self, prompt_hash: str, response: str) -> None:
        """Save response to cache.
        
        Args:
            prompt_hash: Hash of the prompt.
            response: Response to cache.
        """
        cache_file = self.cache_dir / f"{prompt_hash}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"response": response}, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using Claude API.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            The generated response as a string.
        """
        # Generate a simple hash of the prompt for caching
        prompt_hash = str(hash(prompt))
        
        # Check cache first
        cached = self._cached_response(prompt_hash)
        if cached:
            logger.info("Using cached response")
            return cached
        
        # Prepare the API request
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Default parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            # Make the API request with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["content"][0]["text"]
                    
                    # Save to cache
                    self._save_to_cache(prompt_hash, content)
                    return content
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    break
                    
            return f"Error generating response. Please try again later. [Status: {response.status_code}]"
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return "Sorry, I encountered an error connecting to the language model. Please try again."


class GemmaProvider(LLMProvider):
    """Gemma provider using Google's Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """Initialize the Gemma provider.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            model: The Gemini model to use.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.cache_dir = Path("data/cache/gemma")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_name(self) -> str:
        """Return the name of the provider."""
        return f"Gemma ({self.model})"
    
    @lru_cache(maxsize=100)
    def _cached_response(self, prompt_hash: str) -> Optional[str]:
        """Get cached response if available."""
        cache_file = self.cache_dir / f"{prompt_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)["response"]
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
        return None
    
    def _save_to_cache(self, prompt_hash: str, response: str) -> None:
        """Save response to cache."""
        cache_file = self.cache_dir / f"{prompt_hash}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"response": response}, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using Gemini API.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            The generated response as a string.
        """
        # Generate a simple hash of the prompt for caching
        prompt_hash = str(hash(prompt))
        
        # Check cache first
        cached = self._cached_response(prompt_hash)
        if cached:
            logger.info("Using cached response")
            return cached
        
        # Default parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        params = {
            "key": self.api_key
        }
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topK": kwargs.get("top_k", 40),
                "topP": kwargs.get("top_p", 0.95)
            }
        }
        
        try:
            # Make the API request with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                response = requests.post(
                    self.api_url,
                    params=params,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Save to cache
                    self._save_to_cache(prompt_hash, content)
                    return content
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    break
                    
            return f"Error generating response. Please try again later. [Status: {response.status_code}]"
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return "Sorry, I encountered an error connecting to the language model. Please try again."


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference."""
    
    def __init__(self, model: str = "mistral", host: str = "http://localhost:11434"):
        """Initialize the Ollama provider.
        
        Args:
            model: The Ollama model to use.
            host: The Ollama API host URL.
        """
        self.model = model
        self.api_url = f"{host}/api/generate"
        self.cache_dir = Path(f"data/cache/ollama/{model}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_name(self) -> str:
        """Return the name of the provider."""
        return f"Ollama ({self.model})"
    
    @lru_cache(maxsize=100)
    def _cached_response(self, prompt_hash: str) -> Optional[str]:
        """Get cached response if available."""
        cache_file = self.cache_dir / f"{prompt_hash}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)["response"]
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")
        return None
    
    def _save_to_cache(self, prompt_hash: str, response: str) -> None:
        """Save response to cache."""
        cache_file = self.cache_dir / f"{prompt_hash}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump({"response": response}, f)
        except Exception as e:
            logger.warning(f"Error writing to cache: {e}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response using Ollama API.
        
        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            The generated response as a string.
        """
        # Generate a simple hash of the prompt for caching
        prompt_hash = str(hash(prompt))
        
        # Check cache first
        cached = self._cached_response(prompt_hash)
        if cached:
            logger.info("Using cached response")
            return cached
        
        # Prepare the parameters
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 40),
            "top_p": kwargs.get("top_p", 0.95)
        }
        
        try:
            response = requests.post(self.api_url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("response", "")
                
                # Save to cache
                self._save_to_cache(prompt_hash, content)
                return content
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error generating response. Please try again later. [Status: {response.status_code}]"
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            return "Sorry, I encountered an error connecting to the language model. Please try again."


class LLMFactory:
    """Factory class for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> LLMProvider:
        """Create an LLM provider.
        
        Args:
            provider_type: The type of provider to create ('claude', 'gemma', 'ollama').
            **kwargs: Additional parameters for the provider.
            
        Returns:
            An LLM provider instance.
            
        Raises:
            ValueError: If the provider type is not supported.
        """
        provider_type = provider_type.lower()
        
        if provider_type == "claude":
            return ClaudeProvider(**kwargs)
        elif provider_type == "gemma":
            return GemmaProvider(**kwargs)
        elif provider_type == "ollama":
            return OllamaProvider(**kwargs)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}") 