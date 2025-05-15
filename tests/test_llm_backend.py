import pytest
from unittest.mock import patch, MagicMock
import requests

from main import LLMBackend

class TestLLMBackend:
    def _create_backend(self, config, mock_openai=None):
        """Helper method to create and return a backend instance with given config"""
        backend = LLMBackend(config)
        # Verify common expected attributes
        assert hasattr(backend, 'logger')
        assert backend.config == config
        return backend

    def _setup_mock_openai_response(self, mock_openai, content="Generated documentation"):
        """Helper method to set up mock OpenAI client and response"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content=content),
                index=0,
                finish_reason="stop"
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        return mock_client, mock_response

    def _setup_mock_http_response(self, mock_post, content="Generated documentation", raise_error=False):
        """Helper method to set up mock HTTP response"""
        if raise_error:
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
            mock_post.return_value = mock_response
            return mock_response

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        mock_post.return_value = mock_response
        return mock_response

    def _verify_openai_completion(self, mock_openai, mock_client, base_url, api_key, model, prompt):
        """Helper to verify OpenAI API completion was called with expected parameters"""
        mock_openai.assert_called_once_with(
            api_key=api_key,
            base_url=base_url
        )
        mock_client.chat.completions.create.assert_called_once_with(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

    @patch("main.OpenAI")
    def test_init(self, mock_openai, sample_config):
        """Test LLMBackend initialization"""
        backend = self._create_backend(sample_config, mock_openai)
        assert backend.backend == "perplexity"

    @patch("main.OpenAI")
    def test_init_default_backend(self, mock_openai):
        """Test LLMBackend initialization with default backend"""
        config = {"perplexity": {"api_key": "fake-key"}}  # No backend specified
        backend = LLMBackend(config)
        assert backend.backend == "perplexity"  # Should default to perplexity

    @patch("main.OpenAI")
    @patch("main.requests.post")
    def test_generate_completion_dispatches_correctly(self, mock_post, mock_openai, sample_config):
        """Test that generate_completion dispatches to the correct handler"""
        backends_to_test = {
            "perplexity": "_perplexity_completion",
            "lmstudio": "_lmstudio_completion",
            "huggingface": "_huggingface_completion",
            "mistral": "_mistral_completion"
        }

        # Set up mock responses
        self._setup_mock_openai_response(mock_openai, "mock response")
        self._setup_mock_http_response(mock_post, "mock response")

        for backend_name, method_name in backends_to_test.items():
            # Create a config with this backend and appropriate API keys
            config = sample_config.copy()
            config["backend"] = backend_name
            config["perplexity"] = {"api_key": "fake-key", "model": "test-model"}
            config["mistral"] = {"api_key": "fake-key", "model": "test-model"}
            config["lmstudio"] = {"port": 1234, "model": "test-model"}
            config["huggingface"] = {"api_key": "fake-key", "model": "test-model"}

            # Create the backend and patch its methods
            with patch.object(LLMBackend, method_name, return_value=f"Response from {method_name}"):
                backend = self._create_backend(config, mock_openai)

                # Call generate_completion
                result = backend.generate_completion("Test prompt")

                # Check the correct handler was called
                method = getattr(backend, method_name)
                method.assert_called_once_with("Test prompt")
                assert result == f"Response from {method_name}"

    @patch("main.OpenAI")
    def test_generate_completion_unsupported_backend(self, mock_openai):
        """Test generate_completion with an unsupported backend"""
        config = {"backend": "unsupported_backend"}
        backend = self._create_backend(config, mock_openai)

        with pytest.raises(ValueError, match="Unsupported backend: unsupported_backend"):
            backend.generate_completion("Test prompt")

    @patch("main.OpenAI")
    @patch("requests.post")
    def test_retries_on_http_error(self, mock_post, mock_openai):
        """Test that retry logic works when HTTP errors occur"""
        config = {
            "backend": "lmstudio",
            "lmstudio": {"port": 5000}
        }

        # First call raises an error, second call succeeds
        error_response = self._setup_mock_http_response(mock_post, raise_error=True)
        success_response = MagicMock()
        success_response.json.return_value = {
            "choices": [{"message": {"content": "Success after retry"}}]
        }

        mock_post.side_effect = [error_response, success_response]

        # Test the method
        backend = self._create_backend(config, mock_openai)
        result = backend._lmstudio_completion("Test prompt")

        # Verify the result
        assert result == "Success after retry"
        assert mock_post.call_count == 2  # Should be called twice due to retry

    @patch("main.OpenAI")
    def test_perplexity_completion(self, mock_openai):
        """Test the Perplexity backend completion method"""
        # Setup config
        config = {
            "backend": "perplexity",
            "perplexity": {
                "api_key": "fake-key",
                "model": "pplx-7b-online"
            }
        }

        # Setup mock response
        mock_client, _ = self._setup_mock_openai_response(mock_openai, "Generated documentation")

        # Test the method
        backend = self._create_backend(config, mock_openai)
        result = backend._perplexity_completion("Test prompt")

        # Verify the result
        assert result == "Generated documentation"
        self._verify_openai_completion(
            mock_openai,
            mock_client,
            "https://api.perplexity.ai",
            config['perplexity']['api_key'],
            config['perplexity']['model'],
            "Test prompt"
        )

    @patch("main.OpenAI")
    @patch("requests.post")
    def test_lmstudio_completion_non_streaming(self, mock_post, mock_openai):
        """Test the LM Studio backend completion without streaming"""
        config = {
            "backend": "lmstudio",
            "lmstudio": {
                "port": 5000,
                "model": "test-model",
                "temperature": 0.5
            }
        }

        # Setup mock response
        self._setup_mock_http_response(mock_post, "Generated documentation")

        # Test the method
        backend = self._create_backend(config, mock_openai)
        result = backend._lmstudio_completion("Test prompt")

        # Verify the result
        assert result == "Generated documentation"
        mock_post.assert_called_once_with(
            "http://localhost:5000/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Test prompt"}],
                "model": "test-model",
                "temperature": 0.5
            }
        )

    @patch("main.OpenAI")
    @patch("requests.post")
    def test_lmstudio_completion_with_streaming(self, mock_post, mock_openai):
        """Test the LM Studio backend completion with streaming"""
        config = {
            "backend": "lmstudio",
            "lmstudio": {
                "port": 5000,
                "model": "test-model",
                "temperature": 0.5,
                "streaming": True,
                "printstream": False
            }
        }

        # Setup mocks for streaming response
        backend = self._create_backend(config, mock_openai)
        backend._handle_lmstudio_streaming = MagicMock(return_value="Streamed content")

        # Test the method
        result = backend._lmstudio_completion("Test prompt")

        # Verify the result
        assert result == "Streamed content"
        backend._handle_lmstudio_streaming.assert_called_once_with(
            "http://localhost:5000/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "Test prompt"}],
                "model": "test-model",
                "temperature": 0.5
            },
            False
        )

    @patch("main.OpenAI")
    @patch("requests.post")
    def test_handle_lmstudio_streaming(self, mock_post, mock_openai):
        """Test handling of LM Studio streaming responses"""
        config = {"backend": "lmstudio"}
        backend = self._create_backend(config, mock_openai)

        # Create a mock response that simulates streaming
        mock_response = MagicMock()
        # Note: iter_lines with decode_unicode=True returns strings, not bytes
        mock_response.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"Test"}}]}',
            'data: {"choices":[{"delta":{"content":" response"}}]}',
            'data: [DONE]'
        ]
        mock_post.return_value = mock_response

        # Test the method
        result = backend._handle_lmstudio_streaming(
            "http://localhost:1234/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Test"}]},
            False
        )

        # Verify the result
        assert result == "Test response"
        mock_post.assert_called_once_with(
            "http://localhost:1234/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Test"}], "stream": True},
            stream=True
        )

    @patch("main.OpenAI")
    @patch("requests.post")
    def test_huggingface_completion(self, mock_post, mock_openai):
        """Test the HuggingFace backend completion method"""
        config = {
            "backend": "huggingface",
            "huggingface": {
                "api_key": "hf-fake-key",
                "model": "bigscience/bloom",
                "temperature": 0.8,
                "max_new_tokens": 1024
            }
        }

        # Setup mock response - HuggingFace has a different response format
        mock_response = MagicMock()
        mock_response.json.return_value = [{"generated_text": "Generated text from HuggingFace"}]
        mock_post.return_value = mock_response

        # Test the method
        backend = self._create_backend(config, mock_openai)
        result = backend._huggingface_completion("Test prompt")

        # Verify the result
        assert result == "Generated text from HuggingFace"
        mock_post.assert_called_once_with(
            "https://api-inference.huggingface.co/models/bigscience/bloom",
            headers={
                "Authorization": "Bearer hf-fake-key",
                "Content-Type": "application/json"
            },
            json={
                "inputs": "Test prompt",
                "parameters": {
                    "temperature": 0.8,
                    "max_new_tokens": 1024,
                    "return_full_text": False
                }
            }
        )

    @patch("main.OpenAI")
    def test_mistral_completion(self, mock_openai):
        """Test the Mistral backend completion method"""
        config = {
            "backend": "mistral",
            "mistral": {
                "api_key": "mistral-fake-key",
                "model": "mistral-medium",
                "temperature": 0.6,
                "max_tokens": 2048
            }
        }

        # Setup mock response
        mock_client, _ = self._setup_mock_openai_response(mock_openai, "Generated text from Mistral")

        # Test the method
        backend = self._create_backend(config, mock_openai)
        result = backend._mistral_completion("Test prompt")

        # Verify the result
        assert result == "Generated text from Mistral"

        # Verify OpenAI client configuration
        mock_openai.assert_called_once_with(
            api_key=config['mistral']['api_key'],
            base_url="https://api.mistral.ai/v1"
        )

        # Verify parameters sent to API
        mock_client.chat.completions.create.assert_called_once_with(
            model=config['mistral']['model'],
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.6,
            max_tokens=2048
        )

    @pytest.mark.parametrize('mock_dependencies', [
        {'openai_content': 'Custom API response', 'http_content': {'choices': [{'message': {'content': 'Custom HTTP response'}}]}}
    ], indirect=True)
    def test_customized_mock_dependencies(self, mock_dependencies):
        """Test that the enhanced mock_dependencies fixture can be customized"""
        # Create a simple config
        perplexity_config = {
            "backend": "perplexity",
            "perplexity": {
                "api_key": "fake-key",
                "model": "test-model"
            }
        }

        # Set up the backend using actual dependency injection
        with patch("main.OpenAI", return_value=mock_dependencies["openai_client"]):
            # Create the backend
            backend = LLMBackend(perplexity_config)

            # Test OpenAI-based completion
            result = backend._perplexity_completion("Test prompt")

            # Verify the result contains our custom response
            assert result == "Custom API response"

        # Test HTTP-based completion with the fixture
        with patch("main.requests.post", return_value=mock_dependencies["http_response"]):
            # Use a different backend that uses HTTP
            http_config = {
                "backend": "lmstudio",
                "lmstudio": {
                    "port": 5000
                }
            }

            # Create a new backend with HTTP config
            http_backend = LLMBackend(http_config)

            # Test the completion
            http_result = http_backend._lmstudio_completion("Test prompt")

            # Verify custom response
            assert http_result == "Custom HTTP response"
