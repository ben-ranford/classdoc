
import yaml
import pytest
import requests
from unittest.mock import MagicMock


class MockOpenAIResponse:
    """Mock OpenAI API response for testing."""
    
    def __init__(self, content="Mocked response content"):
        self.choices = [
            type('MockChoice', (), {
                'message': type('MockMessage', (), {
                    'content': content
                }),
                'index': 0,
                'finish_reason': 'stop'
            })
        ]
        self.id = "mock-response-id"
        self.created = 1234567890
        self.model = "mock-model"
        self.usage = {
            'prompt_tokens': 10,
            'completion_tokens': 20,
            'total_tokens': 30
        }

def _setup_http_mocks(monkeypatch, http_content, http_status, raise_http_error, streaming_chunks):
    """Set up HTTP request mocks and return the mock response."""
    # Mock requests library for all HTTP methods
    mock_response = MagicMock()
    mock_response.json.return_value = http_content
    mock_response.status_code = http_status

    # Configure raise_for_status behavior
    if raise_http_error:
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{http_status} Error")
    else:
        mock_response.raise_for_status = MagicMock()

    mock_response.iter_lines.return_value = streaming_chunks

    # Mock all request methods to prevent real network calls
    response_factory = _create_response_factory(mock_response)
    for method in ['get', 'post', 'put', 'delete', 'patch']:
        monkeypatch.setattr(f"requests.{method}", response_factory)
    
    return mock_response


def _setup_openai_mocks(monkeypatch, openai_content):
    """Set up OpenAI client mocks and return the mock client."""
    # Create mock OpenAI client with proper structure
    mock_client = MagicMock()
    mock_chat = MagicMock()
    mock_completions = MagicMock()

    # Set up return values for chat.completions.create
    mock_completions.create.return_value = MockOpenAIResponse(openai_content)
    mock_chat.completions = mock_completions
    mock_client.chat = mock_chat

    # Ensure any attempted real API calls will fail early and clearly
    openai_constructor = _create_openai_constructor(mock_client)
    monkeypatch.setattr("openai.OpenAI", openai_constructor)
    
    return mock_client


def _create_response_factory(predefined_response):
    """Create a factory that returns the predefined response or customized ones."""
    def response_factory(*args, **kwargs):
        # Check if this call has a custom response in the params
        call_specific = kwargs.pop('_mock_response', None)
        if call_specific:
            for key, value in call_specific.items():
                setattr(predefined_response, key, value)

                # Special handling for json method which is a function
                if key == 'json_return':
                    predefined_response.json.return_value = value
                # Special handling for raise_for_status
                elif key == 'raise_for_status_error':
                    if value:
                        predefined_response.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{value}")
                    else:
                        predefined_response.raise_for_status.side_effect = None
        return predefined_response
    return response_factory


def _create_openai_constructor(mock_client):
    """Create OpenAI constructor that prevents real API calls."""
    def openai_constructor(**kwargs):
        # This prevents real API calls from being attempted
        if 'fake-key' not in kwargs.get('api_key', ''):
            raise ValueError("Attempted real API call in test! Use a fake key.")
        return mock_client
    return openai_constructor


def _extract_mock_parameters(request):
    """Extract and return mock parameters with defaults."""
    params = getattr(request, 'param', {})
    return {
        'openai_content': params.get('openai_content', "Mocked response content"),
        'http_content': params.get('http_content', {"choices": [{"message": {"content": "Mocked response from requests"}}]}),
        'streaming_chunks': params.get('streaming_chunks', [
            b'data: {"choices":[{"delta":{"content":"Test"}}]}',
            b'data: {"choices":[{"delta":{"content":" response"}}]}',
            b'data: [DONE]'
        ]),
        'http_status': params.get('http_status', 200),
        'raise_http_error': params.get('raise_http_error', False)
    }


@pytest.fixture
def mock_dependencies(monkeypatch, request):
    """
    Fixture to mock external dependencies with customizable responses.

    Parameters can be customized via indirect parameterization:
    @pytest.mark.parametrize('mock_dependencies', [
        {'openai_content': 'Custom response', 'http_content': {'key': 'value'}}
    ], indirect=True)
    """
    # Extract parameters
    params = _extract_mock_parameters(request)
    
    # Set up OpenAI mocks
    mock_client = _setup_openai_mocks(monkeypatch, params['openai_content'])
    
    # Set up HTTP mocks
    mock_response = _setup_http_mocks(
        monkeypatch, 
        params['http_content'], 
        params['http_status'], 
        params['raise_http_error'], 
        params['streaming_chunks']
    )

    # Return configured mocks for further customization in tests
    return {
        "openai_client": mock_client,
        "openai_response": MockOpenAIResponse,
        "http_response": mock_response,
        "create_mock_response": lambda **kwargs: MagicMock(**kwargs)
    }

@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary"""
    return {
        "output_directory": "test_docs",
        "base_directory": "test_src",
        "backend": "perplexity",
        "perplexity": {
            "api_key": "fake-key",
            "model": "pplx-7b-online"
        },
        "excluded_dirs": ["venv", ".git", "node_modules"]
    }

@pytest.fixture
def mock_config_file(sample_config, tmp_path):
    """Create a temporary config file with sample configuration"""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file

@pytest.fixture
def sample_py_file(tmp_path):
    """Create a sample Python file for testing"""
    file_path = tmp_path / "sample.py"
    content = "def hello_world():\n    return 'Hello, World!'\n"
    with open(file_path, 'w') as f:
        f.write(content)
    return file_path

@pytest.fixture
def template_content():
    """Return sample template content"""
    return "# {filename}\n\n## Overview\n\n{overview}\n\n## Usage\n\n{usage}"

@pytest.fixture
def mock_template_file(template_content, tmp_path):
    """Create a temporary template file"""
    template_file = tmp_path / "template.md"
    with open(template_file, 'w') as f:
        f.write(template_content)
    return template_file

@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project structure for integration testing"""
    # Create a sample project structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create some Python files
    py_file1 = src_dir / "main.py"
    py_file1.write_text("def main():\n    print('Hello, world!')\n")

    py_file2 = src_dir / "utils.py"
    py_file2.write_text("def helper():\n    return 'Helper function'\n")

    # Create subdirectories
    subdir = src_dir / "submodule"
    subdir.mkdir()

    py_file3 = subdir / "helper.py"
    py_file3.write_text("class Helper:\n    def __init__(self):\n        pass\n")

    # Create config file
    config = {
        "output_directory": str(tmp_path / "docs"),
        "base_directory": str(src_dir),
        "backend": "perplexity",
        "perplexity": {
            "api_key": "fake-key",
            "model": "pplx-7b-online"
        }
    }

    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    # Create template file
    template_file = tmp_path / "template.md"
    template_file.write_text("# {filename}\n\n## Overview\n\n{overview}\n\n## Usage\n\n{usage}")

    return {
        "root": tmp_path,
        "src_dir": src_dir,
        "config_file": config_file,
        "template_file": template_file,
        "py_files": [py_file1, py_file2, py_file3]
    }
