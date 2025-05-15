# LLMBackend

## Overview
The `LLMBackend` class is designed to interact with various Large Language Model (LLM) backends to generate text completions. It supports backends like Perplexity, LM Studio, HuggingFace, and Mistral. The class initializes with a configuration and provides methods to generate completions using the specified backend.

## Dependencies
- os
- pathlib.Path
- logging
- yaml
- openai.OpenAI
- requests
- argparse
- tenacity

## Definition
```python
class LLMBackend:
    def __init__(self, config):
        ...
```

## Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| config | dict | Configuration dictionary containing backend settings. |
| backend | str | The name of the backend to use (e.g., 'perplexity', 'lmstudio'). |
| logger | logging.Logger | Logger instance for logging messages. |
| backend_handlers | dict | Dictionary mapping backend names to their respective completion methods. |

## Methods
### \_\_init\_\_(config)
```python
def __init__(self, config: dict):
    """
    Initialize the LLMBackend with a configuration dictionary.

    Args:
        config: Configuration dictionary containing backend settings.
    """
```

### generate_completion(prompt)
```python
def generate_completion(self, prompt: str) -> str:
    """
    Completion dispatcher.

    Args:
        prompt: The input prompt for which completion is to be generated.

    Returns:
        The generated completion text.

    Raises:
        ValueError: If the specified backend is unsupported.
    """
```

### _perplexity_completion(prompt)
```python
def _perplexity_completion(self, prompt: str) -> str:
    """
    Generate completion using the Perplexity API.

    Args:
        prompt: The input prompt for which completion is to be generated.

    Returns:
        The generated completion text.

    Raises:
        requests.exceptions.HTTPError: If there is an HTTP error during the request.
    """
```

### _lmstudio_completion(prompt)
```python
def _lmstudio_completion(self, prompt: str) -> str:
    """
    Generate completion using the LM Studio API.

    Args:
        prompt: The input prompt for which completion is to be generated.

    Returns:
        The generated completion text.

    Raises:
        requests.exceptions.HTTPError: If there is an HTTP error during the request.
    """
```

### _handle_lmstudio_streaming(url, payload, printstream)
```python
def _handle_lmstudio_streaming(self, url: str, payload: dict, printstream: bool) -> str:
    """
    Handle streaming response from the LM Studio API.

    Args:
        url: The URL endpoint for the LM Studio API.
        payload: The payload to send with the request.
        printstream: Whether to print the streaming response.

    Returns:
        The generated completion text.

    Raises:
        requests.exceptions.HTTPError: If there is an HTTP error during the request.
    """
```

### _huggingface_completion(prompt)
```python
def _huggingface_completion(self, prompt: str) -> str:
    """
    Generate completion using the HuggingFace API.

    Args:
        prompt: The input prompt for which completion is to be generated.

    Returns:
        The generated completion text.

    Raises:
        requests.exceptions.HTTPError: If there is an HTTP error during the request.
    """
```

### _mistral_completion(prompt)
```python
def _mistral_completion(self, prompt: str) -> str:
    """
    Generate completion using the Mistral AI API.

    Args:
        prompt: The input prompt for which completion is to be generated.

    Returns:
        The generated completion text.

    Raises:
        requests.exceptions.HTTPError: If there is an HTTP error during the request.
    """
```

# DocumentationGenerator

## Overview
The `DocumentationGenerator` class is designed to generate markdown documentation for code files. It reads configuration from a YAML file, interacts with the `LLMBackend` to generate completions, and writes the documentation to the specified output directory.

## Dependencies
- os
- pathlib.Path
- logging
- yaml
- argparse

## Definition
```python
class DocumentationGenerator:
    def __init__(self, config_path="config.yaml"):
        ...
```

## Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| config | dict | Configuration dictionary loaded from the YAML file. |
| docs_dir | pathlib.Path | Directory path where documentation will be saved. |
| base_dir | pathlib.Path | Base directory for the project. |
| supported_extensions | list | List of supported file extensions for documentation generation. |
| excluded_dirs | list | List of directories to exclude from processing. |
| llm_backend | LLMBackend | Instance of the LLMBackend class for generating completions. |
| logger | logging.Logger | Logger instance for logging messages. |

## Methods
### \_\_init\_\_(config_path)
```python
def __init__(self, config_path: str = "config.yaml"):
    """
    Initialize the DocumentationGenerator with a configuration file path.

    Args:
        config_path: Path to the configuration YAML file (default: 'config.yaml').
    """
```

### _load_config(config_path)
```python
def _load_config(self, config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        The loaded configuration dictionary.

    Raises:
        ValueError: If there is an error loading the configuration file.
    """
```

### _setup_logging()
```python
def _setup_logging(self):
    """
    Configure logging.
    """
```

### _create_docs_directory()
```python
def _create_docs_directory(self):
    """
    Ensures the output directory exists, creating all intermediate directories if necessary.
    """
```

### _should_skip_directory(dir_path)
```python
def _should_skip_directory(self, dir_path: str) -> bool:
    """
    Check if a directory should be skipped based on exclusion rules.

    Args:
        dir_path: The path to the directory.

    Returns:
        True if the directory should be skipped, False otherwise.
    """
```

### _get_relative_path(file_path, input_dir)
```python
def _get_relative_path(self, file_path: str, input_dir: str = None) -> pathlib.Path:
    """
    Get the relative path based on the input or base directory.

    Args:
        file_path: The path to the file.
        input_dir: The input directory path (optional).

    Returns:
        The relative path.
    """
```

### _generate_doc_path(file_path, input_dir)
```python
def _generate_doc_path(self, file_path: str, input_dir: str = None) -> pathlib.Path:
    """
    Generate the destination path for a documentation file.

    Args:
        file_path: The path to the file.
        input_dir: The input directory path (optional).

    Returns:
        The generated documentation file path.
    """
```

### _get_file_content(file_path)
```python
def _get_file_content(self, file_path: str) -> str:
    """
    Read content from a file with error handling.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file.

    Raises:
        Exception: If there is an error reading the file.
    """
```

### _get_template_content(template_path)
```python
def _get_template_content(self, template_path: str) -> str:
    """
    Read content from a template file with error handling.

    Args:
        template_path: The path to the template file.

    Returns:
        The content of the template file.

    Raises:
        Exception: If there is an error reading the template file.
    """
```

### _generate_documentation(file_content, file_path, template, input_dir)
```python
def _generate_documentation(self, file_content: str, file_path: str, template: str, input_dir: str = None) -> str:
    """
    Generate documentation for a file using the LLM backend.

    Args:
        file_content: The content of the file.
        file_path: The path to the file.
        template: The template content for documentation.
        input_dir: The input directory path (optional).

    Returns:
        The generated documentation content.

    Raises:
        Exception: If there is an error generating the documentation.
    """
```

### _write_documentation(doc_content, doc_path)
```python
def _write_documentation(self, doc_content: str, doc_path: pathlib.Path) -> bool:
    """
    Write documentation content to a file.

    Args:
        doc_content: The documentation content to write.
        doc_path: The path to the documentation file.

    Returns:
        True if the documentation was written successfully, False otherwise.
    """
```

### _regenerate_single_file(file_path, input_dir)
```python
def _regenerate_single_file(self, file_path: str, input_dir: str = None) -> bool:
    """
    Generate documentation for a single file.

    Args:
        file_path: The path to the file.
        input_dir: The input directory path (optional).

    Returns:
        True if the documentation was generated successfully, False otherwise.
    """
```

### _process_single_file(file_path, input_dir)
```python
def _process_single_file(self, file_path: str, input_dir: str = None) -> bool:
    """
    Process a single file and generate documentation.

    Args:
        file_path: The path to the file.
        input_dir: The input directory path (optional).

    Returns:
        True if the documentation was generated successfully, False otherwise.
    """
```

### _process_directory(source_dir, input_dir)
```python
def _process_directory(self, source_dir: str, input_dir: str = None):
    """
    Walks the directory tree and processes files for documentation generation.

    Args:
        source_dir: The source directory path.
        input_dir: The input directory path (optional).
    """
```

### generate_docs(source_dir, single_file, input_dir)
```python
def generate_docs(self, source_dir: str = None, single_file: str = None, input_dir: str = None):
    """
    Entrypoint to begin documentation generation.

    Args:
        source_dir: The source directory path (optional).
        single_file: The path to a single file to regenerate its documentation (optional).
        input_dir: The input directory path (optional).
    """
```

## Usage Examples
To use the `DocumentationGenerator` class, you can create an instance and call the `generate_docs` method, specifying the source directory or a single file to process.

```python
generator = DocumentationGenerator(config_path='config.yaml')
generator.generate_docs(source_dir='src')
```

To use the `LLMBackend` class directly, you can create an instance with a configuration dictionary and call the `generate_completion` method with a prompt.

```python
config = {
    'backend': 'perplexity',
    'perplexity': {
        'api_key': 'your_api_key',
        'model': 'your_model'
    }
}
backend = LLMBackend(config)
completion = backend.generate_completion('Your prompt here')
print(completion)
```