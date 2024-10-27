# DocumentationGenerator

## Overview
The `DocumentationGenerator` class is responsible for generating markdown documentation for supported files (currently `.py`, `.js`, and `.ts`) based on their content. It uses a Language Model (LLM) backend to generate documentation templates for the given files.

## Dependencies
- `os`: For operating system related functionalities.
- `pathlib`: To handle file paths and directories.
- `logging`: For logging purposes.
- `yaml`: To load the configuration file.
- `openai`: For interacting with the Perplexity AI API (if configured).
- `requests`: For making HTTP requests to LMStudio API (if configured).
- `argparse`: To handle command-line arguments.

## Definition
```python
class DocumentationGenerator:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        #... rest of the constructor...
```
## Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| config | dict | The configuration loaded from the YAML file. |
| docs_dir | `pathlib.Path` | The directory where generated documentation will be stored. |
| base_dir | `pathlib.Path` | The base directory from which files will be processed. |
| supported_extensions | list[str] | List of supported file extensions for documentation generation. |
| excluded_dirs | list[str] | List of directories to exclude from processing. |
| llm_backend | `LLMBackend` | The Language Model backend used for generating documentation templates. |
| logger | `logging.Logger` | The logger instance for logging purposes. |

## Methods
### `__init__(self, config_path="config.yaml")`
Initializes the `DocumentationGenerator` instance with the given configuration file path.

### `_load_config(self, config_path)`
Loads and returns the configuration from the given YAML file path.

### `_setup_logging(self)`
Configures and sets up logging for the instance using `logging.basicConfig`.

### `_create_docs_directory(self)`
Creates the documentation directory if it doesn't exist.

### `_should_skip_directory(self, dir_path)`
Checks if a directory should be skipped based on exclusion rules.

### `_generate_doc_path(self, file_path, input_dir=None)`
Generates and returns the documentation path based on the given file path and input directory.

### `_get_file_content(self, file_path)`
Returns the content of the given file path.

### `_get_template_content(self, template_path)`
Returns the content of the given template path.

### `_generate_documentation(self, file_content, file_path, template, input_dir=None)`
Generates and returns the markdown documentation content for the given file using the LLM backend and template.

### `_write_documentation(self, doc_content, doc_path)`
Writes the generated documentation content to the given file path.

### `_regenerate_single_file(self, file_path, input_dir=None)`
Processes a single file for documentation regeneration.

### `generate_docs(self, source_dir=None, single_file=None, input_dir=None)`
Generates documentation for files in the specified or default directories. If `single_file` is provided, regenerates documentation only for that file.

## Usage Examples

1. **Regenerating documentation for a single file:**
   ```
   python main.py -f path/to/single_file.py
   ```

2. **Processing files in a specific input directory:**
   ```
   python main.py -i path/to/input_directory
   ```

3. **Using a custom configuration file:**
   ```
   python main.py -c path/to/custom_config.yaml
   ```