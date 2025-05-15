import os
import json
from pathlib import Path
import logging
import yaml
from openai import OpenAI
import requests
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMBackend:
    def __init__(self, config):
        self.config = config
        self.backend = config.get('backend', 'perplexity')
        self.logger = logging.getLogger(__name__)

        # Map backends to dict.
        self.backend_handlers = {
            'perplexity': self._perplexity_completion,
            'lmstudio': self._lmstudio_completion,
            'huggingface': self._huggingface_completion,
            'mistral': self._mistral_completion
        }

    def generate_completion(self, prompt):
        """Completion dispatcher."""
        handler = self.backend_handlers.get(self.backend)
        if not handler:
            raise ValueError(f"Unsupported backend: {self.backend}")
        return handler(prompt)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
    )
    def _perplexity_completion(self, prompt):
        """pplx is the OpenAI SDK, just with a custom endpoint."""
        client = OpenAI(
            api_key=self.config['perplexity']['api_key'],
            base_url="https://api.perplexity.ai"
        )
        response = client.chat.completions.create(
            model=self.config['perplexity']['model'],
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
    )
    def _lmstudio_completion(self, prompt):
        """LM Studio..."""
        lmstudio_cfg = self.config.get('lmstudio', {})
        url = f"http://localhost:{lmstudio_cfg.get('port', 1234)}/v1/chat/completions"
        streaming = lmstudio_cfg.get('streaming', False)
        printstream = lmstudio_cfg.get('printstream', False)

        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": lmstudio_cfg.get('model', 'default_model'),
            "temperature": lmstudio_cfg.get('temperature', 0.7)
        }

        if streaming:
            return self._handle_lmstudio_streaming(url, payload, printstream)
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

    def _handle_lmstudio_streaming(self, url, payload, printstream):
        """LM Studio streaming response handler."""
        payload["stream"] = True
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        content = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            try:
                data = line.strip()
                if data.startswith("data:"):
                    data = data[5:].strip()
                if data == "[DONE]":
                    break
                    
                chunk = json.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                content += delta

                if printstream and delta:
                    print(delta, end="", flush=True)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error during streaming: {str(e)}")
                continue
            except Exception as e:
                self.logger.error(f"Unexpected error during streaming: {str(e)}")
                continue

        if printstream:
            print(flush=True)
        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
    )
    def _huggingface_completion(self, prompt):
        """Generate completion using HuggingFace API."""
        hf_config = self.config.get('huggingface', {})
        headers = {
            "Authorization": f"Bearer {hf_config['api_key']}",
            "Content-Type": "application/json"
        }
        api_url = f"https://api-inference.huggingface.co/models/{hf_config['model']}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": hf_config.get('temperature', 0.7),
                "max_new_tokens": hf_config.get('max_new_tokens', 8192),
                "return_full_text": False
            }
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]['generated_text']

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
    )
    def _mistral_completion(self, prompt):
        """Generate completion using Mistral AI API."""
        mistral_config = self.config.get('mistral', {})
        client = OpenAI(
            api_key=mistral_config['api_key'],
            base_url="https://api.mistral.ai/v1"
        )
        response = client.chat.completions.create(
            model=mistral_config['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=mistral_config.get('temperature', 0.7),
            max_tokens=mistral_config.get('max_tokens', 8192)
        )
        return response.choices[0].message.content

class DocumentationGenerator:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.docs_dir = Path(self.config.get('output_directory', 'docs'))

        # Check if base_directory is a placeholder or not set
        base_dir_config = self.config.get('base_directory')
        if not base_dir_config or '/path/to/' in base_dir_config:
            self.base_dir = Path(os.getcwd())
        else:
            self.base_dir = Path(base_dir_config)

        self.supported_extensions = ['.py', '.js', '.ts']
        self.excluded_dirs = self.config.get('excluded_dirs', [
            'venv', '.venv', 'env', '.env', 'node_modules', '.git',
            '__pycache__', '.pytest_cache', '.idea', '.vscode'
        ])
        self.llm_backend = LLMBackend(self.config)
        self._setup_logging()

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, FileNotFoundError) as e:
            raise ValueError(f"Error loading config file {config_path}: {str(e)}")

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _create_docs_directory(self):
        """Ensures output directory exists - using parents=True to create all intermediate directories."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def _should_skip_directory(self, dir_path):
        """Check if directory should be skipped based on exclusion rules."""
        dir_name = Path(dir_path).name
        return (dir_name.startswith('.') or
                dir_name in self.excluded_dirs or
                self.docs_dir.name in Path(dir_path).parts)

    def _get_relative_path(self, file_path, input_dir=None):
        """Get relative path based on input or base directory."""
        try:
            base = Path(input_dir).resolve() if input_dir else self.base_dir
            return Path(file_path).resolve().relative_to(base)
        except ValueError:
            self.logger.warning(f"File {file_path} is not in base directory. Using filename only.")
            return Path(Path(file_path).name)

    def _generate_doc_path(self, file_path, input_dir=None):
        """Generate the destination path for a documentation file."""
        relative_path = self._get_relative_path(file_path, input_dir)
        doc_path = self.docs_dir / relative_path.with_suffix('.md')
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        return doc_path

    def _get_file_content(self, file_path):
        """Read content from a file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def _get_template_content(self, template_path):
        """Read content from a template file with error handling."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading template {template_path}: {str(e)}")
            raise

    def _generate_documentation(self, file_content, file_path, template, input_dir=None):
        """Generate documentation for a file using the LLM backend."""
        relative_path = self._get_relative_path(file_path, input_dir)

        prompt = f"""Generate comprehensive markdown documentation for the following code file. IF the file content itself is EMPTY DO NOT ELABORATE FURTHER WITH ANY OF THE ABOVE OR BELOW & do not generate documentation. do not provide any further information. Do not provide placeholders. Do not reference the instructions.

        When printing the file path, ensure you don't include anything above the project root path (i.e. /users/username/...). Don't output the code for the entire file, just relevant usage examples (If applicable).

        Format:
        {template}

        File path (from the project root): {relative_path}

        Code:
        {file_content}
        """

        try:
            return self.llm_backend.generate_completion(prompt)
        except Exception as e:
            self.logger.error(f"Error generating documentation for {file_path}: {str(e)}")
            return None

    def _write_documentation(self, doc_content, doc_path):
        """Write documentation content to a file."""
        try:
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            self.logger.info(f"Generated documentation: {doc_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error writing documentation to {doc_path}: {str(e)}")
            return False

    def _regenerate_single_file(self, file_path, input_dir=None):
        """Generate documentation for a single file."""
        try:
            file_content = self._get_file_content(file_path)
            template = self._get_template_content("template.md")
            doc_content = self._generate_documentation(file_content, file_path, template, input_dir)

            if doc_content:
                doc_path = self._generate_doc_path(file_path, input_dir)
                return self._write_documentation(doc_content, doc_path)
            return False
        except Exception as e:
            self.logger.error(f"Error regenerating documentation for {file_path}: {str(e)}")
            return False

    def _process_single_file(self, file_path, input_dir=None):
        """Process a single file and generate documentation."""
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"File {file_path} does not exist.")
            return False

        if file_path.suffix not in self.supported_extensions:
            self.logger.error(f"File {file_path} has unsupported extension. "
                            f"Supported extensions: {self.supported_extensions}")
            return False

        self.logger.info(f"Regenerating documentation for file: {file_path}")
        return self._regenerate_single_file(file_path, input_dir)

    def _process_directory(self, source_dir, input_dir=None):
        """Walks directory tree."""
        files_processed = 0
        files_succeeded = 0

        for root, dirs, files in os.walk(source_dir):
            # Filter out directories to skip
            dirs[:] = [d for d in dirs if not self._should_skip_directory(os.path.join(root, d))]

            for file in files:
                file_path = Path(root) / file

                if file_path.suffix not in self.supported_extensions:
                    continue

                files_processed += 1
                self.logger.info(f"Processing: {file_path}")

                if self._regenerate_single_file(file_path, input_dir):
                    files_succeeded += 1

        self.logger.info(f"Documentation generation completed. "
                        f"Processed: {files_processed}, "
                        f"Succeeded: {files_succeeded}, "
                        f"Failed: {files_processed - files_succeeded}")

    def generate_docs(self, source_dir=None, single_file=None, input_dir=None):
        """Entrypoint to begin generation."""
        self._create_docs_directory()

        # Handle single file mode
        if single_file:
            success = self._process_single_file(single_file, input_dir)
            if success:
                self.logger.info("Single file documentation regeneration completed successfully.")
            return

        # Handle directory mode
        # Handle directory mode - determine which directory to use
        if input_dir:
            source_dir = Path(input_dir)
        elif source_dir:
            source_dir = Path(source_dir)
        else:
            source_dir = self.base_dir
            self.logger.info(f"No input directory specified, using current directory: {self.base_dir}")

        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Output directory: {self.docs_dir}")
        self.logger.info(f"Source directory: {source_dir}")

        self._process_directory(source_dir, input_dir)

def main():
    """Command-line interface with proper argument parsing and error handling."""
    parser = argparse.ArgumentParser(description='Documentation Generator')
    parser.add_argument('--file', '-f',
                       help='Specify a single file to regenerate its documentation')
    parser.add_argument('--input', '-i',
                       help='Specify an input directory to process (defaults to current directory if not specified)')
    parser.add_argument('--config', '-c',
                       default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    args = parser.parse_args()

    # Set up logging level based on verbose flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        generator = DocumentationGenerator(config_path=args.config)
        generator.generate_docs(single_file=args.file, input_dir=args.input)
        logger.info("Documentation generation completed successfully")
    except Exception as e:
        logger.error(f"Documentation generation failed: {str(e)}")
        if args.verbose:
            logger.exception("Detailed error information:")
        return 1
    return 0

if __name__ == "__main__":
    exit_code = main()
    import sys
    sys.exit(exit_code)
