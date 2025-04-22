import os
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
        
    def generate_completion(self, prompt):
        if self.backend == 'perplexity':
            return self._perplexity_completion(prompt)
        elif self.backend == 'lmstudio':
            return self._lmstudio_completion(prompt)
        elif self.backend == 'huggingface':
            return self._huggingface_completion(prompt)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
    )
    def _perplexity_completion(self, prompt):
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
        url = f"http://localhost:{self.config['lmstudio']['port']}/v1/chat/completions"
        lmstudio_cfg = self.config.get('lmstudio', {})
        streaming = lmstudio_cfg.get('streaming', False)
        printstream = lmstudio_cfg.get('printstream', False)
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.config['lmstudio']['model'],
            "temperature": 0.7
        }
        if streaming:
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
                    import json
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
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.exceptions.HTTPError)
    )
    def _huggingface_completion(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.config['huggingface']['api_key']}",
            "Content-Type": "application/json"
        }
        api_url = f"https://api-inference.huggingface.co/models/{self.config['huggingface']['model']}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.config['huggingface'].get('temperature', 0.7),
                "max_new_tokens": self.config['huggingface'].get('max_new_tokens', 8192),
                "return_full_text": False
            }
        }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()[0]['generated_text']

class DocumentationGenerator:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.docs_dir = Path(self.config.get('output_directory', 'docs'))
        self.base_dir = Path(self.config.get('base_directory', os.getcwd()))
        self.supported_extensions = ['.py', '.js', '.ts']
        self.excluded_dirs = self.config.get('excluded_dirs', [
            'venv', '.venv', 'env', '.env', 'node_modules', '.git', 
            '__pycache__', '.pytest_cache', '.idea', '.vscode'
        ])
        self.llm_backend = LLMBackend(self.config)
        self._setup_logging()
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _create_docs_directory(self):
        self.docs_dir.mkdir(exist_ok=True)

    def _should_skip_directory(self, dir_path):
        """Check if directory should be skipped based on exclusion rules."""
        dir_name = Path(dir_path).name
        return (dir_name.startswith('.') or 
                dir_name in self.excluded_dirs or 
                self.docs_dir.name in Path(dir_path).parts)

    def _generate_doc_path(self, file_path, input_dir=None):
        try:
            if input_dir:
                relative_path = Path(file_path).resolve().relative_to(Path(input_dir).resolve())
            else:
                relative_path = Path(file_path).resolve().relative_to(self.base_dir)
        except ValueError:
            relative_path = Path(file_path).name
            self.logger.warning(f"File {file_path} is not in base directory {self.base_dir}. "
                              f"Using filename only.")
            
        doc_path = self.docs_dir / relative_path.with_suffix('.md')
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        return doc_path

    def _get_file_content(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_template_content(self, template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _generate_documentation(self, file_content, file_path, template, input_dir=None):
        try:
            if input_dir:
                doc_path = Path(file_path).resolve().relative_to(Path(input_dir).resolve())
            else:
                doc_path = Path(file_path).resolve().relative_to(self.base_dir)
        except ValueError:
            doc_path = Path(file_path).name

        prompt = f"""Generate comprehensive markdown documentation for the following code file. IF the file content itself is EMPTY DO NOT ELABORATE FURTHER WITH ANY OF THE ABOVE OR BELOW & do not generate documentation. do not provide any further information. Do not provide placeholders. Do not reference the instructions.

        When printing the file path, ensure you don't include anything above the project root path (i.e. /users/username/...). Don't output the code for the entire file, just relevant usage examples (If applicable).

        Format:
        {template}
        
        File path (from the project root): {doc_path}
        
        Code:
        {file_content}
        """
        
        try:
            return self.llm_backend.generate_completion(prompt)
        except Exception as e:
            self.logger.error(f"Error generating documentation for {file_path}: {str(e)}")
            return None

    def _write_documentation(self, doc_content, doc_path):
        try:
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            self.logger.info(f"Generated documentation: {doc_path}")
        except Exception as e:
            self.logger.error(f"Error writing documentation to {doc_path}: {str(e)}")

    def _regenerate_single_file(self, file_path, input_dir=None):
        try:
            file_content = self._get_file_content(file_path)
            template = self._get_template_content("template.md")
            doc_content = self._generate_documentation(file_content, file_path, template, input_dir)
            
            if doc_content:
                doc_path = self._generate_doc_path(file_path, input_dir)
                self._write_documentation(doc_content, doc_path)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error regenerating documentation for {file_path}: {str(e)}")
            return False

    def generate_docs(self, source_dir=None, single_file=None, input_dir=None):
        self._create_docs_directory()
        
        if single_file:
            single_file_path = Path(single_file)
            if not single_file_path.exists():
                self.logger.error(f"File {single_file} does not exist.")
                return
            if single_file_path.suffix not in self.supported_extensions:
                self.logger.error(f"File {single_file} has unsupported extension. "
                                f"Supported extensions: {self.supported_extensions}")
                return
            
            self.logger.info(f"Regenerating documentation for single file: {single_file_path}")
            success = self._regenerate_single_file(single_file_path, input_dir)
            if success:
                self.logger.info("Single file documentation regeneration completed successfully.")
            return

        if input_dir:
            source_dir = Path(input_dir)
        elif source_dir is None:
            source_dir = self.base_dir
        else:
            source_dir = Path(source_dir)
            
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info(f"Output directory: {self.docs_dir}")
        self.logger.info(f"Source directory: {source_dir}")
        
        files_processed = 0
        files_succeeded = 0
        
        for root, dirs, files in os.walk(source_dir):
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

def main():
    parser = argparse.ArgumentParser(description='Documentation Generator')
    parser.add_argument('--file', '-f', 
                       help='Specify a single file to regenerate its documentation')
    parser.add_argument('--input', '-i',
                       help='Specify an input directory to process')
    parser.add_argument('--config', '-c', 
                       default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    try:
        generator = DocumentationGenerator(config_path=args.config)
        generator.generate_docs(single_file=args.file, input_dir=args.input)
    except Exception as e:
        logging.error(f"Documentation generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()