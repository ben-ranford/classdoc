import os
from pathlib import Path
import logging
import yaml
from openai import OpenAI
import requests

class LLMBackend:
    def __init__(self, config):
        self.config = config
        self.backend = config.get('backend', 'perplexity')
        
    def generate_completion(self, prompt):
        if self.backend == 'perplexity':
            return self._perplexity_completion(prompt)
        elif self.backend == 'lmstudio':
            return self._lmstudio_completion(prompt)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

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

    def _lmstudio_completion(self, prompt):
        url = f"http://localhost:{self.config['lmstudio']['port']}/v1/chat/completions"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.config['lmstudio']['model'],
            "temperature": 0.7
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

class DocumentationGenerator:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.docs_dir = Path(self.config.get('output_directory', 'docs'))
        self.supported_extensions = ['.py', '.js', '.ts']
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

    def _generate_doc_path(self, file_path):
        relative_path = file_path.relative_to(Path.cwd())
        doc_path = self.docs_dir / f"{relative_path.with_suffix('.md')}"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        return doc_path

    def _get_file_content(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _get_template_content(self, template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _generate_documentation(self, file_content, file_path, template):
        prompt = f"""Generate comprehensive markdown documentation for the following code file.
        Include:
        - File overview
        - Dependencies
        - Classes and methods with descriptions
        - Usage examples
        - Any important notes

        Format:
        {template}
        
        File path: {file_path}
        
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

    def generate_docs(self, source_dir="."):
        self._create_docs_directory()
        
        for root, _, files in os.walk(source_dir):
            if self.docs_dir.name in root:
                continue
                
            for file in files:
                file_path = Path(root) / file
                
                if file_path.suffix not in self.supported_extensions:
                    continue
                    
                self.logger.info(f"Processing: {file_path}")
                
                file_content = self._get_file_content(file_path)
                template = self._get_template_content("template.md")
                doc_content = self._generate_documentation(file_content, file_path, template)
                
                if doc_content:
                    doc_path = self._generate_doc_path(file_path)
                    self._write_documentation(doc_content, doc_path)

def main():
    try:
        generator = DocumentationGenerator()
        generator.generate_docs()
    except Exception as e:
        logging.error(f"Documentation generation failed: {str(e)}")

if __name__ == "__main__":
    main()