# ClassDoc

ClassDoc is a documentation generator that uses an LLM to automatically create markdown documentation for your codebase. It supports Python, JavaScript, and TypeScript files, maintaining your project's directory structure in the generated documentation. Currently this tool doesn't take in any kind of context outside of the file it's working on, so some stuff may be wrong. You should probably double check whatever it outputs regardless.

## Supported (Tested) formats

- Python (.py)
- JavaScript (.js)
- TypeScript (.ts)

Change the code at your own risk to include whatever formats you want to use it with.

Want to see an example? head to docs/main.md. Generated using mistral-nemo-instruct-2407 (MLX) using LM Studio.

## Installation

### Prerequisites

- Python 3.13 or higher
- Poetry

#### Setting up a virtual environment

First, make sure you have `python` and `pip` installed. Then, install and activate a virtual environment:

```bash
# On macOS/Linux:
python3 -m venv venv && source venv/bin/activate
# On Windows (Command Prompt):
python -m venv venv && venv\Scripts\activate
# On Windows (PowerShell):
python -m venv venv && venv\Scripts\activate.ps1
```

Once your virtual environment is activated, install Poetry using pip:
`pip install --upgrade pip && pip install poetry`

#### Setting up the tool

Now, clone this repository:

```bash
git clone https://github.com/ben-ranford/ClassDoc.git
cd ClassDoc
```

#### Usage

1. Create a template.md file to customize the documentation format, or use the provided default.
2. Configure your settings in config.yaml. A sample is provided in config.sample.yaml.
3. Run the tool using python main.py --input {your project root} (you can also run it against a single file using --file/-f)

### Sample

#### Lotsa options

`python script.py --input /path/to/input/directory --config custom_config.yaml`

#### Plain and simple

`python script.py --input /path/to/input/directory`
