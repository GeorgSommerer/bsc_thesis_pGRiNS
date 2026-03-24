# Project Setup

## Clone the repository
```bash
git clone https://github.com/MoltenEcdysone09/GRiNS.git
cd GRiNS
```

## Pre-requisite
Poetry is used to manage dependencies and virtual environments for this project. Ensure you have `python >=3.12` and `poetry >=2.0.0`. Poetry can be installed from package manager or refer to [Installation Guide](https://python-poetry.org/docs/#installation).

## Setting up dependencies through poetry
- Make sure all other virtual environments are deactived. Otherwise poetry overwrites into the environment.
```bash
deactivate
```
- Include (all-groups) to install packages required for documentation and testing.
- Include (all-extras) to install with GPU support. CUDA system libraries may need to be installed seperately.
```bash
poetry sync --all-groups --all-extras
```

## Changing dependencies
### Interactively
- Add packages with
```
poetry add <pacage-name>
```
- Remove packages with
```
poetry remove <package-name>
```
### Manually
- Edit the `pyproject.toml` file
- Lock the package versions
```bash
poetry lock
```
- Sync poetry dependencies
```bash
poetry sync --all-groups --all-extras
```

## Running 
- Running python
```bash
poetry run python
```
- Running scripts
```bash
poetry run <script-name>
```

## Further details
Refer to [poetry's documentation](https://python-poetry.org/docs/)