# Testing

## Local testing
Local testing requires the `test` group packages to be installed.

### Flake8
- Checks for syntax errors
```bash
poetry run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```
### Pytest
- Checks if the package can be imported without errors
```bash
poetry run pytest tests/
```

## GitHub CI/CD
- Refer to [GitHub Documentation](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python). 
- Custom script for project at [python-app.yml](https://github.com/MoltenEcdysone09/GRiNS/blob/main/.github/workflows/python-app.yml)