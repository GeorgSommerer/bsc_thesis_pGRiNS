# Building

## Local build with poetry

### Clean build
```bash
poetry build --clean
```

### Install locally built package
- Local virutal environment recommended. 
- Include `--force-reinstall` if the version number of built package isn't changed.
```bash
source /path/to/venv/bin/activate
pip install 'dist/grins-x.x.x-py3-none-any.whl[cuda12]' --force-reinstall
```

## Github CI/CD
- Refer to [GitHub Documentation](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python#publishing-to-pypi). 
- Requires [PyPI account](https://pypi.org/) and [OpenID Connect configuration](https://pypi.org/manage/account/publishing/)
- Custom script for project at [python-publish.yml](https://github.com/MoltenEcdysone09/GRiNS/blob/main/.github/workflows/python-publish.yml). 