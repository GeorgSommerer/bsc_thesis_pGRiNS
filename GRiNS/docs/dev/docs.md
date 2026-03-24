# Documentation

Documentation is made using `mkdocs` with additional plugins. The `docs` group packages needs to be installed to build and deploy the documentation website.

## Local server for development
- Start the server
```bash 
poetry run mkdocs serve
```
- The website should be accessible in <http://127.0.0.1:8000/>

## Deploy to GitHub Pages
- Refer to [MkDocs Documentation](https://www.mkdocs.org/user-guide/deploying-your-docs/#github-pages). 
- WARNING: Automatically pushes and deploys to GitHub when run
```bash 
poetry run mkdocs gh-deploy
```