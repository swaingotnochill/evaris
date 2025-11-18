# Release Checklist

Use this checklist for each release to ensure nothing is missed.

## For Each Release

### 1. Preparation

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with new version and date
- [ ] Update `README.md` if needed
- [ ] Review dependencies in `pyproject.toml`
- [ ] Commit all changes: `git commit -am "Prepare release vX.Y.Z"`

### 2. Quality Checks

- [ ] Run tests: `pytest --cov=evaris --cov-fail-under=90`
- [ ] Format code: `black evaris tests`
- [ ] Lint code: `ruff check evaris tests --fix`
- [ ] Type check: `mypy evaris`
- [ ] Check for security issues: `pip-audit` (if installed)

### 3. Build Package

- [ ] Clean old builds: `rm -rf dist/ build/ *.egg-info/`
- [ ] Build package: `python -m build`
- [ ] Check package: `twine check dist/*`
- [ ] Verify package contents: `unzip -l dist/*.whl`

### 4. Test on TestPyPI

- [ ] Upload to TestPyPI: `twine upload --repository testpypi dist/*`
- [ ] View on TestPyPI: https://test.pypi.org/project/evaris/
- [ ] Test installation:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple/ evaris
  ```
- [ ] Test basic functionality
- [ ] Test optional dependencies: `pip install evaris[tracing]`

### 5. Publish to PyPI

- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] View on PyPI: https://pypi.org/project/evaris/
- [ ] Test installation: `pip install evaris`
- [ ] Verify basic functionality

### 6. Git and GitHub

- [ ] Create git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Push changes: `git push origin main`
- [ ] Push tags: `git push origin vX.Y.Z`
- [ ] Create GitHub release: https://github.com/swaingotnochill/evaris/releases/new
- [ ] Upload wheel and source to GitHub release (optional)

### 7. Post-Release

- [ ] Update version to next dev version in `pyproject.toml` (e.g., `0.2.0-dev`)
- [ ] Commit: `git commit -am "Bump version to X.Y.Z-dev"`
- [ ] Monitor PyPI downloads: https://pepy.tech/project/evaris
- [ ] Announce release (optional):
  - [ ] GitHub Discussions
  - [ ] Twitter/X
  - [ ] Reddit
  - [ ] Blog post

## Quick Commands

```bash
./publish.sh test   # Test on TestPyPI
./publish.sh prod   # Publish to PyPI

python -m build && twine check dist/* && twine upload --repository testpypi dist/*
```

## Emergency Rollback

To yank a release (not delete, just mark as bad):

```bash
# Yank from PyPI (keeps package but discourages installation)
pip install yank
yank evaris X.Y.Z "Reason for yanking"
```
Note: Do not delete or re-upload the same version!

