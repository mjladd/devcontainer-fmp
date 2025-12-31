# UV Setup Documentation

## Overview

This devcontainer has been optimized to use `uv` for Python dependency management instead of pip. UV is a faster, more efficient Python package installer and resolver.

## Key Changes

### 1. Project Structure
- **pyproject.toml**: All dependencies are now defined in this file following PEP 621 standards
- **.python-version**: Specifies Python 3.11 for the project
- **requirements.txt**: Kept for reference but no longer used by the devcontainer

### 2. Dockerfile Optimizations
- UV is installed via official Docker image (`ghcr.io/astral-sh/uv:latest`)
- Virtual environment is created at `/opt/venv` during build
- Dependencies are installed using `uv pip install`
- PATH is configured to use the virtual environment automatically

### 3. DevContainer Configuration
- Python interpreter points to `/opt/venv/bin/python`
- Environment variables set for virtual environment activation
- Post-create command installs the project in editable mode
- Added Ruff extension for modern Python linting/formatting

## Using UV in the Container

### Installing New Dependencies
```bash
# Add a dependency
uv pip install package-name

# Add to pyproject.toml
# Edit pyproject.toml and add the package to the dependencies list
# Then run:
uv pip install -r pyproject.toml
```

### Updating Dependencies
```bash
# Update all dependencies
uv pip install --upgrade -r pyproject.toml

# Update a specific package
uv pip install --upgrade package-name
```

### Removing Dependencies
```bash
# Remove a package
uv pip uninstall package-name

# Don't forget to remove it from pyproject.toml
```

### Syncing Dependencies
```bash
# Sync the environment with pyproject.toml
uv pip install -r pyproject.toml
```

## Benefits of UV

1. **Speed**: UV is 10-100x faster than pip for dependency resolution
2. **Reproducibility**: Better lockfile support and dependency resolution
3. **Modern Standards**: Uses pyproject.toml instead of requirements.txt
4. **Better Caching**: More efficient caching mechanisms
5. **Compatibility**: Drop-in replacement for pip commands

## Migration Notes

- The virtual environment location changed from `.venv` to `/opt/venv`
- UV commands are backwards compatible with pip (e.g., `uv pip install` works like `pip install`)
