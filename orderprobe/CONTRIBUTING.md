# Contributing to OrderProbe

Thank you for your interest in contributing to OrderProbe! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/Zhaolu-K/OrderProbe.git
   cd OrderProbe
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## 🛠️ Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Code Quality

Run the quality checks:

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .

# Run tests
pytest
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation
- `test:` for tests
- `refactor:` for code refactoring

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## 📋 Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints for function parameters and return values
- Write docstrings for all public functions
- Keep functions small and focused

### Naming Conventions
- Use descriptive variable and function names
- Use snake_case for variables and functions
- Use PascalCase for classes
- Use UPPER_CASE for constants

### Documentation
- Update README files for any new features
- Add docstrings to all public functions
- Include examples in documentation

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_specific.py

# Run with coverage
pytest --cov=idiom_eval --cov-report=html
```

### Writing Tests
- Place tests in `tests/` directory
- Use descriptive test names
- Test both positive and negative cases
- Mock external dependencies when appropriate

## 📊 Adding New Metrics

When adding new evaluation metrics:

1. **Create the calculator** in the appropriate module (Sacc, Slogic, etc.)
2. **Add comprehensive documentation** in the module's README
3. **Update the main README** with the new metric
4. **Add tests** for the new functionality
5. **Update setup.py** if new dependencies are required

### Example Structure for New Metric

```python
# your_metric.py
def calculate_your_metric(predictions, references):
    """
    Calculate your new metric.

    Args:
        predictions: List of model predictions
        references: List of reference explanations

    Returns:
        Dict with metric scores
    """
    # Implementation here
    pass
```

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to reproduce**: Step-by-step instructions
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, dependencies
6. **Code sample**: Minimal code to reproduce the issue

## 💡 Feature Requests

For feature requests, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the problem** you're trying to solve
3. **Explain your proposed solution**
4. **Consider alternative approaches**
5. **Include example use cases**

## 📚 Documentation

### Building Documentation
```bash
# Install docs dependencies
pip install sphinx sphinx-rtd-theme

# Build docs
cd docs
make html
```

### Writing Documentation
- Use clear, concise language
- Include code examples
- Explain concepts thoroughly
- Update docs with code changes

## 🤝 Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## 📞 Getting Help

If you need help:

1. Check the documentation first
2. Search existing issues
3. Ask questions in discussions
4. Contact maintainers for sensitive issues

Thank you for contributing to the Idiom Evaluation Framework! 🎉
