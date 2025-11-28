# Contributing to Medulloblastoma G3/G4 Classification

Thank you for your interest in contributing to this project! We welcome contributions from researchers, developers, and clinicians who want to advance pediatric cancer research.

## Ways to Contribute

- 🐛 **Report bugs** and issues
- 💡 **Suggest new features** or improvements
- 📝 **Improve documentation** and examples
- 🧪 **Add tests** to increase code coverage
- 🔬 **Contribute scientific insights** and validation
- 💻 **Submit code improvements** and optimizations

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/bitsxlamarato-medulloblastoma.git
cd bitsxlamarato-medulloblastoma

# Add upstream remote
git remote add upstream https://github.com/bsc-health/bitsxlamarato-medulloblastoma.git
```

### 2. Set Up Development Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,docs]"

# Verify installation
python -c "import medulloblastoma; print('✅ Development setup complete!')"
```

### 3. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create a new feature branch
git checkout -b feature/your-feature-name
```

## Development Workflow

### Code Style and Quality

We use several tools to maintain code quality:

```bash
# Format code with black and isort
make format

# Run linting checks  
make lint

# Run tests
make test

# Run tests with coverage
make test-cov
```

### Code Standards

- **Python version**: 3.10+
- **Code style**: Black formatter (line length 99)
- **Import sorting**: isort with black profile
- **Linting**: flake8 for code quality
- **Type hints**: Use type hints for all new functions
- **Docstrings**: NumPy style docstrings for all public functions

Example function with proper documentation:

```python
from typing import Union, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np

def process_expression_data(
    data: pd.DataFrame,
    variance_threshold: float = 0.1,
    zero_threshold: float = 0.2,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, dict]:
    """
    Process gene expression data with filtering and normalization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Gene expression data with samples as rows and genes as columns
    variance_threshold : float, optional
        Minimum variance threshold for gene filtering, by default 0.1
    zero_threshold : float, optional
        Maximum fraction of zero values allowed per gene, by default 0.2
    method : str, optional
        Processing method ('standard' or 'robust'), by default 'standard'
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Processed data and processing statistics
        
    Examples
    --------
    >>> processed_data, stats = process_expression_data(
    ...     data, variance_threshold=0.05, method='robust'
    ... )
    >>> print(f"Filtered {stats['genes_removed']} low-variance genes")
    """
    # Implementation here
    pass
```

### Testing

We use pytest for testing. Write tests for all new functionality:

```python
# tests/test_new_feature.py
import pytest
import pandas as pd
import numpy as np
from medulloblastoma.new_module import new_function

class TestNewFunction:
    """Test new functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(np.random.randn(100, 50))
    
    def test_new_function_basic(self, sample_data):
        """Test basic functionality."""
        result = new_function(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] <= sample_data.shape[0]  # May filter samples
        
    def test_new_function_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty data
        with pytest.raises(ValueError):
            new_function(pd.DataFrame())
            
        # Test invalid parameters
        with pytest.raises(ValueError):
            new_function(sample_data, invalid_param=-1)
```

Run tests before submitting:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_new_feature.py -v

# Run with coverage
pytest --cov=medulloblastoma tests/
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Environment**: OS, Python version, package versions
- **Reproduction steps**: Minimal code to reproduce the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traceback if applicable

```markdown
**Bug Description**
Brief description of the issue

**Environment**
- OS: Windows 11 / macOS 13 / Ubuntu 22.04
- Python version: 3.11.2
- Package version: 0.1.0

**Steps to Reproduce**
1. Load data with `load_data(...)`
2. Run preprocessing with `preprocess(...)`
3. Error occurs

**Expected Behavior**
Should process data without errors

**Actual Behavior**
Raises ValueError: ...

**Error Traceback**
```
Traceback (most recent call last):
  ...
```

### Feature Requests

For feature requests, please include:

- **Problem description**: What problem would this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought of
- **Use case**: Real-world examples where this would be useful
- **Scientific rationale**: For research features, include scientific justification

### Documentation Improvements

Documentation contributions are highly valued:

- Fix typos and grammar
- Add missing examples
- Improve API documentation
- Create tutorials for specific use cases
- Add scientific background information

### Scientific Contributions

We welcome contributions from domain experts:

- **Validation studies**: Test on new datasets
- **Method improvements**: Better algorithms or preprocessing
- **Biological insights**: Interpretation of results
- **Literature reviews**: Relevant research summaries
- **Benchmarking**: Comparison with other tools

## Submission Process

### 1. Make Your Changes

- Keep commits focused and atomic
- Write clear commit messages
- Add tests for new functionality
- Update documentation as needed

```bash
# Make your changes
# ... edit files ...

# Run quality checks
make format
make lint
make test

# Commit your changes
git add .
git commit -m "Add new feature: improved outlier detection

- Implement robust outlier detection method
- Add comprehensive tests
- Update documentation with examples
- Improve performance by 2x on large datasets"
```

### 2. Update Documentation

If you've added new features:

- Add docstrings to new functions
- Update the README if needed
- Add examples to docs/examples/
- Update API reference if needed

### 3. Run Final Checks

```bash
# Make sure everything works
make clean
make requirements-dev
make test
make lint

# Build documentation to check for errors
make docs
```

### 4. Submit Pull Request

```bash
# Push your changes
git push origin feature/your-feature-name

# Create pull request on GitHub
# Include description of changes and motivation
```

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what changes were made and why
- **Tests**: Include relevant tests
- **Documentation**: Update docs for any user-facing changes
- **Breaking changes**: Clearly document any breaking changes
- **Related issues**: Link to related issues/feature requests

#### Pull Request Template

```markdown
## Summary
Brief description of the changes

## Motivation
Why are these changes needed?

## Changes Made
- [ ] Added new feature X
- [ ] Fixed bug in Y
- [ ] Updated documentation for Z
- [ ] Added tests for A, B, C

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Updated function docstrings
- [ ] Updated README if needed
- [ ] Added examples if applicable

## Breaking Changes
List any breaking changes (if none, remove this section)

## Related Issues
Closes #123
Relates to #456
```

## Review Process

1. **Automated checks**: GitHub Actions runs tests and linting
2. **Code review**: Maintainers review code quality and design
3. **Testing**: Verify that changes work as expected
4. **Documentation review**: Check that documentation is clear and complete
5. **Scientific review**: For research contributions, validate scientific accuracy

## Community Guidelines

### Code of Conduct

- Be respectful and professional
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Credit others for their contributions
- Maintain patient confidentiality and data privacy

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions
- **Email**: For security issues or private matters

## Scientific Contribution Guidelines

### Data Privacy

- Never commit patient data to the repository
- Use synthetic or anonymized data for examples
- Follow institutional review board (IRB) guidelines
- Respect data use agreements and licenses

### Research Ethics

- Properly cite all sources and methods
- Acknowledge collaborators and funding sources
- Follow reproducible research practices
- Consider clinical implications of changes

### Validation

For scientific contributions, please include:

- **Validation on external datasets** when possible
- **Statistical analysis** of improvements
- **Comparison with existing methods**
- **Clinical relevance** assessment

## Development Tips

### Setting Up IDE

For VS Code users, recommended extensions:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter", 
    "ms-python.isort",
    "ms-python.flake8",
    "ms-toolsai.jupyter"
  ]
}
```

### Debugging

```python
# Use loguru for logging during development
from loguru import logger

def debug_function(data):
    logger.debug(f"Input data shape: {data.shape}")
    
    result = process_data(data)
    
    logger.info(f"Processing complete: {result.shape}")
    return result
```

### Performance Testing

```python
# Use time measurements for performance testing
import time
import numpy as np

def benchmark_function():
    """Benchmark new vs old implementation."""
    data = generate_test_data()
    
    # Old method
    start_time = time.time()
    result_old = old_method(data)
    time_old = time.time() - start_time
    
    # New method
    start_time = time.time()
    result_new = new_method(data)
    time_new = time.time() - start_time
    
    speedup = time_old / time_new
    print(f"Speedup: {speedup:.2f}x ({time_old:.2f}s → {time_new:.2f}s)")
    
    # Verify results are equivalent
    np.testing.assert_allclose(result_old, result_new)
```

## Recognition

Contributors will be recognized in:

- **README.md**: Contributors section
- **Release notes**: Major contributions highlighted
- **Academic papers**: Co-authorship for significant scientific contributions
- **Presentations**: Conference presentations and posters

## Questions?

If you have questions about contributing:

- Check existing [GitHub Issues](https://github.com/bsc-health/bitsxlamarato-medulloblastoma/issues)
- Start a [GitHub Discussion](https://github.com/bsc-health/bitsxlamarato-medulloblastoma/discussions)
- Email the maintainers: jose.estragues@bsc.es, guillermo.prolcastelo@bsc.es

Thank you for contributing to pediatric cancer research! 🧬❤️
