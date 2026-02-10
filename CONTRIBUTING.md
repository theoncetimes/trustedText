# Contributing to TrustedText

Thank you for your interest in contributing to TrustedText! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in [GitHub Issues]
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU/CPU)

### Contributing Code

1. **Fork the Repository**
   ```bash
   git clone https://github.com/theoncetimes/trustedText.git
   cd trustedText
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Set Up Development Environment**
   ```bash
   # Install dependencies
   pip install torch sentence-transformers scikit-learn numpy
   
   # Or use uv
   uv pip install -r pyproject.toml
   ```

4. **Make Your Changes**
   - Write clean, readable code
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed

5. **Test Your Changes**
   ```bash
   # Test training pipeline
   python trustedText.py
   
   # Test inference
   python scripts/load_model.py
   
   # Test API
   cd app && python api.py
   ```

6. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

7. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a PR on GitHub with a clear description.

## Development Guidelines

### Code Style

- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Keep functions focused and modular
- Write descriptive variable names

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update relevant docs/ files
- Include usage examples

### Testing

- Test on different platforms (CUDA, MPS, CPU) if possible
- Verify GPU detection works correctly
- Test with various text inputs
- Check performance metrics

## Areas for Contribution

### High Priority

- [ ] Additional embedding models support
- [ ] Multilingual text detection
- [ ] Performance optimizations
- [ ] Better visualization tools
- [ ] Comprehensive test suite

### Feature Ideas

- [ ] Batch processing improvements
- [ ] API authentication
- [ ] Model fine-tuning utilities
- [ ] Export to ONNX/TensorRT
- [ ] Confidence calibration
- [ ] Explainability features

### Documentation

- [ ] More usage examples
- [ ] Video tutorials
- [ ] Jupyter notebooks
- [ ] Deployment guides
- [ ] Benchmarking results

## Data Contributions

**Important**: Do not commit training data to the repository.

Instead, you can:
- Share data collection methodologies
- Contribute data preprocessing scripts
- Document data sources (with proper attribution)
- Share performance metrics on different datasets

## Code Review Process

1. All submissions require review
2. Maintainers will review your PR
3. Address any requested changes
4. Once approved, your PR will be merged

## Questions?

Feel free to:
- Open a discussion on GitHub
- Ask in pull request comments
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions make TrustedText better for everyone. We appreciate your time and effort!
