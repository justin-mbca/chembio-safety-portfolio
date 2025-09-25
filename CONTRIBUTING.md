# Contributing to ChemBio-SafeGuard

Thank you for your interest in contributing to the Chemical & Biological Safety System! This project aims to create robust AI safety measures for preventing misuse of chemical and biological knowledge in large language models.

## Development Setup

### Prerequisites
- Python 3.8+ 
- PyTorch 1.9+
- Transformers 4.21+
- FastAPI 0.85+

### Installation

```bash
# Clone the repository
git clone https://github.com/justin-mbca/chembio-safety-portfolio.git
cd chembio-safety-portfolio

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Demo

```bash
# Run complete portfolio demonstration
python demo_script.py

# Start API server
python simple_api.py

# Run API tests
python test_api.py
```

## Project Structure

```
chembio-safety-portfolio/
├── risk_classifier.py      # Core ML model and safety middleware
├── training_pipeline.py    # Training and evaluation framework
├── main.py                # Production FastAPI server
├── simple_api.py           # Simplified API for testing
├── demo_script.py          # Complete demonstration script
├── test_api.py             # API testing suite
├── requirements.txt        # Dependencies
├── README.md              # Main documentation
└── .gitignore             # Git ignore rules
```

## Contributing Guidelines

### Code Quality
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write comprehensive tests for new features

### Safety Considerations
- All safety-related code must be thoroughly tested
- Document potential security implications
- Follow responsible disclosure for vulnerabilities
- Ensure adversarial robustness testing

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Issue Reporting

When reporting issues:
- Use a clear and descriptive title
- Provide steps to reproduce the problem
- Include relevant error messages and logs
- Specify your environment (OS, Python version, etc.)

## Code of Conduct

This project is committed to fostering a welcoming and inclusive environment. We expect all contributors to:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Security

This project deals with AI safety and security. Please report security vulnerabilities privately to the maintainers rather than opening public issues.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for questions about contributing or reach out to the maintainers directly.
