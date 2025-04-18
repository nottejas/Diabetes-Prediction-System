# Contributing to Diabetes Prediction System

Thank you for considering contributing to the Diabetes Prediction System! This document provides guidelines and instructions for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/DiabetesPrediction.git
   cd DiabetesPrediction
   ```
3. Set up the development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Development Workflow

1. Create a branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
   or
   ```bash
   git checkout -b fix/issue-description
   ```

2. Make your changes

3. Run tests to ensure your changes don't break existing functionality
   
4. Commit your changes with meaningful commit messages:
   ```bash
   git commit -m "Add feature: description of the feature"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request to the main repository

## Coding Standards

- Follow PEP 8 style guidelines for Python code
- Include docstrings for all functions, classes, and modules
- Write clear, descriptive variable and function names
- Add comments for complex logic

## Project Structure

Please maintain the existing project structure:

- `app.py`: Flask web application
- `clean_dataset.py`: Data cleaning pipeline
- `data_preprocessing.py`: Data preprocessing for model training
- `model_training.py`: Model training and evaluation
- `models.py`: Database models
- `recommendation_system.py`: Recommendation system logic

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the requirements.txt if you add new dependencies
3. The PR should work on Python 3.6+
4. Make sure all tests pass
5. Be responsive to feedback and questions in your PR

## Code of Conduct

- Be respectful and inclusive
- Be constructive in feedback
- Focus on the issue, not the person

## Questions?

If you have any questions, feel free to open an issue for discussion.

Thank you for contributing to the Diabetes Prediction System! 