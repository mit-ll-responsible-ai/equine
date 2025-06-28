# Coverage Report

The test coverage report is automatically generated and deployed with the documentation.

When viewing the deployed documentation at [https://mit-ll-responsible-ai.github.io/equine/](https://mit-ll-responsible-ai.github.io/equine/), you can access the coverage report at:

**https://mit-ll-responsible-ai.github.io/equine/coverage/**

This report shows:
- Line-by-line coverage highlighting
- Coverage percentages by file and function
- Missing lines highlighted in red
- Interactive HTML coverage details

The coverage report is updated automatically when code is pushed to the main branch.

## Local Development

To generate coverage reports locally:

```bash
# Generate HTML coverage report
pytest --cov=src/equine --cov-report=html:htmlcov

# Open the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```
