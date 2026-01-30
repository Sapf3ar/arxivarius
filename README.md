## Testing Documentation

### Overview
This document outlines the testing strategy used for the functionalities within the main.py file, including unit tests and integration tests.

### Unit Tests
- **Functionality Tested:** Initialization of session keys, file upload mechanism, and document downloading.
- **Testing Framework:** pytest.
- **Test Cases: **
    - Valid and invalid paper links were tested to ensure proper handling by the `download_papers` function.

### Integration Tests
- Integration tests will be implemented in future iterations to ensure that functionalities involving external services work correctly.

### Running Tests
To run the tests, execute `pytest` in the project directory.