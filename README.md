# Execution Plan Generator Backend

A Flask-based backend service that generates execution plans using LLM (Large Language Model) technology. This service helps create structured plans based on user requirements, questions, and additional context.

## Features

- Generate execution plans based on user input
- Support for custom document formats
- Configurable plan depth
- CORS-enabled API endpoints
- Environment variable configuration
- Async processing support

## Prerequisites

- Python 3.x
- pip (Python package manager)
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

### API Endpoints

#### Generate Plan
- **URL**: `/generate-plan`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "user_doc_format": "markdown",
    "user_question": "How to implement feature X",
    "user_requirements": ["req1", "req2"],
    "additional_context": "Project context",
    "max_depth": 2
  }
  ```
- **Parameters**:
  - `user_doc_format` (required): Format of the document
  - `user_question` (required): User's question
  - `user_requirements` (optional): Array of requirements
  - `additional_context` (optional): Additional context for plan generation
  - `max_depth` (optional): Maximum depth of the execution plan (default: 1)

Example request using curl:
```bash
curl -X POST http://localhost:5000/generate-plan \
  -H "Content-Type: application/json" \
  -d '{
    "user_doc_format": "markdown",
    "user_question": "How to implement feature X",
    "user_requirements": ["req1", "req2"],
    "additional_context": "Project context",
    "max_depth": 2
  }'
```

## Project Structure

- `app.py`: Main Flask application
- `ExecutionPlanGenerator.py`: Core plan generation logic
- `ExecutionPlan.py`: Execution plan data structure
- `LLMAgent.py`: LLM interaction handling
- `utils.py`: Utility functions
- `requirements.txt`: Project dependencies

## Deployment

The project is configured for deployment on AWS Elastic Beanstalk with the following files:
- `.ebextensions/`: Elastic Beanstalk configuration
- `.elasticbeanstalk/`: Elastic Beanstalk deployment settings
- `Procfile`: Process definition for the application
- `.ebignore`: Files to ignore during deployment

## Development

- The project uses Flask for the web framework
- CORS is enabled for all routes
- Async processing is implemented for better performance
- Environment variables are managed using python-dotenv

## License

[Add your license information here] 