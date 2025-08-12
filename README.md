# Test Case Quality Evaluator

A Python script that evaluates the quality of test cases against user stories using Google's Gemini LLM with two comprehensive rubrics: Structural Coverage Index (SCI) and RUST-lite.

## Features

- **SCI Scoring**: Evaluates 7 structural criteria (0-2 scale each)
- **RUST-lite Scoring**: Evaluates 11 quality criteria (1-5 scale each) across Readability, Understandability, Specificity, and Technical Adequacy
- **Combined Scoring**: Calculates overall quality score and verdict
- **Batch Processing**: Evaluate multiple test cases at once
- **JSON Export**: Export detailed results to JSON format
- **Error Handling**: Robust error handling with fallback scores

## Installation

1. Clone or download the script files
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Gemini API key:
   ```bash
   # Option 1: Environment variable
   export GEMINI_API_KEY="your-api-key-here"
   
   # Option 2: Create .env file
   echo "GEMINI_API_KEY=your-api-key-here" > .env
   ```

## Usage

### Basic Usage

```python
from test_case_evaluator import TestCaseEvaluator

# Initialize evaluator
evaluator = TestCaseEvaluator()

# Define user story
user_story = "As a user, I want to log in so that I can access my account."

# Define test cases
test_cases = [
    {
        "preconditions": "User is on login page",
        "steps": "1. Enter username\n2. Enter password\n3. Click login button",
        "expected_results": "User is logged in and redirected to dashboard"
    }
]

# Evaluate test cases
results = evaluator.evaluate_test_cases(user_story, test_cases)

# Export to JSON
evaluator.export_to_json(results, "my_results")
```

### Command Line Usage

```bash
python test_case_evaluator.py
```

## Evaluation Rubrics

### 1. Structural Coverage Index (SCI)

Each criterion scored 0-2 (0=Poor, 1=Fair, 2=Good):

- **traceability**: Steps/results map to role, goal, reason in user story
- **coverage_breadth**: Covers all major scenarios implied in story
- **coverage_depth**: Has happy path + at least one negative or edge case
- **clarity_precision**: Steps/results unambiguous, measurable
- **completeness**: Has preconditions, steps, expected results
- **variety_types**: Covers functional, non-functional, UX, security, performance as applicable
- **consistency**: Consistent terminology/format across cases

**SCI_percent = (sum of 7 scores ÷ 14) × 100**

### 2. RUST-lite Rubric

Each sub-question scored 1-5 (1=Poor, 5=Excellent):

#### R (Readability)
- `readability_overall`: Overall readability and flow
- `grammar_clarity`: Grammar, spelling, and clarity

#### U (Understandability)
- `accuracy_to_intent`: Accurately reflects user story intent
- `clarity_comprehensibility`: Easy to understand and follow

#### S (Specificity)
- `specific_information`: Contains specific, actionable information
- `acceptance_criteria_defined`: Clear acceptance criteria
- `coverage_of_case_types`: Covers various test case types

#### T (Technical Adequacy)
- `coverage_technical_reqs`: Covers technical requirements
- `compliance_best_practices`: Follows testing best practices
- `translating_needs`: Translates user needs to testable criteria
- `ambiguity_in_technical_details`: Technical details are clear and unambiguous

**RUST_overall = (R_avg + U_avg + S_avg + T_avg) / 4**

### 3. Combined Score & Verdict

```
combined_score = (SCI_percent + (RUST_overall × 20)) / 2
verdict = High Quality (≥85), Adequate (≥70), Needs Improvement (<70)
```

## Output Format

### JSON Output
```json
{
  "user_story": "As a user...",
  "test_case_results": [
    {
      "test_case": {...},
      "sci_scores": {...},
      "sci_percent": 85.71,
      "rust_scores": {...},
      "rust_overall": 4.2,
      "combined_score": 87.71,
      "verdict": "High Quality"
    }
  ],
  "aggregated_results": {
    "sci_percent_average": 82.14,
    "rust_overall_average": 4.1,
    "combined_score_average": 85.14,
    "overall_verdict": "High Quality"
  }
}
```

### JSON Output
The JSON includes:
- All test case details (preconditions, steps, expected results)
- Individual SCI and RUST scores
- Combined scores and verdicts
- Evaluation summary with averages

## Configuration

- **Model**: Uses Gemini 2.5 Flash by default
- **Temperature**: Set to 0 for deterministic outputs
- **Response Format**: Forces JSON output for consistent parsing

## Error Handling

- Graceful fallback to default scores if API calls fail
- JSON parsing error handling
- Comprehensive error messages for debugging

## Requirements

- Python 3.7+
- Gemini API key
- Internet connection for API calls

## License

This script is provided as-is for educational and testing purposes.
