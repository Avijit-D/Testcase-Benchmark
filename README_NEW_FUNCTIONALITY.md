# Test Case Evaluator - New Functionality

## Overview

The `test_case_evaluator.py` script has been completely refactored to accept JSON input files containing multiple `(user_story, test_case)` pairs and evaluate each independently using Google's Gemini LLM.

## Key Changes

### 1. Command-Line Interface
- Added `argparse` for command-line arguments
- Script now runs as: `python test_case_evaluator.py --input <file> --output <prefix>`

### 2. Input File Support
- **JSON Format**: List of objects with keys `"user_story"`, `"inputs"`, `"preconditions"`, `"steps"`, `"expected_results"`

### 3. Independent Processing
- Each row/object is treated independently (no more grouping by user story)
- Each `(user_story, test_case)` pair gets its own evaluation

### 4. Output Files
- **JSON Output**: Full detailed results with timestamps and evaluation IDs
- **Appending**: New evaluations are appended to existing files
- **Backward Compatibility**: Handles both old and new file formats

### 5. Enhanced Features
- Gemini temperature set to 0 for deterministic output
- Connection testing before evaluation starts
- Proper error handling and validation
- Environment variable validation for API key

## Usage Examples

### Basic Usage
```bash
# Evaluate JSON input file
python test_case_evaluator.py --input test_cases.json --output evaluation_results
```

### Input File Format

#### JSON Format (`example_input.json`)
```json
[
  {
    "user_story": "As a user, I want to reset my password...",
    "preconditions": "User is on login page...",
    "steps": "1. Click link...",
    "expected_results": "Password updated..."
  }
]
```

## Output Structure

### JSON Output Structure
```json
{
  "evaluation_summary": {
    "total_evaluations": 2,
    "first_evaluation_timestamp": "2024-01-15 10:30:00"
  },
  "evaluations": [
    {
      "evaluation_summary": {
        "total_test_cases": 3,
        "evaluation_timestamp": "2024-01-15 10:30:00",
        "evaluation_id": "20240115_103000"
      },
      "individual_results": [
        {
          "user_story": "...",
          "preconditions": "...",
          "steps": "...",
          "expected_results": "...",
          "sci_scores": {...},
          "sci_percent": 85.7,
          "rust_scores": {...},
          "rust_overall": 4.2,
          "combined_score": 87.7,
          "verdict": "High Quality"
        }
      ]
    }
  ]
}
```

## Requirements

### Environment Setup
1. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

2. Install required packages:
   ```bash
   pip install google-generativeai python-dotenv
   ```

### API Key Validation
- Script checks for `GEMINI_API_KEY` environment variable
- Tests Gemini connection before starting evaluation
- Stops execution if API key is missing or invalid

## Scoring System

### SCI (Structural Coverage Index) - 7 criteria, scored 0-2
1. **Traceability**: Test case clearly traces to user story requirements
2. **Coverage Breadth**: Covers multiple aspects of the user story
3. **Coverage Depth**: Provides thorough testing of specific functionality
4. **Clarity & Precision**: Clear, unambiguous, and precise
5. **Completeness**: Includes all necessary information
6. **Variety of Test Types**: Demonstrates different testing approaches
7. **Consistency**: Follows consistent format and style

### RUST-lite - 11 criteria, scored 1-5
- **Readability (R)**: Overall flow, grammar, and clarity
- **Understandability (U)**: Accuracy to user intent, clarity, specific information
- **Specificity (S)**: Acceptance criteria, case types, technical requirements
- **Testability (T)**: Best practices, translating needs, technical details

### Combined Score
Formula: `(SCI_percent + (RUST_overall * 20)) / 2`

### Verdicts
- **High Quality**: ≥85
- **Adequate**: 70-84
- **Needs Improvement**: <70

## Error Handling

The script includes comprehensive error handling for:
- Missing or invalid API keys
- File not found errors
- Invalid file formats
- Missing required columns/keys
- Gemini API connection failures
- JSON parsing errors

## Example Files

- `example_input.json` - Sample JSON input file

## Running the Script

1. **Set up environment**:
   ```bash
   # Create .env file with your API key
   echo "GEMINI_API_KEY=your_key_here" > .env
   ```

2. **Run evaluation**:
   ```bash
   python test_case_evaluator.py --input example_input.json --output results
   ```

3. **Check outputs**:
   - `results.json` - Detailed results

## Benefits of New Structure

1. **Scalability**: Process hundreds of test cases from files
2. **Automation**: No manual input required
3. **Consistency**: Deterministic output with temperature=0
4. **Flexibility**: Support for JSON format
5. **Traceability**: Each result includes the original user story
6. **Batch Processing**: Evaluate multiple test cases in one run
7. **Data Export**: Structured output for further analysis
8. **Appending Results**: Multiple evaluation runs stored in same file
9. **Timestamp Tracking**: Each evaluation has unique timestamp and ID

## Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   ✗ Error: GEMINI_API_KEY environment variable is not set.
   ```
   Solution: Create `.env` file with your API key

2. **Connection Test Failed**:
   ```
   ✗ Gemini connection failed: [error details]
   ```
   Solution: Check API key validity and internet connection

3. **File Format Error**:
   ```
   ✗ Validation error: Input file must be JSON format
   ```
   Solution: Ensure file has .json extension

4. **Missing Keys**:
   ```
   ✗ Validation error: Missing required keys in JSON object: {'user_story'}
   ```
   Solution: Check JSON structure matches required format

## Future Enhancements

Potential improvements for future versions:
- Batch processing with progress bars
- Custom scoring weights
- Integration with test management systems
- Web interface for easier interaction
- Support for different LLM providers
