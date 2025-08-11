#!/usr/bin/env python3
"""
Test Case Quality Evaluator using Google's Gemini LLM

This script evaluates the quality of test cases against user stories using:
1. Structural Coverage Index (SCI) - 7 criteria scored 0-2
2. RUST-lite rubric - 11 criteria scored 1-5
3. Combined scoring and quality verdicts

"""

import os
import json
import argparse
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class TestCaseEvaluator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it to the constructor.")
        
        # Configure Gemini with temperature=0 for deterministic output
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prompt templates for SCI evaluation
        self.sci_prompt_template = """You are an expert test case evaluator. Evaluate the following test case against the user story using the Structural & Coverage Index (SCI) rubric.

User Story: {user_story}

Inputs: {inputs}

Test Case:
Preconditions: {preconditions}
Steps: {steps}
Expected Results: {expected_results}

SCI Rubric (score each criterion 0-2):
1. Traceability: Test case clearly traces back to user story requirements
2. Coverage Breadth: Test case covers multiple aspects of the user story
3. Coverage Depth: Test case provides thorough testing of specific functionality
4. Clarity & Precision: Test case is clear, unambiguous, and precise
5. Completeness: Test case includes all necessary information
6. Variety of Test Types: Test case demonstrates different testing approaches
7. Consistency: Test case follows consistent format and style

Return ONLY a JSON object with integer scores (0-2)(0: not present, 1: present but not fully, 2: present and fully) for each criterion, no explanations:
{{
    "traceability": 0,
    "coverage_breadth": 0,
    "coverage_depth": 0,
    "clarity_precision": 0,
    "completeness": 0,
    "variety_types": 0,
    "consistency": 0
}}"""

        # Prompt template for RUST-lite evaluation
        self.rust_prompt_template = """You are an expert test case evaluator. Evaluate the following test case against the user story using the RUST-lite rubric.

User Story: {user_story}

Inputs: {inputs}

Test Case:
Preconditions: {preconditions}
Steps: {steps}
Expected Results: {expected_results}

RUST-lite Rubric (score each sub-question 1-5):
READABILITY (R):
- Overall readability and flow: 1-5
- Grammar and clarity: 1-5

UNDERSTANDABILITY (U):
- Accuracy to user intent: 1-5
- Clarity and comprehensibility: 1-5
- Specific information provided: 1-5

SPECIFICITY (S):
- Acceptance criteria clearly defined: 1-5
- Coverage of different case types: 1-5
- Coverage of technical requirements: 1-5

TESTABILITY (T):
- Compliance with testing best practices: 1-5
- Translating user needs to testable criteria: 1-5
- Ambiguity in technical details: 1-5

Return ONLY a JSON object with integer scores (1-5) for each sub-question, no explanations:
{{
    "readability_overall": 1,
    "grammar_clarity": 1,
    "accuracy_to_intent": 1,
    "clarity_comprehensibility": 1,
    "specific_information": 1,
    "acceptance_criteria_defined": 1,
    "coverage_of_case_types": 1,
    "coverage_technical_reqs": 1,
    "compliance_best_practices": 1,
    "translating_needs": 1,
    "ambiguity_in_technical_details": 1
}}"""

    def test_connection(self) -> bool:
        """Test Gemini connection before starting evaluation."""
        try:
            # Set temperature=0 for deterministic output
            test_response = self.model.generate_content("Connection test: reply with OK", generation_config=genai.types.GenerationConfig(temperature=0))
            print(f"✓ Gemini connection test successful: {test_response.text.strip()}")
            return True
        except Exception as e:
            print(f"✗ Gemini connection failed: {e}")
            return False

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini API with the given prompt and return parsed JSON response."""
        try:
            # Set temperature=0 for deterministic output
            response = self.model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0))
            
            # Extract text content from response
            if hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            # Clean the response text to extract JSON
            response_text = response_text.strip()
            
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text}")
            return None
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    def _get_default_scores(self) -> Dict[str, Any]:
        """Return default SCI scores when API call fails."""
        return {
            "traceability": 1,
            "coverage_breadth": 1,
            "coverage_depth": 1,
            "clarity_precision": 1,
            "completeness": 1,
            "variety_types": 1,
            "consistency": 1
        }

    def _get_default_rust_scores(self) -> Dict[str, Any]:
        """Return default RUST scores when API call fails."""
        return {
            "readability_overall": 3,
            "grammar_clarity": 3,
            "accuracy_to_intent": 3,
            "clarity_comprehensibility": 3,
            "specific_information": 3,
            "acceptance_criteria_defined": 3,
            "coverage_of_case_types": 3,
            "coverage_technical_reqs": 3,
            "compliance_best_practices": 3,
            "translating_needs": 3,
            "ambiguity_in_technical_details": 3
        }

    def calculate_sci_score(self, sci_scores: Dict[str, int]) -> float:
        """Calculate SCI percentage from individual scores."""
        total_score = sum(sci_scores.values())
        return (total_score / 14) * 100

    def calculate_rust_score(self, rust_scores: Dict[str, int]) -> float:
        """Calculate RUST overall score from individual scores."""
        # Group scores by RUST categories
        readability = (rust_scores["readability_overall"] + rust_scores["grammar_clarity"]) / 2
        understandability = (rust_scores["accuracy_to_intent"] + rust_scores["clarity_comprehensibility"] + rust_scores["specific_information"]) / 3
        specificity = (rust_scores["acceptance_criteria_defined"] + rust_scores["coverage_of_case_types"] + rust_scores["coverage_technical_reqs"]) / 3
        testability = (rust_scores["compliance_best_practices"] + rust_scores["translating_needs"] + rust_scores["ambiguity_in_technical_details"]) / 3
        
        return (readability + understandability + specificity + testability) / 4

    def calculate_combined_score(self, sci_percent: float, rust_overall: float) -> float:
        """Calculate combined score: (SCI_percent + (RUST_overall * 20)) / 2"""
        return (sci_percent + (rust_overall * 20)) / 2

    def get_verdict(self, combined_score: float) -> str:
        """Determine verdict based on combined score."""
        if combined_score >= 85:
            return "High Quality"
        elif combined_score >= 70:
            return "Adequate"
        else:
            return "Needs Improvement"

    def evaluate_test_case(self, user_story: str, test_case: Dict[str, str]) -> Dict[str, Any]:
        """Evaluate a single test case using both SCI and RUST-lite rubrics."""
        # Format test case for prompts
        preconditions = test_case.get('preconditions', '')
        steps = test_case.get('steps', '')
        expected_results = test_case.get('expected_results', '')
        inputs = test_case.get('inputs', '') # Get the new 'inputs' field
        
        # Get SCI scores
        sci_prompt = self.sci_prompt_template.format(
            user_story=user_story,
            inputs=inputs, # Include inputs in the prompt
            preconditions=preconditions,
            steps=steps,
            expected_results=expected_results
        )
        sci_scores = self._call_gemini(sci_prompt)
        
        # Use default SCI scores if API call failed
        if not sci_scores:
            sci_scores = self._get_default_scores()
        
        # Get RUST scores
        rust_prompt = self.rust_prompt_template.format(
            user_story=user_story,
            inputs=inputs, # Include inputs in the prompt
            preconditions=preconditions,
            steps=steps,
            expected_results=expected_results
        )
        rust_scores = self._call_gemini(rust_prompt)
        
        # Use default RUST scores if API call failed
        if not rust_scores:
            rust_scores = self._get_default_rust_scores()
        
        # Calculate scores
        sci_percent = self.calculate_sci_score(sci_scores)
        rust_overall = self.calculate_rust_score(rust_scores)
        combined_score = self.calculate_combined_score(sci_percent, rust_overall)
        verdict = self.get_verdict(combined_score)
        
        return {
            "user_story": user_story,
            "preconditions": preconditions,
            "steps": steps,
            "expected_results": expected_results,
            "inputs": inputs, # Include inputs in the result
            "sci_scores": sci_scores,
            "sci_percent": sci_percent,
            "rust_scores": rust_scores,
            "rust_overall": rust_overall,
            "combined_score": combined_score,
            "verdict": verdict
        }

    def read_input_file(self, input_file: str) -> List[Dict[str, str]]:
        """Read test cases from JSON input file."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        file_extension = input_file.lower().split('.')[-1]
        
        if file_extension == 'json':
            return self._read_json(input_file)
        else:
            raise ValueError("Input file must be JSON format")
    
    def _read_json(self, json_file: str) -> List[Dict[str, str]]:
        """Read test cases from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")
        
        test_cases = []
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each item in JSON must be an object")
            
            # Validate required keys
            required_keys = {'user_story', 'inputs', 'preconditions', 'steps', 'expected_results'}
            if not required_keys.issubset(set(item.keys())):
                missing = required_keys - set(item.keys())
                raise ValueError(f"Missing required keys in JSON object: {missing}")
            
            test_cases.append({
                'user_story': str(item['user_story']).strip(),
                'inputs': str(item['inputs']).strip(),
                'preconditions': str(item['preconditions']).strip(),
                'steps': str(item['steps']).strip(),
                'expected_results': str(item['expected_results']).strip()
            })
        
        return test_cases

    def evaluate_all_test_cases(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Evaluate all test cases independently."""
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Evaluating test case {i}/{len(test_cases)}...")
            result = self.evaluate_test_case(
                test_case['user_story'], 
                test_case
            )
            results.append(result)
            
            # Print scores in terminal for easy tracking
            print(f"  📊 SCI Score: {result['sci_percent']:.1f}%")
            print(f"  📊 RUST Score: {result['rust_overall']:.1f}/5.0")
            print(f"  📊 Combined Score: {result['combined_score']:.1f}/100")
            print(f"  🎯 Verdict: {result['verdict']}")
            print(f"  📝 User Story: {result['user_story'][:80]}{'...' if len(result['user_story']) > 80 else ''}")
            print()
        
        return results

    def export_to_json(self, results: List[Dict[str, Any]], output_prefix: str):
        """Export results to JSON format with full details, appending to existing file if it exists."""
        json_filename = f"{output_prefix}.json"
        current_timestamp = datetime.now()
        
        # Create new evaluation data with timestamp
        new_evaluation_data = {
            "evaluation_summary": {
                "total_test_cases": len(results),
                "evaluation_timestamp": str(current_timestamp),
                "evaluation_id": current_timestamp.strftime("%Y%m%d_%H%M%S")
            },
            "individual_results": results
        }
        
        # Check if file exists and load existing data
        if os.path.exists(json_filename):
            try:
                with open(json_filename, 'r', encoding='utf-8') as jsonfile:
                    existing_data = json.load(jsonfile)
                
                # If existing data is a list, convert to new format
                if isinstance(existing_data, list):
                    existing_data = {
                        "evaluation_summary": {
                            "total_evaluations": 1,
                            "first_evaluation_timestamp": str(current_timestamp)
                        },
                        "evaluations": [{
                            "evaluation_summary": {
                                "total_test_cases": len(existing_data),
                                "evaluation_timestamp": str(current_timestamp),
                                "evaluation_id": current_timestamp.strftime("%Y%m%d_%H%M%S")
                            },
                            "individual_results": existing_data
                        }]
                    }
                
                # If existing data has evaluations list, append to it
                if "evaluations" in existing_data:
                    existing_data["evaluations"].append(new_evaluation_data)
                    existing_data["evaluation_summary"]["total_evaluations"] = len(existing_data["evaluations"])
                else:
                    # Convert old format to new format
                    existing_data = {
                        "evaluation_summary": {
                            "total_evaluations": 2,
                            "first_evaluation_timestamp": existing_data.get("evaluation_summary", {}).get("evaluation_timestamp", str(current_timestamp))
                        },
                        "evaluations": [existing_data, new_evaluation_data]
                    }
                
                output_data = existing_data
                print(f"✓ Appending new evaluation to existing file: {json_filename}")
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠ Warning: Could not read existing file {json_filename}: {e}")
                print("Creating new file with current results only.")
                output_data = {
                    "evaluation_summary": {
                        "total_evaluations": 1,
                        "first_evaluation_timestamp": str(current_timestamp)
                    },
                    "evaluations": [new_evaluation_data]
                }
        else:
            # Create new file with first evaluation
            output_data = {
                "evaluation_summary": {
                    "total_evaluations": 1,
                    "first_evaluation_timestamp": str(current_timestamp)
                },
                "evaluations": [new_evaluation_data]
            }
            print(f"✓ Creating new results file: {json_filename}")
        
        # Write the combined data
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(output_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"✓ Results exported to {json_filename}")
        print(f"📊 Total evaluations in file: {output_data['evaluation_summary']['total_evaluations']}")
        return json_filename

def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test Case Quality Evaluator using Google's Gemini LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_case_evaluator.py --input test_cases.json --output evaluation_results
        """
    )
    
    parser.add_argument(
        '--input', 
        required=True,
        help='Path to JSON input file containing test cases'
    )
    
    parser.add_argument(
        '--output', 
        required=True,
        help='Prefix for output file (e.g., "results" creates results.json)'
    )
    
    args = parser.parse_args()
    
    try:
        # Check if GEMINI_API_KEY is set
        if not os.getenv('GEMINI_API_KEY'):
            print("✗ Error: GEMINI_API_KEY environment variable is not set.")
            print("Please set it in your .env file or environment variables.")
            return 1
        
        # Initialize evaluator
        print("Initializing Test Case Evaluator...")
        evaluator = TestCaseEvaluator()
        
        # Test Gemini connection
        print("Testing Gemini connection...")
        if not evaluator.test_connection():
            print("✗ Cannot proceed without a valid Gemini connection.")
            return 1
        
        # Read input file
        print(f"Reading input file: {args.input}")
        test_cases = evaluator.read_input_file(args.input)
        print(f"✓ Loaded {len(test_cases)} test cases from {args.input}")
        
        # Evaluate all test cases
        print("\nStarting evaluation...")
        results = evaluator.evaluate_all_test_cases(test_cases)
        print(f"✓ Completed evaluation of {len(results)} test cases")
        
        # Export results
        print("\nExporting results...")
        json_file = evaluator.export_to_json(results, args.output)
        
        print(f"\n🎉 Evaluation complete!")
        print(f"📊 Results saved to: {json_file}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"✗ File error: {e}")
        return 1
    except ValueError as e:
        print(f"✗ Validation error: {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
