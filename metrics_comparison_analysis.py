import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import re

class MetricsComparisonAnalyzer:
    def __init__(self):
        self.previous_data = None
        self.mild_data = None
        self.severe_data = None
        self.comparison_df = None
        
    def load_data(self):
        """Load all three datasets"""
        try:
            # Load previous project data from testcase db.txt
            with open('testcase db.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Extract the JSON data starting from "previous project data benchmark result"
            start_idx = content.find('previous project data benchmark result')
            if start_idx != -1:
                json_start = content.find('{', start_idx)
                if json_start != -1:
                    json_content = content[json_start:]
                    # Find the end of the JSON (before "degraded_mild dataset v1")
                    end_marker = json_content.find('degraded_mild dataset v1')
                    if end_marker != -1:
                        json_content = json_content[:end_marker].strip()
                        # Remove trailing commas and clean up
                        json_content = re.sub(r',\s*$', '', json_content)
                        self.previous_data = json.loads(json_content)
                    else:
                        # Try to parse the entire remaining content
                        self.previous_data = json.loads(json_content)
                else:
                    print("Could not find JSON start in previous project data")
            else:
                print("Could not find 'previous project data benchmark result' marker")
                
        except Exception as e:
            print(f"Error loading previous project data: {e}")
            
        # Load mild and severe datasets
        try:
            with open('output_mild.json.json', 'r') as f:
                self.mild_data = json.load(f)
        except Exception as e:
            print(f"Error loading mild data: {e}")
            
        try:
            with open('output_severe.json.json', 'r') as f:
                self.severe_data = json.load(f)
        except Exception as e:
            print(f"Error loading severe data: {e}")
    
    def extract_user_stories(self, data, dataset_name: str) -> List[Dict]:
        """Extract user stories and metrics from dataset"""
        stories = []
        
        if not data or 'evaluations' not in data:
            return stories
            
        for eval_item in data['evaluations']:
            if 'individual_results' in eval_item:
                for result in eval_item['individual_results']:
                    story = {
                        'dataset': dataset_name,
                        'user_story': result.get('user_story', ''),
                        'sci_percent': result.get('sci_percent', 0),
                        'rust_overall': result.get('rust_overall', 0),
                        'combined_score': result.get('combined_score', 0),
                        'verdict': result.get('verdict', ''),
                        'sci_scores': result.get('sci_scores', {}),
                        'rust_scores': result.get('rust_scores', {})
                    }
                    stories.append(story)
        
        return stories
    
    def map_user_stories(self) -> pd.DataFrame:
        """Map user stories between datasets and create comparison dataframe"""
        previous_stories = self.extract_user_stories(self.previous_data, 'Previous Project')
        mild_stories = self.extract_user_stories(self.mild_data, 'Mild Degradation')
        severe_stories = self.extract_user_stories(self.severe_data, 'Severe Degradation')
        
        # Create mapping based on user story similarity
        mapped_data = []
        
        for prev_story in previous_stories:
            prev_text = prev_story['user_story']
            
            # Find matching stories in mild and severe datasets
            mild_match = self.find_best_match(prev_text, mild_stories)
            severe_match = self.find_best_match(prev_text, severe_stories)
            
            if mild_match and severe_match:
                mapped_data.append({
                    'user_story': prev_text,
                    'previous_sci': prev_story['sci_percent'],
                    'previous_rust': prev_story['rust_overall'],
                    'previous_combined': prev_story['combined_score'],
                    'previous_verdict': prev_story['verdict'],
                    'mild_sci': mild_match['sci_percent'],
                    'mild_rust': mild_match['rust_overall'],
                    'mild_combined': mild_match['combined_score'],
                    'mild_verdict': mild_match['verdict'],
                    'severe_sci': severe_match['sci_percent'],
                    'severe_rust': severe_match['rust_overall'],
                    'severe_combined': severe_match['combined_score'],
                    'severe_verdict': severe_match['verdict'],
                    'sci_drop_mild': prev_story['sci_percent'] - mild_match['sci_percent'],
                    'rust_drop_mild': prev_story['rust_overall'] - mild_match['rust_overall'],
                    'combined_drop_mild': prev_story['combined_score'] - mild_match['combined_score'],
                    'sci_drop_severe': prev_story['sci_percent'] - severe_match['sci_percent'],
                    'rust_drop_severe': prev_story['rust_overall'] - severe_match['rust_overall'],
                    'combined_drop_severe': prev_story['combined_score'] - severe_match['combined_score']
                })
        
        self.comparison_df = pd.DataFrame(mapped_data)
        return self.comparison_df
    
    def find_best_match(self, target_story: str, candidate_stories: List[Dict]) -> Dict:
        """Find the best matching user story based on text similarity"""
        if not candidate_stories:
            return None
            
        # Simple keyword matching - look for common phrases
        target_lower = target_story.lower()
        
        for candidate in candidate_stories:
            candidate_lower = candidate['user_story'].lower()
            
            # Check for exact matches first
            if target_lower == candidate_lower:
                return candidate
                
            # Check for key phrase matches
            key_phrases = [
                '12-19-2017 deletions processed',
                'redesign the Resources page',
                'report to the Agencies about user testing',
                'round 2 of DABS or FABS landing page edits',
                'round 2 of Homepage edits',
                'round 3 of the Help page edits',
                'log better',
                'FABS submission to be modified',
                'New Relic to provide useful data',
                'round 2 of the Help page edits'
            ]
            
            for phrase in key_phrases:
                if phrase.lower() in target_lower and phrase.lower() in candidate_lower:
                    return candidate
        
        # If no exact match, return the first candidate (fallback)
        return candidate_stories[0] if candidate_stories else None
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the quality comparison"""
        if self.comparison_df is None or self.comparison_df.empty:
            print("No comparison data available")
            return
            
        # Create separate figures for better readability
        
        # Figure 1: Combined Score Comparison
        self.create_combined_score_figure()
        
        # Figure 2: SCI and RUST Score Comparison
        self.create_sci_rust_figure()
        
        # Figure 3: Average Metrics Comparison
        self.create_average_metrics_figure()
        
        # Create detailed analysis report
        self.create_analysis_report()
        
        # Analyze cases where degraded mild performed better
        self.analyze_improvement_cases()
    
    def create_combined_score_figure(self):
        """Create a focused figure for combined score comparison"""
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(self.comparison_df))
        width = 0.25
        
        plt.bar(x - width, self.comparison_df['previous_combined'], width, label='Original Dataset', 
               color='#2E8B57', alpha=0.8)
        plt.bar(x, self.comparison_df['mild_combined'], width, label='Mild Degradation', 
               color='#FFA500', alpha=0.8)
        plt.bar(x + width, self.comparison_df['severe_combined'], width, label='Severe Degradation', 
               color='#DC143C', alpha=0.8)
        
        plt.xlabel('Test Cases', fontsize=12)
        plt.ylabel('Combined Score', fontsize=12)
        plt.title('Combined Score Comparison: Original vs Degraded Datasets', fontsize=14, fontweight='bold')
        plt.xticks(x, [f'TC{i+1}' for i in range(len(self.comparison_df))], rotation=45)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (prev, mild, severe) in enumerate(zip(self.comparison_df['previous_combined'], 
                                                    self.comparison_df['mild_combined'], 
                                                    self.comparison_df['severe_combined'])):
            plt.text(i - width, prev + 1, f'{prev:.1f}', ha='center', va='bottom', fontsize=9)
            plt.text(i, mild + 1, f'{mild:.1f}', ha='center', va='bottom', fontsize=9)
            plt.text(i + width, severe + 1, f'{severe:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('combined_score_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sci_rust_figure(self):
        """Create a focused figure for SCI and RUST score comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        x = np.arange(len(self.comparison_df))
        width = 0.25
        
        # SCI Score Comparison
        ax1.bar(x - width, self.comparison_df['previous_sci'], width, label='Original Dataset', 
               color='#2E8B57', alpha=0.8)
        ax1.bar(x, self.comparison_df['mild_sci'], width, label='Mild Degradation', 
               color='#FFA500', alpha=0.8)
        ax1.bar(x + width, self.comparison_df['severe_sci'], width, label='Severe Degradation', 
               color='#DC143C', alpha=0.8)
        
        ax1.set_xlabel('Test Cases', fontsize=12)
        ax1.set_ylabel('SCI Score (%)', fontsize=12)
        ax1.set_title('SCI Score Comparison', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'TC{i+1}' for i in range(len(self.comparison_df))], rotation=45)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels for SCI
        for i, (prev, mild, severe) in enumerate(zip(self.comparison_df['previous_sci'], 
                                                    self.comparison_df['mild_sci'], 
                                                    self.comparison_df['severe_sci'])):
            ax1.text(i - width, prev + 2, f'{prev:.1f}%', ha='center', va='bottom', fontsize=9)
            ax1.text(i, mild + 2, f'{mild:.1f}%', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width, severe + 2, f'{severe:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # RUST Score Comparison
        ax2.bar(x - width, self.comparison_df['previous_rust'], width, label='Original Dataset', 
               color='#2E8B57', alpha=0.8)
        ax2.bar(x, self.comparison_df['mild_rust'], width, label='Mild Degradation', 
               color='#FFA500', alpha=0.8)
        ax2.bar(x + width, self.comparison_df['severe_rust'], width, label='Severe Degradation', 
               color='#DC143C', alpha=0.8)
        
        ax2.set_xlabel('Test Cases', fontsize=12)
        ax2.set_ylabel('RUST Score', fontsize=12)
        ax2.set_title('RUST Score Comparison', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'TC{i+1}' for i in range(len(self.comparison_df))], rotation=45)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels for RUST
        for i, (prev, mild, severe) in enumerate(zip(self.comparison_df['previous_rust'], 
                                                    self.comparison_df['mild_rust'], 
                                                    self.comparison_df['severe_rust'])):
            ax2.text(i - width, prev + 0.1, f'{prev:.1f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i, mild + 0.1, f'{mild:.1f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i + width, severe + 0.1, f'{severe:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('sci_rust_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_average_metrics_figure(self):
        """Create a focused figure for average metrics comparison"""
        plt.figure(figsize=(10, 8))
        
        metrics = ['SCI', 'RUST', 'Combined']
        x = np.arange(len(metrics))
        width = 0.25
        
        # Calculate average scores for each dataset
        prev_avg = [
            self.comparison_df['previous_sci'].mean(),
            self.comparison_df['previous_rust'].mean() * 20,  # Scale RUST to 0-100
            self.comparison_df['previous_combined'].mean()
        ]
        mild_avg = [
            self.comparison_df['mild_sci'].mean(),
            self.comparison_df['mild_rust'].mean() * 20,
            self.comparison_df['mild_combined'].mean()
        ]
        severe_avg = [
            self.comparison_df['severe_sci'].mean(),
            self.comparison_df['severe_rust'].mean() * 20,
            self.comparison_df['severe_combined'].mean()
        ]
        
        plt.bar(x - width, prev_avg, width, label='Original Dataset', color='#2E8B57', alpha=0.8)
        plt.bar(x, mild_avg, width, label='Mild Degradation', color='#FFA500', alpha=0.8)
        plt.bar(x + width, severe_avg, width, label='Severe Degradation', color='#DC143C', alpha=0.8)
        
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.title('Average Metrics Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, metrics)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (prev, mild, severe) in enumerate(zip(prev_avg, mild_avg, severe_avg)):
            plt.text(i - width, prev + 2, f'{prev:.1f}', ha='center', va='bottom', fontsize=10)
            plt.text(i, mild + 2, f'{mild:.1f}', ha='center', va='bottom', fontsize=10)
            plt.text(i + width, severe + 2, f'{severe:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('average_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_improvement_cases(self):
        """Analyze cases where degraded mild performed better than original dataset"""
        if self.comparison_df is None or self.comparison_df.empty:
            return
            
        print("\n" + "=" * 80)
        print("ANALYSIS OF CASES WHERE MILD DEGRADATION PERFORMED BETTER")
        print("=" * 80)
        
        # Find cases where mild degradation performed better
        improvement_cases = []
        
        for i, row in self.comparison_df.iterrows():
            mild_better_sci = row['mild_sci'] > row['previous_sci']
            mild_better_rust = row['mild_rust'] > row['previous_rust']
            mild_better_combined = row['mild_combined'] > row['previous_combined']
            
            if mild_better_sci or mild_better_rust or mild_better_combined:
                improvement_cases.append({
                    'test_case': i + 1,
                    'user_story': row['user_story'][:80] + "...",
                    'sci_improvement': mild_better_sci,
                    'rust_improvement': mild_better_rust,
                    'combined_improvement': mild_better_combined,
                    'sci_change': row['mild_sci'] - row['previous_sci'],
                    'rust_change': row['mild_rust'] - row['previous_rust'],
                    'combined_change': row['mild_combined'] - row['previous_combined']
                })
        
        if not improvement_cases:
            print("No cases found where mild degradation performed better than the original dataset.")
            return
        
        print(f"Found {len(improvement_cases)} test case(s) where mild degradation showed improvement:")
        print()
        
        for case in improvement_cases:
            print(f"Test Case {case['test_case']}:")
            print(f"  User Story: {case['user_story']}")
            print(f"  SCI Improvement: {'Yes' if case['sci_improvement'] else 'No'} (Change: {case['sci_change']:+.1f}%)")
            print(f"  RUST Improvement: {'Yes' if case['rust_improvement'] else 'No'} (Change: {case['rust_change']:+.1f})")
            print(f"  Combined Improvement: {'Yes' if case['combined_improvement'] else 'No'} (Change: {case['combined_change']:+.1f})")
            print()
        
        # Analyze which specific metrics were most affected
        print("DETAILED METRIC ANALYSIS FOR IMPROVEMENT CASES:")
        print("-" * 60)
        
        for case in improvement_cases:
            tc_idx = case['test_case'] - 1
            prev_row = self.comparison_df.iloc[tc_idx]
            
            print(f"\nTest Case {case['test_case']} - Detailed Metric Analysis:")
            
            # Analyze SCI metrics
            if case['sci_improvement']:
                print("  SCI Metrics Analysis:")
                sci_metrics = ['traceability', 'coverage_breadth', 'coverage_depth', 
                              'clarity_precision', 'completeness', 'variety_types', 'consistency']
                
                # Get the original SCI scores from the data
                if 'sci_scores' in self.previous_data['evaluations'][0]['individual_results'][tc_idx]:
                    prev_sci_scores = self.previous_data['evaluations'][0]['individual_results'][tc_idx]['sci_scores']
                    
                    # Get mild SCI scores
                    if 'sci_scores' in self.mild_data['evaluations'][0]['individual_results'][tc_idx]:
                        mild_sci_scores = self.mild_data['evaluations'][0]['individual_results'][tc_idx]['sci_scores']
                        
                        for metric in sci_metrics:
                            if metric in prev_sci_scores and metric in mild_sci_scores:
                                change = mild_sci_scores[metric] - prev_sci_scores[metric]
                                if change > 0:
                                    print(f"    {metric.capitalize()}: +{change} (Improved)")
                                elif change < 0:
                                    print(f"    {metric.capitalize()}: {change} (Declined)")
                                else:
                                    print(f"    {metric.capitalize()}: No change")
            
            # Analyze RUST metrics
            if case['rust_improvement']:
                print("  RUST Metrics Analysis:")
                rust_metrics = ['readability_overall', 'grammar_clarity', 'accuracy_to_intent',
                               'clarity_comprehensibility', 'specific_information', 'acceptance_criteria_defined',
                               'coverage_of_case_types', 'coverage_technical_reqs', 'compliance_best_practices',
                               'translating_needs', 'ambiguity_in_technical_details']
                
                # Get the original RUST scores from the data
                if 'rust_scores' in self.previous_data['evaluations'][0]['individual_results'][tc_idx]:
                    prev_rust_scores = self.previous_data['evaluations'][0]['individual_results'][tc_idx]['rust_scores']
                    
                    # Get mild RUST scores
                    if 'rust_scores' in self.mild_data['evaluations'][0]['individual_results'][tc_idx]:
                        mild_rust_scores = self.mild_data['evaluations'][0]['individual_results'][tc_idx]['rust_scores']
                        
                        for metric in rust_metrics:
                            if metric in prev_rust_scores and metric in mild_rust_scores:
                                change = mild_rust_scores[metric] - prev_rust_scores[metric]
                                if change > 0:
                                    print(f"    {metric.replace('_', ' ').title()}: +{change} (Improved)")
                                elif change < 0:
                                    print(f"    {metric.replace('_', ' ').title()}: {change} (Declined)")
                                else:
                                    print(f"    {metric.replace('_', ' ').title()}: No change")
        
        print("\n" + "=" * 80)
        
        # Save improvement analysis to file
        self.save_improvement_analysis(improvement_cases)
    
    def save_improvement_analysis(self, improvement_cases):
        """Save the improvement analysis to a file"""
        report = []
        report.append("=" * 80)
        report.append("IMPROVEMENT CASES ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        report.append(f"Found {len(improvement_cases)} test case(s) where mild degradation showed improvement:")
        report.append("")
        
        for case in improvement_cases:
            report.append(f"Test Case {case['test_case']}:")
            report.append(f"  User Story: {case['user_story']}")
            report.append(f"  SCI Improvement: {'Yes' if case['sci_improvement'] else 'No'} (Change: {case['sci_change']:+.1f}%)")
            report.append(f"  RUST Improvement: {'Yes' if case['rust_improvement'] else 'No'} (Change: {case['rust_change']:+.1f})")
            report.append(f"  Combined Improvement: {'Yes' if case['combined_improvement'] else 'No'} (Change: {case['combined_change']:+.1f})")
            report.append("")
        
        with open('improvement_cases_analysis.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Improvement analysis saved to 'improvement_cases_analysis.txt'")
    
    def create_analysis_report(self):
        """Create a detailed analysis report"""
        if self.comparison_df is None or self.comparison_df.empty:
            return
            
        report = []
        report.append("=" * 80)
        report.append("QUALITY METRICS COMPARISON ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS:")
        report.append("-" * 40)
        report.append(f"Total Test Cases Analyzed: {len(self.comparison_df)}")
        report.append("")
        
        # Average scores
        report.append("AVERAGE SCORES:")
        report.append("-" * 40)
        report.append(f"Original Dataset - SCI: {self.comparison_df['previous_sci'].mean():.2f}%")
        report.append(f"Original Dataset - RUST: {self.comparison_df['previous_rust'].mean():.2f}/5.0")
        report.append(f"Original Dataset - Combined: {self.comparison_df['previous_combined'].mean():.2f}/100")
        report.append("")
        report.append(f"Mild Degradation - SCI: {self.comparison_df['mild_sci'].mean():.2f}%")
        report.append(f"Mild Degradation - RUST: {self.comparison_df['mild_rust'].mean():.2f}/5.0")
        report.append(f"Mild Degradation - Combined: {self.comparison_df['mild_combined'].mean():.2f}/100")
        report.append("")
        report.append(f"Severe Degradation - SCI: {self.comparison_df['severe_sci'].mean():.2f}%")
        report.append(f"Severe Degradation - RUST: {self.comparison_df['severe_rust'].mean():.2f}/5.0")
        report.append(f"Severe Degradation - Combined: {self.comparison_df['severe_combined'].mean():.2f}/100")
        report.append("")
        
        # Quality degradation analysis
        report.append("QUALITY DEGRADATION ANALYSIS:")
        report.append("-" * 40)
        
        mild_sci_drop = self.comparison_df['sci_drop_mild'].mean()
        mild_rust_drop = self.comparison_df['rust_drop_mild'].mean()
        mild_combined_drop = self.comparison_df['combined_drop_mild'].mean()
        
        severe_sci_drop = self.comparison_df['sci_drop_severe'].mean()
        severe_rust_drop = self.comparison_df['rust_drop_severe'].mean()
        severe_combined_drop = self.comparison_df['combined_drop_severe'].mean()
        
        report.append(f"Mild Degradation Average Drops:")
        report.append(f"  - SCI Score: {mild_sci_drop:.2f}%")
        report.append(f"  - RUST Score: {mild_rust_drop:.2f}/5.0")
        report.append(f"  - Combined Score: {mild_combined_drop:.2f}/100")
        report.append("")
        
        report.append(f"Severe Degradation Average Drops:")
        report.append(f"  - SCI Score: {severe_sci_drop:.2f}%")
        report.append(f"  - RUST Score: {severe_rust_drop:.2f}/5.0")
        report.append(f"  - Combined Score: {severe_combined_drop:.2f}/100")
        report.append("")
        
        # Individual test case analysis
        report.append("INDIVIDUAL TEST CASE ANALYSIS:")
        report.append("-" * 40)
        
        for i, row in self.comparison_df.iterrows():
            report.append(f"Test Case {i+1}:")
            report.append(f"  User Story: {row['user_story'][:80]}...")
            report.append(f"  Original: SCI={row['previous_sci']:.1f}%, RUST={row['previous_rust']:.1f}, Combined={row['previous_combined']:.1f}")
            report.append(f"  Mild: SCI={row['mild_sci']:.1f}%, RUST={row['mild_rust']:.1f}, Combined={row['mild_combined']:.1f}")
            report.append(f"  Severe: SCI={row['severe_sci']:.1f}%, RUST={row['severe_rust']:.1f}, Combined={row['severe_combined']:.1f}")
            report.append(f"  Quality Drop (Mild): {row['combined_drop_mild']:.1f}")
            report.append(f"  Quality Drop (Severe): {row['combined_drop_severe']:.1f}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        if mild_combined_drop > 10:
            report.append("WARNING: Mild degradation shows significant quality drop (>10 points)")
        if severe_combined_drop > 20:
            report.append("CRITICAL: Severe degradation shows critical quality drop (>20 points)")
        
        if mild_sci_drop > 15:
            report.append("INFO: SCI scores show notable degradation in mild dataset")
        if severe_sci_drop > 30:
            report.append("INFO: SCI scores show severe degradation in severe dataset")
            
        if mild_rust_drop > 1:
            report.append("INFO: RUST scores indicate readability/understanding issues in mild dataset")
        if severe_rust_drop > 2:
            report.append("INFO: RUST scores indicate critical readability/understanding issues in severe dataset")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        with open('quality_metrics_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Analysis report saved to 'quality_metrics_analysis_report.txt'")
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("QUALITY METRICS COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Original Dataset Average Combined Score: {self.comparison_df['previous_combined'].mean():.1f}/100")
        print(f"Mild Degradation Average Combined Score: {self.comparison_df['mild_combined'].mean():.1f}/100")
        print(f"Severe Degradation Average Combined Score: {self.comparison_df['severe_combined'].mean():.1f}/100")
        print(f"Average Quality Drop (Mild): {mild_combined_drop:.1f} points")
        print(f"Average Quality Drop (Severe): {severe_combined_drop:.1f} points")
        print("=" * 80)

def main():
    """Main function to run the analysis"""
    print("Initializing Quality Metrics Comparison Analyzer...")
    
    analyzer = MetricsComparisonAnalyzer()
    
    print("Loading datasets...")
    analyzer.load_data()
    
    print("Mapping user stories between datasets...")
    comparison_df = analyzer.map_user_stories()
    
    if comparison_df is not None and not comparison_df.empty:
        print(f"Successfully mapped {len(comparison_df)} test cases")
        
        print("Creating visualizations...")
        analyzer.create_visualizations()
        
        print("Analysis complete! Check the generated files:")
        print("- quality_metrics_comparison.png (visualizations)")
        print("- quality_metrics_analysis_report.txt (detailed report)")
    else:
        print("No comparison data could be generated. Please check the input files.")

if __name__ == "__main__":
    main()
