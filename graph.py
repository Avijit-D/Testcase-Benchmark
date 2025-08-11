import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ====== CONFIG ======
JSON_FILE = "project1/results.json"  # Path to your results file
ID_FIELD = "user_story"  # Key used to match same test cases across runs
# ====================

# Load JSON
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle both old and new format
if "evaluations" in data:
    # New format with evaluations list
    evaluations = data["evaluations"]
else:
    # Old format where data is directly the evaluations list
    evaluations = data

# Organize results by test case ID across all runs
testcase_history = defaultdict(list)

for run_index, run in enumerate(evaluations, start=1):
    for tc in run["individual_results"]:
        testcase_history[tc[ID_FIELD]].append({
            "iteration": run_index,
            "sci_percent": tc.get("sci_percent", None),
            "rust_overall": tc.get("rust_overall", None),
            "combined_score": tc.get("combined_score", None),
            "sci_scores": tc.get("sci_scores", {}),
            "rust_scores": tc.get("rust_scores", {})
        })

# Function to print summary statistics
def print_summary():
    print(f"\n📊 EVALUATION SUMMARY")
    print(f"=" * 50)
    print(f"Total evaluations: {len(evaluations)}")
    print(f"Total test cases: {len(testcase_history)}")
    
    # Calculate averages across all evaluations
    all_sci_scores = []
    all_rust_scores = []
    all_combined_scores = []
    
    for tc_id, history in testcase_history.items():
        for h in history:
            all_sci_scores.append(h["sci_percent"])
            all_rust_scores.append(h["rust_overall"])
            all_combined_scores.append(h["combined_score"])
    
    if all_sci_scores:
        print(f"Average SCI Score: {sum(all_sci_scores)/len(all_sci_scores):.1f}%")
        print(f"Average RUST Score: {sum(all_rust_scores)/len(all_rust_scores):.1f}/5.0")
        print(f"Average Combined Score: {sum(all_combined_scores)/len(all_combined_scores):.1f}/100")
        
        # Count verdicts
        verdicts = {}
        for score in all_combined_scores:
            if score >= 85:
                verdict = "High Quality"
            elif score >= 70:
                verdict = "Adequate"
            else:
                verdict = "Needs Improvement"
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        
        print(f"\nVerdict Distribution:")
        for verdict, count in verdicts.items():
            percentage = (count / len(all_combined_scores)) * 100
            print(f"  {verdict}: {count} ({percentage:.1f}%)")
    
    print(f"\n📈 Generating comparison plots...")

# Function to create comparison plots
def create_comparison_plots():
    # Get all unique test case IDs
    test_case_ids = list(testcase_history.keys())
    
    # Create a mapping for shorter IDs
    id_mapping = {}
    for i, tc_id in enumerate(test_case_ids):
        # Create a shorter ID (TC_1, TC_2, etc.)
        short_id = f"TC_{i+1}"
        id_mapping[tc_id] = short_id
    
    # Prepare data for plotting
    max_iterations = max(len(history) for history in testcase_history.values())
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    
    # Colors for different iterations
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot for each iteration
    for iteration in range(1, max_iterations + 1):
        sci_scores = []
        rust_scores = []
        combined_scores = []
        
        for tc_id in test_case_ids:
            # Find the score for this iteration
            score_data = None
            for h in testcase_history[tc_id]:
                if h["iteration"] == iteration:
                    score_data = h
                    break
            
            if score_data:
                sci_scores.append(score_data["sci_percent"])
                rust_scores.append(score_data["rust_overall"])
                combined_scores.append(score_data["combined_score"])
            else:
                # If no data for this iteration, use None
                sci_scores.append(None)
                rust_scores.append(None)
                combined_scores.append(None)
        
        # Plot SCI scores
        valid_indices = [i for i, score in enumerate(sci_scores) if score is not None]
        valid_scores = [score for score in sci_scores if score is not None]
        if valid_scores:
            ax1.plot(valid_indices, valid_scores, marker='o', linewidth=2, 
                    label=f'Iteration {iteration}', color=colors[(iteration-1) % len(colors)])
        
        # Plot RUST scores
        valid_indices = [i for i, score in enumerate(rust_scores) if score is not None]
        valid_scores = [score for score in rust_scores if score is not None]
        if valid_scores:
            ax2.plot(valid_indices, valid_scores, marker='s', linewidth=2, 
                    label=f'Iteration {iteration}', color=colors[(iteration-1) % len(colors)])
        
        # Plot Combined scores
        valid_indices = [i for i, score in enumerate(combined_scores) if score is not None]
        valid_scores = [score for score in combined_scores if score is not None]
        if valid_scores:
            ax3.plot(valid_indices, valid_scores, marker='^', linewidth=2, 
                    label=f'Iteration {iteration}', color=colors[(iteration-1) % len(colors)])
    
    # Configure SCI plot
    ax1.set_title('SCI Score Comparison Across Iterations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('SCI Score (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Configure RUST plot
    ax2.set_title('RUST Score Comparison Across Iterations', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RUST Score (1-5)', fontsize=12)
    ax2.set_ylim(0, 5)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Configure Combined plot
    ax3.set_title('Combined Score Comparison Across Iterations', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Test Case ID', fontsize=12)
    ax3.set_ylabel('Combined Score (0-100)', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Set x-axis labels for all plots
    x_positions = list(range(len(test_case_ids)))
    x_labels = [id_mapping[tc_id] for tc_id in test_case_ids]
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('test_case_comparison2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plot: test_case_comparison.png")
    
    # Create a legend file with test case mappings
    with open('test_case_mapping.txt', 'w', encoding='utf-8') as f:
        f.write("Test Case ID Mapping:\n")
        f.write("=" * 50 + "\n")
        for tc_id, short_id in id_mapping.items():
            f.write(f"{short_id}: {tc_id[:100]}...\n")
    print(f"✓ Saved test case mapping: test_case_mapping.txt")

# Function to create detailed comparison table
def create_comparison_table():
    # Create a DataFrame for easy comparison
    comparison_data = []
    
    for tc_id, history in testcase_history.items():
        row = {"Test Case": tc_id[:50] + "..." if len(tc_id) > 50 else tc_id}
        
        for h in history:
            iteration = h["iteration"]
            row[f"Iteration_{iteration}_SCI"] = f"{h['sci_percent']:.1f}%"
            row[f"Iteration_{iteration}_RUST"] = f"{h['rust_overall']:.1f}"
            row[f"Iteration_{iteration}_Combined"] = f"{h['combined_score']:.1f}"
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df.to_csv('test_case_comparison.csv', index=False)
    print(f"✓ Saved comparison table: test_case_comparison.csv")

# Function to create variance analysis plots
def create_variance_analysis():
    print(f"\n📊 Analyzing metric variance across iterations...")
    
    # Collect all SCI and RUST scores across all test cases and iterations
    sci_metrics = {
        'traceability': [],
        'coverage_breadth': [],
        'coverage_depth': [],
        'clarity_precision': [],
        'completeness': [],
        'variety_types': [],
        'consistency': []
    }
    
    rust_metrics = {
        'readability_overall': [],
        'grammar_clarity': [],
        'accuracy_to_intent': [],
        'clarity_comprehensibility': [],
        'specific_information': [],
        'acceptance_criteria_defined': [],
        'coverage_of_case_types': [],
        'coverage_technical_reqs': [],
        'compliance_best_practices': [],
        'translating_needs': [],
        'ambiguity_in_technical_details': []
    }
    
    # Collect data for each metric
    for tc_id, history in testcase_history.items():
        for h in history:
            # Add SCI metrics
            for metric, value in h["sci_scores"].items():
                if metric in sci_metrics:
                    sci_metrics[metric].append(value)
            
            # Add RUST metrics
            for metric, value in h["rust_scores"].items():
                if metric in rust_metrics:
                    rust_metrics[metric].append(value)
    
    # Calculate statistics for each metric
    import numpy as np
    
    sci_stats = {}
    for metric, values in sci_metrics.items():
        if values:
            sci_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'variance': np.var(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    rust_stats = {}
    for metric, values in rust_metrics.items():
        if values:
            rust_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'variance': np.var(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    # Create variance analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # SCI Standard Deviation
    sci_metrics_list = list(sci_stats.keys())
    sci_stds = [sci_stats[metric]['std'] for metric in sci_metrics_list]
    sci_means = [sci_stats[metric]['mean'] for metric in sci_metrics_list]
    
    bars1 = ax1.bar(sci_metrics_list, sci_stds, color='skyblue', alpha=0.7)
    ax1.set_title('SCI Metrics - Standard Deviation', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Standard Deviation', fontsize=12)
    ax1.set_xticklabels(sci_metrics_list, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, std_val in zip(bars1, sci_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{std_val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # RUST Standard Deviation
    rust_metrics_list = list(rust_stats.keys())
    rust_stds = [rust_stats[metric]['std'] for metric in rust_metrics_list]
    rust_means = [rust_stats[metric]['mean'] for metric in rust_metrics_list]
    
    bars2 = ax2.bar(rust_metrics_list, rust_stds, color='lightcoral', alpha=0.7)
    ax2.set_title('RUST Metrics - Standard Deviation', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Standard Deviation', fontsize=12)
    ax2.set_xticklabels(rust_metrics_list, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, std_val in zip(bars2, rust_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{std_val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # SCI Coefficient of Variation (CV = std/mean)
    sci_cv = [sci_stats[metric]['std'] / sci_stats[metric]['mean'] if sci_stats[metric]['mean'] > 0 else 0 
              for metric in sci_metrics_list]
    
    bars3 = ax3.bar(sci_metrics_list, sci_cv, color='lightgreen', alpha=0.7)
    ax3.set_title('SCI Metrics - Coefficient of Variation (CV)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Coefficient of Variation (std/mean)', fontsize=12)
    ax3.set_xticklabels(sci_metrics_list, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cv_val in zip(bars3, sci_cv):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cv_val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # RUST Coefficient of Variation
    rust_cv = [rust_stats[metric]['std'] / rust_stats[metric]['mean'] if rust_stats[metric]['mean'] > 0 else 0 
               for metric in rust_metrics_list]
    
    bars4 = ax4.bar(rust_metrics_list, rust_cv, color='gold', alpha=0.7)
    ax4.set_title('RUST Metrics - Coefficient of Variation (CV)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Coefficient of Variation (std/mean)', fontsize=12)
    ax4.set_xticklabels(rust_metrics_list, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cv_val in zip(bars4, rust_cv):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cv_val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('metric_variance_analysis2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved variance analysis plot: metric_variance_analysis.png")
    
    # Create detailed variance report
    with open('metric_variance_report.txt', 'w', encoding='utf-8') as f:
        f.write("METRIC VARIANCE ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("SCI METRICS ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for metric in sci_metrics_list:
            stats = sci_stats[metric]
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Std Dev: {stats['std']:.2f}\n")
            f.write(f"  Variance: {stats['variance']:.2f}\n")
            f.write(f"  CV: {stats['std']/stats['mean']:.2f}\n")
            f.write(f"  Range: {stats['min']:.0f} - {stats['max']:.0f}\n")
            f.write(f"  Count: {stats['count']}\n\n")
        
        f.write("RUST METRICS ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for metric in rust_metrics_list:
            stats = rust_stats[metric]
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Std Dev: {stats['std']:.2f}\n")
            f.write(f"  Variance: {stats['variance']:.2f}\n")
            f.write(f"  CV: {stats['std']/stats['mean']:.2f}\n")
            f.write(f"  Range: {stats['min']:.0f} - {stats['max']:.0f}\n")
            f.write(f"  Count: {stats['count']}\n\n")
        
        # Find most and least variable metrics
        f.write("MOST VARIABLE METRICS:\n")
        f.write("-" * 30 + "\n")
        
        all_metrics = []
        for metric in sci_metrics_list:
            all_metrics.append(('SCI', metric, sci_stats[metric]['std'], sci_stats[metric]['std']/sci_stats[metric]['mean']))
        for metric in rust_metrics_list:
            all_metrics.append(('RUST', metric, rust_stats[metric]['std'], rust_stats[metric]['std']/rust_stats[metric]['mean']))
        
        # Sort by standard deviation
        all_metrics.sort(key=lambda x: x[2], reverse=True)
        
        f.write("By Standard Deviation:\n")
        for i, (category, metric, std, cv) in enumerate(all_metrics[:5]):
            f.write(f"  {i+1}. {category} - {metric}: {std:.2f}\n")
        
        f.write("\nBy Coefficient of Variation:\n")
        all_metrics.sort(key=lambda x: x[3], reverse=True)
        for i, (category, metric, std, cv) in enumerate(all_metrics[:5]):
            f.write(f"  {i+1}. {category} - {metric}: {cv:.2f}\n")
    
    print(f"✓ Saved variance report: metric_variance_report.txt")

# Run analysis
print_summary()
create_comparison_plots()
create_comparison_table()
create_variance_analysis()
print(f"\n🎉 Analysis complete! Check the generated files:")
print(f"  📊 test_case_comparison.png - Visual comparison of all test cases")
print(f"  📋 test_case_comparison.csv - Detailed comparison table")
print(f"  📝 test_case_mapping.txt - Mapping of short IDs to full test case descriptions")
print(f"  📊 metric_variance_analysis.png - Variance analysis of individual metrics")
print(f"  📋 metric_variance_report.txt - Detailed variance statistics")
