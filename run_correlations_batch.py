#!/usr/bin/env python3
"""
Script to run compute_correlations_save_datasets.py for multiple seeds and generate a summary.

Usage:
python run_correlations_batch.py --project <project> --dataset_name <dataset_name> 
                                --model <model> --job_type <job_type> 
                                [--gt_job_type <gt_job_type>]
"""

import argparse
import subprocess
import os
import sys
import numpy as np
import json
from datetime import datetime


def run_single_seed(project, dataset_name, model, job_type, gt_job_type, seed):
    """Run compute_correlations_save_datasets.py for a single seed and capture output."""
    
    cmd = [
        'python', 'compute_correlations_save_datasets.py',
        '--project', project,
        '--dataset_name', dataset_name,
        '--model', model,
        '--job_type', job_type,
        '--gt_job_type', gt_job_type,
        '--seed', str(seed)
    ]
    
    print(f"Running seed {seed}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output to extract correlation values
        output_lines = result.stdout.split('\n')
        pearson_corr = None
        spearman_corr = None
        lower_outliers = None
        upper_outliers = None
        
        for line in output_lines:
            if "Pearson Correlation:" in line:
                # Extract value from "Pearson Correlation: 0.1234 (p-value: 1.23e-04)"
                parts = line.split(":")
                if len(parts) > 1:
                    value_part = parts[1].split("(")[0].strip()
                    pearson_corr = float(value_part)
            elif "Spearman Correlation:" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    value_part = parts[1].split("(")[0].strip()
                    spearman_corr = float(value_part)
            elif "Lower Outliers Fraction:" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    lower_outliers = float(parts[1].strip())
            elif "Upper Outliers Fraction:" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    upper_outliers = float(parts[1].strip())
        
        return {
            'seed': seed,
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'lower_outliers_fraction': lower_outliers,
            'upper_outliers_fraction': upper_outliers,
            'success': True,
            'error': None
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error running seed {seed}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {
            'seed': seed,
            'pearson_correlation': None,
            'spearman_correlation': None,
            'lower_outliers_fraction': None,
            'upper_outliers_fraction': None,
            'success': False,
            'error': str(e)
        }


def generate_summary(results, model_save_path):
    """Generate summary statistics and save to file."""
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) == 0:
        print("No successful runs to summarize!")
        return
    
    # Extract values for statistics
    pearson_values = [r['pearson_correlation'] for r in successful_results if r['pearson_correlation'] is not None]
    spearman_values = [r['spearman_correlation'] for r in successful_results if r['spearman_correlation'] is not None]
    lower_outliers_values = [r['lower_outliers_fraction'] for r in successful_results if r['lower_outliers_fraction'] is not None]
    upper_outliers_values = [r['upper_outliers_fraction'] for r in successful_results if r['upper_outliers_fraction'] is not None]
    
    # Calculate statistics
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_seeds': len(results),
        'successful_seeds': len(successful_results),
        'failed_seeds': len(results) - len(successful_results),
        'individual_results': results,
        'statistics': {}
    }
    
    if pearson_values:
        summary['statistics']['pearson_correlation'] = {
            'mean': np.mean(pearson_values),
            'std': np.std(pearson_values, ddof=1) if len(pearson_values) > 1 else 0.0,
            'min': np.min(pearson_values),
            'max': np.max(pearson_values),
            'count': len(pearson_values)
        }
    
    if spearman_values:
        summary['statistics']['spearman_correlation'] = {
            'mean': np.mean(spearman_values),
            'std': np.std(spearman_values, ddof=1) if len(spearman_values) > 1 else 0.0,
            'min': np.min(spearman_values),
            'max': np.max(spearman_values),
            'count': len(spearman_values)
        }
    
    if lower_outliers_values:
        summary['statistics']['lower_outliers_fraction'] = {
            'mean': np.mean(lower_outliers_values),
            'std': np.std(lower_outliers_values, ddof=1) if len(lower_outliers_values) > 1 else 0.0,
            'min': np.min(lower_outliers_values),
            'max': np.max(lower_outliers_values),
            'count': len(lower_outliers_values)
        }
    
    if upper_outliers_values:
        summary['statistics']['upper_outliers_fraction'] = {
            'mean': np.mean(upper_outliers_values),
            'std': np.std(upper_outliers_values, ddof=1) if len(upper_outliers_values) > 1 else 0.0,
            'min': np.min(upper_outliers_values),
            'max': np.max(upper_outliers_values),
            'count': len(upper_outliers_values)
        }
    
    # Save summary to JSON file
    os.makedirs(model_save_path, exist_ok=True)
    summary_file = os.path.join(model_save_path, 'correlation_summary.json')
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total seeds processed: {summary['total_seeds']}")
    print(f"Successful: {summary['successful_seeds']}")
    print(f"Failed: {summary['failed_seeds']}")
    print()
    
    for metric, stats in summary['statistics'].items():
        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Count: {stats['count']}")
        print()
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Run correlations for multiple seeds and generate summary')
    parser.add_argument('--project', required=True, help='Project name')
    parser.add_argument('--dataset_name', required=True, help='Dataset name')
    parser.add_argument('--model', required=True, choices=['PT', 'PT+ADT', 'SARA'], help='Model type')
    parser.add_argument('--job_type', required=True, help='Job type for the specified model')
    parser.add_argument('--gt_job_type', default='NewNorm', help='Job type for ground truth (default: NewNorm)')
    
    args = parser.parse_args()
    
    # Define seeds to process
    seeds = [231, 107, 93, 1, 123, 827, 67, 42]
    
    print(f"Processing correlations for {len(seeds)} seeds...")
    print(f"Project: {args.project}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model}")
    print(f"Job Type: {args.job_type}")
    print(f"GT Job Type: {args.gt_job_type}")
    print()
    
    # Run for each seed
    results = []
    for seed in seeds:
        result = run_single_seed(args.project, args.dataset_name, args.model, 
                                args.job_type, args.gt_job_type, seed)
        results.append(result)
    
    # Determine model save path (same logic as in compute_correlations_save_datasets.py)
    if args.model in ['PT', 'PT+ADT']:
        model_name = 'PreferenceTransformer'
    elif args.model == 'SARA':
        model_name = 'SimilarityRewards'
    
    model_group = f"{args.dataset_name}_{model_name}"
    base_save_path = "relabeled_offlinedatasets"
    
    # Handle None job_type for directory structure
    job_type_processed = args.job_type
    if job_type_processed == "None":
        job_type_processed = None
    job_folder = "null" if job_type_processed is None else job_type_processed
    
    model_save_path = os.path.join(base_save_path, args.project, args.dataset_name, model_group, job_folder)
    
    # Generate and save summary
    generate_summary(results, model_save_path)
    
    return 0


if __name__ == "__main__":
    exit(main())