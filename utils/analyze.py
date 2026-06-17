#!/usr/bin/env python3
"""
Analyze judge_results from model outputs.

Generates detailed analysis files with per-dataset statistics and sample Q&A.
Usage: python analyze.py <results_folder>
"""

import json
import os
import sys
import random
from pathlib import Path
from collections import defaultdict


def analyze_results(results_folder):
    """
    Analyze judge results in the given folder with per-dataset breakdown.
    
    Args:
        results_folder: Path to the results folder containing judge_results
    """
    results_path = Path(results_folder)
    judge_results_path = results_path / "judge_results"
    
    if not judge_results_path.exists():
        print(f"Error: judge_results folder not found in {results_folder}")
        sys.exit(1)
    
    # Structure: judge_results/{any_brand,trade_dress}/{model_name}/{brand_category}/{dataset}.jsonl
    
    # Collect detailed statistics per model, dataset
    model_data = defaultdict(lambda: defaultdict(lambda: {
        'mentioned': {'total': 0, 'count': 0, 'examples': []},
        'trade_dress': {'total': 0, 'count': 0, 'examples': []},
    }))
    
    # Process any_brand (mentions)
    any_brand_path = judge_results_path / "any_brand"
    if any_brand_path.exists():
        for model_dir in any_brand_path.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                for brand_category in model_dir.iterdir():
                    if brand_category.is_dir():
                        for dataset_file in brand_category.glob("*.jsonl"):
                            dataset_name = dataset_file.stem
                            
                            with open(dataset_file, 'r') as f:
                                for line in f:
                                    if line.strip():
                                        try:
                                            obj = json.loads(line)
                                            parsed = obj.get('parsed')
                                            if parsed is not None:
                                                mentioned = parsed.get('mentioned', False)
                                                model_data[model_name][dataset_name]['mentioned']['total'] += 1
                                                if mentioned:
                                                    model_data[model_name][dataset_name]['mentioned']['count'] += 1
                                                    model_data[model_name][dataset_name]['mentioned']['examples'].append(obj)
                                        except json.JSONDecodeError:
                                            pass
    
    # Process trade_dress
    trade_dress_path = judge_results_path / "trade_dress"
    if trade_dress_path.exists():
        for model_dir in trade_dress_path.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                for brand_category in model_dir.iterdir():
                    if brand_category.is_dir():
                        for dataset_file in brand_category.glob("*.jsonl"):
                            dataset_name = dataset_file.stem
                            
                            with open(dataset_file, 'r') as f:
                                for line in f:
                                    if line.strip():
                                        try:
                                            obj = json.loads(line)
                                            parsed = obj.get('parsed')
                                            if parsed is not None:
                                                trade_dress = parsed.get('trade_dress_present', False)
                                                model_data[model_name][dataset_name]['trade_dress']['total'] += 1
                                                if trade_dress:
                                                    model_data[model_name][dataset_name]['trade_dress']['count'] += 1
                                                    model_data[model_name][dataset_name]['trade_dress']['examples'].append(obj)
                                        except json.JSONDecodeError:
                                            pass
    
    # Generate detailed output
    output_file = results_path / "analysis_results.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("DETAILED MODEL ANALYSIS RESULTS\n")
        f.write("=" * 100 + "\n\n")
        
        for model_name in sorted(model_data.keys()):
            f.write("\n")
            f.write("█" * 100 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("█" * 100 + "\n\n")
            
            for dataset_name in sorted(model_data[model_name].keys()):
                stats = model_data[model_name][dataset_name]
                
                f.write(f"📄 {dataset_name}.jsonl\n")
                f.write("-" * 100 + "\n")
                
                # Mentions statistics
                if stats['mentioned']['total'] > 0:
                    mention_percent = (stats['mentioned']['count'] / stats['mentioned']['total']) * 100
                    f.write(f"  📌 Mentions: {mention_percent:6.2f}% ({stats['mentioned']['count']}/{stats['mentioned']['total']})\n")
                else:
                    f.write(f"  📌 Mentions: No data\n")
                
                # Trade dress statistics
                if stats['trade_dress']['total'] > 0:
                    trade_percent = (stats['trade_dress']['count'] / stats['trade_dress']['total']) * 100
                    f.write(f"  🎨 Trade Dress: {trade_percent:6.2f}% ({stats['trade_dress']['count']}/{stats['trade_dress']['total']})\n")
                else:
                    f.write(f"  🎨 Trade Dress: No data\n")
                
                # Sample examples
                f.write("\n  📋 Sample Examples:\n")
                
                # Sample mentions
                if stats['mentioned']['examples']:
                    examples = random.sample(stats['mentioned']['examples'], min(2, len(stats['mentioned']['examples'])))
                    f.write(f"\n    [Mentions Examples]\n")
                    for i, example in enumerate(examples, 1):
                        f.write(f"\n    Example {i}:\n")
                        question = example.get('question', 'N/A').replace('\n', ' ')
                        response = example.get('response', 'N/A').replace('\n', ' ')
                        f.write(f"      Q: {question}\n")
                        f.write(f"      A: {response}\n")
                
                # Sample trade_dress
                if stats['trade_dress']['examples']:
                    examples = random.sample(stats['trade_dress']['examples'], min(2, len(stats['trade_dress']['examples'])))
                    f.write(f"\n    [Trade Dress Examples]\n")
                    for i, example in enumerate(examples, 1):
                        f.write(f"\n    Example {i}:\n")
                        question = example.get('question', 'N/A').replace('\n', ' ')
                        response = example.get('response', 'N/A').replace('\n', ' ')
                        f.write(f"      Q: {question}\n")
                        f.write(f"      A: {response}\n")
                
                f.write("\n" + "-" * 100 + "\n\n")
        
        # Generate summary tables for each model
        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("SUMMARY TABLES BY MODEL\n")
        f.write("=" * 100 + "\n\n")
        
        for model_name in sorted(model_data.keys()):
            f.write("\n")
            f.write(f"📊 {model_name}\n")
            f.write("-" * 100 + "\n")
            
            # Calculate column widths
            max_dataset_len = max(len(name) for name in model_data[model_name].keys())
            dataset_width = max(max_dataset_len, 30)
            mention_width = 20
            trade_width = 20
            
            # Header
            f.write(f"{'Dataset':<{dataset_width}} | {'Mentions':<{mention_width}} | {'Trade Dress':<{trade_width}}\n")
            f.write("-" * (dataset_width + mention_width + trade_width + 6) + "\n")
            
            # Data rows
            for dataset_name in sorted(model_data[model_name].keys()):
                stats = model_data[model_name][dataset_name]
                
                # Mentions
                if stats['mentioned']['total'] > 0:
                    mention_percent = (stats['mentioned']['count'] / stats['mentioned']['total']) * 100
                    mention_str = f"{mention_percent:6.2f}% ({stats['mentioned']['count']}/{stats['mentioned']['total']})"
                else:
                    mention_str = "No data"
                
                # Trade dress
                if stats['trade_dress']['total'] > 0:
                    trade_percent = (stats['trade_dress']['count'] / stats['trade_dress']['total']) * 100
                    trade_str = f"{trade_percent:6.2f}% ({stats['trade_dress']['count']}/{stats['trade_dress']['total']})"
                else:
                    trade_str = "No data"
                
                f.write(f"{dataset_name:<{dataset_width}} | {mention_str:<{mention_width}} | {trade_str:<{trade_width}}\n")
            
            f.write("\n")
    
    print(f"✅ Analysis complete. Results saved to: {output_file}")
    
    # Print summary to console
    print("\n" + "=" * 100)
    print("ANALYSIS SUMMARY")
    print("=" * 100 + "\n")
    
    for model_name in sorted(model_data.keys()):
        print(f"\n🔹 MODEL: {model_name}")
        print("-" * 100)
        
        for dataset_name in sorted(model_data[model_name].keys()):
            stats = model_data[model_name][dataset_name]
            
            print(f"  📄 {dataset_name}")
            
            if stats['mentioned']['total'] > 0:
                mention_percent = (stats['mentioned']['count'] / stats['mentioned']['total']) * 100
                print(f"     📌 Mentions: {mention_percent:6.2f}% ({stats['mentioned']['count']}/{stats['mentioned']['total']})")
            
            if stats['trade_dress']['total'] > 0:
                trade_percent = (stats['trade_dress']['count'] / stats['trade_dress']['total']) * 100
                print(f"     🎨 Trade Dress: {trade_percent:6.2f}% ({stats['trade_dress']['count']}/{stats['trade_dress']['total']})")
            
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <results_folder>")
        print("Example: python analyze.py /net/scratch/.../experiments_results_NPO/model_outputs_prompts_by_brand_seed42_2026-06-16_23-50-15")
        sys.exit(1)
    
    results_folder = sys.argv[1]
    
    if not os.path.exists(results_folder):
        print(f"Error: Folder not found: {results_folder}")
        sys.exit(1)
    
    analyze_results(results_folder)
