#!/usr/bin/env python3
"""
Visualization and Analysis Tools for CodeQL Query Refinement Experiments

This module provides tools for visualizing experiment data, generating reports,
and analyzing performance trends across iterations.

Features:
- Performance trend graphs over time
- Regression analysis visualization  
- False positive/negative analysis
- Comparative analysis between experiments
- Export capabilities for reports
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from datetime import datetime

from experiment_data_model import (
    Experiment, TestRun, ExperimentStorage, 
    PerformanceMetrics, RegressionAnalysis
)


class ExperimentVisualizer:
    """Main class for generating experiment visualizations"""
    
    def __init__(self, storage_dir: str = "./experiment_data"):
        self.storage = ExperimentStorage(storage_dir)
        
        # Set up matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_performance_trends(self, experiment: Experiment, 
                               metrics: List[str] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot performance metrics trends over iterations"""
        
        if metrics is None:
            metrics = ['f1_score', 'precision', 'recall', 'accuracy']
        
        # Extract data
        iterations = []
        timestamps = []
        metric_data = {metric: [] for metric in metrics}
        
        for run in experiment.test_runs:
            if run.performance_metrics:
                iterations.append(run.iteration_number)
                timestamps.append(run.timestamp)
                
                for metric in metrics:
                    value = getattr(run.performance_metrics, metric, 0.0)
                    metric_data[metric].append(value)
        
        if not iterations:
            print("No performance data to plot")
            return None
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Metrics vs Iterations
        for metric in metrics:
            ax1.plot(iterations, metric_data[metric], 
                    marker='o', linewidth=2, label=metric.replace('_', ' ').title())
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Performance Trends - {experiment.name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Metrics vs Time
        for metric in metrics:
            ax2.plot(timestamps, metric_data[metric], 
                    marker='s', linewidth=2, alpha=0.7, label=metric.replace('_', ' ').title())
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Score')
        ax2.set_title('Performance Trends Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Format time axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_regression_analysis(self, experiment: Experiment,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot regression analysis showing improvements and regressions"""
        
        # Extract regression data
        iterations = []
        f1_deltas = []
        precision_deltas = []
        recall_deltas = []
        net_changes = []
        regressions = []
        improvements = []
        
        for run in experiment.test_runs:
            if run.regression_analysis:
                reg = run.regression_analysis
                iterations.append(run.iteration_number)
                f1_deltas.append(reg.f1_delta)
                precision_deltas.append(reg.precision_delta)
                recall_deltas.append(reg.recall_delta)
                net_changes.append(reg.net_change_score)
                regressions.append(reg.total_regressions)
                improvements.append(reg.total_improvements)
        
        if not iterations:
            print("No regression data to plot")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Metric deltas
        ax1.bar([i-0.2 for i in iterations], f1_deltas, width=0.2, label='F1', alpha=0.7)
        ax1.bar(iterations, precision_deltas, width=0.2, label='Precision', alpha=0.7)
        ax1.bar([i+0.2 for i in iterations], recall_deltas, width=0.2, label='Recall', alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Performance Change')
        ax1.set_title('Performance Changes by Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Net change score
        colors = ['green' if x > 0 else 'red' for x in net_changes]
        ax2.bar(iterations, net_changes, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Net Change Score')
        ax2.set_title('Overall Performance Change')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regressions vs Improvements
        ax3.bar([i-0.2 for i in iterations], regressions, width=0.4, 
               label='Regressions', color='red', alpha=0.7)
        ax3.bar([i+0.2 for i in iterations], improvements, width=0.4, 
               label='Improvements', color='green', alpha=0.7)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Number of Testcases')
        ax3.set_title('Testcase-level Changes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative net change
        cumulative_net = []
        cumsum = 0
        for improvement, regression in zip(improvements, regressions):
            cumsum += improvement - regression
            cumulative_net.append(cumsum)
        
        ax4.plot(iterations, cumulative_net, marker='o', linewidth=2, color='blue')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Cumulative Net Improvement')
        ax4.set_title('Cumulative Progress')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Regression Analysis - {experiment.name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_false_positive_negative_analysis(self, experiment: Experiment,
                                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot false positive and negative trends"""
        
        # Extract FP/FN data
        iterations = []
        false_positives = []
        false_negatives = []
        total_fps = []
        total_fns = []
        
        for run in experiment.test_runs:
            if run.performance_metrics:
                iterations.append(run.iteration_number)
                fps = run.get_false_positives()
                fns = run.get_false_negatives()
                
                false_positives.append(len(fps))
                false_negatives.append(len(fns))
                
                # Group by CWE
                fp_by_cwe = {}
                fn_by_cwe = {}
                
                for fp in fps:
                    fp_by_cwe[fp.cwe_id] = fp_by_cwe.get(fp.cwe_id, 0) + 1
                for fn in fns:
                    fn_by_cwe[fn.cwe_id] = fn_by_cwe.get(fn.cwe_id, 0) + 1
                
                total_fps.append(fp_by_cwe)
                total_fns.append(fn_by_cwe)
        
        if not iterations:
            print("No FP/FN data to plot")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: FP/FN trends
        ax1.plot(iterations, false_positives, marker='o', linewidth=2, 
                label='False Positives', color='red')
        ax1.plot(iterations, false_negatives, marker='s', linewidth=2, 
                label='False Negatives', color='orange')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Count')
        ax1.set_title('False Positives and Negatives Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: FP/FN ratio
        ratios = []
        for fp, fn in zip(false_positives, false_negatives):
            total = fp + fn
            if total > 0:
                ratios.append(fp / total)
            else:
                ratios.append(0)
        
        ax2.bar(iterations, ratios, alpha=0.7, color='purple')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('FP / (FP + FN)')
        ax2.set_title('False Positive Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Total errors trend
        total_errors = [fp + fn for fp, fn in zip(false_positives, false_negatives)]
        ax3.bar(iterations, total_errors, alpha=0.7, color='gray')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Total Errors (FP + FN)')
        ax3.set_title('Total Classification Errors')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error reduction rate
        if len(total_errors) > 1:
            reduction_rates = []
            for i in range(1, len(total_errors)):
                prev_errors = total_errors[i-1]
                curr_errors = total_errors[i]
                if prev_errors > 0:
                    reduction_rate = (prev_errors - curr_errors) / prev_errors
                else:
                    reduction_rate = 0
                reduction_rates.append(reduction_rate)
            
            colors = ['green' if x > 0 else 'red' for x in reduction_rates]
            ax4.bar(iterations[1:], reduction_rates, color=colors, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Error Reduction Rate')
            ax4.set_title('Error Reduction Between Iterations')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'False Positive/Negative Analysis - {experiment.name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_execution_time_analysis(self, experiment: Experiment,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Plot execution time analysis"""
        
        # Extract execution data
        iterations = []
        total_times = []
        avg_times = []
        max_times = []
        min_times = []
        
        for run in experiment.test_runs:
            if run.execution_stats:
                stats = run.execution_stats
                iterations.append(run.iteration_number)
                total_times.append(stats.total_execution_time)
                avg_times.append(stats.average_execution_time)
                max_times.append(stats.max_execution_time)
                min_times.append(stats.min_execution_time)
        
        if not iterations:
            print("No execution time data to plot")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Total execution time
        ax1.bar(iterations, total_times, alpha=0.7, color='blue')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Total Time (seconds)')
        ax1.set_title('Total Execution Time per Iteration')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average execution time with error bars
        errors = [[avg - min_t for avg, min_t in zip(avg_times, min_times)],
                 [max_t - avg for avg, max_t in zip(avg_times, max_times)]]
        
        ax2.errorbar(iterations, avg_times, yerr=errors, marker='o', 
                    linewidth=2, capsize=5, color='green')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Average Time (seconds)')
        ax2.set_title('Average Execution Time (with min/max range)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time efficiency (results per second)
        efficiency = []
        for run in experiment.test_runs:
            if run.execution_stats and run.performance_metrics:
                total_results = (run.performance_metrics.true_positives + 
                               run.performance_metrics.true_negatives)
                if run.execution_stats.total_execution_time > 0:
                    eff = total_results / run.execution_stats.total_execution_time
                else:
                    eff = 0
                efficiency.append(eff)
        
        if efficiency:
            ax3.plot(iterations, efficiency, marker='s', linewidth=2, color='purple')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Correct Results per Second')
            ax3.set_title('Query Efficiency')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Execution success rate
        success_rates = []
        for run in experiment.test_runs:
            if run.execution_stats:
                total_executions = (run.execution_stats.successful_executions + 
                                  run.execution_stats.failed_executions)
                if total_executions > 0:
                    success_rate = run.execution_stats.successful_executions / total_executions
                else:
                    success_rate = 0
                success_rates.append(success_rate)
        
        if success_rates:
            ax4.bar(iterations, success_rates, alpha=0.7, color='orange')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Success Rate')
            ax4.set_title('Query Execution Success Rate')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Execution Time Analysis - {experiment.name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_comprehensive_report(self, experiment_id: str, 
                                    output_dir: str = "./reports") -> str:
        """Generate a comprehensive report with all visualizations"""
        
        experiment = self.storage.load_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_dir = output_path / f"report_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir.mkdir(exist_ok=True)
        
        print(f"Generating comprehensive report for {experiment.name}...")
        
        # Generate all visualizations
        print("  Creating performance trends plot...")
        fig1 = self.plot_performance_trends(experiment, 
                                          save_path=str(report_dir / "performance_trends.png"))
        if fig1:
            plt.close(fig1)
        
        print("  Creating regression analysis plot...")
        fig2 = self.plot_regression_analysis(experiment,
                                           save_path=str(report_dir / "regression_analysis.png"))
        if fig2:
            plt.close(fig2)
        
        print("  Creating FP/FN analysis plot...")
        fig3 = self.plot_false_positive_negative_analysis(experiment,
                                                         save_path=str(report_dir / "fp_fn_analysis.png"))
        if fig3:
            plt.close(fig3)
        
        print("  Creating execution time analysis plot...")
        fig4 = self.plot_execution_time_analysis(experiment,
                                                save_path=str(report_dir / "execution_time_analysis.png"))
        if fig4:
            plt.close(fig4)
        
        # Generate summary data
        print("  Creating summary data...")
        summary_data = self.generate_summary_data(experiment)
        
        with open(report_dir / "summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Generate markdown report
        print("  Creating markdown report...")
        markdown_report = self.generate_markdown_report(experiment, summary_data)
        
        with open(report_dir / "report.md", 'w') as f:
            f.write(markdown_report)
        
        print(f"✅ Report generated in: {report_dir}")
        return str(report_dir)
    
    def generate_summary_data(self, experiment: Experiment) -> Dict[str, Any]:
        """Generate summary statistics for the experiment"""
        
        if not experiment.test_runs:
            return {"error": "No test runs found"}
        
        # Basic info
        summary = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "target_cwe": experiment.target_cwe,
            "total_iterations": len(experiment.test_runs),
            "duration": str(experiment.test_runs[-1].timestamp - experiment.created_timestamp),
        }
        
        # Performance progression
        runs_with_metrics = [r for r in experiment.test_runs if r.performance_metrics]
        if runs_with_metrics:
            first_run = runs_with_metrics[0]
            last_run = runs_with_metrics[-1]
            best_run = max(runs_with_metrics, key=lambda r: r.performance_metrics.f1_score)
            
            summary["performance"] = {
                "initial_f1": first_run.performance_metrics.f1_score,
                "final_f1": last_run.performance_metrics.f1_score,
                "best_f1": best_run.performance_metrics.f1_score,
                "best_iteration": best_run.iteration_number,
                "improvement": last_run.performance_metrics.f1_score - first_run.performance_metrics.f1_score
            }
        
        # Regression summary
        regressions = [r.regression_analysis for r in experiment.test_runs if r.regression_analysis]
        if regressions:
            total_regressions = sum(r.total_regressions for r in regressions)
            total_improvements = sum(r.total_improvements for r in regressions)
            
            summary["regression_analysis"] = {
                "total_regressions": total_regressions,
                "total_improvements": total_improvements,
                "net_improvement": total_improvements - total_regressions,
                "average_net_change": sum(r.net_change_score for r in regressions) / len(regressions)
            }
        
        # Trends
        f1_slope, f1_direction = experiment.calculate_trend('f1_score')
        precision_slope, precision_direction = experiment.calculate_trend('precision')
        
        summary["trends"] = {
            "f1_trend": {"slope": f1_slope, "direction": f1_direction},
            "precision_trend": {"slope": precision_slope, "direction": precision_direction}
        }
        
        return summary
    
    def generate_markdown_report(self, experiment: Experiment, summary_data: Dict[str, Any]) -> str:
        """Generate a markdown report"""
        
        md = f"""# Experiment Report: {experiment.name}

**Experiment ID:** {experiment.experiment_id}  
**Target CWE:** {experiment.target_cwe}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

{experiment.description}

- **Total Iterations:** {summary_data.get('total_iterations', 'N/A')}
- **Duration:** {summary_data.get('duration', 'N/A')}
- **3-Shot Generation:** {'Yes' if experiment.use_three_shot_generation else 'No'}
- **FP/FN Analysis:** {'Yes' if experiment.use_fp_fn_analysis else 'No'}

## Performance Summary

"""
        
        if 'performance' in summary_data:
            perf = summary_data['performance']
            md += f"""
- **Initial F1 Score:** {perf['initial_f1']:.3f}
- **Final F1 Score:** {perf['final_f1']:.3f}
- **Best F1 Score:** {perf['best_f1']:.3f} (Iteration {perf['best_iteration']})
- **Overall Improvement:** {perf['improvement']:+.3f}

"""
        
        md += """## Visualizations

### Performance Trends
![Performance Trends](performance_trends.png)

### Regression Analysis  
![Regression Analysis](regression_analysis.png)

### False Positive/Negative Analysis
![FP/FN Analysis](fp_fn_analysis.png)

### Execution Time Analysis
![Execution Time Analysis](execution_time_analysis.png)

## Iteration Details

"""
        
        # Add iteration details
        for run in experiment.test_runs:
            md += f"""### Iteration {run.iteration_number}
**Timestamp:** {run.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            if run.performance_metrics:
                m = run.performance_metrics
                md += f"""
**Performance:**
- F1 Score: {m.f1_score:.3f}
- Precision: {m.precision:.3f}  
- Recall: {m.recall:.3f}
- Accuracy: {m.accuracy:.3f}
"""
            
            if run.regression_analysis:
                r = run.regression_analysis
                md += f"""
**Changes from Previous:**
- F1 Change: {r.f1_delta:+.3f}
- Testcase Regressions: {r.total_regressions}
- Testcase Improvements: {r.total_improvements}
"""
            
            if run.fp_fn_analysis:
                fp_count = len(run.get_false_positives())
                fn_count = len(run.get_false_negatives())
                md += f"""
**Error Analysis:**
- False Positives: {fp_count}
- False Negatives: {fn_count}
"""
            
            md += "\n"
        
        md += """## Conclusions

"""
        
        if 'trends' in summary_data:
            trends = summary_data['trends']
            md += f"""
**Performance Trends:**
- F1 Score: {trends['f1_trend']['direction']} (slope: {trends['f1_trend']['slope']:+.4f})
- Precision: {trends['precision_trend']['direction']} (slope: {trends['precision_trend']['slope']:+.4f})
"""
        
        if 'regression_analysis' in summary_data:
            reg = summary_data['regression_analysis']
            md += f"""
**Regression Summary:**
- Total Improvements: {reg['total_improvements']}
- Total Regressions: {reg['total_regressions']}
- Net Improvement: {reg['net_improvement']}
"""
        
        return md
    
    def compare_experiments(self, experiment_ids: List[str],
                          save_path: Optional[str] = None) -> plt.Figure:
        """Compare multiple experiments"""
        
        experiments = []
        for exp_id in experiment_ids:
            exp = self.storage.load_experiment(exp_id)
            if exp:
                experiments.append(exp)
        
        if not experiments:
            print("No valid experiments found")
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Final F1 scores comparison
        names = [exp.name[:20] + "..." if len(exp.name) > 20 else exp.name for exp in experiments]
        final_f1s = []
        
        for exp in experiments:
            runs_with_metrics = [r for r in exp.test_runs if r.performance_metrics]
            if runs_with_metrics:
                final_f1s.append(runs_with_metrics[-1].performance_metrics.f1_score)
            else:
                final_f1s.append(0)
        
        ax1.bar(names, final_f1s, alpha=0.7)
        ax1.set_ylabel('Final F1 Score')
        ax1.set_title('Final Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best F1 scores comparison
        best_f1s = []
        for exp in experiments:
            runs_with_metrics = [r for r in exp.test_runs if r.performance_metrics]
            if runs_with_metrics:
                best_f1 = max(r.performance_metrics.f1_score for r in runs_with_metrics)
                best_f1s.append(best_f1)
            else:
                best_f1s.append(0)
        
        ax2.bar(names, best_f1s, alpha=0.7, color='green')
        ax2.set_ylabel('Best F1 Score')
        ax2.set_title('Best Performance Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Iteration count comparison
        iteration_counts = [len(exp.test_runs) for exp in experiments]
        ax3.bar(names, iteration_counts, alpha=0.7, color='orange')
        ax3.set_ylabel('Number of Iterations')
        ax3.set_title('Iteration Count Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: F1 progression for all experiments
        for exp in experiments:
            iterations = []
            f1_scores = []
            
            for run in exp.test_runs:
                if run.performance_metrics:
                    iterations.append(run.iteration_number)
                    f1_scores.append(run.performance_metrics.f1_score)
            
            if iterations:
                exp_name = exp.name[:15] + "..." if len(exp.name) > 15 else exp.name
                ax4.plot(iterations, f1_scores, marker='o', linewidth=2, label=exp_name)
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score Progression Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.suptitle('Experiment Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations and reports for CodeQL experiments"
    )
    
    parser.add_argument('--experiment-id', help='Experiment ID to visualize')
    parser.add_argument('--list-experiments', action='store_true', help='List all experiments')
    parser.add_argument('--compare', nargs='+', help='Compare multiple experiments by ID')
    parser.add_argument('--generate-report', action='store_true', help='Generate comprehensive report')
    parser.add_argument('--output-dir', default='./reports', help='Output directory for reports')
    parser.add_argument('--storage-dir', default='./experiment_data', help='Experiment data directory')
    
    args = parser.parse_args()
    
    visualizer = ExperimentVisualizer(args.storage_dir)
    
    if args.list_experiments:
        experiments = visualizer.storage.list_experiments()
        print("Available experiments:")
        for exp_id in experiments:
            exp = visualizer.storage.load_experiment(exp_id)
            if exp:
                print(f"  {exp_id}: {exp.name} ({len(exp.test_runs)} iterations)")
        return
    
    if args.compare:
        print(f"Comparing experiments: {args.compare}")
        fig = visualizer.compare_experiments(args.compare)
        if fig:
            plt.show()
        return
    
    if not args.experiment_id:
        print("Please specify --experiment-id or use --list-experiments")
        return
    
    experiment = visualizer.storage.load_experiment(args.experiment_id)
    if not experiment:
        print(f"Experiment {args.experiment_id} not found")
        return
    
    if args.generate_report:
        report_path = visualizer.generate_comprehensive_report(args.experiment_id, args.output_dir)
        print(f"Report generated: {report_path}")
    else:
        # Show individual plots
        print("Generating visualizations...")
        
        fig1 = visualizer.plot_performance_trends(experiment)
        fig2 = visualizer.plot_regression_analysis(experiment)
        fig3 = visualizer.plot_false_positive_negative_analysis(experiment)
        fig4 = visualizer.plot_execution_time_analysis(experiment)
        
        plt.show()


if __name__ == "__main__":
    main()
