#!/usr/bin/env python3
"""
Publication-Quality Visualization Suite for Dyslexia Time Cost Analysis
Generates figures for manuscript and presentations
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style parameters
plt.style.use('seaborn-v0_8-whitegrid')
try:
    sns.set_palette("husl")
except:
    # Fallback for older seaborn versions
    sns.set_style("whitegrid")

class DyslexiaVisualizationSuite:
    """
    Comprehensive visualization suite for dyslexia reading analysis
    """
    
    def __init__(self, config):
        self.config = config
        self.figure_dir = Path(config.get('FIGURE_DIR', 'results/figures'))
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Set global plotting parameters
        self.dpi = config.get('FIGURE_DPI', 300)
        self.format = config.get('FIGURE_FORMAT', 'pdf')
        self.palette = config.get('COLOR_PALETTE', 'viridis')
        
        # Color scheme for groups
        self.group_colors = {
            'Control': '#2E86AB',
            'Dyslexic': '#F24236',
            'control': '#2E86AB',
            'dyslexic': '#F24236'
        }
        
    def create_publication_figures(self, data: pd.DataFrame, 
                                 statistical_results: Dict) -> Dict[str, str]:
        """
        Create all publication-ready figures
        
        Returns:
        --------
        Dict mapping figure names to file paths
        """
        logger.info("Creating publication figures...")
        
        figure_paths = {}
        
        # Figure 1: Descriptive statistics and group differences
        figure_paths['figure_1'] = self._create_descriptive_figure(data)
        
        # Figure 2: Feature effects on eye measures
        figure_paths['figure_2'] = self._create_feature_effects_figure(
            data, statistical_results.get('hypothesis_1', {})
        )
        
        # Figure 3: Dyslexic amplification effects
        figure_paths['figure_3'] = self._create_amplification_figure(
            data, statistical_results.get('hypothesis_2', {})
        )
        
        # Figure 4: Variance decomposition
        figure_paths['figure_4'] = self._create_variance_decomposition_figure(
            statistical_results.get('hypothesis_3', {})
        )
        
        # Figure 5: Model diagnostics and validation
        figure_paths['figure_5'] = self._create_diagnostic_figure(
            data, statistical_results
        )
        
        # Supplementary figures
        figure_paths['supp_correlations'] = self._create_correlation_matrix(data)
        figure_paths['supp_distributions'] = self._create_distribution_figure(data)
        
        logger.info(f"Created {len(figure_paths)} figures")
        return figure_paths
    
    def _create_descriptive_figure(self, data: pd.DataFrame) -> str:
        """Figure 1: Descriptive statistics and basic group differences"""
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Descriptive Statistics and Group Differences', fontsize=16, fontweight='bold')
        
        # Prepare data
        plot_data = data.copy()
        plot_data['Group'] = plot_data['dyslexic'].map({True: 'Dyslexic', False: 'Control'})
        
        # Panel A: Total Reading Time distribution
        ax = axes[0, 0]
        for group in ['Control', 'Dyslexic']:
            group_data = plot_data[plot_data['Group'] == group]['total_reading_time']
            ax.hist(group_data, alpha=0.7, label=group, bins=30, 
                   color=self.group_colors[group], density=True)
        ax.set_xlabel('Total Reading Time (ms)')
        ax.set_ylabel('Density')
        ax.set_title('A. Reading Time Distributions')
        ax.legend()
        
        # Panel B: Eye measure comparisons
        ax = axes[0, 1]
        eye_measures = ['first_fixation_duration', 'gaze_duration', 'total_reading_time']
        eye_measures = [m for m in eye_measures if m in data.columns]
        
        group_means = []
        for measure in eye_measures:
            control_mean = plot_data[plot_data['Group'] == 'Control'][measure].mean()
            dyslexic_mean = plot_data[plot_data['Group'] == 'Dyslexic'][measure].mean()
            group_means.append([control_mean, dyslexic_mean])
        
        x = np.arange(len(eye_measures))
        width = 0.35
        
        control_means = [means[0] for means in group_means]
        dyslexic_means = [means[1] for means in group_means]
        
        ax.bar(x - width/2, control_means, width, label='Control', 
               color=self.group_colors['Control'], alpha=0.8)
        ax.bar(x + width/2, dyslexic_means, width, label='Dyslexic', 
               color=self.group_colors['Dyslexic'], alpha=0.8)
        
        ax.set_xlabel('Eye Measures')
        ax.set_ylabel('Duration (ms)')
        ax.set_title('B. Group Differences in Eye Measures')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in eye_measures], rotation=45)
        ax.legend()
        
        # Panel C: Word-level feature distributions
        ax = axes[0, 2]
        features = ['word_length', 'log_frequency', 'predictability']
        features = [f for f in features if f in data.columns]
        
        box_data = []
        labels = []
        for feature in features:
            box_data.append(plot_data[feature].dropna())
            labels.append(feature.replace('_', ' ').title())
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(self.group_colors['Control'])
            patch.set_alpha(0.7)
        
        ax.set_title('C. Word Feature Distributions')
        ax.tick_params(axis='x', rotation=45)
        
        # Panel D: Length vs Reading Time by group
        ax = axes[1, 0]
        for group in ['Control', 'Dyslexic']:
            group_data = plot_data[plot_data['Group'] == group]
            ax.scatter(group_data['word_length'], group_data['total_reading_time'], 
                      alpha=0.6, label=group, color=self.group_colors[group], s=20)
        
        ax.set_xlabel('Word Length (characters)')
        ax.set_ylabel('Total Reading Time (ms)')
        ax.set_title('D. Length Effect by Group')
        ax.legend()
        
        # Panel E: Frequency vs Reading Time by group
        ax = axes[1, 1]
        for group in ['Control', 'Dyslexic']:
            group_data = plot_data[plot_data['Group'] == group]
            ax.scatter(group_data['log_frequency'], group_data['total_reading_time'], 
                      alpha=0.6, label=group, color=self.group_colors[group], s=20)
        
        ax.set_xlabel('Log Frequency')
        ax.set_ylabel('Total Reading Time (ms)')
        ax.set_title('E. Frequency Effect by Group')
        ax.legend()
        
        # Panel F: Effect size summary
        ax = axes[1, 2]
        # Calculate Cohen's d for key comparisons
        effect_sizes = []
        comparisons = []
        
        for measure in eye_measures:
            control_data = plot_data[plot_data['Group'] == 'Control'][measure]
            dyslexic_data = plot_data[plot_data['Group'] == 'Dyslexic'][measure]
            
            # Cohen's d
            pooled_std = np.sqrt((control_data.var() + dyslexic_data.var()) / 2)
            cohens_d = (dyslexic_data.mean() - control_data.mean()) / pooled_std
            
            effect_sizes.append(cohens_d)
            comparisons.append(measure.replace('_', ' ').title())
        
        bars = ax.barh(comparisons, effect_sizes, color=self.group_colors['Dyslexic'], alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
        
        ax.set_xlabel("Cohen's d (Dyslexic - Control)")
        ax.set_title('F. Effect Sizes')
        ax.legend(fontsize=8)
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.figure_dir / f'figure_1_descriptive.{self.format}'
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def _create_feature_effects_figure(self, data: pd.DataFrame, 
                                     hypothesis_1_results: Dict) -> str:
        """Figure 2: Feature effects on eye measures"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Feature Effects on Eye Measures (Hypothesis 1)', fontsize=16, fontweight='bold')
        
        features = ['word_length', 'log_frequency', 'launch_site_distance', 'predictability']
        feature_labels = ['Word Length', 'Log Frequency', 'Launch Distance', 'Predictability']
        
        # Panel A: Length effects
        ax = axes[0, 0]
        if 'word_length' in data.columns and 'total_reading_time' in data.columns:
            # Bin by length for cleaner visualization
            data_binned = data.copy()
            data_binned['length_bin'] = pd.cut(data['word_length'], bins=5, labels=['1-3', '4-5', '6-7', '8-10', '11+'])
            
            length_means = data_binned.groupby(['length_bin', 'dyslexic'])['total_reading_time'].mean().unstack()
            length_stds = data_binned.groupby(['length_bin', 'dyslexic'])['total_reading_time'].std().unstack()
            
            x = np.arange(len(length_means.index))
            width = 0.35
            
            ax.bar(x - width/2, length_means[False], width, yerr=length_stds[False], 
                   label='Control', color=self.group_colors['Control'], alpha=0.8, capsize=5)
            ax.bar(x + width/2, length_means[True], width, yerr=length_stds[True], 
                   label='Dyslexic', color=self.group_colors['Dyslexic'], alpha=0.8, capsize=5)
            
            ax.set_xlabel('Word Length (characters)')
            ax.set_ylabel('Reading Time (ms)')
            ax.set_title('A. Word Length Effect')
            ax.set_xticks(x)
            ax.set_xticklabels(length_means.index)
            ax.legend()
        
        # Panel B: Frequency effects
        ax = axes[0, 1]
        if 'log_frequency' in data.columns and 'total_reading_time' in data.columns:
            # Bin by frequency quartiles
            data_binned = data.copy()
            data_binned['freq_quartile'] = pd.qcut(data['log_frequency'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
            
            freq_means = data_binned.groupby(['freq_quartile', 'dyslexic'])['total_reading_time'].mean().unstack()
            freq_stds = data_binned.groupby(['freq_quartile', 'dyslexic'])['total_reading_time'].std().unstack()
            
            x = np.arange(len(freq_means.index))
            width = 0.35
            
            ax.bar(x - width/2, freq_means[False], width, yerr=freq_stds[False], 
                   label='Control', color=self.group_colors['Control'], alpha=0.8, capsize=5)
            ax.bar(x + width/2, freq_means[True], width, yerr=freq_stds[True], 
                   label='Dyslexic', color=self.group_colors['Dyslexic'], alpha=0.8, capsize=5)
            
            ax.set_xlabel('Word Frequency')
            ax.set_ylabel('Reading Time (ms)')
            ax.set_title('B. Frequency Effect')
            ax.set_xticks(x)
            ax.set_xticklabels(freq_means.index, rotation=45)
            ax.legend()
        
        # Panel C: Preview effects
        ax = axes[1, 0]
        if 'launch_site_distance' in data.columns:
            # Bin by launch distance
            data_binned = data.copy()
            data_binned['launch_bin'] = pd.cut(data['launch_site_distance'], 
                                             bins=[0, 2, 4, 8, np.inf], 
                                             labels=['0-2', '2-4', '4-8', '8+'])
            
            launch_means = data_binned.groupby(['launch_bin', 'dyslexic'])['total_reading_time'].mean().unstack()
            launch_stds = data_binned.groupby(['launch_bin', 'dyslexic'])['total_reading_time'].std().unstack()
            
            x = np.arange(len(launch_means.index))
            width = 0.35
            
            ax.bar(x - width/2, launch_means[False], width, yerr=launch_stds[False], 
                   label='Control', color=self.group_colors['Control'], alpha=0.8, capsize=5)
            ax.bar(x + width/2, launch_means[True], width, yerr=launch_stds[True], 
                   label='Dyslexic', color=self.group_colors['Dyslexic'], alpha=0.8, capsize=5)
            
            ax.set_xlabel('Launch Site Distance')
            ax.set_ylabel('Reading Time (ms)')
            ax.set_title('C. Preview/Launch Distance Effect')
            ax.set_xticks(x)
            ax.set_xticklabels(launch_means.index)
            ax.legend()
        
        # Panel D: Predictability effects
        ax = axes[1, 1]
        if 'predictability' in data.columns:
            # Bin by predictability quartiles
            data_binned = data.copy()
            data_binned['pred_quartile'] = pd.qcut(data['predictability'], q=4, 
                                                  labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
            
            pred_means = data_binned.groupby(['pred_quartile', 'dyslexic'])['total_reading_time'].mean().unstack()
            pred_stds = data_binned.groupby(['pred_quartile', 'dyslexic'])['total_reading_time'].std().unstack()
            
            x = np.arange(len(pred_means.index))
            width = 0.35
            
            ax.bar(x - width/2, pred_means[False], width, yerr=pred_stds[False], 
                   label='Control', color=self.group_colors['Control'], alpha=0.8, capsize=5)
            ax.bar(x + width/2, pred_means[True], width, yerr=pred_stds[True], 
                   label='Dyslexic', color=self.group_colors['Dyslexic'], alpha=0.8, capsize=5)
            
            ax.set_xlabel('Word Predictability')
            ax.set_ylabel('Reading Time (ms)')
            ax.set_title('D. Predictability Effect')
            ax.set_xticks(x)
            ax.set_xticklabels(pred_means.index, rotation=45)
            ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.figure_dir / f'figure_2_feature_effects.{self.format}'
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def _create_amplification_figure(self, data: pd.DataFrame, 
                                   hypothesis_2_results: Dict) -> str:
        """Figure 3: Dyslexic amplification effects"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Dyslexic Amplification Effects (Hypothesis 2)', fontsize=16, fontweight='bold')
        
        # Panel A: Length × Group interaction
        ax = axes[0, 0]
        if 'word_length' in data.columns:
            # Create length bins
            data_binned = data.copy()
            data_binned['length_bin'] = pd.cut(data['word_length'], bins=5)
            
            # Calculate slopes for each group
            control_data = data[data['dyslexic'] == False]
            dyslexic_data = data[data['dyslexic'] == True]
            
            if len(control_data) > 0 and len(dyslexic_data) > 0:
                # Regression lines
                control_slope, control_intercept, _, _, _ = stats.linregress(
                    control_data['word_length'], control_data['total_reading_time']
                )
                dyslexic_slope, dyslexic_intercept, _, _, _ = stats.linregress(
                    dyslexic_data['word_length'], dyslexic_data['total_reading_time']
                )
                
                # Plot data points
                ax.scatter(control_data['word_length'], control_data['total_reading_time'], 
                          alpha=0.3, color=self.group_colors['Control'], s=10, label='Control')
                ax.scatter(dyslexic_data['word_length'], dyslexic_data['total_reading_time'], 
                          alpha=0.3, color=self.group_colors['Dyslexic'], s=10, label='Dyslexic')
                
                # Plot regression lines
                x_range = np.linspace(data['word_length'].min(), data['word_length'].max(), 100)
                ax.plot(x_range, control_slope * x_range + control_intercept, 
                       color=self.group_colors['Control'], linewidth=2, 
                       label=f'Control (slope={control_slope:.1f})')
                ax.plot(x_range, dyslexic_slope * x_range + dyslexic_intercept, 
                       color=self.group_colors['Dyslexic'], linewidth=2, 
                       label=f'Dyslexic (slope={dyslexic_slope:.1f})')
            
            ax.set_xlabel('Word Length')
            ax.set_ylabel('Total Reading Time (ms)')
            ax.set_title('A. Length × Group Interaction')
            ax.legend()
        
        # Panel B: Frequency × Group interaction
        ax = axes[0, 1]
        if 'log_frequency' in data.columns:
            control_data = data[data['dyslexic'] == False]
            dyslexic_data = data[data['dyslexic'] == True]
            
            if len(control_data) > 0 and len(dyslexic_data) > 0:
                # Regression lines for frequency effect
                control_slope, control_intercept, _, _, _ = stats.linregress(
                    control_data['log_frequency'], control_data['total_reading_time']
                )
                dyslexic_slope, dyslexic_intercept, _, _, _ = stats.linregress(
                    dyslexic_data['log_frequency'], dyslexic_data['total_reading_time']
                )
                
                # Plot data points
                ax.scatter(control_data['log_frequency'], control_data['total_reading_time'], 
                          alpha=0.3, color=self.group_colors['Control'], s=10, label='Control')
                ax.scatter(dyslexic_data['log_frequency'], dyslexic_data['total_reading_time'], 
                          alpha=0.3, color=self.group_colors['Dyslexic'], s=10, label='Dyslexic')
                
                # Plot regression lines
                x_range = np.linspace(data['log_frequency'].min(), data['log_frequency'].max(), 100)
                ax.plot(x_range, control_slope * x_range + control_intercept, 
                       color=self.group_colors['Control'], linewidth=2, 
                       label=f'Control (slope={control_slope:.1f})')
                ax.plot(x_range, dyslexic_slope * x_range + dyslexic_intercept, 
                       color=self.group_colors['Dyslexic'], linewidth=2, 
                       label=f'Dyslexic (slope={dyslexic_slope:.1f})')
            
            ax.set_xlabel('Log Frequency')
            ax.set_ylabel('Total Reading Time (ms)')
            ax.set_title('B. Frequency × Group Interaction')
            ax.legend()
        
        # Panel C: Interaction effect sizes
        ax = axes[1, 0]
        if hypothesis_2_results:
            # Extract interaction coefficients from statistical results
            interaction_effects = {}
            
            for measure, model_result in hypothesis_2_results.items():
                if 'interactions' in model_result:
                    for predictor, effect in model_result['interactions'].items():
                        if predictor not in interaction_effects:
                            interaction_effects[predictor] = []
                        interaction_effects[predictor].append(effect)
            
            # Average effects across measures
            predictors = list(interaction_effects.keys())
            effects = [np.mean(interaction_effects[pred]) for pred in predictors]
            
            if predictors and effects:
                bars = ax.barh(predictors, effects, color=self.group_colors['Dyslexic'], alpha=0.7)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
                ax.set_xlabel('Interaction Effect Size')
                ax.set_title('C. Group × Feature Interactions')
                
                # Add effect magnitude indicators
                for i, (pred, effect) in enumerate(zip(predictors, effects)):
                    ax.text(effect + 0.01 if effect > 0 else effect - 0.01, i, 
                           f'{effect:.3f}', va='center', 
                           ha='left' if effect > 0 else 'right')
        
        # Panel D: Preview benefit comparison
        ax = axes[1, 1]
        if 'preview_benefit_score' in data.columns:
            # Compare preview benefits between groups
            control_preview = data[data['dyslexic'] == False]['preview_benefit_score']
            dyslexic_preview = data[data['dyslexic'] == True]['preview_benefit_score']
            
            # Box plots
            box_data = [control_preview.dropna(), dyslexic_preview.dropna()]
            bp = ax.boxplot(box_data, labels=['Control', 'Dyslexic'], patch_artist=True)
            
            bp['boxes'][0].set_facecolor(self.group_colors['Control'])
            bp['boxes'][1].set_facecolor(self.group_colors['Dyslexic'])
            
            ax.set_ylabel('Preview Benefit Score')
            ax.set_title('D. Preview Benefit by Group')
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.figure_dir / f'figure_3_amplification.{self.format}'
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def _create_variance_decomposition_figure(self, hypothesis_3_results: Dict) -> str:
        """Figure 4: Variance decomposition and model comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Variance Decomposition and Model Comparison (Hypothesis 3)', 
                    fontsize=16, fontweight='bold')
        
        # Panel A: Progressive R² increase
        ax = axes[0, 0]
        
        if hypothesis_3_results:
            measure = 'total_reading_time'  # Focus on main outcome
            if measure in hypothesis_3_results:
                variance_data = hypothesis_3_results[measure].get('variance_explained', {})
                
                if variance_data:
                    # Extract progressive R² values
                    baseline_r2 = variance_data.get('baseline_r2', 0)
                    progressive_r2 = variance_data.get('progressive_r2', {})
                    
                    model_names = ['Baseline'] + list(progressive_r2.keys())
                    r2_values = [baseline_r2] + list(progressive_r2.values())
                    
                    # Create stepped plot
                    x = np.arange(len(model_names))
                    ax.plot(x, r2_values, 'o-', linewidth=2, markersize=8, 
                           color=self.group_colors['Control'])
                    ax.fill_between(x, 0, r2_values, alpha=0.3, color=self.group_colors['Control'])
                    
                    ax.set_xlabel('Model Complexity')
                    ax.set_ylabel('R² (Variance Explained)')
                    ax.set_title('A. Progressive Model Performance')
                    ax.set_xticks(x)
                    ax.set_xticklabels(model_names, rotation=45)
                    ax.set_ylim(0, 1)
        
        # Panel B: R² change by feature addition
        ax = axes[0, 1]
        
        if hypothesis_3_results and measure in hypothesis_3_results:
            variance_data = hypothesis_3_results[measure].get('variance_explained', {})
            r2_changes = variance_data.get('r2_change', {})
            
            if r2_changes:
                features = list(r2_changes.keys())
                changes = list(r2_changes.values())
                
                bars = ax.bar(features, changes, color=self.group_colors['Dyslexic'], alpha=0.7)
                ax.set_xlabel('Feature Added')
                ax.set_ylabel('ΔR² (Additional Variance)')
                ax.set_title('B. Incremental Variance Explained')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, change in zip(bars, changes):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{change:.3f}', ha='center', va='bottom')
        
        # Panel C: Model comparison across measures
        ax = axes[1, 0]
        
        measures = ['total_reading_time', 'gaze_duration', 'first_fixation_duration']
        r2_by_measure = []
        measure_labels = []
        
        for measure in measures:
            if measure in hypothesis_3_results:
                variance_data = hypothesis_3_results[measure].get('variance_explained', {})
                full_r2 = variance_data.get('full_r2', 0)
                r2_by_measure.append(full_r2)
                measure_labels.append(measure.replace('_', ' ').title())
        
        if r2_by_measure:
            bars = ax.bar(measure_labels, r2_by_measure, 
                         color=[self.group_colors['Control'], self.group_colors['Dyslexic'], '#FFA500'], 
                         alpha=0.7)
            ax.set_xlabel('Eye Measure')
            ax.set_ylabel('Full Model R²')
            ax.set_title('C. Model Performance by Measure')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, r2 in zip(bars, r2_by_measure):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{r2:.3f}', ha='center', va='bottom')
        
        # Panel D: Group difference reduction
        ax = axes[1, 1]
        
        if hypothesis_3_results:
            # Calculate group difference reduction
            baseline_explained = []
            full_explained = []
            measure_names = []
            
            for measure in measures:
                if measure in hypothesis_3_results:
                    variance_data = hypothesis_3_results[measure].get('variance_explained', {})
                    baseline_r2 = variance_data.get('baseline_r2', 0)
                    full_r2 = variance_data.get('full_r2', 0)
                    
                    baseline_explained.append(baseline_r2)
                    full_explained.append(full_r2)
                    measure_names.append(measure.replace('_', ' ').title())
            
            if baseline_explained and full_explained:
                x = np.arange(len(measure_names))
                width = 0.35
                
                ax.bar(x - width/2, baseline_explained, width, label='Baseline (Group Only)', 
                       color=self.group_colors['Control'], alpha=0.7)
                ax.bar(x + width/2, full_explained, width, label='Full Model', 
                       color=self.group_colors['Dyslexic'], alpha=0.7)
                
                ax.set_xlabel('Eye Measure')
                ax.set_ylabel('Variance Explained (R²)')
                ax.set_title('D. Baseline vs Full Model')
                ax.set_xticks(x)
                ax.set_xticklabels(measure_names, rotation=45)
                ax.legend()
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.figure_dir / f'figure_4_variance_decomposition.{self.format}'
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def _create_diagnostic_figure(self, data: pd.DataFrame, 
                                statistical_results: Dict) -> str:
        """Figure 5: Model diagnostics and validation"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Diagnostics and Validation', fontsize=16, fontweight='bold')
        
        # Panel A: Residual plots
        ax = axes[0, 0]
        
        # Get residuals from main model (if available)
        if 'hypothesis_3' in statistical_results:
            trt_results = statistical_results['hypothesis_3'].get('total_reading_time', {})
            full_model = trt_results.get('full', {})
            
            if 'fitted_values' in full_model and 'residuals' in full_model:
                fitted = full_model['fitted_values']
                residuals = full_model['residuals']
                
                ax.scatter(fitted, residuals, alpha=0.6, s=20, color=self.group_colors['Control'])
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel('Fitted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('A. Residuals vs Fitted')
                
                # Add smoothed line
                try:
                    from scipy.interpolate import make_interp_spline
                    sorted_indices = np.argsort(fitted)
                    fitted_sorted = fitted[sorted_indices]
                    residuals_sorted = residuals[sorted_indices]
                    
                    # Create smooth line
                    spline = make_interp_spline(fitted_sorted, residuals_sorted, k=3)
                    fitted_smooth = np.linspace(fitted_sorted.min(), fitted_sorted.max(), 100)
                    residuals_smooth = spline(fitted_smooth)
                    ax.plot(fitted_smooth, residuals_smooth, color='red', linewidth=2)
                except:
                    pass
        
        # Panel B: Q-Q plot
        ax = axes[1, 0]
        
        if 'hypothesis_3' in statistical_results:
            trt_results = statistical_results['hypothesis_3'].get('total_reading_time', {})
            full_model = trt_results.get('full', {})
            
            if 'residuals' in full_model:
                residuals = full_model['residuals']
                stats.probplot(residuals, dist="norm", plot=ax)
                ax.set_title('B. Q-Q Plot (Normal)')
                ax.grid(True, alpha=0.3)
        
        # Panel C: Cross-validation results
        ax = axes[0, 1]
        
        if 'validation' in statistical_results:
            validation_results = statistical_results['validation']
            
            measures = []
            cv_scores = []
            cv_stds = []
            
            for measure, val_data in validation_results.items():
                if 'mean_cv_score' in val_data:
                    measures.append(measure.replace('_', ' ').title())
                    cv_scores.append(val_data['mean_cv_score'])
                    cv_stds.append(val_data.get('std_cv_score', 0))
            
            if measures and cv_scores:
                bars = ax.bar(measures, cv_scores, yerr=cv_stds, capsize=5,
                             color=self.group_colors['Dyslexic'], alpha=0.7)
                ax.set_xlabel('Eye Measure')
                ax.set_ylabel('Cross-Validation R²')
                ax.set_title('C. Cross-Validation Performance')
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylim(0, 1)
                
                # Add value labels
                for bar, score in zip(bars, cv_scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # Panel D: Feature importance
        ax = axes[1, 1]
        
        if 'validation' in statistical_results:
            # Get feature importance from cross-validation
            feature_importance = {}
            
            for measure, val_data in validation_results.items():
                importance_data = val_data.get('feature_importance', {})
                for feature, importance in importance_data.items():
                    if feature not in feature_importance:
                        feature_importance[feature] = []
                    feature_importance[feature].append(importance)
            
            # Average across measures
            features = []
            importances = []
            for feature, importance_list in feature_importance.items():
                if importance_list:
                    features.append(feature.replace('_c', '').replace('_', ' ').title())
                    importances.append(np.mean(importance_list))
            
            if features and importances:
                # Sort by importance
                sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
                features, importances = zip(*sorted_pairs)
                
                bars = ax.barh(features, importances, color=self.group_colors['Control'], alpha=0.7)
                ax.set_xlabel('Feature Importance (|Coefficient|)')
                ax.set_title('D. Feature Importance')
                
                # Add value labels
                for bar, importance in zip(bars, importances):
                    width = bar.get_width()
                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                           f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.figure_dir / f'figure_5_diagnostics.{self.format}'
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def _create_correlation_matrix(self, data: pd.DataFrame) -> str:
        """Supplementary: Feature correlation matrix"""
        
        # Select numeric columns for correlation
        numeric_cols = ['word_length', 'log_frequency', 'launch_site_distance', 
                       'predictability', 'first_fixation_duration', 'gaze_duration', 
                       'total_reading_time', 'preview_benefit_score']
        
        available_cols = [col for col in numeric_cols if col in data.columns]
        
        if len(available_cols) < 3:
            logger.warning("Insufficient numeric columns for correlation matrix")
            return ""
        
        # Compute correlation matrix
        corr_data = data[available_cols].corr()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, ax=ax, fmt='.2f')
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.figure_dir / f'supplementary_correlations.{self.format}'
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def _create_distribution_figure(self, data: pd.DataFrame) -> str:
        """Supplementary: Variable distributions by group"""
        
        # Key variables to plot
        variables = ['word_length', 'log_frequency', 'predictability', 'total_reading_time']
        available_vars = [var for var in variables if var in data.columns]
        
        if len(available_vars) < 2:
            logger.warning("Insufficient variables for distribution plots")
            return ""
        
        # Create subplots
        n_vars = len(available_vars)
        n_cols = 2
        n_rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Variable Distributions by Group', fontsize=16, fontweight='bold')
        
        # Prepare data
        plot_data = data.copy()
        plot_data['Group'] = plot_data['dyslexic'].map({True: 'Dyslexic', False: 'Control'})
        
        for i, var in enumerate(available_vars):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Create overlapping histograms
            for group in ['Control', 'Dyslexic']:
                group_data = plot_data[plot_data['Group'] == group][var].dropna()
                ax.hist(group_data, alpha=0.7, label=group, bins=30, 
                       color=self.group_colors[group], density=True)
            
            ax.set_xlabel(var.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.set_title(f'{var.replace("_", " ").title()} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(available_vars), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        
        # Save figure
        filepath = self.figure_dir / f'supplementary_distributions.{self.format}'
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        return str(filepath)
    
    def create_interactive_dashboard(self, data: pd.DataFrame) -> str:
        """Create interactive Plotly dashboard"""
        
        # Prepare data
        plot_data = data.copy()
        plot_data['Group'] = plot_data['dyslexic'].map({True: 'Dyslexic', False: 'Control'})
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Length vs Reading Time', 'Frequency vs Reading Time',
                           'Predictability vs Reading Time', 'Feature Distributions'],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Panel 1: Length effect
        for group in ['Control', 'Dyslexic']:
            group_data = plot_data[plot_data['Group'] == group]
            fig.add_trace(
                go.Scatter(
                    x=group_data['word_length'],
                    y=group_data['total_reading_time'],
                    mode='markers',
                    name=f'{group} (Length)',
                    marker=dict(color=self.group_colors[group], size=5, opacity=0.6),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Panel 2: Frequency effect
        for group in ['Control', 'Dyslexic']:
            group_data = plot_data[plot_data['Group'] == group]
            fig.add_trace(
                go.Scatter(
                    x=group_data['log_frequency'],
                    y=group_data['total_reading_time'],
                    mode='markers',
                    name=f'{group} (Freq)',
                    marker=dict(color=self.group_colors[group], size=5, opacity=0.6),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Panel 3: Predictability effect
        if 'predictability' in plot_data.columns:
            for group in ['Control', 'Dyslexic']:
                group_data = plot_data[plot_data['Group'] == group]
                fig.add_trace(
                    go.Scatter(
                        x=group_data['predictability'],
                        y=group_data['total_reading_time'],
                        mode='markers',
                        name=f'{group} (Pred)',
                        marker=dict(color=self.group_colors[group], size=5, opacity=0.6),
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Panel 4: Histograms
        for group in ['Control', 'Dyslexic']:
            group_data = plot_data[plot_data['Group'] == group]
            fig.add_trace(
                go.Histogram(
                    x=group_data['total_reading_time'],
                    name=f'{group} (Hist)',
                    marker=dict(color=self.group_colors[group], opacity=0.7),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Interactive Dyslexia Reading Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Word Length", row=1, col=1)
        fig.update_xaxes(title_text="Log Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Predictability", row=2, col=1)
        fig.update_xaxes(title_text="Total Reading Time (ms)", row=2, col=2)
        
        fig.update_yaxes(title_text="Total Reading Time (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Total Reading Time (ms)", row=1, col=2)
        fig.update_yaxes(title_text="Total Reading Time (ms)", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        # Save interactive figure
        filepath = self.figure_dir / 'interactive_dashboard.html'
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def create_summary_table(self, statistical_results: Dict) -> pd.DataFrame:
        """Create publication-ready summary table"""
        
        summary_data = []
        
        # Extract key results from statistical analysis
        if 'hypothesis_1' in statistical_results:
            h1_results = statistical_results['hypothesis_1']
            
            for measure, model_data in h1_results.items():
                if isinstance(model_data, dict) and 'coefficients' in model_data:
                    coeffs = model_data['coefficients']
                    pvals = model_data.get('pvalues', {})
                    
                    for predictor in ['word_length_c', 'log_frequency_c', 'launch_site_distance_c', 'predictability_c']:
                        if predictor in coeffs:
                            summary_data.append({
                                'Hypothesis': 'H1: Feature Effects',
                                'Eye Measure': measure.replace('_', ' ').title(),
                                'Predictor': predictor.replace('_c', '').replace('_', ' ').title(),
                                'Coefficient': f"{coeffs[predictor]:.3f}",
                                'P-value': f"{pvals.get(predictor, np.nan):.3f}" if predictor in pvals else "—",
                                'Significance': "***" if pvals.get(predictor, 1) < 0.001 else 
                                              "**" if pvals.get(predictor, 1) < 0.01 else
                                              "*" if pvals.get(predictor, 1) < 0.05 else "ns"
                            })
        
        # Add interaction effects from Hypothesis 2
        if 'hypothesis_2' in statistical_results:
            h2_results = statistical_results['hypothesis_2']
            
            for measure, model_data in h2_results.items():
                if 'interactions' in model_data:
                    interactions = model_data['interactions']
                    
                    for predictor, effect in interactions.items():
                        summary_data.append({
                            'Hypothesis': 'H2: Group Interactions',
                            'Eye Measure': measure.replace('_', ' ').title(),
                            'Predictor': f"{predictor.replace('_c', '').replace('_', ' ').title()} × Group",
                            'Coefficient': f"{effect:.3f}",
                            'P-value': "—",  # Would need to extract from model
                            'Significance': "—"
                        })
        
        # Add variance decomposition from Hypothesis 3
        if 'hypothesis_3' in statistical_results:
            h3_results = statistical_results['hypothesis_3']
            
            for measure, model_data in h3_results.items():
                if 'variance_explained' in model_data:
                    var_data = model_data['variance_explained']
                    
                    summary_data.append({
                        'Hypothesis': 'H3: Variance Decomposition',
                        'Eye Measure': measure.replace('_', ' ').title(),
                        'Predictor': 'Baseline R²',
                        'Coefficient': f"{var_data.get('baseline_r2', 0):.3f}",
                        'P-value': "—",
                        'Significance': "—"
                    })
                    
                    summary_data.append({
                        'Hypothesis': 'H3: Variance Decomposition',
                        'Eye Measure': measure.replace('_', ' ').title(),
                        'Predictor': 'Full Model R²',
                        'Coefficient': f"{var_data.get('full_r2', 0):.3f}",
                        'P-value': "—",
                        'Significance': "—"
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary table
        filepath = self.figure_dir / 'summary_table.csv'
        summary_df.to_csv(filepath, index=False)
        
        return summary_df
    
    def save_all_figures(self, data: pd.DataFrame, statistical_results: Dict) -> Dict[str, str]:
        """Save all figures and return file paths"""
        
        logger.info("Generating all publication figures...")
        
        # Create main publication figures
        figure_paths = self.create_publication_figures(data, statistical_results)
        
        # Create interactive dashboard
        figure_paths['interactive'] = self.create_interactive_dashboard(data)
        
        # Create summary table
        summary_table = self.create_summary_table(statistical_results)
        figure_paths['summary_table'] = str(self.figure_dir / 'summary_table.csv')
        
        logger.info(f"All figures saved to {self.figure_dir}")
        return figure_paths

def main():
    """Test visualization with sample data"""
    pass

if __name__ == "__main__":
    main()