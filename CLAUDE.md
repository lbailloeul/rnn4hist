# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROOT/C++ project for analyzing Single Photon Sensor (SPS) histograms using RNN models for histogram fitting. The project focuses on photoelectron spectroscopy analysis, specifically fitting histograms with Generalized Poisson (GP) distributions to extract gain parameters from SPS data.

## Architecture

The codebase consists of two main directories:

- **data-generation/**: Contains C++ ROOT macros for data generation and analysis
- **cnn-rnn-models/**: Directory for machine learning models (currently empty)

### Core Components

**Header File:**
- `sps_fit.h`: Function declarations for GP fitting, statistical analysis, and canvas utilities

**Analysis Scripts:**
- `sps_fit.C`: Original GP fitting implementation with basic analysis functions
- `sps_hist.C`: Histogram generation script that creates synthetic SPS data across different μ_p (occupancy) values  
- `mu_p_analysis.C`: Advanced analysis script that performs systematic studies of gain measurements vs occupancy parameters

### Key Functions

**Generalized Poisson Fitting:**
- `generpoiss()`: Implements the GP probability distribution for SPS analysis
- `GP()`: Main fitting function that performs peak finding, Gaussian pre-fitting, and GP parameter estimation
- `params_stats()`: Handles statistical display and parameter visualization on ROOT canvases

**Analysis Workflow:**
1. `sps_hist.C` generates synthetic histograms with varying μ_p values (0.5 to 3.0)
2. `mu_p_analysis.C` systematically fits all generated histograms and produces comprehensive analysis plots
3. Results include gain measurements, ratios, differences, and RMSE calculations vs truth values

## Development Commands

### Compilation and Execution
```bash
# Compile and run histogram generation
root -l -q sps_hist.C

# Run analysis on generated histograms  
root -l -q mu_p_analysis.C

# Individual function execution (legacy)
root -l -q sps_fit.C
```

### File Dependencies
- ROOT framework must be installed and configured
- Generated ROOT files follow naming pattern: `histos_mu_X.XX.root`
- Analysis outputs: `mu_p_analysis_results.tsv`, `mu_p_analysis_summary.pdf/png`

## Key Parameters

**Physics Constants:**
- Truth gain: 12.2 (defined in `mu_p_analysis.C:39`)
- Crosstalk parameter (λ): ~0.1
- Number of photoelectron peaks fitted: typically 7-8

**Analysis Range:**
- μ_p (high LED): 0.5 to 3.0 in 30 steps
- μ_p (low LED): fixed at 0.5
- Histogram range: 0-300 ADU, 300 bins

## Data Flow

1. **Generation**: `sps_hist.C` creates multiple ROOT files with synthetic SPS histograms
2. **Fitting**: `mu_p_analysis.C` loads histograms, performs GP fits, extracts gain parameters
3. **Analysis**: Statistical analysis of gain vs occupancy, including error propagation and RMSE calculations
4. **Visualization**: Multi-panel plots showing fitted gains, ratios, differences, and RMSE metrics