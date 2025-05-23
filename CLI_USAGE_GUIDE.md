# 🎓 Image Processing CLI Explorer - Usage Guide

## 🚀 Quick Start

### Launch the CLI
```bash
# Method 1: Direct launch
python image_processing_cli.py

# Method 2: Using the launcher
python explore.py
```

## 🎯 Core Features

### 📚 Educational Content Access
- **`explain <concept>`** - Get detailed theoretical explanations
- **`define <term>`** - Quick definitions of technical terms
- **`code <technique>`** - View implementation examples
- **`demo <technique>`** - Run interactive demonstrations

### 🧭 Navigation System
- **`tp<number>`** - Enter specific TP (e.g., `tp1`, `tp2`, ..., `tp11`)
- **`list`** - Show all available TPs with descriptions
- **`back`** - Return to previous level
- **`home`** - Return to main menu

### 🔍 Discovery Tools
- **`search <keyword>`** - Find content across all TPs
- **`random`** - Explore random concepts for serendipitous learning
- **`tip`** - Get educational tips and insights

### 📊 Session Management
- **`status`** - View current session information
- **`history`** - See command history
- **`achievements`** - Track learning progress
- **`clear`** - Clear screen for fresh start

## 🎨 Example Learning Sessions

### Session 1: Understanding Convolution
```
[Home]> define convolution
[Home]> explain convolution
[Home]> code convolution
[Home]> tp2
[TP02]> demo convolution
[TP02]> plots
```

### Session 2: Exploring Fourier Analysis
```
[Home]> search fourier
[Home]> tp3
[TP03]> explain fourier_transform
[TP03]> code fourier_transform
[TP03]> demo fft_2d
```

### Session 3: Random Discovery
```
[Home]> random
[Home]> tip
[Home]> search edge
[Home]> explain edge_detection
```

## 🎯 TP Shortcuts

Instead of typing `tp<number>`, you can use these shortcuts:
- **`basic`** → TP01: Basic Operations
- **`spatial`** or **`filtering`** → TP02: Spatial Filtering
- **`fourier`** or **`frequency`** → TP03: Fourier Analysis
- **`segmentation`** → TP04: Image Segmentation
- **`histogram`** or **`enhancement`** → TP05: Histogram Enhancement
- **`restoration`** → TP06: Image Restoration
- **`registration`** → TP07: Image Registration
- **`noise`** → TP08: Noise and Filtering
- **`medical`** → TP09: Medical Segmentation
- **`shape`** or **`classification`** → TP10: Shape Classification
- **`multiscale`** or **`pyramid`** → TP11: Multiscale Analysis

## 📖 Available Definitions

The CLI includes definitions for these key terms:
- `pixel`, `kernel`, `convolution`, `fourier_transform`
- `histogram`, `segmentation`, `enhancement`, `restoration`
- `registration`, `noise`, `edge`, `gradient`
- `morphology`, `threshold`, `clustering`, `feature`
- `filter`, `frequency_domain`, `spatial_domain`, `psf`

## 🧮 Available Explanations

Detailed explanations are available for:
- **`convolution`** - Mathematical operation and applications
- **`fourier_transform`** - Frequency domain analysis
- **`histogram_equalization`** - Contrast enhancement technique

## 💻 Code Examples

Implementation examples available for:
- **`convolution`** - Complete convolution implementation
- **`histogram_equalization`** - Contrast enhancement code

## 🎬 Interactive Demos

Run demonstrations for any TP:
- **`demo <technique>`** - Executes the main.py for current TP
- Generates visualizations and saves to plots directory
- Shows real-time processing results

## 🏆 Achievement System

Unlock achievements by:
- **🎓 Explorer** - Use 10+ commands
- **🔍 Discoverer** - Try 5+ different commands
- **🕵️ Detective** - Use search functionality
- **🚀 Navigator** - Explore TP modules

## 💡 Pro Tips

1. **Case Insensitive**: All commands work in any case
2. **Partial Matching**: Use shortcuts like `basic` instead of `basic_operations`
3. **Tab Completion**: May work in some terminals
4. **Random Learning**: Use `random` for unexpected discoveries
5. **Progressive Learning**: Start with `list`, then explore specific TPs
6. **Visual Learning**: Always check `plots` after running demos

## 🎨 Visual Features

- **Colored Output**: Different colors for different content types
- **ASCII Art**: Beautiful welcome banner and formatting
- **Icons**: Emojis for visual appeal and quick recognition
- **Progress Tracking**: Breadcrumbs show current location
- **Inspirational Quotes**: Motivational messages about image processing

## 🔧 Technical Requirements

- **Python 3.6+**
- **Dependencies**: numpy, matplotlib, scikit-image, scipy
- **Terminal**: Works in any terminal with color support
- **Platform**: Cross-platform (Windows, macOS, Linux)

## 🎯 Learning Pathways

### Beginner Path
1. `list` → `tp1` → `explain` basic concepts
2. `tp2` → Learn about filtering
3. `tp5` → Understand enhancement

### Intermediate Path
1. `tp3` → Fourier analysis
2. `tp4` → Segmentation techniques
3. `tp6` → Image restoration

### Advanced Path
1. `tp10` → Machine learning classification
2. `tp11` → Multiscale analysis
3. `tp9` → Medical applications

## 🚀 Getting Started Checklist

- [ ] Launch CLI with `python image_processing_cli.py`
- [ ] Type `help` to see all commands
- [ ] Use `list` to see all TPs
- [ ] Try `define pixel` for your first definition
- [ ] Explore `tp1` for basic operations
- [ ] Use `random` for serendipitous discovery
- [ ] Check `achievements` to track progress
- [ ] Use `tip` for learning insights

## 🎓 Educational Philosophy

This CLI is designed to make learning image processing:
- **Interactive** - Hands-on exploration
- **Visual** - Beautiful, engaging interface
- **Progressive** - Build knowledge step by step
- **Inspiring** - Motivational quotes and achievements
- **Comprehensive** - Theory, code, and practice combined

Happy exploring! 🌟
