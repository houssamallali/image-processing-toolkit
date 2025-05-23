#!/usr/bin/env python3
"""
Interactive Image Processing CLI - Educational Explorer
A beautiful, inspiring command-line interface for learning image processing concepts.

Author: Professional Image Processing Project
Date: 2024
"""

import os
import sys
import random
import subprocess
from pathlib import Path
from datetime import datetime
import importlib.util

# Color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    PURPLE = '\033[35m'
    ORANGE = '\033[33m'

class ImageProcessingCLI:
    def __init__(self):
        self.current_tp = None
        self.session_start = datetime.now()
        self.commands_used = []
        self.achievements = set()

        # TP structure with beautiful descriptions
        self.tps = {
            '01': {
                'name': 'Basic Operations',
                'description': 'Foundation of digital image representation and manipulation',
                'icon': '🎨',
                'techniques': ['image_loading', 'rgb_analysis', 'compression', 'visualization'],
                'color': Colors.CYAN
            },
            '02': {
                'name': 'Spatial Filtering',
                'description': 'Convolution-based image enhancement and noise reduction',
                'icon': '🔍',
                'techniques': ['convolution', 'lowpass_filtering', 'highpass_filtering', 'enhancement'],
                'color': Colors.GREEN
            },
            '03': {
                'name': 'Fourier Analysis',
                'description': 'Frequency domain transformations and filtering',
                'icon': '🌊',
                'techniques': ['fft_2d', 'frequency_filtering', 'inverse_fourier', 'applications'],
                'color': Colors.BLUE
            },
            '04': {
                'name': 'Image Segmentation',
                'description': 'Partitioning images into meaningful regions',
                'icon': '🧩',
                'techniques': ['thresholding', 'k_means', 'otsu_method', 'advanced_clustering'],
                'color': Colors.PURPLE
            },
            '05': {
                'name': 'Histogram Enhancement',
                'description': 'Intensity distribution manipulation for better contrast',
                'icon': '📊',
                'techniques': ['histogram_equalization', 'contrast_stretching', 'gamma_correction', 'matching'],
                'color': Colors.YELLOW
            },
            '06': {
                'name': 'Image Restoration',
                'description': 'Recovering degraded images through deconvolution',
                'icon': '🔧',
                'techniques': ['deblurring', 'psf_estimation', 'wiener_filtering', 'iterative_methods'],
                'color': Colors.ORANGE
            },
            '07': {
                'name': 'Image Registration',
                'description': 'Aligning multiple images for comparison and analysis',
                'icon': '🎯',
                'techniques': ['feature_matching', 'transformation_estimation', 'alignment', 'evaluation'],
                'color': Colors.RED
            },
            '08': {
                'name': 'Noise and Filtering',
                'description': 'Understanding and removing various types of image noise',
                'icon': '🔇',
                'techniques': ['noise_modeling', 'statistical_analysis', 'adaptive_filtering', 'evaluation'],
                'color': Colors.CYAN
            },
            '09': {
                'name': 'Medical Segmentation',
                'description': 'Specialized techniques for medical image analysis',
                'icon': '🏥',
                'techniques': ['follicle_detection', 'morphological_operations', 'region_growing', 'validation'],
                'color': Colors.GREEN
            },
            '10': {
                'name': 'Shape Classification',
                'description': 'Machine learning approaches for shape recognition',
                'icon': '🤖',
                'techniques': ['feature_extraction', 'svm_classification', 'neural_networks', 'evaluation'],
                'color': Colors.BLUE
            },
            '11': {
                'name': 'Multiscale Analysis',
                'description': 'Pyramid decomposition and multi-resolution processing',
                'icon': '🏔️',
                'techniques': ['pyramid_construction', 'wavelet_transform', 'scale_space', 'reconstruction'],
                'color': Colors.PURPLE
            }
        }

        # Inspirational quotes
        self.quotes = [
            "Vision is the art of seeing what is invisible to others. - Jonathan Swift",
            "The eye sees only what the mind is prepared to comprehend. - Robertson Davies",
            "In the world of pixels, every image tells a story waiting to be discovered.",
            "Image processing is the bridge between human perception and machine understanding.",
            "Every filter applied is a new perspective gained.",
            "In the frequency domain, we see the hidden rhythms of visual information.",
            "Segmentation is the art of finding meaning in the chaos of pixels.",
            "Through enhancement, we reveal the beauty hidden in darkness.",
            "Restoration brings back memories thought lost forever.",
            "In registration, we align not just images, but understanding itself."
        ]

        # Technical definitions
        self.definitions = {
            'pixel': "The smallest unit of a digital image, representing a single point of color/intensity.",
            'kernel': "A small matrix used in convolution operations to apply filters to images.",
            'convolution': "A mathematical operation that applies a kernel to each pixel neighborhood.",
            'fourier_transform': "Mathematical technique to decompose signals into frequency components.",
            'histogram': "A graph showing the distribution of pixel intensities in an image.",
            'segmentation': "The process of partitioning an image into meaningful regions or objects.",
            'enhancement': "Techniques to improve image quality and visual appearance.",
            'restoration': "Methods to recover original image from degraded observations.",
            'registration': "The process of aligning multiple images of the same scene.",
            'noise': "Random variations in pixel values that degrade image quality.",
            'edge': "Boundaries between regions with different intensities or colors.",
            'gradient': "Rate of change in pixel intensity, used for edge detection.",
            'morphology': "Image processing based on shape and structure analysis.",
            'threshold': "A value used to separate pixels into different classes.",
            'clustering': "Grouping pixels with similar characteristics together.",
            'feature': "Distinctive characteristics extracted from images for analysis.",
            'filter': "An operation that modifies pixel values based on local neighborhoods.",
            'frequency_domain': "Representation of images in terms of frequency components.",
            'spatial_domain': "Standard representation of images as 2D pixel arrays.",
            'psf': "Point Spread Function - describes how a point source is blurred by imaging system."
        }

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_banner(self):
        """Display the beautiful welcome banner."""
        banner = f"""
{Colors.BOLD}{Colors.PURPLE}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██╗███╗   ███╗ █████╗  ██████╗ ███████╗██████╗ ██████╗ ██╗███████╗       ║
║    ██║████╗ ████║██╔══██╗██╔════╝ ██╔════╝██╔══██╗██╔══██╗██║██╔════╝       ║
║    ██║██╔████╔██║███████║██║  ███╗█████╗  ██████╔╝██████╔╝██║█████╗         ║
║    ██║██║╚██╔╝██║██╔══██║██║   ██║██╔══╝  ██╔══██╗██╔══██╗██║██╔══╝         ║
║    ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝███████╗██║  ██║██║  ██║██║███████╗       ║
║    ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝       ║
║                                                                              ║
║                    ✨ ELEGANT VISUAL PROCESSING STUDIO ✨                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.END}

{Colors.BOLD}{Colors.CYAN}✨ Welcome to IMAGERRIE - Where Art Meets Science! ✨{Colors.END}

{Colors.GREEN}� Explore 11 elegant modules covering the entire spectrum of visual processing
🎭 Dive deep into the artistry of algorithms and mathematical beauty
� Visualize concepts through stunning demonstrations and graceful plots
� Learn through refined examples and sophisticated applications{Colors.END}

{Colors.PURPLE}💫 "{random.choice(self.quotes)}"{Colors.END}

{Colors.BOLD}{Colors.PURPLE}    🌹 ═══ Crafted with Elegance and Precision ═══ 🌹{Colors.END}
"""
        print(banner)

    def print_help(self):
        """Display comprehensive help information."""
        help_text = f"""
{Colors.BOLD}{Colors.CYAN}🚀 NAVIGATION COMMANDS{Colors.END}
{Colors.GREEN}
📚 Content Access:
  tp<number>              - Enter specific TP (e.g., tp1, tp2, ..., tp11)
  list                    - Show all available TPs
  explain <concept>       - Get theoretical explanation
  define <term>           - Get definition of technical term
  show plots             - Display available visualizations
  code <technique>       - Show implementation code
  demo <technique>       - Run interactive demonstration

🔍 Discovery:
  search <keyword>       - Find content across all TPs
  random                 - Explore random concept
  tip                    - Get tip of the day
  achievements           - View your learning progress

🧭 Navigation:
  back                   - Go back to previous level
  home                   - Return to main menu
  help                   - Show this help message
  clear                  - Clear screen
  exit/quit              - Exit the application

📊 Session Info:
  status                 - Show current session information
  history                - View command history
{Colors.END}

{Colors.YELLOW}💡 Pro Tips:
• Use tab completion for commands (if supported by your terminal)
• Commands are case-insensitive
• You can use partial matches for TP names (e.g., 'basic' for basic_operations)
• Type 'random' for serendipitous learning discoveries!{Colors.END}
"""
        print(help_text)

    def print_tp_overview(self):
        """Display beautiful overview of all TPs."""
        print(f"\n{Colors.BOLD}{Colors.HEADER}🎓 IMAGE PROCESSING LEARNING MODULES{Colors.END}\n")

        for tp_id, tp_info in self.tps.items():
            color = tp_info['color']
            icon = tp_info['icon']
            name = tp_info['name']
            desc = tp_info['description']

            print(f"{color}{Colors.BOLD}{icon} TP{tp_id}: {name}{Colors.END}")
            print(f"   {Colors.CYAN}{desc}{Colors.END}")
            print(f"   {Colors.YELLOW}Techniques: {', '.join(tp_info['techniques'])}{Colors.END}\n")

    def get_random_tip(self):
        """Return a random educational tip."""
        tips = [
            "🔍 Convolution is like sliding a magnifying glass over an image to reveal patterns!",
            "🌊 The Fourier Transform reveals the 'musical notes' that make up an image.",
            "🎯 Histogram equalization spreads pixel intensities like butter on toast - evenly!",
            "🧩 Image segmentation is like solving a jigsaw puzzle - finding pieces that belong together.",
            "🔧 Image restoration is digital archaeology - recovering lost details from degraded images.",
            "📊 A histogram is an image's fingerprint - it tells you the story of its intensities.",
            "🎨 Edge detection finds the 'outlines' that our brain uses to recognize objects.",
            "🔄 Morphological operations are like digital sculpting tools for binary images.",
            "🎭 Filters are like Instagram effects, but with mathematical precision!",
            "🌈 Color spaces are different languages for describing the same visual information."
        ]
        return random.choice(tips)

    def search_content(self, keyword):
        """Search for content across all TPs."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}🔍 Searching for '{keyword}'...{Colors.END}\n")

        results = []
        keyword_lower = keyword.lower()

        # Search through TP names and descriptions
        for tp_id, tp_info in self.tps.items():
            if (keyword_lower in tp_info['name'].lower() or
                keyword_lower in tp_info['description'].lower()):
                results.append(f"{tp_info['icon']} TP{tp_id}: {tp_info['name']}")

        # Search through techniques
        for tp_id, tp_info in self.tps.items():
            for technique in tp_info['techniques']:
                if keyword_lower in technique.lower():
                    results.append(f"   🔧 {technique} (in TP{tp_id}: {tp_info['name']})")

        if results:
            print(f"{Colors.GREEN}Found {len(results)} results:{Colors.END}")
            for result in results:
                print(f"  {result}")
        else:
            print(f"{Colors.YELLOW}No results found for '{keyword}'. Try broader terms like 'filter', 'transform', or 'enhancement'.{Colors.END}")

    def show_achievements(self):
        """Display user achievements and progress."""
        print(f"\n{Colors.BOLD}{Colors.PURPLE}🏆 YOUR LEARNING ACHIEVEMENTS{Colors.END}\n")

        total_commands = len(self.commands_used)
        unique_commands = len(set(self.commands_used))
        session_time = datetime.now() - self.session_start

        print(f"{Colors.GREEN}📊 Session Statistics:{Colors.END}")
        print(f"   ⏱️  Session Duration: {session_time}")
        print(f"   🎯 Commands Used: {total_commands}")
        print(f"   🌟 Unique Commands: {unique_commands}")

        # Achievement logic
        achievements = []
        if total_commands >= 10:
            achievements.append("🎓 Explorer - Used 10+ commands")
        if unique_commands >= 5:
            achievements.append("🔍 Discoverer - Tried 5+ different commands")
        if 'search' in self.commands_used:
            achievements.append("🕵️ Detective - Used search functionality")
        if any(cmd.startswith('tp') for cmd in self.commands_used):
            achievements.append("🚀 Navigator - Explored TP modules")

        if achievements:
            print(f"\n{Colors.YELLOW}🏅 Unlocked Achievements:{Colors.END}")
            for achievement in achievements:
                print(f"   {achievement}")
        else:
            print(f"\n{Colors.CYAN}💡 Keep exploring to unlock achievements!{Colors.END}")

    def show_status(self):
        """Show current session status."""
        session_time = datetime.now() - self.session_start
        print(f"\n{Colors.BOLD}{Colors.CYAN}📊 SESSION STATUS{Colors.END}")
        print(f"{Colors.GREEN}Current Location: {f'TP{self.current_tp}' if self.current_tp else 'Home'}{Colors.END}")
        print(f"{Colors.GREEN}Session Duration: {session_time}{Colors.END}")
        print(f"{Colors.GREEN}Commands Used: {len(self.commands_used)}{Colors.END}")

    def show_history(self):
        """Show command history."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}📜 COMMAND HISTORY{Colors.END}")
        if self.commands_used:
            for i, cmd in enumerate(self.commands_used[-10:], 1):  # Show last 10 commands
                print(f"{Colors.YELLOW}{i:2d}. {cmd}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}No commands used yet.{Colors.END}")

    def show_tp_details(self, tp_id):
        """Show detailed information about a specific TP."""
        tp_info = self.tps[tp_id]
        color = tp_info['color']

        print(f"\n{color}{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{color}{Colors.BOLD}{tp_info['icon']} TP{tp_id}: {tp_info['name'].upper()}{Colors.END}")
        print(f"{color}{Colors.BOLD}{'='*80}{Colors.END}")

        print(f"\n{Colors.CYAN}📖 Description:{Colors.END}")
        print(f"   {tp_info['description']}")

        print(f"\n{Colors.GREEN}🔧 Available Techniques:{Colors.END}")
        for i, technique in enumerate(tp_info['techniques'], 1):
            print(f"   {i}. {technique.replace('_', ' ').title()}")

        print(f"\n{Colors.YELLOW}💡 Available Commands:{Colors.END}")
        print(f"   • explain <technique>  - Get theoretical explanation")
        print(f"   • code <technique>     - View implementation")
        print(f"   • demo <technique>     - Run demonstration")
        print(f"   • plots               - Show visualizations")
        print(f"   • back                - Return to main menu")

    def explain_concept(self, concept):
        """Provide theoretical explanation of a concept."""
        explanations = {
            'convolution': """
🔍 CONVOLUTION - The Heart of Image Processing

Convolution is a mathematical operation that combines two functions to produce a third.
In image processing, it's like sliding a small matrix (kernel) over the entire image,
computing the weighted sum at each position.

🧮 Mathematical Definition:
(f * g)(x,y) = ΣΣ f(i,j) × g(x-i, y-j)

🎯 Key Applications:
• Blurring and smoothing (Gaussian kernel)
• Edge detection (Sobel, Laplacian kernels)
• Sharpening (high-pass kernels)
• Noise reduction (averaging kernels)

💡 Think of it as: Each pixel's new value is influenced by its neighbors,
weighted according to the kernel pattern!
""",
            'fourier_transform': """
🌊 FOURIER TRANSFORM - Seeing Images in Frequency Domain

The Fourier Transform decomposes an image into its frequency components,
revealing periodic patterns and enabling frequency-based filtering.

🧮 2D Fourier Transform:
F(u,v) = ΣΣ f(x,y) × e^(-j2π(ux/M + vy/N))

🎯 Key Insights:
• Low frequencies = smooth regions, overall structure
• High frequencies = edges, fine details, noise
• Phase contains shape information
• Magnitude contains energy distribution

💡 Applications:
• Frequency domain filtering
• Image compression (JPEG)
• Pattern recognition
• Noise analysis
""",
            'histogram_equalization': """
📊 HISTOGRAM EQUALIZATION - Spreading the Light

Histogram equalization redistributes pixel intensities to achieve
uniform distribution across the entire intensity range.

🎯 Goal: Transform image so its histogram is approximately flat

🧮 Process:
1. Compute cumulative distribution function (CDF)
2. Normalize CDF to [0, 255] range
3. Map each pixel through the transformation

💡 Effect:
• Enhances contrast in low-contrast images
• Reveals hidden details in dark/bright regions
• May over-enhance some regions

🌟 Best for: Medical images, satellite imagery, low-light photos
"""
        }

        concept_key = concept.lower().replace(' ', '_')
        if concept_key in explanations:
            print(f"\n{Colors.BOLD}{Colors.BLUE}{explanations[concept_key]}{Colors.END}")
        else:
            print(f"\n{Colors.YELLOW}📚 Explanation for '{concept}' not found. Try: convolution, fourier_transform, histogram_equalization{Colors.END}")

    def define_term(self, term):
        """Provide definition of a technical term."""
        term_key = term.lower().replace(' ', '_')
        if term_key in self.definitions:
            print(f"\n{Colors.BOLD}{Colors.CYAN}📖 DEFINITION: {term.upper()}{Colors.END}")
            print(f"{Colors.GREEN}{self.definitions[term_key]}{Colors.END}")
        else:
            print(f"\n{Colors.YELLOW}📖 Definition for '{term}' not found.{Colors.END}")
            print(f"{Colors.CYAN}Available terms: {', '.join(list(self.definitions.keys())[:10])}...{Colors.END}")

    def show_code_example(self, technique):
        """Show code implementation for a technique."""
        code_examples = {
            'convolution': """
🐍 CONVOLUTION IMPLEMENTATION

```python
import numpy as np
from scipy.signal import convolve2d

def apply_convolution(image, kernel):
    \"\"\"Apply convolution with proper boundary handling.\"\"\"
    if len(image.shape) == 3:
        # Apply to each channel separately
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = convolve2d(
                image[:, :, i], kernel,
                mode='same', boundary='symm'
            )
        return result
    else:
        return convolve2d(image, kernel, mode='same', boundary='symm')

# Example kernels
gaussian_3x3 = (1/16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
```
""",
            'histogram_equalization': """
🐍 HISTOGRAM EQUALIZATION IMPLEMENTATION

```python
import numpy as np

def histogram_equalization(image):
    \"\"\"Apply histogram equalization to enhance contrast.\"\"\"
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = image.copy()

    # Compute histogram
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])

    # Compute cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]

    # Apply transformation
    equalized = np.interp(gray.flatten(), bins[:-1], cdf_normalized)

    return equalized.reshape(gray.shape).astype(np.uint8)
```
"""
        }

        technique_key = technique.lower().replace(' ', '_')
        if technique_key in code_examples:
            print(f"\n{Colors.GREEN}{code_examples[technique_key]}{Colors.END}")
        else:
            print(f"\n{Colors.YELLOW}💻 Code example for '{technique}' not found. Try: convolution, histogram_equalization{Colors.END}")

    def explore_random(self):
        """Explore a random concept for serendipitous learning."""
        random_tp = random.choice(list(self.tps.keys()))
        tp_info = self.tps[random_tp]
        random_technique = random.choice(tp_info['techniques'])

        print(f"\n{Colors.BOLD}{Colors.PURPLE}🎲 RANDOM DISCOVERY{Colors.END}")
        print(f"{tp_info['color']}{tp_info['icon']} TP{random_tp}: {tp_info['name']}{Colors.END}")
        print(f"{Colors.CYAN}🔧 Technique: {random_technique.replace('_', ' ').title()}{Colors.END}")
        print(f"{Colors.YELLOW}💡 {self.get_random_tip()}{Colors.END}")

    def run_demo(self, technique):
        """Run an interactive demonstration."""
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎬 RUNNING DEMO: {technique.upper()}{Colors.END}")

        if self.current_tp:
            tp_dir = f"{self.current_tp.zfill(2)}_" + self.tps[self.current_tp]['name'].lower().replace(' ', '_')
            main_file = Path(tp_dir) / "main.py"

            if main_file.exists():
                print(f"{Colors.CYAN}🚀 Executing {main_file}...{Colors.END}")
                try:
                    subprocess.run([sys.executable, str(main_file)], check=True)
                    print(f"{Colors.GREEN}✅ Demo completed successfully!{Colors.END}")
                except subprocess.CalledProcessError as e:
                    print(f"{Colors.RED}❌ Demo failed: {e}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}📁 Demo file not found: {main_file}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}💡 Please select a TP first (e.g., 'tp1') to run demos.{Colors.END}")

    def show_plots(self):
        """Display available plots for current TP."""
        if not self.current_tp:
            print(f"{Colors.YELLOW}💡 Please select a TP first to view plots.{Colors.END}")
            return

        tp_name = self.tps[self.current_tp]['name'].lower().replace(' ', '_')
        plots_dir = Path("plots") / f"{self.current_tp.zfill(2)}_{tp_name}"

        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            if plot_files:
                print(f"\n{Colors.BOLD}{Colors.YELLOW}📊 AVAILABLE PLOTS{Colors.END}")
                for i, plot_file in enumerate(plot_files, 1):
                    print(f"   {i}. {plot_file.name}")
                print(f"\n{Colors.CYAN}💡 Plots are saved in: {plots_dir}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}📊 No plots found in {plots_dir}{Colors.END}")
        else:
            print(f"{Colors.YELLOW}📁 Plots directory not found: {plots_dir}{Colors.END}")

    def run(self):
        """Main application loop."""
        self.clear_screen()
        self.print_banner()

        print(f"\n{Colors.BOLD}Type 'help' for commands or 'list' to see all modules. Let's begin your journey!{Colors.END}\n")

        while True:
            try:
                # Show current location
                location = f"TP{self.current_tp}" if self.current_tp else "Home"
                prompt = f"{Colors.BOLD}{Colors.BLUE}[{location}]>{Colors.END} "

                command = input(prompt).strip().lower()

                if not command:
                    continue

                self.commands_used.append(command)

                # Handle exit commands
                if command in ['exit', 'quit', 'q']:
                    self.show_achievements()
                    print(f"\n{Colors.GREEN}✨ Thank you for exploring image processing! Keep learning and creating! ✨{Colors.END}\n")
                    break

                # Handle navigation commands
                elif command == 'help':
                    self.print_help()

                elif command == 'clear':
                    self.clear_screen()
                    self.print_banner()

                elif command == 'list':
                    self.print_tp_overview()

                elif command == 'home':
                    self.current_tp = None
                    print(f"{Colors.GREEN}🏠 Returned to home{Colors.END}")

                elif command == 'tip':
                    print(f"\n{Colors.YELLOW}💡 {self.get_random_tip()}{Colors.END}\n")

                elif command == 'achievements':
                    self.show_achievements()

                elif command == 'status':
                    self.show_status()

                elif command == 'history':
                    self.show_history()

                elif command.startswith('search '):
                    keyword = command[7:]
                    self.search_content(keyword)

                elif command.startswith('tp') and len(command) >= 3:
                    tp_num = command[2:].zfill(2)
                    if tp_num in self.tps:
                        self.current_tp = tp_num
                        self.show_tp_details(tp_num)
                    else:
                        print(f"{Colors.RED}❌ TP{tp_num} not found. Use 'list' to see available TPs.{Colors.END}")

                elif command == 'random':
                    self.explore_random()

                elif command.startswith('explain '):
                    concept = command[8:]
                    self.explain_concept(concept)

                elif command.startswith('define '):
                    term = command[7:]
                    self.define_term(term)

                elif command.startswith('code '):
                    technique = command[5:]
                    self.show_code_example(technique)

                elif command.startswith('demo '):
                    technique = command[5:]
                    self.run_demo(technique)

                elif command == 'plots' or command == 'show plots':
                    self.show_plots()

                elif command == 'back':
                    if self.current_tp:
                        self.current_tp = None
                        print(f"{Colors.GREEN}🔙 Returned to main menu{Colors.END}")
                    else:
                        print(f"{Colors.YELLOW}💡 Already at main menu{Colors.END}")

                # Handle TP name shortcuts
                elif command in ['basic', 'basic_operations']:
                    self.current_tp = '01'
                    self.show_tp_details('01')

                elif command in ['spatial', 'filtering', 'spatial_filtering']:
                    self.current_tp = '02'
                    self.show_tp_details('02')

                elif command in ['fourier', 'frequency', 'fourier_analysis']:
                    self.current_tp = '03'
                    self.show_tp_details('03')

                elif command in ['segmentation', 'image_segmentation']:
                    self.current_tp = '04'
                    self.show_tp_details('04')

                elif command in ['histogram', 'enhancement', 'histogram_enhancement']:
                    self.current_tp = '05'
                    self.show_tp_details('05')

                elif command in ['restoration', 'image_restoration']:
                    self.current_tp = '06'
                    self.show_tp_details('06')

                elif command in ['registration', 'image_registration']:
                    self.current_tp = '07'
                    self.show_tp_details('07')

                elif command in ['noise', 'noise_filtering']:
                    self.current_tp = '08'
                    self.show_tp_details('08')

                elif command in ['medical', 'medical_segmentation']:
                    self.current_tp = '09'
                    self.show_tp_details('09')

                elif command in ['shape', 'classification', 'shape_classification']:
                    self.current_tp = '10'
                    self.show_tp_details('10')

                elif command in ['multiscale', 'pyramid', 'multiscale_analysis']:
                    self.current_tp = '11'
                    self.show_tp_details('11')

                else:
                    print(f"{Colors.YELLOW}❓ Unknown command '{command}'. Type 'help' for available commands.{Colors.END}")

            except KeyboardInterrupt:
                print(f"\n\n{Colors.GREEN}✨ Goodbye! Keep exploring the world of image processing! ✨{Colors.END}\n")
                break
            except Exception as e:
                print(f"{Colors.RED}❌ An error occurred: {e}{Colors.END}")

def main():
    """Entry point for the CLI application."""
    try:
        cli = ImageProcessingCLI()
        cli.run()
    except Exception as e:
        print(f"{Colors.RED}❌ Failed to start CLI: {e}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()
