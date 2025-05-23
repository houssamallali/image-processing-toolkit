#!/usr/bin/env python3
import os
import sys
import importlib.util
import argparse


def available_tp_numbers():
    """Return sorted TP numbers found in the project."""
    numbers = []
    for d in os.listdir():
        if d.startswith('TP') and '_' in d:
            num = d[2:d.index('_')]
            if num.isdigit():
                numbers.append(int(num))
    return sorted(numbers)

def import_script(script_path):
    """Import a Python script as a module."""
    script_name = os.path.basename(script_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def list_scripts():
    """List all available TP scripts dynamically."""
    script_dirs = [d for d in os.listdir() if d.startswith('TP') and os.path.isdir(d)]
    all_scripts = {}
    
    for dir_name in script_dirs:
        all_scripts[dir_name] = []
        if os.path.exists(dir_name):
            for file in os.listdir(dir_name):
                if file.endswith('.py') and not file.startswith('__'):
                    all_scripts[dir_name].append(file)
    
    return all_scripts

def print_available_scripts():
    """Print all available scripts in a formatted way."""
    scripts = list_scripts()
    print("\nAvailable scripts:")
    print("=================")
    
    for dir_name, files in scripts.items():
        if files:
            print(f"\n{dir_name}:")
            for i, file in enumerate(sorted(files), 1):
                print(f"  {i}. {file}")
    
    print("\nExamples:")
    print("  python run.py --list")
    print("  python run.py TP2_Filtering/lowpass.py")
    print("  python run.py TP3_FourierAnalysis/inversefourier.py --save-plots")
    print("  python run.py --all-tp 2")

def run_script(script_path, save_plots=False):
    """Run a specific script and handle plot saving if requested."""
    original_dir = os.getcwd()
    
    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found.")
        return False
    
    script_dir = os.path.dirname(script_path)
    script_name = os.path.basename(script_path)
    
    # Change to the script's directory
    if script_dir:
        os.chdir(script_dir)
    
    print(f"Running: {script_path}")
    
    # If save_plots is True, modify matplotlib to save plots
    if save_plots:
        import matplotlib
        import matplotlib.pyplot as plt
        
        # Store the original show function
        original_show = plt.show
        
        # Create a directory for saving plots
        plots_dir = os.path.join(original_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Define a custom show function that saves plots
        def custom_show():
            base_name = os.path.splitext(script_name)[0]
            fig_nums = plt.get_fignums()
            for i, fig_num in enumerate(fig_nums):
                fig = plt.figure(fig_num)
                output_path = os.path.join(plots_dir, f"{base_name}_plot{i+1}.png")
                fig.savefig(output_path)
                print(f"Saved plot to: {output_path}")
            # Call the original show function
            original_show()
        
        # Replace the show function
        plt.show = custom_show
    
    try:
        # Import and execute the script
        import_script(script_name)
        success = True
    except Exception as e:
        print(f"Error running script: {e}")
        success = False
    
    # Restore the current directory
    os.chdir(original_dir)
    return success

def run_all_in_tp(tp_number, save_plots=False):
    """Run all scripts in a specific TP directory."""
    tp_dir = f"TP{tp_number}_"

    # Determine the full directory name
    full_dir = None
    for dirname in os.listdir():
        if dirname.startswith(tp_dir) and os.path.isdir(dirname):
            full_dir = dirname
            break
    
    if not full_dir:
        print(f"Error: Directory for TP{tp_number} not found.")
        return False
    
    success = True
    for file in os.listdir(full_dir):
        if file.endswith('.py') and not file.startswith('__'):
            script_path = os.path.join(full_dir, file)
            success = run_script(script_path, save_plots) and success
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Run image processing scripts.')
    parser.add_argument('script', nargs='?', help='Path to the script to run')
    parser.add_argument('--list', action='store_true', help='List all available scripts')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files in addition to displaying them')
    parser.add_argument('--all-tp', type=int, choices=available_tp_numbers(),
                        help='Run all scripts in the specified TP number')
    
    args = parser.parse_args()
    
    if args.list:
        print_available_scripts()
        return 0
    
    if args.all_tp:
        success = run_all_in_tp(args.all_tp, args.save_plots)
        return 0 if success else 1
    
    if not args.script:
        parser.print_help()
        print("\nNo script specified. Use --list to see available scripts.")
        return 1
    
    success = run_script(args.script, args.save_plots)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 