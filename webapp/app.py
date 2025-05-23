from flask import Flask, render_template, send_from_directory, abort
import os
import glob
import run

app = Flask(__name__)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')

@app.route('/')
def index():
    scripts = run.list_scripts()
    return render_template('index.html', scripts=scripts)

@app.route('/run/<path:script>')
def run_script(script):
    # sanitize script path
    script_path = os.path.normpath(script)
    if '..' in script_path or script_path.startswith('/'):
        abort(400)
    success = run.run_script(script_path, save_plots=True)
    if not success:
        abort(500)
    base_name = os.path.splitext(os.path.basename(script_path))[0]
    pattern = os.path.join(PLOTS_DIR, f"{base_name}_plot*.png")
    images = [os.path.basename(p) for p in sorted(glob.glob(pattern))]
    return render_template('plots.html', script=script_path, images=images)

@app.route('/plots/<path:filename>')
def plot_file(filename):
    return send_from_directory(PLOTS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)
