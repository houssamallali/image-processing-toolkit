from flask import Flask, render_template, redirect, url_for
import os
import subprocess

app = Flask(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')

# Ensure plots are accessible via the static folder
STATIC_PLOTS = os.path.join(app.static_folder, 'plots')
if not os.path.islink(STATIC_PLOTS):
    try:
        os.makedirs(os.path.dirname(STATIC_PLOTS), exist_ok=True)
        if os.path.exists(STATIC_PLOTS):
            os.remove(STATIC_PLOTS)
        os.symlink(PLOTS_DIR, STATIC_PLOTS)
    except OSError:
        pass


def list_scripts():
    scripts = {}
    for entry in os.listdir(ROOT_DIR):
        if entry.startswith('TP') and os.path.isdir(os.path.join(ROOT_DIR, entry)):
            files = []
            for f in os.listdir(os.path.join(ROOT_DIR, entry)):
                if f.endswith('.py') and not f.startswith('__'):
                    files.append(f)
            scripts[entry] = sorted(files)
    return scripts


@app.route('/')
def index():
    scripts = list_scripts()
    return render_template('index.html', scripts=scripts)


@app.route('/run/<tp>/<script>')
def run_script(tp, script):
    script_path = os.path.join(ROOT_DIR, tp, script)
    if not os.path.exists(script_path):
        return f'Script {script_path} not found', 404

    cmd = ['python', os.path.join(ROOT_DIR, 'run.py'), os.path.join(tp, script), '--save-plots']
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f'Error running script: {e}', 500
    return redirect(url_for('show_plots', tp=tp, script=script))


@app.route('/plots/<tp>/<script>')
def show_plots(tp, script):
    base = os.path.splitext(script)[0]
    images = []
    tp_plot_dir = os.path.join(PLOTS_DIR)
    for f in os.listdir(tp_plot_dir):
        if f.startswith(base):
            images.append(url_for('static', filename=f'plots/{f}'))
    return render_template('plots.html', images=images, tp=tp, script=script)


if __name__ == '__main__':
    app.run(debug=True)
