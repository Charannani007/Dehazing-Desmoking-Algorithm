import os
import cv2
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from image_dehazer import image_dehazer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'static/dehazed_images/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('dehaze_image', filename=filename))
    return render_template('index.html')

@app.route('/dehaze/<filename>')
def dehaze_image(filename):
    input_image_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_folder = app.config['OUTPUT_FOLDER']
    frame = cv2.imread(input_image_file)
    if frame is None:
        return "Failed to read the input image", 400

    output_details = []
    parameter_sets = [
        {'airlightEstimation_windowSze': 15, 'boundaryConstraint_windowSze': 3, 'C0': 20, 'C1': 300,
         'regularize_lambda': 0.1, 'sigma': 0.5, 'delta': 0.85, 'description': 'Baseline'},
        {'airlightEstimation_windowSze': 25, 'boundaryConstraint_windowSze': 3, 'C0': 20, 'C1': 300,
         'regularize_lambda': 0.1, 'sigma': 0.5, 'delta': 0.85, 'description': 'Increased Airlight Estimation Window Size'},
        {'boundaryConstraint_windowSze': 5, 'airlightEstimation_windowSze': 15, 'C0': 20, 'C1': 300,
         'regularize_lambda': 0.1, 'sigma': 0.5, 'delta': 0.85, 'description': 'Increased Boundary Constraint Window Size'},
        {'regularize_lambda': 0.2, 'airlightEstimation_windowSze': 15, 'boundaryConstraint_windowSze': 3, 'C0': 20, 'C1': 300,
         'sigma': 0.5, 'delta': 0.85, 'description': 'Increased Regularization Lambda'},
        {'sigma': 1.0, 'airlightEstimation_windowSze': 15, 'boundaryConstraint_windowSze': 3, 'C0': 20, 'C1': 300,
         'regularize_lambda': 0.1, 'delta': 0.85, 'description': 'Increased Sigma'},
        {'delta': 0.9, 'airlightEstimation_windowSze': 15, 'boundaryConstraint_windowSze': 3, 'C0': 20, 'C1': 300,
         'regularize_lambda': 0.1, 'sigma': 0.5, 'description': 'Increased Delta'}
    ]
    dehazer = image_dehazer()

    for idx, params in enumerate(parameter_sets):
        frame = cv2.imread(input_image_file)
        HazeCorrectedImg, TransmissionMap = dehazer.remove_haze(
            frame,
            airlightEstimation_windowSze=params['airlightEstimation_windowSze'],
            boundaryConstraint_windowSze=params['boundaryConstraint_windowSze'],
            C0=params['C0'],
            C1=params['C1'],
            regularize_lambda=params['regularize_lambda'],
            sigma=params['sigma'],
            delta=params['delta'],
            showHazeTransmissionMap=False
        )
        output_filename = f'dehazed_{idx}.jpg'
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, HazeCorrectedImg)
        output_details.append({
            'output_path': output_filename,
            'description': params['description'],
            'parameters': params,
        })

    return render_template('results.html', output_details=output_details, filename=filename)


if __name__ == '__main__':
    app.run(debug=True)
