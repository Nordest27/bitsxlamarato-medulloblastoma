import io
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
from medulloblastoma.modeling.predict import predict_proba

_app = Flask(__name__, template_folder='templates', static_folder='static')


@_app.route('/')
def endpointHome():
    return render_template('index.html')


@_app.route('/public/<path:filename>')
def send_file(filename):
    return send_from_directory('public', filename)

@_app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON from the request
    json_data = request.get_json()
    # Extract CSV text from the 'rawCSV' field
    csv_string = json_data['rawCSV']

    csv_buffer = io.StringIO(csv_string)
    data = pd.read_csv(csv_buffer, index_col=0)
    model_path = os.path.join('app', 'data', 'statistical', 'best_model.pth')
    logreg_path = os.path.join('app', 'data', 'statistical', 'logreg_save.joblib')

    all_data = pd.read_csv(os.path.join('data', 'processed', 'g3g4_statistical.csv'), index_col=0)

    # We concat all the training data to ensure we normalize the value correctly
    result = predict_proba(data=pd.concat([data, all_data]), model_path=model_path, mid_dim=2048, features=32, logreg_path=logreg_path)

    return jsonify({
        "P(G4)": round(result[0], 4),
        "MYC": data['ENSG00000136997_at'][0],
        "TP53": data['ENSG00000141510_at'][0],
        "SNCAIP": data['ENSG00000064692_at'][0]
    })

if __name__ == '__main__':
    _app.run(debug=False)