from flask import Flask, render_template, request, jsonify

_app = Flask(__name__, template_folder='templates', static_folder='static')


@_app.route('/')
def endpointHome():
    return render_template('index.html')

@_app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    return jsonify({
        "rValue": 0.0,
        "rV1": 0.0,
        "rV2": 0.0,
        "rV3": 0.0
    })

if __name__ == '__main__':
    _app.run(debug=False)