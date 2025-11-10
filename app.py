# %%
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import pickle

# %%
# Load model
model = pickle.load(open('model.pkl', 'rb'))

# %%
# Initialize Flask
app = Flask(__name__)

# %%

# HTML template
html_template = """
<h2>Banknote Authenticator</h2>
<form action="/predict" method="post">
  Feature 1: <input type="text" name="f1"><br>
  Feature 2: <input type="text" name="f2"><br>
  Feature 3: <input type="text" name="f3"><br>
  Feature 4: <input type="text" name="f4"><br>
  <input type="submit" value="Predict">
</form>
{% if prediction_text %}
  <h3>{{ prediction_text }}</h3>
{% endif %}
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])[0]
    result = "REAL banknote 🟢" if prediction == 0 else "FAKE banknote 🔴"
    return render_template_string(html_template, prediction_text=f'This banknote is {result}')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])[0]
    return jsonify(int(prediction))

# Run Flask in Jupyter without reloader
app.run(debug=True, use_reloader=False)


# %%



