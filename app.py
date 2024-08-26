from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the model and scaler
df = pd.read_csv('mixed_data_set_with_quality.csv')
feature_cols = ['casting_temperature', 'cooling_temp', 'casting_speed', 'entry_temp',
                'emulsion_temp', 'emulsion_pressure', 'emulsion_concentration', 'quench_pressure']
target_col = 'quality'
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(df[feature_cols], df[target_col])

scaler = StandardScaler()
scaler.fit(df[feature_cols])

min_range = {'casting_temperature': 680, 'cooling_temp': 20, 'casting_speed': 6, 'entry_temp': 400,
             'emulsion_temp': 40, 'emulsion_pressure': 1, 'emulsion_concentration': 2, 'quench_pressure': 1}
max_range = {'casting_temperature': 750, 'cooling_temp': 35, 'casting_speed': 12, 'entry_temp': 500,
             'emulsion_temp': 60, 'emulsion_pressure': 3, 'emulsion_concentration': 5, 'quench_pressure': 2}

def calculate_cooling_water_temp(casting_temp):
    return max(20, min(0.5 * casting_temp + 2, 35))

def calculate_casting_speed(casting_temp):
    return max(6, min(0.8 * casting_temp + 1000, 12))

def calculate_entry_temp(casting_temp):
    return max(400, min(450, 500))

def calculate_emulsion_temp(casting_temp):
    return max(40, min(50, 60))

def calculate_emulsion_pressure(casting_temp):
    return max(1, min(2, 3))

def calculate_emulsion_concentration(casting_temp):
    return max(2, min(3, 5))

def calculate_quench_pressure(casting_temp):
    return max(1, min(1.5, 2))

def suggest_adjustments(parameters, min_range, max_range):
    suggestions = {}
    function_mapping = {
        'cooling_temp': calculate_cooling_water_temp,
        'casting_speed': calculate_casting_speed,
        'entry_temp': calculate_entry_temp,
        'emulsion_temp': calculate_emulsion_temp,
        'emulsion_pressure': calculate_emulsion_pressure,
        'emulsion_concentration': calculate_emulsion_concentration,
        'quench_pressure': calculate_quench_pressure,
    }

    for param, value in parameters.items():
        if param == 'casting_temperature':
            continue

        func = function_mapping.get(param)
        if func:
            optimal_value = func(parameters['casting_temperature'])

            optimal_value = max(min_range[param], min(optimal_value, max_range[param]))
            percentage_change = ((optimal_value - value) / value) * 100
            suggestions[param] = {
                'optimal_value': optimal_value,
                'percentage_change': percentage_change
            }
    return suggestions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            # You can add logic to process the CSV file here
        else:
            return "Please upload a CSV file."

        casting_temperature = float(request.form['casting_temperature'])
        cooling_temp = float(request.form['cooling_temp'])
        casting_speed = float(request.form['casting_speed'])
        entry_temp = float(request.form['entry_temp'])
        emulsion_temp = float(request.form['emulsion_temp'])
        emulsion_pressure = float(request.form['emulsion_pressure'])
        emulsion_concentration = float(request.form['emulsion_concentration'])
        quench_pressure = float(request.form['quench_pressure'])

        parameters = {
            'casting_temperature': casting_temperature,
            'cooling_temp': cooling_temp,
            'casting_speed': casting_speed,
            'entry_temp': entry_temp,
            'emulsion_temp': emulsion_temp,
            'emulsion_pressure': emulsion_pressure,
            'emulsion_concentration': emulsion_concentration,
            'quench_pressure': quench_pressure
        }

        input_df = pd.DataFrame(parameters, index=[0])
        input_scaled = scaler.transform(input_df)
        quality_pred = RF.predict(input_scaled)[0]

        suggestions = {}
        if quality_pred != 'high':
            suggestions = suggest_adjustments(parameters, min_range, max_range)

        return render_template('results.html', quality_pred=quality_pred, suggestions=suggestions)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
