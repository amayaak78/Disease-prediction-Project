from flask import Flask, render_template, request,flash
import pickle
import pandas as pd
import numpy as np

# create an object for flask
app = Flask(__name__)
# open the file iris
model = pickle.load(open('pred.pkl', 'rb'))


# For displaying the home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('front.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = pd.read_csv("D:\\project\\Symptom-severity.csv")
    data1 = pd.read_csv("D:\\project\\output.csv")
    data3 = pd.read_csv("D:\\project\\dataset.csv")
    # fetching the values of SL,SW,PL and PW from the webpage and convert it to float
    if request.form.get('Symptom1') == "None" and request.form.get('Symptom2') == "None" and request.form.get('Symptom3') == "None" and request.form.get('Symptom4') == "None" and request.form.get('Symptom5') == "None" and request.form.get('Symptom6') == "None" and request.form.get('Symptom7') == "None" and request.form.get('Symptom8') == "None" and request.form.get('Symptom9') == "None":
        error_message="whoops!!", "Enter Symptoms Please"
        return render_template('front.html',error_statement=error_message)
    elif request.form.get('Symptom1') == "None" or request.form.get('Symptom2') == "None" or request.form.get('Symptom3') == "None" or request.form.get('Symptom4') == "None" or request.form.get('Symptom5') == "None" :
        error_message="oops!!", "First 5 Symptoms are mandatory"
        return render_template('front.html',error_statement=error_message)
    else:
        usersymptoms = [x for x in request.form.values()]
        symptoms = usersymptoms
        a = np.array(data["Symptom"])
        b = np.array(data["weight"])

        for j in range(len(usersymptoms)):
            if usersymptoms[j] not in a:
                usersymptoms[j]=0
        for j in range(len(usersymptoms)):
            for k in range(len(a)):
                if usersymptoms[j] == a[k]:
                    usersymptoms[j] = b[k]

        sym_weight = sum(usersymptoms)
        nulls = [0, 0, 0, 0, 0, 0, 0, 0]
        psy = [usersymptoms + nulls]

        # Prediction based on the model stored in pickle file
        prediction = model.predict(psy)
        out = prediction[0]


        des = data1[data1['Disease'] == out]['Description'].values[0]
        prec1 = data1[data1['Disease'] == out]['Precaution_1'].values[0]
        prec2 = data1[data1['Disease'] == out]['Precaution_2'].values[0]
        prec3 = data1[data1['Disease'] == out]['Precaution_3'].values[0]
        prec4 = data1[data1['Disease'] == out]['Precaution_4'].values[0]
        tydr = data1[data1['Disease'] == out]['Type of Doctors'].values[0]
        link = data1[data1['Disease'] == out]['Link '].values[0]
        severity = data1[data1['Disease'] == out]['Totalweight'].values[0]
        sev = (sym_weight / severity) * 100
        sev1 = round(sev)
        return render_template('prediction.html', prediction_text=out, pred1=des, pred2=prec1, pred3=prec2, pred4=prec3,
                           pred5=prec4, pred6=tydr, pred7=link, pred8=sev1)


# For running the current module
if __name__ == '__main__':
    app.run(debug=True)
