from flask import Flask, render_template, request
import numpy as np
import joblib
app = Flask(__name__)
model=joblib.load("random_f_model")

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/upload')
def upload():
    return render_template("upload.html")
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    # print(int_features)
    final_features = [np.array(int_features)]
    # print(final_features)
    prediction = model.predict(final_features)
    result=(prediction[0]).astype("int")
    # print(result)
    # output=f"12AM-1AM: {result[0]} | 1AM-2AM: {result[1]} | 2AM-3AM: {result[2]} | 3AM-4AM: {result[3]}" \
    #        f"| 4AM-5AM: {result[4]} | 5AM-6AM: {result[5]} | 6AM-7AM: {result[6]}" \
    #        f"| 7AM-8AM: {result[7]} | 8AM-9AM: {result[8]} | 9AM-10AM: {result[9]}" \
    #        f"| 10AM-11AM: {result[10]} | 11AM-12PM: {result[11]} | 12PM-1PM: {result[12]}" \
    #        f"| 1PM-2PM: {result[13]} | 2PM-3PM: {result[14]} | 3PM-4PM: {result[15]}" \
    #        f"| 4PM-5PM: {result[16]} | 5PM-6PM: {result[17]} | 6PM-7PM: {result[18]}" \
    #        f"| 7PM-8PM: {result[19]} | 8PM-9PM: {result[20]} | 9PM-10PM: {result[21]}" \
    #        f"| 10PM-11PM: {result[22]} | 11PM-12AM: {result[23]}"
    return render_template("result.html",prediction_text=f"{result[0]}",
                           prediction_text1=f"{result[1]}",
                           prediction_text2=f"{result[2]}",
                           prediction_text3=f"{result[3]}",
                           prediction_text4=f"{result[4]}",
                           prediction_text5=f"{result[5]}",
                           prediction_text6=f"{result[6]}",
                           prediction_text7=f"{result[7]}",
                           prediction_text8=f"{result[8]}",
                           prediction_text9=f"{result[9]}",
                           prediction_text10=f"{result[10]}",
                           prediction_text11=f"{result[11]}",
                           prediction_text12=f"{result[12]}",
                           prediction_text13=f"{result[13]}",
                           prediction_text14=f"{result[14]}",
                           prediction_text15=f"{result[15]}",
                           prediction_text16=f"{result[16]}",
                           prediction_text17=f"{result[17]}",
                           prediction_text18=f"{result[18]}",
                           prediction_text19=f"{result[19]}",
                           prediction_text20=f"{result[20]}",
                           prediction_text21=f"{result[21]}",
                           prediction_text22=f"{result[22]}",
                           prediction_text23=f"{result[23]}")

if __name__=="__main__":
    app.run(debug=True)

