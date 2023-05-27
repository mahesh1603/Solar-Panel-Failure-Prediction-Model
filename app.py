from flask import Flask, render_template, request
import re
import pandas as pd
import pickle
import joblib


model = pickle.load(open('rfc.pkl','rb'))
Median_Imputation = joblib.load('Median_Imputation')
Outlier_Winsorizer = joblib.load('Outlier_Winsorizer')
RobustScalar = joblib.load('RobustScalar')



#define flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_excel(f)
        newMedian_Imputation = pd.DataFrame(Median_Imputation.transform(data), 
                                            columns = data.select_dtypes(exclude = ['object']).columns)
        newMedian_Imputation[['Ipv', 'Vpv', 'Vdc', 
                              'ia', 'ib', 'ic', 
                              'va', 'vb', 'vc', 'Iabc', 
                              'If', 'Vabc', 
                              'Vf']]= pd.DataFrame(Outlier_Winsorizer.transform(newMedian_Imputation[['Ipv', 'Vpv', 
                                                                                                      'Vdc', 'ia', 
                                                                                                      'ib', 'ic', 
                                                                                                      'va', 'vb', 
                                                                                                      'vc', 'Iabc', 
                                                                                                      'If', 'Vabc', 'Vf']]))
        newRobustScalar = pd.DataFrame(RobustScalar.transform(newMedian_Imputation), columns = data.select_dtypes(exclude = ['object']).columns)
        predictions = pd.DataFrame(model.predict(newRobustScalar),columns = ['Label'])
        final = pd.concat([predictions, data], axis = 1)
        
        return render_template("new.html", Y = final.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
