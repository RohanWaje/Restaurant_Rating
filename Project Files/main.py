
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import logging
import pymongo
from pymongo import MongoClient
from datetime import datetime
import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from textblob import TextBlob
import ast

app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET','POST'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("home.html")


def ConfigureDataBase():

    client = MongoClient("mongodb+srv://covid:Password#10@cluster0-txvtu.mongodb.net/test?retryWrites=true&w=majority")
    return client.get_database('ML')

@app.route('/single_order',methods=['GET','POST'])  # route to display the home page
@cross_origin()

def single_order():
    return render_template("single_order.html")

@app.route('/bulk_order',methods=['GET','POST'])  # route to display the home page
@cross_origin()

def bulk_order():
    return render_template("bulk_order.html")

@app.route('/single_predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def single_predict():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Ranking = float(request.form['Ranking'])
            Number_of_Reviews = float(request.form['Number of Reviews'])
            Paris_City = float(request.form['Paris_City'])
            Reviews = str(request.form['Reviews'])
            London_City = float(request.form['London_City'])
            Milan_City = float(request.form['Milan_City'])
            Rome_City = float(request.form['Rome_City'])
            Berlin_City = float(request.form['Berlin_City'])
            Madrid_City = float(request.form['Madrid_City'])
            Barcelona_City = float(request.form['Barcelona_City'])
            Lisbon_City = float(request.form['Lisbon_City'])
            Prague_City = float(request.form['Prague_City'])
            Vienna_City = float(request.form['Vienna_City'])
			
			
			
            single_dict = [{"Ranking":Ranking,"Number of Reviews":Number_of_Reviews,"Paris_City":Paris_City,"Reviews":Reviews,
                           "London_City":London_City,"Milan_City":Milan_City,"Rome_City":Rome_City,"Berlin_City":Berlin_City,
                           "Madrid_City":Madrid_City,"Barcelona_City":Barcelona_City,"Lisbon_City":Lisbon_City,"Prague_City":Prague_City,
                           "Vienna_City":Vienna_City}]



            data = pd.DataFrame(single_dict)
            data['Reviews'] =data['Reviews'].str.replace(('\W+'),' ')
            data['Reviews'] =data['Reviews'].str.lower()
            stop = stopwords.words('english')

            data['Reviews'] =data['Reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

            st = PorterStemmer()
            data['Reviews'] = data['Reviews'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

            def senti(x):
                return TextBlob(x).sentiment.polarity

            data['Sentiment'] = data["Reviews"].apply(senti).round(2)
            data.drop(['Reviews'],axis=1,inplace=True)

            filename = 'Rest_Rate_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            prediction=loaded_model.predict(data)
            print('Prediction is', prediction)

            #Storing the record into database

            sin_output = pd.DataFrame(prediction)
            sin_output["Pred"] = sin_output
            sin_output.drop([0], inplace=True, axis=1)
            data = data.reset_index()
            

            output = pd.merge(data,sin_output['Pred'],how = 'left',left_index = True, right_index = True)
            output=output.drop(['index'], axis=1)

            output['DT']=datetime.now()
			

            db = ConfigureDataBase()
            collection=db['Restaurant_Rating']

            data_dict = output.to_dict(orient='records')
            collection.insert_many(data_dict)

            

            		
            # showing the prediction results in a UI
            return render_template('single_order_result.html',prediction=prediction[0])
        except Exception as e:
            print('The Exception message is: ',e)
            logging.debug(e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('home.html')
		
@app.route('/bulk_predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def bulk_predict():
    if request.method == 'POST':
        try:
            file = request.form['bulk']
            data = pd.read_excel(file)

            data['Reviews'] =data['Reviews'].str.replace(('\W+'),' ')
            data['Reviews'] =data['Reviews'].str.lower()
            stop = stopwords.words('english')
            data['Reviews'] =data['Reviews'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

            st = PorterStemmer()
            data['Reviews'] = data['Reviews'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

            def senti(x):
                return TextBlob(x).sentiment.polarity

            data['Sentiment'] = data["Reviews"].apply(senti).round(2)
            data.drop(['Reviews'],axis=1,inplace=True)


			            
            filename = 'Rest_Rate_model.pickle'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            pred_excel=loaded_model.predict(data)
            print('Prediction is', pred_excel)
            bulk_output = pd.DataFrame(pred_excel)
            bulk_output["Pred"] = bulk_output
            bulk_output.drop([0], inplace=True, axis=1)
            bulk_input = data.reset_index()

            bulk_output = pd.merge(bulk_input,bulk_output['Pred'],how = 'left',left_index = True, right_index = True)
            bulk_output=bulk_output.drop(['index'], axis=1)

            bulk_output['Date_Time'] = datetime.now()

            # storing the results into database

            db = ConfigureDataBase()
            collection=db['Restaurant_Rating']

            data_dict = bulk_output.to_dict(orient='records')
            collection.insert_many(data_dict)
			
            # showing the prediction results in a UI
            return render_template('bulk_order_result.html',pred_bulk=pred_excel)
        except Exception as e:
            print('The Exception message is: ',e)
            logging.debug(e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('home.html')



if __name__ == "__main__":

    logging.basicConfig(filename='Restaurant_Rating.log', level=logging.DEBUG)
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True) # running the app