from flask import Flask,jsonify,request
from flask_restful import Api,Resource
import pickle
import pandas as pd 
from fbprophet import Prophet

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/getBeansPrice', methods=['POST'])
def getPrice():


        # user inputs
        data = request.get_json();

        user_date = data.get('date', '');

         # pickle file name
        file_name = 'prophet_pickle_file.pkl'

        # load the pickle file 
        loaded_model = pickle.load(open(file_name,'rb'))

        future_date = pd.DataFrame({'ds':[user_date]})

        # forecast using the model
        out = loaded_model.predict(future_date);

        print(out.yhat[0])

        # convert to jsons
        return {
        "vegetable-code" : "TY-BEANS",
        "vegetable-name" : "beans",
        "area" : "nuwaraeliya",
        "price-output": out.yhat[0],
        "status-code": 200
        }


@app.route('/getCarrotPrice', methods=['POST'])
def getCarrotPrice():


        # user inputs
        data = request.get_json();

        user_date = data.get('date', '');

         # pickle file name
        file_name = 'prophet-carrot.pkl'

        # load the pickle file 
        loaded_model = pickle.load(open(file_name,'rb'))

        future_date = pd.DataFrame({'ds':[user_date]})

        # forecast using the model
        out = loaded_model.predict(future_date);

        print(out.yhat[0])

        # convert to jsons
        return {
        "vegetable-code" : "TY-CARROT",
        "vegetable-name" : "carrot",
        "area" : "nuwaraeliya",
        "price-output": out.yhat[0],
        "status-code": 200
        }



@app.route('/getTomatoPrice', methods=['POST'])
def getTomatoPrice():


        # user inputs
        data = request.get_json();

        user_date = data.get('date', '');

         # pickle file name
        file_name = 'prophet-tomato.pkl'

        # load the pickle file 
        loaded_model = pickle.load(open(file_name,'rb'))

        future_date = pd.DataFrame({'ds':[user_date]})

        # forecast using the model
        out = loaded_model.predict(future_date);

        print(out.yhat[0])

        # convert to jsons
        return {
        "vegetable-code" : "TY-TOMATO",
        "vegetable-name" : "tomato",
        "area" : "nuwaraeliya",
        "price-output": out.yhat[0],
        "status-code": 200
        }


@app.route('/getBeansPriceRangeWeeks', methods=['POST'])
def getBeansRangePrice():


        # user inputs
        data = request.get_json();

        start_date = data.get('startdate', '');
        end_date = data.get('enddate', '');

         # pickle file name
        file_name = 'prophet_pickle_file.pkl'

        # load the pickle file 
        loaded_model = pickle.load(open(file_name,'rb'))

        future = pd.DataFrame({'ds': pd.date_range(start=start_date, end= end_date, freq='W')})
        
        # forecast using the model
        out = loaded_model.predict(future);

        print(out.yhat[0])

        responseData = out.yhat;

        print(responseData.to_json());

        c = responseData.to_json();


       # convert to jsons
        return {
        "vegetable-code" : "TY-BEANS",
        "vegetable-name" : "beans",
        "area" : "nuwaraeliya",
        "price-output": c,
        "status-code": 200
        }


@app.route('/getCarrotPriceRangeMonths', methods=['POST'])
def getCarrotRangePrice():


        # user inputs
        data = request.get_json();

        start_date = data.get('startdate', '');
        end_date = data.get('enddate', '');

         # pickle file name
        file_name = 'prophet-carrot.pkl'

        # load the pickle file 
        loaded_model = pickle.load(open(file_name,'rb'))

        future = pd.DataFrame({'ds': pd.date_range(start=start_date, end= end_date, freq='M')})
        
        # forecast using the model
        out = loaded_model.predict(future);

        print(out.yhat[0])

        responseData = out.yhat;

        print(responseData.to_json());

        c = responseData.to_json();


       # convert to jsons
        return {
        "vegetable-code" : "TY-CARROT",
        "vegetable-name" : "carrot",
        "area" : "nuwaraeliya",
        "price-output": c,
        "status-code": 200
        }


@app.route('/getTomatoPriceRangeMonths', methods=['POST'])
def getTomatoRangePrice():


        # user inputs
        data = request.get_json();

        start_date = data.get('startdate', '');
        end_date = data.get('enddate', '');

         # pickle file name
        file_name = 'prophet-tomato.pkl'

        # load the pickle file 
        loaded_model = pickle.load(open(file_name,'rb'))

        future = pd.DataFrame({'ds': pd.date_range(start=start_date, end= end_date, freq='M')})
        
        # forecast using the model
        out = loaded_model.predict(future);

        print(out.yhat[0])

        responseData = out.yhat;

        print(responseData.to_json());

        c = responseData.to_json();


       # convert to jsons
        return {
        "vegetable-code" : "TY-TOMATO",
        "vegetable-name" : "tomato",
        "area" : "nuwaraeliya",
        "price-output": c,
        "status-code": 200
        }

if __name__ == '__main__':
        app.run(debug=True)