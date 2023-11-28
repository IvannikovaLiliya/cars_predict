from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
from io import BytesIO

import uvicorn
import pickle
import pandas as pd
import numpy as np

predictor = pickle.load(open('car_price_predictor.pkl', 'rb'))

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


    def predict(self):
        df = pd.DataFrame({
               "name": self.name,
               "year": self.year,
               "selling_price": self.selling_price,
               "km_driven": self.km_driven,
               "fuel": self.fuel,
               "seller_type": self.seller_type,
               "transmission": self.transmission,
               "owner": self.owner,
               "mileage": self.mileage,
               "engine": self.engine,
               "max_power": self.max_power,
               "torque": self. torque,
               "seats": self.seats
            }, index=[0])

        # обработка полей с метриками
        df['mileage'] = df['mileage'].str.replace('kmpl|km/kg', '', regex=True).astype(float)
        df['max_power'] = pd.to_numeric(df['max_power'].str.replace('bhp', '', regex=True), errors='coerce')
        df['engine'] = df['engine'].str.replace('CC', '', regex=True).astype(float)
        df.drop('torque', axis=1, inplace = True)
        df['seats'] = df['seats'].astype('category')
        df['auto_mark'] = df['name'].str.split(' ', expand=True)[0]

        def fuel_pr(var):
            if var == 'Petrol':
                new_var = 'Petrol'
            elif var == 'Diesel':
                new_var = 'Diesel'
            else:
                new_var = 'Other'

            return new_var

        df['fuel'] = df['fuel'].apply(lambda x: fuel_pr(x))

        def transmission_pr(var):
            if var == 'Automatic':
                new_var = 1
            else:
                new_var = 0

            return new_var

        df['transmission'] = df['transmission'].apply(lambda x: transmission_pr(x))

        def owner_pr(var):
            if var == 'Test Drive Car':
                new_var = 5
            elif var == 'First Owner':
                new_var = 4
            elif var == 'Second Owner':
                new_var = 3
            elif var == 'Third Owner':
                new_var = 2
            else:
                new_var = 1

            return new_var

        df['owner'] = df['owner'].apply(lambda x: owner_pr(x))
        # Land Rover скорректировали
        df.loc[df['auto_mark'] == 'Land', 'auto_mark'] = 'Land Rover'

        df_countries = pd.read_csv('country.csv', sep=';')
        df_country = df.merge(df_countries, on=['auto_mark'], how='left')
        df_country['country'].fillna('None', inplace=True)

        preds = predictor.predict(df_country)

        return np.e ** preds


class Items(BaseModel):
    objects: List[Item]


@app.get('/')
# возвращаем Hello!
def root():
    return print('Hello!')


@app.post("/predict_item", response_model=float)
async def predict_item(item: Item) -> float:
    return item.predict()

@app.post('/predict_items')
async def predict_items(file: UploadFile) -> FileResponse:
    content = file.file.read() #считываем байтовое содержимое
    buffer = BytesIO(content) #создаем буфер типа BytesIO
    df = pd.read_csv(buffer, index_col=0)
    buffer.close()
    await file.close()  # закрывается именно сам файл

    preds = []
    for row in range(df.shape[0]):
        dict_row = df.iloc[row].to_dict()
        preds.append(*Item(**dict_row).predict())
    df['pred'] = preds

    df.to_csv('items_predict.csv')
    response = FileResponse(path='items_predict.csv', media_type='text/csv', filename='all_predicts.csv')
    return response

uvicorn.run(app, host="0.0.0.0", port=8000)