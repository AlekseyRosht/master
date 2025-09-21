# main.py
import logging
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import io
from typing import List
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Глобальная переменная для модели
model = None

# Определение категориальных признаков
CATEGORICAL_FEATURES = [
    'Age',
    'Diabetes',
    'Obesity',
    'Alcohol Consumption',
    'Diet',
    'Physical Activity Days Per Week',
    'Sleep Hours Per Day',
    'Gender'
]

class PredictionResponse(BaseModel):
    id: int
    prediction: int

# Корневой endpoint
@app.get("/")
async def root():
    logger.info("Корневой endpoint вызван")
    return {"message": "API для предсказания риска сердечного приступа готов к работе"}

# Endpoint для предсказаний
@app.post("/predict/", response_model=List[PredictionResponse])
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Модель не загружена")
            
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Обработка данных
        df = preprocess_data(df)
        
        # Предсказания
        predictions = model.predict(df)
        
        # Формирование результата
        result = [
            PredictionResponse(id=i, prediction=int(pred))
            for i, pred in enumerate(predictions)
        ]
        
        return result
    
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def convert_float_to_int(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'float32']).columns
    for col in columns:
        if df[col].notna().all() and (df[col] % 1 == 0).all():
            df[col] = df[col].astype(int)
        elif df[col].notna().sum() > 0 and (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].fillna(-1).astype(int)
    return df

def preprocess_data(df):
    columns_to_drop = [
        'Family History', 
        'Smoking', 
        'Exercise Hours Per Week', 
        'Previous Heart Problems', 
        'Medication Use', 
        'Stress Level'
    ]
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df = convert_float_to_int(df)
    
    df['Age'] = df['Age'].astype(str).replace('nan', 'Unknown')
    df['Sleep Hours Per Day'] = df['Sleep Hours Per Day'].astype(str).replace('nan', 'Unknown')
    
    gender_mapping = {'Male': 1, 'Female': 0}
    df['Gender'] = df['Gender'].replace(gender_mapping)
    df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce').astype('int64')
    
    return df

# Функция для обучения модели
def train_model():
    train_path = r"C:\Users\Alex\Desktop\masterskay\heart_train.csv"
    df_train = pd.read_csv(train_path)
    
    df_train = preprocess_data(df_train)
    
    X = df_train.drop('Heart Attack Risk (Binary)', axis=1)
    y = df_train['Heart Attack Risk (Binary)']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    undersampler = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    
    model = CatBoostClassifier(
        cat_features=CATEGORICAL_FEATURES,
        iterations=1000,
        learning_rate=0.05,
        verbose=100,
        random_state=42
    )
    
    model.fit(X_train_resampled, y_train_resampled, verbose=False)
    model.save_model('catboost_model.cbm')
    return model

# Загрузка модели при старте приложения
@app.on_event("startup")
async def startup_event():
    global model
    try:
        if not os.path.exists('catboost_model.cbm'):
            logger.info("Модель не найдена, начинаем обучение...")
            model = train_model()
        else:
            logger.info("Загружаем существующую модель...")
            model = CatBoostClassifier()
            model.load_model('catboost_model.cbm')
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        raise RuntimeError(f"Не удалось загрузить модель: {str(e)}")

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)