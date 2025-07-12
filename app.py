from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import uvicorn
import os
from dotenv import load_dotenv
from google import genai
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime


# ------------------ Cargar modelo y escalador ------------------
with open("modelos_pkl/diabetes.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("modelos_pkl/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------ Clase para entrada ------------------
class DiabetesInput(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    HvyAlcoholConsump: int
    GenHlth: int
    DiffWalk: int
    Age: int

mapping = {
    0: "No diabetes",
    1: "Prediabetes",
    2: "Diabetes"
}

# ------------------ FastAPI App ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# ------------------ Configurar Gemini ------------------
load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
conexion = os.getenv("DATABASE_URL")

# Create a new client and connect to the server
mongo_client = MongoClient(conexion, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    mongo_client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

mydb = mongo_client["Cluster0"]
collection = mydb["predicciones"]

# Predicción
@app.post("/predecir")
def predecir(datos: DiabetesInput):
    entrada = [[
        datos.HighBP,
        datos.HighChol,
        datos.CholCheck,
        datos.BMI,
        datos.Smoker,
        datos.Stroke,
        datos.HeartDiseaseorAttack,
        datos.PhysActivity,
        datos.HvyAlcoholConsump,
        datos.GenHlth,
        datos.DiffWalk,
        datos.Age
    ]]

    entrada_escalada = scaler.transform(entrada)
    prediccion = modelo.predict(entrada_escalada)
    resultado = mapping[int(prediccion[0])]

    # Guardar en MongoDB
    collection.insert_one({
        "datos": datos.dict(),
        "prediccion_num": int(prediccion[0]),
        "diagnostico": resultado,
        "fecha": datetime.utcnow()
    })

    return {
        "prediccion_num": int(prediccion[0]),
        "diagnostico": resultado
    }

# Explicación con Gemini
@app.post("/explicacion")
def explicacion(datos: DiabetesInput):
    prompt = f"""

Eres un coach de salud personal y motivador. Tu objetivo es interpretar el resultado de un análisis de riesgo de diabetes y dar una explicación útil y alentadora.

Analiza los siguientes datos del paciente y el diagnóstico del modelo.
- Destaca los factores de salud positivos que el paciente tiene a su favor.
- Identifica 1 o 2 áreas clave en las que puede enfocarse para mejorar su bienestar.
- Usa un tono amigable, empático y que motive al usuario a tomar acción.
- No uses jerga médica.
- Presión alta: {datos.HighBP}
- Colesterol alto: {datos.HighChol}
- Chequeo colesterol: {datos.CholCheck}
- BMI: {datos.BMI}
- Fuma: {datos.Smoker}
- Derrame cerebral: {datos.Stroke}
- Enfermedad cardíaca: {datos.HeartDiseaseorAttack}
- Actividad física: {datos.PhysActivity}
- Consumo excesivo alcohol: {datos.HvyAlcoholConsump}
- Salud general: {datos.GenHlth}
- Dificultad para caminar: {datos.DiffWalk}
- Edad: {datos.Age}
"""

    # Llamar a Gemini para generar explicación
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    explicacion = response.text.strip()
    return {"explicacion": explicacion}

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Servir archivos estáticos desde templates/
app.mount("/", StaticFiles(directory="templates", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)