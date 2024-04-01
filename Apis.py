from fastapi import FastAPI
from uvicorn import Server
from Backend import predict
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = [
    "http://localhost:3000",  # React frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/{model}/{battery}/{cycle}")
async def read_prediction(model: str, battery: str, cycle: str):
    print(model, battery, cycle)
    battery=battery.replace("$","")
    cycle=cycle.replace("$","")
    model=model.replace("$","")
    print(model, battery, cycle)
    print(type(battery), type(cycle))
    battery = int(battery.replace("$", ""))
    cycle = int(cycle.replace("$", ""))
    prediction = predict(model, battery, cycle)
    return {"prediction": prediction}


if __name__ == "__main__":
    run("Apis:app", host="localhost", port=8000, reload=True)
