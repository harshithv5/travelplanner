from fastapi import FastAPI
from app.api import router

app = FastAPI(title='TravelStack AI')
app.include_router(router)

@app.get('/')
def root():
    return {'message': 'TravelStack AI running'}
