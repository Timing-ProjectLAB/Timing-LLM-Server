from fastapi import FastAPI
from api.chatbot import router as chatbot_router
from core.vectorstore import load_all_vectorstores

app = FastAPI()

@app.on_event("startup")
def startup_event():
    app.state.vectorstores = load_all_vectorstores()
    print(app.state.vectorstores)

app.include_router(chatbot_router)
