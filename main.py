from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.chatbot import router as chatbot_router
from core.vectorstore import load_all_vectorstores

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    app.state.vectorstores = load_all_vectorstores()
    print(app.state.vectorstores)
    print("Server Starting up...")
    yield 
    print("Server shutting down...")


# lifespan 주입
app = FastAPI(lifespan=lifespan)

# Router 등록
app.include_router(chatbot_router)