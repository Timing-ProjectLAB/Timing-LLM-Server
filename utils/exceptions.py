from fastapi import HTTPException

def raise_400(message: str):
    raise HTTPException(status_code=400, detail={
        "error": {
            "code": "LLM40001",
            "message": message
        }
    })

def raise_422(message: str):
    raise HTTPException(status_code=422, detail={
        "error": {
            "code": "LLM42201",
            "message": message
        }
    })

def raise_500(message: str):
    raise HTTPException(status_code=500, detail={
        "error": {
            "code": "LLM50001",
            "message": message
        }
    })
