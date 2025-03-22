from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import os, json
from generate import generate_output_json

app = FastAPI(title="Automesh API")

@app.get("/")
async def root():
    return {"message": "Automesh API test"}

@app.post("/generate")
async def generate(payload: Dict[str, Any]):
    """
    Expects a JSON payload with a 'prompt' field. 
    The endpoint loads a sample data point, runs the generation process using the provided prompt,
    and returns a JSON response in the desired format.
    """
    try:
        # Get the prompt from the payload
        prompt = payload.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing prompt in request body")
        
        # Optionally, you could load other data or configurations from the payload.
        # For this example, we assume a fixed data file is used.
        file_path = os.path.join(os.getcwd(), "sample_data.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Data file not found")
        
        # The generate_output_json function internally loads the data file,
        # runs the model, and constructs the output.
        output_json = generate_output_json(prompt, edge_threshold=0.5)
        return JSONResponse(content=output_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
