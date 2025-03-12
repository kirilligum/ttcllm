import os
import yaml
import litellm
from fastapi import FastAPI
from litellm.proxy.server import router as litellm_router

# Load configuration from config/config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

# Initialize FastAPI app
app = FastAPI()

# Register the LiteLLM proxy router
app.include_router(litellm_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])
