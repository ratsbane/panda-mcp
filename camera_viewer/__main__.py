import uvicorn
from camera_viewer.app import app

uvicorn.run(app, host="0.0.0.0", port=8080)
