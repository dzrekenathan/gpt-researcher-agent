import json
import os
import shutil
from typing import Dict, List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from gpt_researcher.utils.enum import ReportType, Tone, ReportSource
from backend.server.websocket_manager import WebSocketManager, run_agent
from backend.server.server_utils import (
    get_config_dict,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_websocket_communication
)


from gpt_researcher.utils.logging_config import setup_research_logging

import logging

# Get logger instance
logger = logging.getLogger(__name__)

# Don't override parent logger settings
logger.propagate = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Only log to console
    ]
)

# Models

# Pydantic model for the new endpoint
class GenerateReportRequest(BaseModel):
    prompt: str
    tone: Tone
    report_source: ReportSource
    report_type: ReportType

class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str


class ConfigRequest(BaseModel):
    ANTHROPIC_API_KEY: str
    TAVILY_API_KEY: str
    LANGCHAIN_TRACING_V2: str
    LANGCHAIN_API_KEY: str
    OPENAI_API_KEY: str
    DOC_PATH: str
    RETRIEVER: str
    GOOGLE_API_KEY: str = ''
    GOOGLE_CX_KEY: str = ''
    BING_API_KEY: str = ''
    SEARCHAPI_API_KEY: str = ''
    SERPAPI_API_KEY: str = ''
    SERPER_API_KEY: str = ''
    SEARX_URL: str = ''
    XAI_API_KEY: str
    DEEPSEEK_API_KEY: str


# App initialization
app = FastAPI()

# Static files and templates
app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")
templates = Jinja2Templates(directory="./frontend")

# WebSocket manager
manager = WebSocketManager()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

# Startup event


@app.get('/health')
async def root():
    return {
        "status": "ok",
        "version": "0.0.1",
    }

@app.on_event("startup")
def startup_event():
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    os.makedirs(DOC_PATH, exist_ok=True)
    

# Routes

# Create a folder to upload files per user's given folder name
# @app.post("/create-folder/{folder_name}")
# async def create_folder(folder_name: str):
#     """
#     Create a new folder under the DOC_PATH directory.
#     """
#     folder_path = os.path.join(DOC_PATH, folder_name)
#     try:
#         os.makedirs(folder_path, exist_ok=True)
#         return {"status": "success", "message": f"Folder '{folder_name}' created successfully."}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-folder/{folder_name}")
async def create_folder(folder_name: str):
    """
    Create a new folder under the DOC_PATH directory.
    """
    folder_path = os.path.join(DOC_PATH, folder_name)
    
    # Check if folder already exists
    if os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder '{folder_name}' already exists.")
    
    try:
        os.makedirs(folder_path, exist_ok=True)
        return {"status": "success", "message": f"Folder '{folder_name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-folders")
async def list_folders():
    """
    List all folders under the DOC_PATH directory.
    """
    try:
        # Get all directories under DOC_PATH
        folders = [f for f in os.listdir(DOC_PATH) if os.path.isdir(os.path.join(DOC_PATH, f))]
        return {"status": "success", "folders": folders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "report": None})


@app.get("/{folder_name}/files")
async def list_files(folder_name: str = None):
    """
    List files in a specific folder under the DOC_PATH directory.
    """
    folder_path = os.path.join(DOC_PATH, folder_name) if folder_name else DOC_PATH
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' does not exist.")

    files = os.listdir(folder_path)
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    return await execute_multi_agents(manager)


#New upload file option
@app.post("/upload/{folder_name}")
async def upload_file(folder_name: str, file: UploadFile = File(...)):
    """
    Upload a file to a specific folder under the DOC_PATH directory.
    """
    folder_path = os.path.join(DOC_PATH, folder_name)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' does not exist.")

    return await handle_file_upload(file, folder_path)

# @app.delete("/files/{filename}")
# async def delete_file(filename: str):
#     return await handle_file_deletion(filename, DOC_PATH)

@app.delete("/folders/{folder_name}")
async def delete_folder(folder_name: str):
    """
    Delete an entire folder and all its contents.
    Args:
        folder_name (str): The name of the folder to delete.
    Returns:
        dict: A message indicating success or failure.
    """
    try:
        # Construct the full folder path
        folder_path = os.path.join(DOC_PATH, folder_name)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found.")
        
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
        return {"status": "success", "message": f"Folder '{folder_name}' deleted successfully."}
    
    except Exception as e:
        logger.error(f"Error deleting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{folder_name}/{filename}")
async def delete_file(folder_name: str, filename: str):
    """
    Delete a specific file in a folder.
    Args:
        folder_name (str): The name of the folder containing the file.
        filename (str): The name of the file to delete.
    Returns:
        dict: A message indicating success or failure.
    """
    try:
        # Construct the full file path
        file_path = os.path.join(DOC_PATH, folder_name, filename)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found in folder '{folder_name}'.")
        
        # Delete the file
        os.remove(file_path)
        return {"status": "success", "message": f"File '{filename}' deleted successfully from folder '{folder_name}'."}
    
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)



@app.get("/generate-report")
async def generate_report(
    prompt: str,  # Query parameter for the task/prompt
    tone: Tone,  # Query parameter for the tone
    report_source: ReportSource,  # Query parameter for the report source
    report_type: ReportType,  # Query parameter for the report type
    folder_name: str = None,  # Optional query parameter for the folder name
    tavily_api_key: str = None,  # Optional query parameter for the Tavily API key
    open_ai_key: str = None  # Optional query parameter for the OpenAI API key
):
    """
    Generate a report using query parameters.
    """
    try:
        # If the user provides a Tavily API key, update the environment variables
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key

        if open_ai_key:
            os.environ["OPENAI_API_KEY"] = open_ai_key

        # Determine the document path
        if folder_name:
            document_path = os.path.join(DOC_PATH, folder_name)
            if not os.path.exists(document_path):
                raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' does not exist.")
        else:
            document_path = DOC_PATH

        # Get the list of documents in the folder
        document_urls = [os.path.join(document_path, f) for f in os.listdir(document_path) if os.path.isfile(os.path.join(document_path, f))]

        # Run the research task synchronously
        report = await run_agent(
            task=prompt,
            report_type=report_type.value,  # Pass the enum value
            report_source=report_source.value,  # Pass the enum value
            tone=tone.value,  # Pass the enum value
            websocket=None,  # No WebSocket for this endpoint
            headers=None,
            source_urls=[],  # No specific URLs
            document_urls=document_urls,  # Use documents from the specified folder
            query_domains=[],  # Add query domains if needed
            config_path="default"  # Add config path if needed
        )

        # Return the generated report
        return {"status": "success", "report": report}

    except Exception as e:
        # Handle errors
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/generate-report")
# async def generate_report(request: GenerateReportRequest, folder_name: str = None, 
#                           tavily_api_key: str = None, open_ai_key
#                           : str = None,):
#     """
#     Generate a report using the specified folder's documents.
#     """
#     try:
#         # Extract inputs from the request
        
#         tone = request.tone
#         report_source = request.report_source
#         report_type = request.report_type

#         # If the user provides a Tavily API key, update the environment variables
#         if tavily_api_key:
#             os.environ["TAVILY_API_KEY"] = tavily_api_key

#         if open_ai_key:
#             os.environ["OPENAI_API_KEY"] = open_ai_key

#         # Determine the document path
#         if folder_name:
#             document_path = os.path.join(DOC_PATH, folder_name)
#             if not os.path.exists(document_path):
#                 raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' does not exist.")
#         else:
#             document_path = DOC_PATH

#         # Get the list of documents in the folder
#         document_urls = [os.path.join(document_path, f) for f in os.listdir(document_path) if os.path.isfile(os.path.join(document_path, f))]

#         # Run the research task synchronously
#         report = await run_agent(
#             task=task,
#             report_type=report_type.value,  # Pass the enum value
#             report_source=report_source.value,  # Pass the enum value
#             tone=tone.value,  # Pass the enum value
#             websocket=None,  # No WebSocket for this endpoint
#             headers=None,
#             source_urls=[],  # No specific URLs
#             document_urls=document_urls,  # Use documents from the specified folder
#             query_domains=[],  # Add query domains if needed
#             config_path="default"  # Add config path if needed
#         )

#         # Return the generated report
#         return {"status": "success", "report": report}

#     except Exception as e:
#         # Handle errors
#         raise HTTPException(status_code=500, detail=str(e))
    


# {
#   "prompt": "What is the exams project?",
#   "tone": "Analytical (critical evaluation and detailed examination of data and theories)",
#   "report_source": "local",
#   "report_type": "detailed_report"
# }