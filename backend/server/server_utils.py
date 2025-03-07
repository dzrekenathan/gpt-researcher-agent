import json
import os
import re
import time
import shutil
from typing import Dict, List, Any
from fastapi.responses import JSONResponse, FileResponse
from gpt_researcher.document.document import DocumentLoader
from gpt_researcher import GPTResearcher
from backend.utils import write_md_to_pdf, write_md_to_word, write_text_to_md
from pathlib import Path
from datetime import datetime
from fastapi import HTTPException
import logging
from fastapi import UploadFile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CustomLogsHandler:
    """Custom handler to capture streaming logs from the research process"""
    def __init__(self, websocket, task: str):
        self.logs = []
        self.websocket = websocket
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")
        self.log_file = os.path.join("outputs", f"{sanitized_filename}.json")
        self.timestamp = datetime.now().isoformat()
        # Initialize log file with metadata
        os.makedirs("outputs", exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "events": [],
                "content": {
                    "query": "",
                    "sources": [],
                    "context": [],
                    "report": "",
                    "costs": 0.0
                }
            }, f, indent=2)

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Store log data and send to websocket"""
        # Send to websocket for real-time display
        if self.websocket:
            await self.websocket.send_json(data)
            
        # Read current log file
        with open(self.log_file, 'r') as f:
            log_data = json.load(f)
            
        # Update appropriate section based on data type
        if data.get('type') == 'logs':
            log_data['events'].append({
                "timestamp": datetime.now().isoformat(),
                "type": "event",
                "data": data
            })
        else:
            # Update content section for other types of data
            log_data['content'].update(data)
            
        # Save updated log file
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        logger.debug(f"Log entry written to: {self.log_file}")


# class Researcher:
#     def __init__(self, query: str, report_type: str = "research_report"):
#         self.query = query
#         self.report_type = report_type
#         # Generate unique ID for this research task
#         self.research_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query)}"
#         # Initialize logs handler with research ID
#         self.logs_handler = CustomLogsHandler(None, self.research_id)
#         self.researcher = GPTResearcher(
#             query=query,
#             report_type=report_type,
#             websocket=self.logs_handler
#         )
# Researcher class updated
class Researcher:
    def __init__(self, query: str, report_type: str = "research_report", folder_name: str = None):
        self.query = query
        self.report_type = report_type
        self.folder_name = folder_name
        self.research_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(query)}"
        self.logs_handler = CustomLogsHandler(None, self.research_id)
        
        # Get documents from specified folder
        self.document_urls = get_folder_documents(folder_name)
        
        self.researcher = GPTResearcher(
            query=query,
            report_type=report_type,
            websocket=self.logs_handler,
            document_urls=self.document_urls
        )

    async def research(self) -> dict:
        """Conduct research and return paths to generated files"""
        await self.researcher.conduct_research()
        report = await self.researcher.write_report()
        
        # Generate the files
        sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{self.query}")
        file_paths = await generate_report_files(report, sanitized_filename)
        
        # Get the JSON log path that was created by CustomLogsHandler
        json_relative_path = os.path.relpath(self.logs_handler.log_file)
        
        return {
            "output": {
                **file_paths,  # Include PDF, DOCX, and MD paths
                "json": json_relative_path
            }
        }

def sanitize_filename(filename: str) -> str:
    # Split into components
    prefix, timestamp, *task_parts = filename.split('_')
    task = '_'.join(task_parts)
    
    # Calculate max length for task portion
    # 255 - len("outputs/") - len("task_") - len(timestamp) - len("_.json") - safety_margin
    max_task_length = 255 - 8 - 5 - 10 - 6 - 10  # ~216 chars for task
    
    # Truncate task if needed
    truncated_task = task[:max_task_length] if len(task) > max_task_length else task
    
    # Reassemble and clean the filename
    sanitized = f"{prefix}_{timestamp}_{truncated_task}"
    return re.sub(r"[^\w\s-]", "", sanitized).strip()


# async def handle_start_command(websocket, data: str, manager):
#     json_data = json.loads(data[6:])
#     (
#         task,
#         report_type,
#         source_urls,
#         document_urls,
#         tone,
#         headers,
#         report_source,
#         query_domains,
#     ) = extract_command_data(json_data)

#     if not task or not report_type:
#         print("Error: Missing task or report_type")
#         return

#     # Create logs handler with websocket and task
#     logs_handler = CustomLogsHandler(websocket, task)
#     # Initialize log content with query
#     await logs_handler.send_json({
#         "query": task,
#         "sources": [],
#         "context": [],
#         "report": ""
#     })

#     sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")

#     report = await manager.start_streaming(
#         task,
#         report_type,
#         report_source,
#         source_urls,
#         document_urls,
#         tone,
#         websocket,
#         headers,
#         query_domains,
#     )
#     report = str(report)
#     file_paths = await generate_report_files(report, sanitized_filename)
#     # Add JSON log path to file_paths
#     file_paths["json"] = os.path.relpath(logs_handler.log_file)
#     await send_file_paths(websocket, file_paths)
# Handle Start Command Updated
async def handle_start_command(websocket, data: str, manager):
    json_data = json.loads(data[6:])
    (
        task,
        report_type,
        source_urls,
        folder_name,  # Added folder name parameter
        tone,
        headers,
        report_source,
        query_domains,
    ) = extract_command_data(json_data)

    # Get documents from specified folder
    document_urls = get_folder_documents(folder_name)
    
    # Rest of the existing code...
    sanitized_filename = sanitize_filename(f"task_{int(time.time())}_{task}")
    
    # Modified to include folder name in report generation
    file_paths = await generate_report_files(report, sanitized_filename, folder_name)
    
    # Return paths relative to output directory
    return {
        "output": {
            **file_paths,
            "json": os.path.relpath(logs_handler.log_file)
        }
    }

async def handle_human_feedback(data: str):
    feedback_data = json.loads(data[14:])  # Remove "human_feedback" prefix
    print(f"Received human feedback: {feedback_data}")
    # TODO: Add logic to forward the feedback to the appropriate agent or update the research state

async def handle_chat(websocket, data: str, manager):
    json_data = json.loads(data[4:])
    print(f"Received chat message: {json_data.get('message')}")
    await manager.chat(json_data.get("message"), websocket)

# async def generate_report_files(report: str, filename: str) -> Dict[str, str]:
#     pdf_path = await write_md_to_pdf(report, filename)
#     docx_path = await write_md_to_word(report, filename)
#     md_path = await write_text_to_md(report, filename)
#     return {"pdf": pdf_path, "docx": docx_path, "md": md_path}
# Generate report files method updated
async def generate_report_files(report: str, filename: str, folder_name: str = None) -> Dict[str, str]:
    """Generate report files in folder-specific output directory"""
    base_output = Path(os.getenv("OUTPUT_PATH", "outputs"))
    
    if folder_name:
        output_dir = base_output / folder_name
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = base_output
        
    sanitized_name = sanitize_filename(filename)
    
    pdf_path = await write_md_to_pdf(report, str(output_dir / sanitized_name))
    docx_path = await write_md_to_word(report, str(output_dir / sanitized_name))
    md_path = await write_text_to_md(report, str(output_dir / sanitized_name))
    
    return {
        "pdf": str(pdf_path.relative_to(base_output)),
        "docx": str(docx_path.relative_to(base_output)),
        "md": str(md_path.relative_to(base_output))
    }


async def send_file_paths(websocket, file_paths: Dict[str, str]):
    await websocket.send_json({"type": "path", "output": file_paths})



def get_config_dict(
    langchain_api_key: str, openai_api_key: str, tavily_api_key: str,
    google_api_key: str, google_cx_key: str, bing_api_key: str,
    searchapi_api_key: str, serpapi_api_key: str, serper_api_key: str, searx_url: str
) -> Dict[str, str]:
    return {
        "LANGCHAIN_API_KEY": langchain_api_key or os.getenv("LANGCHAIN_API_KEY", ""),
        "OPENAI_API_KEY": openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        "TAVILY_API_KEY": tavily_api_key or os.getenv("TAVILY_API_KEY", ""),
        "GOOGLE_API_KEY": google_api_key or os.getenv("GOOGLE_API_KEY", ""),
        "GOOGLE_CX_KEY": google_cx_key or os.getenv("GOOGLE_CX_KEY", ""),
        "BING_API_KEY": bing_api_key or os.getenv("BING_API_KEY", ""),
        "SEARCHAPI_API_KEY": searchapi_api_key or os.getenv("SEARCHAPI_API_KEY", ""),
        "SERPAPI_API_KEY": serpapi_api_key or os.getenv("SERPAPI_API_KEY", ""),
        "SERPER_API_KEY": serper_api_key or os.getenv("SERPER_API_KEY", ""),
        "SEARX_URL": searx_url or os.getenv("SEARX_URL", ""),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "DOC_PATH": os.getenv("DOC_PATH", "./my-docs"),
        "RETRIEVER": os.getenv("RETRIEVER", ""),
        "EMBEDDING_MODEL": os.getenv("OPENAI_EMBEDDING_MODEL", "")
    }


def update_environment_variables(config: Dict[str, str]):
    for key, value in config.items():
        os.environ[key] = value


async def handle_file_upload(file: UploadFile, folder_name: str = None) -> Dict[str, str]:
    """Handle file upload to a specific folder with validation."""
    # Define the base path only once - this is the root directory
    base_path = os.getenv("DOC_PATH", "./my-docs")
    
    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)
    
    # Construct the folder path based on whether a folder_name is provided
    if folder_name:
        folder_path = os.path.join(base_path, folder_name)
    else:
        folder_path = base_path
    
    # Create the specific folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the full file path
    file_path = os.path.join(folder_path, file.filename)
    
    # Normalize the file path to avoid redundant separators
    file_path = os.path.normpath(file_path)
    
    # Check if the file already exists
    if os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File already exists")
    
    try:
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load the newly uploaded document (if needed)
        document_loader = DocumentLoader([file_path])
        await document_loader.load()
        
        # Return the correct path format
        return {"filename": file.filename, "path": file_path.replace("\\", "/")}
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")


async def handle_file_deletion(filename: str, folder_name: str = None) -> JSONResponse:
    """Handle file deletion from a specific folder."""
    base_path = os.getenv("DOC_PATH", "./my-docs")
    
    # Ensure the folder path is correctly constructed
    folder_path = os.path.join(base_path, folder_name) if folder_name else base_path
    
    # Construct the full file path
    file_path = os.path.join(folder_path, os.path.basename(filename))
    
    # Check if the file exists
    if os.path.exists(file_path):
        try:
            # Delete the file
            os.remove(file_path)
            logger.info(f"File deleted: {file_path}")
            return JSONResponse(content={"message": "File deleted successfully"})
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            raise HTTPException(status_code=500, detail="File deletion failed")
    else:
        logger.warning(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

# async def handle_file_deletion(filename: str, DOC_PATH: str) -> JSONResponse:
#     file_path = os.path.join(DOC_PATH, os.path.basename(filename))
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         print(f"File deleted: {file_path}")
#         return JSONResponse(content={"message": "File deleted successfully"})
#     else:
#         print(f"File not found: {file_path}")
#         return JSONResponse(status_code=404, content={"message": "File not found"})


async def execute_multi_agents(manager) -> Any:
    websocket = manager.active_connections[0] if manager.active_connections else None
    if websocket:
        report = await run_research_task("Is AI in a hype cycle?", websocket, stream_output)
        return {"report": report}
    else:
        return JSONResponse(status_code=400, content={"message": "No active WebSocket connection"})


async def handle_websocket_communication(websocket, manager):
    while True:
        try:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            elif data.startswith("start"):
                await handle_start_command(websocket, data, manager)
            elif data.startswith("human_feedback"):
                await handle_human_feedback(data)
            elif data.startswith("chat"):
                await handle_chat(websocket, data, manager)
            else:
                print("Error: Unknown command or not enough parameters provided.")
        except Exception as e:
            print(f"WebSocket error: {e}")
            break


# def extract_command_data(json_data: Dict) -> tuple:
#     return (
#         json_data.get("task"),
#         json_data.get("report_type"),
#         json_data.get("source_urls"),
#         json_data.get("document_urls"),
#         json_data.get("tone"),
#         json_data.get("headers", {}),
#         json_data.get("report_source"),
#         json_data.get("query_domains", []),
#     )

def extract_command_data(json_data: Dict) -> tuple:
    return (
        json_data.get("task"),
        json_data.get("report_type"),
        json_data.get("source_urls"),
        json_data.get("folder_name"),  # Added folder name extraction
        json_data.get("tone"),
        json_data.get("headers", {}),
        json_data.get("report_source"),
        json_data.get("query_domains", []),
    )

# This should get the endpoint to the file location
def get_folder_documents(folder_name: str = None) -> List[str]:
    """Get document paths from specified folder or root DOC_PATH"""
    base_path = os.getenv("DOC_PATH", "./my-docs")
    folder_path = os.path.join(base_path, folder_name) if folder_name else base_path
    
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found")
    
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]