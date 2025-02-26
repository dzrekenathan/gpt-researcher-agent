import asyncio
import os
from typing import List, Union
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.document_loaders import BSHTMLLoader


class DocumentLoader:
    def __init__(self, path: Union[str, List[str]]):
        """
        Initialize the DocumentLoader.

        Args:
            path: Either a single folder path or a list of file paths.
        """
        self.path = path

    async def load(self) -> list:
        """
        Load documents from the specified path(s).

        Returns:
            List of documents with their raw content and metadata.
        """
        tasks = []

        # If path is a list of file paths
        if isinstance(self.path, list):
            for file_path in self.path:
                if os.path.isfile(file_path):  # Ensure it's a valid file
                    file_extension = self._get_file_extension(file_path)
                    tasks.append(self._load_document(file_path, file_extension))

        # If path is a folder path
        elif isinstance(self.path, (str, bytes, os.PathLike)):
            if os.path.isdir(self.path):  # Ensure it's a valid directory
                for root, _, files in os.walk(self.path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_extension = self._get_file_extension(file_path)
                        tasks.append(self._load_document(file_path, file_extension))
            else:
                raise ValueError(f"Path '{self.path}' is not a valid directory.")

        else:
            raise ValueError("Invalid type for path. Expected str, bytes, os.PathLike, or list thereof.")

        # Gather all document loading tasks
        docs = []
        for pages in await asyncio.gather(*tasks):
            for page in pages:
                if page.page_content:
                    docs.append({
                        "raw_content": page.page_content,
                        "url": os.path.basename(page.metadata['source'])
                    })

        if not docs:
            raise ValueError("🤷 Failed to load any documents!")

        return docs

    def _get_file_extension(self, file_path: str) -> str:
        """
        Extract and normalize the file extension.

        Args:
            file_path: Path to the file.

        Returns:
            Normalized file extension (lowercase, without the dot).
        """
        _, file_extension_with_dot = os.path.splitext(file_path)
        return file_extension_with_dot.strip(".").lower()

    async def _load_document(self, file_path: str, file_extension: str) -> list:
        """
        Load a single document based on its file extension.

        Args:
            file_path: Path to the file.
            file_extension: File extension (e.g., "pdf", "docx").

        Returns:
            List of document pages.
        """
        ret_data = []
        try:
            loader_dict = {
                "pdf": PyMuPDFLoader(file_path),
                "txt": TextLoader(file_path),
                "doc": UnstructuredWordDocumentLoader(file_path),
                "docx": UnstructuredWordDocumentLoader(file_path),
                "pptx": UnstructuredPowerPointLoader(file_path),
                "csv": UnstructuredCSVLoader(file_path, mode="elements"),
                "xls": UnstructuredExcelLoader(file_path, mode="elements"),
                "xlsx": UnstructuredExcelLoader(file_path, mode="elements"),
                "md": UnstructuredMarkdownLoader(file_path),
                "html": BSHTMLLoader(file_path),
                "htm": BSHTMLLoader(file_path),
            }

            loader = loader_dict.get(file_extension)
            if loader:
                try:
                    ret_data = loader.load()
                except Exception as e:
                    print(f"Failed to load document: {file_path}")
                    print(e)
            else:
                print(f"Unsupported file type: {file_extension} for file: {file_path}")

        except Exception as e:
            print(f"Failed to load document: {file_path}")
            print(e)

        return ret_data
# import asyncio
# import os
# from typing import List, Union
# from langchain_community.document_loaders import (
#     PyMuPDFLoader,
#     TextLoader,
#     UnstructuredCSVLoader,
#     UnstructuredExcelLoader,
#     UnstructuredMarkdownLoader,
#     UnstructuredPowerPointLoader,
#     UnstructuredWordDocumentLoader
# )
# from langchain_community.document_loaders import BSHTMLLoader


# class DocumentLoader:

#     def __init__(self, path: Union[str, List[str]]):
#         self.path = path

#     async def load(self) -> list:
#         tasks = []
#         if isinstance(self.path, list):
#             for file_path in self.path:
#                 if os.path.isfile(file_path):  # Ensure it's a valid file
#                     filename = os.path.basename(file_path)
#                     file_name, file_extension_with_dot = os.path.splitext(filename)
#                     file_extension = file_extension_with_dot.strip(".").lower()
#                     tasks.append(self._load_document(file_path, file_extension))
                    
#         elif isinstance(self.path, (str, bytes, os.PathLike)):
#             for root, dirs, files in os.walk(self.path):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     file_name, file_extension_with_dot = os.path.splitext(file)
#                     file_extension = file_extension_with_dot.strip(".").lower()
#                     tasks.append(self._load_document(file_path, file_extension))
                    
#         else:
#             raise ValueError("Invalid type for path. Expected str, bytes, os.PathLike, or list thereof.")

#         # for root, dirs, files in os.walk(self.path):
#         #     for file in files:
#         #         file_path = os.path.join(root, file)
#         #         file_name, file_extension_with_dot = os.path.splitext(file_path)
#         #         file_extension = file_extension_with_dot.strip(".")
#         #         tasks.append(self._load_document(file_path, file_extension))

#         docs = []
#         for pages in await asyncio.gather(*tasks):
#             for page in pages:
#                 if page.page_content:
#                     docs.append({
#                         "raw_content": page.page_content,
#                         "url": os.path.basename(page.metadata['source'])
#                     })
                    
#         if not docs:
#             raise ValueError("🤷 Failed to load any documents!")

#         return docs

#     async def _load_document(self, file_path: str, file_extension: str) -> list:
#         ret_data = []
#         try:
#             loader_dict = {
#                 "pdf": PyMuPDFLoader(file_path),
#                 "txt": TextLoader(file_path),
#                 "doc": UnstructuredWordDocumentLoader(file_path),
#                 "docx": UnstructuredWordDocumentLoader(file_path),
#                 "pptx": UnstructuredPowerPointLoader(file_path),
#                 "csv": UnstructuredCSVLoader(file_path, mode="elements"),
#                 "xls": UnstructuredExcelLoader(file_path, mode="elements"),
#                 "xlsx": UnstructuredExcelLoader(file_path, mode="elements"),
#                 "md": UnstructuredMarkdownLoader(file_path),
#                 "html": BSHTMLLoader(file_path),
#                 "htm": BSHTMLLoader(file_path)
#             }

#             loader = loader_dict.get(file_extension, None)
#             if loader:
#                 try:
#                     ret_data = loader.load()
#                 except Exception as e:
#                     print(f"Failed to load HTML document : {file_path}")
#                     print(e)

#         except Exception as e:
#             print(f"Failed to load document : {file_path}")
#             print(e)

#         return ret_data
