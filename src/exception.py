from types import ModuleType
from src.logger import logging

def error_message_detail(error: Exception, error_detail: ModuleType) -> str:
    _, _, exc_tb = error_detail.sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    if exc_tb is not None:

        return f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"
    else:
        file_name = "Unknown"
        line_number = "Unknown"
        return f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail: ModuleType) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message