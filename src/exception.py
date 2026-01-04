import sys
from src.logger import logging

def error_message_detail(error: Exception, error_detail) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: {file_name} at line number: {line_number} with message: {str(error)}"
    else:
        return f"Error occurred with message: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detail=None) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message