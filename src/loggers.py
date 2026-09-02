import logging
import os 
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  #strftime() converts the date/time into a formatted string. adds .log to the end is created as the filename.
logs_path =os.path.join(os.getcwd(),"logs",LOG_FILE)  #Get Current Working Directory. os.path.join() combines all these pieces correctly for your operating system.
os.makedirs(logs_path,exist_ok=True)  #os.makedirs() creates directories/folders .  exist_ok=True If this folder already exists, don't throw an error.

LOG_FILE_PATH =os.path.join(logs_path,LOG_FILE)

logging.basicConfig( #basicConfig() is used to configure Python's logging system
    filename=LOG_FILE_PATH, # Store the logs in this file. 
    format = "[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # This controls how each log message will look.
    level = logging.INFO, #Python writes it into that file
    
    
)


