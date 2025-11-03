import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

project = rf.workspace("cashydroprobe").project("digital-numbers-v3ajx")
version = project.version(1)
dataset = version.download("yolov11")