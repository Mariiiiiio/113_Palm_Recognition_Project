# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="Qc2s8bFhIj8k8rHQdxUn")
project = rf.workspace("feng-chia-university-wlii3").project("gan-v30in")
version = project.version(10)
dataset = version.download("yolov9")
                