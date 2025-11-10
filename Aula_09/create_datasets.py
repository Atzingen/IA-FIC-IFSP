


# import kagglehub
# path = kagglehub.dataset_download("datasets/ahmedhaytham/chess-object-detection-yolov5-for-chess")

# print("Path to dataset files:", path)



# python chessred.py --dataroot 'path/to/save/dataset' --download




# from roboflow import Roboflow
# rf = Roboflow(api_key="")
# project = rf.workspace("3week").project("king-queen-pawn")
# version = project.version(1)
# dataset = version.download("yolov11")




import kagglehub

# Download latest version
path = kagglehub.dataset_download("krithiik/chess-piece-detection-annotated-for-yolo")

print("Path to dataset files:", path)