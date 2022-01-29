Libraries used for this project:
- numpy == 1.21.2
- opencv-python == 4.5.3.56
- opencv-utils == 0.0.2
- cv2-utils == 0.2.0
- math
- cv2
- queue (from queue import Queue)
- os

How to run the code:
script: sudoku.py
function: sudoku(task_number, images_path, predictions_filename_path), where:
	- task_number is the number of the task (e.g: "task1", "task2")
	- images_path is the path to the folder containing the images for the task
	- predictions_filename_path, the path to the directory where the predictions will be written
output: the predictions(output files) will be written in the given directory having ".txt" extension, for example: ("1_predicted.txt", "10_predicted.txt")

Change these variables for each task and run "sudoku.py" script.