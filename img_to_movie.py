import os
import cv2


IMAGES_DIRECTORY = os.path.join("data", "prezentace")

img_array = []
files_to_process = os.listdir(os.path.join(IMAGES_DIRECTORY))
files_to_process.sort()

size = (0, 0)
for repeat in range(1):
    for idx, filename in enumerate(files_to_process):

        img = cv2.imread(os.path.join(IMAGES_DIRECTORY, str(filename)))
        resized = cv2.resize(img, (1920,1080), interpolation=cv2.INTER_AREA)
        height, width, layers = resized.shape
        size = (width, height)
        print(size)
        for i in range(300):
            img_array.append(resized)


out = cv2.VideoWriter('prezentace_programy.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30.30, size)


for i, img in enumerate(img_array):
    out.write(img_array[i])
out.release()
