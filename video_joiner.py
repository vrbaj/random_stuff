import cv2
# A list of the paths of your videos
videos = ["UPRTv3.mp4", "prezentace_programy.mp4"]

# Create a new video
video = cv2.VideoWriter("new_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920,1080))

# Write all the frames sequentially to the new video
for v in videos:
    print("processing ", v)
    curr_v = cv2.VideoCapture(v)
    while curr_v.isOpened():
        r, frame = curr_v.read()    # Get return value and curr frame of curr video
        if not r:
            break
        video.write(frame)          # Write the frame

video.release()