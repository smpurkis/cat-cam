import os
import cv2
from camera.base_camera import BaseCamera
from camera.utils import display_time_text_on_frame


class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.environ.get("OPENCV_CAMERA_SOURCE"):
            Camera.set_video_source(int(os.environ["OPENCV_CAMERA_SOURCE"]))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError("Could not start camera.")

        while True:
            # read current frame
            _, frame = camera.read()

            scale_percent = 75  # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            frame = display_time_text_on_frame(frame=frame)
            yield frame
