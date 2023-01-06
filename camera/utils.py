from typing import Tuple
import cv2
from datetime import datetime


def display_text_on_frame(frame, text: str, text_x_y: Tuple[int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.3

    # Get the size of the text
    (text_width, text_height) = cv2.getTextSize(text, font, font_size, 1)[0]

    # Calculate the coordinates for the text
    text_x = int((frame.shape[1] - text_width) / 2)
    text_y = int((frame.shape[0] + text_height) / 2)

    text_x = text_x_y[0]  # This is the distance from the left border of the image
    text_y = text_x_y[1]  # This is the distance from the top border of the image

    # Overlay the text on the image
    cv2.putText(
        frame, text, (text_x, text_y), font, font_size, (255, 255, 255), 1, cv2.LINE_AA
    )
    return frame


def display_time_text_on_frame(frame):

    # Define the text to be overlaid
    text = str(datetime.now()).split(" ")[1].split(".")[0]
    frame = display_text_on_frame(frame=frame, text=text, text_x_y=(10, 10))
    return frame
