from __future__ import print_function
from pathlib import Path
from twitchstream.outputvideo import TwitchBufferedOutputStream, TwitchOutputStream
import argparse
import time
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple
import dotenv
import os

dotenv.load_dotenv()


def display_text_on_frame(frame, text: str, text_x_y: Tuple[int, int]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.6

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
    frame = display_text_on_frame(frame=frame, text=text, text_x_y=(12, 20))
    return frame


def check_stop_stream():
    if Path("twitch/stop_stream").exists():
        exit(0)


def reset_stream():
    Path("twitch/stop_stream").unlink(missing_ok=True)


def run_twitch():
    # parser = argparse.ArgumentParser(description=__doc__)
    # required = parser.add_argument_group("required arguments")
    # required.add_argument("-u", "--username", help="twitch username", required=True)
    # required.add_argument(
    #     "-o",
    #     "--oauth",
    #     help="twitch oauth "
    #     "(visit https://twitchapps.com/tmi/ "
    #     "to create one for your account)",
    #     required=True,
    # )
    # required.add_argument("-s", "--streamkey", help="twitch streamkey", required=True)
    # args = parser.parse_args()
    reset_stream()
    camera = cv2.VideoCapture(-1)

    # load two streams:
    # * one stream to send the video
    # * one stream to interact with the chat
    height, width = 480, 640

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    with TwitchBufferedOutputStream(
        # twitch_stream_key=args.streamkey,
        twitch_stream_key=os.environ["TWITCH_STREAM_KEY"],
        width=width,
        height=height,
        fps=10.0,
        enable_audio=False,
        verbose=True,
    ) as videostream:
        previous_frame = np.zeros((height, width, 3))

        # frame = np.zeros((480, 640, 3))
        ret, frame = camera.read()

        frame = display_time_text_on_frame(frame=frame)
        frequency = 100
        last_phase = 0

        # The main loop to create videos
        while True:
            check_stop_stream()
            # frame = np.zeros((480, 640, 3))
            ret, frame = camera.read()
            # frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = display_time_text_on_frame(frame=frame)
            frame[:, :, :] = 255 - frame[:, :, :]

            # if not ret:
            #     frame = previous_frame

            # If there are not enough video frames left,
            # add some more.
            if videostream.get_video_frame_buffer_state() < 1:
                # print(frame.shape)
                # cv2.imwrite("image.jpeg", frame)
                videostream.send_video_frame(frame)
            # videostream.send_video_frame(frame)

            # If there are not enough audio fragments left,
            # add some more, but take care to stay in sync with
            # the video! Audio and video buffer separately,
            # so they will go out of sync if the number of video
            # frames does not match the number of audio samples!
            # elif videostream.get_audio_buffer_state() < 30:
            #     x = np.linspace(
            #         last_phase,
            #         last_phase + frequency * 2 * np.pi / videostream.fps,
            #         int(44100 / videostream.fps) + 1,
            #     )
            #     last_phase = x[-1]
            #     audio = np.sin(x[:-1])
            #     videostream.send_audio(audio, audio)

            # If nothing is happening, it is okay to sleep for a while
            # and take some pressure of the CPU. But not too long, if
            # the buffers run dry, audio and video will go out of sync.
            else:
                time.sleep(0.001)
            # time.sleep(0.001)


if __name__ == "__main__":
    run_twitch()
