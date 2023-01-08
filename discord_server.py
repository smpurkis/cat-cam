# bot.py
from pathlib import Path
import time

# print("waiting 60 seconds until network start")
# time.sleep(60)

import asyncio
import os

import discord
from dotenv import load_dotenv
import subprocess as sp
from discord import SyncWebhook
from multiprocessing import Process

# from twitch.play_twitch import run_twitch


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")


intents = discord.Intents.default()
intents.message_content = True

no_exception = False

client = discord.Client(intents=intents)

start_cmd = "bash -x /home/pi/projects/cat_cam/run.sh"
end_gunicorn_cmd = "fuser -k 5000/tcp"
end_localtunnel_cmd = "fuser -k 5000/tcp"
launch_twitch_cmd = "bash -x /home/pi/projects/cat_cam/twitch/run.sh"

help_message = """
/start - start the cat cam, outputs "CatCam ready!" when the cam has fully loaded.
/end or /stop - ends the cat cam, outputs "CatCam has been stopped!" when the cam has been stopped.
/twitch_start - run twitch stream
/twitch_stop - stop twitch stream
/start_debug - /start with stream logs
/end_debug or /stop_debug - /end with stream logs
/reboot - reboot raspberry pi
"""

server_loaded_line = "your url is:"

server_process = None
twitch_process = None

def stop_stream():
    Path("/home/pi/projects/cat_cam/twitch/stop_stream").write_text("s")

def reset_stream():
    Path("/home/pi/projects/cat_cam/twitch/stop_stream").unlink(missing_ok=True)

@client.event
async def on_ready():
    print(f"{client.user} has connected to Discord!")
    webhook = SyncWebhook.from_url(
        "https://discord.com/api/webhooks/1060990017440317499/mJwz-D-YsfcFPmZZuNuh-yybQQlDklAE3EmYjOSv_aeO3lnAJNbmfNcKT6qylKN9cCsR"
    )
    webhook.send("CatCam Bot is up!")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    global server_process
    global twitch_process
    if message.content.startswith("/twitch_start"):
        # process = Process(target=run_twitch)
        # process = Process(target=sp.Popen, args=(launch_twitch_cmd,))
        process = sp.Popen(launch_twitch_cmd.split())
        # process.run()
        twitch_process = process
        await asyncio.sleep(3)
        await message.channel.send(f"View at: https://www.twitch.tv/zyguard7777777")
        await message.channel.send("CatCam ready!")

    if message.content.startswith("/twitch_stop"):
        twitch_process.terminate()
        stop_stream()
        await asyncio.sleep(1)
        await message.channel.send("CatCam has been stopped!")
        reset_stream()

    if message.content.startswith("/start") or message.content.startswith(
        "/start_debug"
    ):
        process = sp.Popen(start_cmd.split(), stdout=sp.PIPE, stderr=sp.STDOUT)
        server_process = process
        for stdout_line in iter(process.stdout.readline, ""):
            stdout_line = stdout_line.decode()
            print(stdout_line.replace("\n", ""))
            if message.content.startswith("/start_debug"):
                await message.channel.send(f"STDOUT: {stdout_line}")
            if server_loaded_line in stdout_line:
                await message.channel.send(f"STDOUT: {stdout_line}")
                break
        await asyncio.sleep(3)
        await message.channel.send("CatCam ready!")

    if (
        message.content.startswith("/end")
        or message.content.startswith("/end_debug")
        or message.content.startswith("/stop")
        or message.content.startswith("/stop_debug")
    ):
        server_process.terminate()
        process = sp.Popen(end_gunicorn_cmd.split(), stdout=sp.PIPE, stderr=sp.STDOUT)
        await asyncio.sleep(1)
        await message.channel.send("CatCam has been stopped!")

    if message.content.startswith("/reboot"):
        await message.channel.send("Triggering Raspberry Pi Reboot!")
        sp.run("sudo reboot".split())

    if message.content.startswith("/help"):
        await message.channel.send(help_message)


no_exception = False

client.run(TOKEN)
