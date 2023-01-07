# bot.py
import time

print("waiting 60 seconds until network start")
time.sleep(60)

import asyncio
import os

import discord
from dotenv import load_dotenv
import subprocess as sp
from threading import Thread

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")



intents = discord.Intents.default()
intents.message_content = True

no_exception = False

# while not no_exception:
#     try:
client = discord.Client(intents=intents)
    #     no_exception = True
    #     break
    # except Exception as e:
    #     pass
    # time.sleep(1)

start_cmd = "bash -x /home/pi/projects/cat_cam/run.sh"
end_gunicorn_cmd = "fuser -k 5000/tcp"
end_localtunnel_cmd = "fuser -k 5000/tcp"

help_message = """
/start - start the cat cam, outputs "CatCam ready!" when the cam has fully loaded.
/end or /stop - ends the cat cam, outputs "CatCam has been stopped!" when the cam has been stopped.
/start_debug - /start with stream logs
/end_debug or /stop_debug - /end with stream logs
"""

server_loaded_line = "your url is:"

server_process = None


@client.event
async def on_ready():
    print(f"{client.user} has connected to Discord!")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    global server_process

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
        await asyncio.sleep(1)
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

    if message.content.startswith("/help"):
        await message.channel.send(help_message)


no_exception = False

# while not no_exception:
#     try:
client.run(TOKEN)
    #     no_exception = True
    #     break
    # except Exception as e:
    #     pass
    # time.sleep(1)
