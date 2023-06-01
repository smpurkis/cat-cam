from pathlib import Path
import time

import asyncio
import os
from typing import Optional

import discord
from dotenv import load_dotenv
import subprocess as sp
from discord import SyncWebhook
from multiprocessing import Process

# from twitch.play_twitch import run_twitch


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if TOKEN is None:
    raise Exception("DISCORD_TOKEN environment variable is not set")

intents = discord.Intents.default()
intents.message_content = True

no_exception = False

client = discord.Client(intents=intents)

start_cmd = "bash -x ./run.sh"
end_gunicorn_cmd = "fuser -k 5000/tcp"
launch_twitch_cmd = "bash -x ./run.sh"

help_message = """
/tstart - run twitch stream
/tstop - stop twitch stream
/reboot - reboot raspberry pi
/help - show this message
"""

server_loaded_line = "your url is:"

server_process = None
twitch_process: Optional[sp.Popen] = None


def stop_stream():
    Path("./twitch/stop_stream").write_text("s")


def reset_stream():
    Path("./twitch/stop_stream").unlink(missing_ok=True)


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
    if message.content.startswith("/tstart"):
        # process = Process(target=run_twitch)
        # process = Process(target=sp.Popen, args=(launch_twitch_cmd,))
        process = sp.Popen(launch_twitch_cmd.split())
        # process.run()
        twitch_process = process
        await asyncio.sleep(3)
        await message.channel.send(f"View at: https://www.twitch.tv/zyguard7777777")
        await message.channel.send("CatCam ready!")

    if message.content.startswith("/tstop"):
        twitch_process.terminate()
        stop_stream()
        await asyncio.sleep(1)
        await message.channel.send("CatCam has been stopped!")
        reset_stream()

    if message.content.startswith("/reboot"):
        await message.channel.send("Triggering Raspberry Pi Reboot!")
        sp.run("sudo reboot".split())

    if message.content.startswith("/help"):
        await message.channel.send(help_message)


no_exception = False

client.run(TOKEN)
