import pylivestream.api as pls

pls.stream_camera(
    ini_file="./pylivestream.json", websites=["twitch"], assume_yes=True, timeout=10
)
