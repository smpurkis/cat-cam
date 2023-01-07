from discord import SyncWebhook

webhook = SyncWebhook.from_url(
    "https://discord.com/api/webhooks/1060990017440317499/mJwz-D-YsfcFPmZZuNuh-yybQQlDklAE3EmYjOSv_aeO3lnAJNbmfNcKT6qylKN9cCsR"
)
webhook.send("Hello World")
