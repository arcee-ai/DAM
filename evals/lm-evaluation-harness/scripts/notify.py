#!/usr/bin/env python
# This is an example of sending a slack notification. For more details see
# official docs:
# https://api.slack.com/messaging/webhooks

import requests
import json
import os

# This URL is tied to a single channel. That can be generalized, or you can
# create a new "app" to use another channel.
WEBHOOK = os.environ.get("WEBHOOK_URL")
if WEBHOOK is None:
    print("Webhook URL not found in WEBHOOK_URL env var. Will just print messages.")


def notify(message):
    headers = {"Content-Type": "application/json"}
    data = json.dumps({"text": message})
    if WEBHOOK is None:
        print(message)
    else:
        requests.post(WEBHOOK, data=data, headers=headers)


if __name__ == "__main__":
    print("Please type your message.")
    message = input("message> ")
    notify(message)
