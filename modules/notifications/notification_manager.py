class NotificationManager:
    def __init__(self):
        self.notifiers = []

    async def send_notification(self, msg):
        print("Sending:", msg)
        return True

    async def flush(self):
        return True

    async def close(self):
        return True
