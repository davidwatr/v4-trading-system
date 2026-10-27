import os
from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

client.cancel_order_by_id('bf1538e6-d421-48bb-b653-38a0fc1a586a')
print('Canceled')
