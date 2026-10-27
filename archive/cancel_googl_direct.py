import os
from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

client.cancel_order_by_id('b8273d9e-56da-40b5-a21d-cb0aefb2f0e3')
print('Canceled')
