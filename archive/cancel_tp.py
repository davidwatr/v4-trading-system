import os
from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

client.cancel_order_by_id('55009486-d4a7-4123-8a8d-60946e0f47ac')
print('Canceled TP order')
