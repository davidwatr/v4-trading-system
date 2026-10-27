import os
from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Cancel specific TSLA sell order by ID
try:
    client.cancel_order_by_id('8ea18a9e-082f-462e-9924-4d169d0a0497')
    print('Canceled TSLA sell order')
except Exception as e:
    print(f'Error: {e}')
