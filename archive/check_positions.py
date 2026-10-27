from alpaca.trading.client import TradingClient
import os
os.environ['APCA_API_KEY_ID'] = 'PK7F31Q9KU3CRWUQYH7Y'
os.environ['APCA_API_SECRET_KEY'] = 'XJh1wE1NylF2h2TzWlqgOu6YyHsQY06PJh0zSsHx'
os.environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets'
client = TradingClient('PK7F31Q9KU3CRWUQYH7Y', 'XJh1wE1NylF2h2TzWlqgOu6YyHsQY06PJh0zSsHx', paper=True)
positions = client.get_all_positions()
print(f'Total positions: {len(positions)}')
for p in positions:
    print(f'  {p.symbol}: {p.qty} shares, Entry: ${p.avg_entry_price}, PnL: ${p.unrealized_pl:.2f}')
