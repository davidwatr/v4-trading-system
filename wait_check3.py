import time
time.sleep(30)

from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

positions = client.get_all_positions()
print('Positions:')
for p in positions:
    print(f'  {p.symbol}: {int(p.qty)} shares')
