from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

# Cancel specific TSLA sell order by ID
try:
    client.cancel_order_by_id('8ea18a9e-082f-462e-9924-4d169d0a0497')
    print('Canceled TSLA sell order')
except Exception as e:
    print(f'Error: {e}')
