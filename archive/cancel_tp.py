from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

client.cancel_order_by_id('55009486-d4a7-4123-8a8d-60946e0f47ac')
print('Canceled TP order')
