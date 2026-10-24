from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

client.cancel_order_by_id('b8273d9e-56da-40b5-a21d-cb0aefb2f0e3')
print('Canceled')
