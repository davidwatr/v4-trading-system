from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

client.cancel_order_by_id('bf1538e6-d421-48bb-b653-38a0fc1a586a')
print('Canceled')
