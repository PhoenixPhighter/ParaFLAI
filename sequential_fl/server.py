import flwr as fl
from pathlib import Path
# from flask import Flask, Response

fl.server.start_server(
    server_address="0.0.0.0:80",
    config=fl.server.ServerConfig(num_rounds=3),
        certificates=(
        Path(".cache/certificates/ca.crt").read_bytes(),
        Path(".cache/certificates/server.pem").read_bytes(),
        Path(".cache/certificates/server.key").read_bytes(),
    )
)

# app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<p>Hello, World!</p>"

# @app.route("/start_server")
# def start_server():
    
#     response = Response('Started Server!', status=200)
#     return response