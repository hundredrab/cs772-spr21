"""
API Server to get results from the model.

Based on a simple BaseHttpServer.

Usage:

python3 server.py -p 8000 -l localhost

-p, --port
"""

import argparse
import json
import pickle
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize


def toTensorFormat(embed_dict):
  new_dict = {}
  for i,j in embed_dict.items():
    new_dict[i] = torch.FloatTensor(j)
  return new_dict

print('-'*80)
pickle_in = open("word2vec.pkl", "rb")
word2vec = pickle.load(pickle_in)
word2vec = toTensorFormat(word2vec)
print('-'*80)

pickle_in = open("glove_dict_tensor.pkl", "rb")
glove = pickle.load(pickle_in)
print('-'*80)


pickle_in = open("fasttext.pkl", "rb")
fasttext = pickle.load(pickle_in)
fasttext = toTensorFormat(fasttext)
print('-'*80)

glove = defaultdict(lambda: torch.tensor(np.zeros(200)), glove)
fasttext = defaultdict(lambda: torch.tensor(np.zeros(300)), fasttext)
word2vec = defaultdict(lambda: torch.tensor(np.zeros(100)), word2vec)


def pad_sent(sent, n=30):
    return sent[:n] + [-1] * (n - len(sent))


class NET0(nn.Module):
    def __init__(self, input_dim=200*30, numClasses=5):
        super(NET0, self).__init__()
        self.input_dim = input_dim
        self.numClasses = numClasses
        self.output_layer = nn.Linear(self.input_dim, self.numClasses)

    def forward(self, text):
        return self.output_layer(text)


class NET(nn.Module):
    def __init__(self, input_dim=200*30, num_layers=1,
                 activation='sigmoid', numClasses=5, embedding='glove'):
        super(NET, self).__init__()
        self.input_dim = input_dim
        self.numClasses = numClasses
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'relu':
            self.activation = torch.nn.functional.relu
        else:
            self.activation = torch.sigmoid
        self.num_layers = num_layers
        self.neurons = 128
        self.layers = [nn.Linear(self.input_dim, 30*self.neurons)]
        self.neurons = (self.neurons//2)
        self.temp = 0
        for i in range(self.num_layers-1):
            self.layers.append(nn.Linear(30*self.neurons*2, 30*(self.neurons)))
            self.neurons = (self.neurons//2)

        self.layers = torch.nn.ModuleList(self.layers)
        self.output_layer = nn.Linear(30*self.neurons*2, self.numClasses)

    def forward(self, text):
        self.temp = self.activation(self.layers[0](text))
        for i in range(1, self.num_layers):
            self.temp = self.activation(self.layers[i](self.temp))
        return self.output_layer(self.temp)


model = NET(num_layers=1, activation='relu')
device = torch.device("cpu")
model = model.to(device)
fn = nn.CrossEntropyLoss()
fn.to(device)
optimizer = optim.Adam(params=model.parameters(), lr=0.0005)


def demo(sent: str, model, embedding='glove'):
    with torch.no_grad():
        text = sent
        text = word_tokenize(text)
        text = pad_sent(text)
        if embedding == 'glove':
            text = [glove[i].tolist() for i in text]
        if embedding == 'fasttext':
            text = [fasttext[i].tolist() for i in text]
        if embedding == 'word2vec':
            text = [word2vec[i].tolist() for i in text]
        text = torch.FloatTensor(np.array(text).flatten())
        out = model(text.float().to(device))

    preds = nn.Softmax(dim=1)(
        out.detach().cpu().reshape(1, -1)).numpy().tolist()[0]
    return preds


c = {
    'fasttext': 300*30,
    'word2vec': 100*30
}

models = {
    'layer0': NET0(),
    'layer1_relu': NET(num_layers=1, activation='relu'),
    'layer1_sigmoid': NET(num_layers=1, activation='sigmoid'),
    'layer2_relu': NET(num_layers=2, activation='relu'),
    'layer2_sigmoid': NET(num_layers=2, activation='sigmoid'),
    # 'layer3_relu': NET(num_layers=3, activation='relu'),
    'layer3_sigmoid': NET(num_layers=3, activation='sigmoid'),

    'layer0_fasttext_300dim': NET0(input_dim=c['fasttext']),
    'layer0_word2vec_100dim': NET0(input_dim=c['word2vec']),
    'layer1_relu_fasttext_300dim': NET(num_layers=1, activation='relu', input_dim=c['fasttext']),
    'layer1_relu_word2vec_100dim': NET(num_layers=1, activation='relu', input_dim=c['word2vec']),
    'layer1_sigmoid_fasttext_300dim': NET(num_layers=1, activation='sigmoid', input_dim=c['fasttext']),
    'layer1_sigmoid_word2vec_100dim': NET(num_layers=1, activation='sigmoid', input_dim=c['word2vec']),
    'layer2_relu_fasttext_300dim': NET(num_layers=2, activation='relu', input_dim=c['fasttext']),
    'layer2_relu_word2vec_100dim': NET(num_layers=2, activation='relu', input_dim=c['word2vec']),
    'layer2_sigmoid_fasttext_300dim': NET(num_layers=2, activation='sigmoid', input_dim=c['fasttext']),
    'layer2_sigmoid_word2vec_100dim': NET(num_layers=2, activation='sigmoid', input_dim=c['word2vec']),
    'layer3_relu_fasttext_300dim': NET(num_layers=3, activation='relu', input_dim=c['fasttext']),
    'layer3_relu_word2vec_100dim': NET(num_layers=3, activation='relu', input_dim=c['word2vec']),
    'layer3_sigmoid_fasttext_300dim': NET(num_layers=3, activation='sigmoid', input_dim=c['fasttext']),
    'layer3_sigmoid_word2vec_100dim': NET(num_layers=3, activation='sigmoid', input_dim=c['word2vec']),
}
# import pdb; pdb.set_trace();
print(models.keys())

for pth in models:
    models[pth].load_state_dict(torch.load(f'models/{pth}.pth', map_location='cpu'))
    print(f"Loaded {pth}")

demo("Very Good", models['layer2_relu'])
#==============================================================


HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:76.0) "
    "Gecko/20100101 Firefox/76.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Content-Type": "application/x-www-form-urlencoded",
    "DNT": "1",
    "Connection": "keep-alive",
}


class APIServer(BaseHTTPRequestHandler):
    """Basic server to handle incoming API requests."""

    def _set_headers(self, status):
        self.send_response(status)
        self.send_header("Content-type", "application/json")

        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "X-Requested-With, Content-type"
        )

        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        data = self.path[1:]
        print(data)
        status, details = get_details(data)
        self._set_headers(status)
        self.wfile.write(details.encode())

    def do_OPTIONS(self):
        """Handle OPTIONS requests."""
        self.send_response(200, "ok")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "X-Requested-With, Content-type"
        )


def run(server_class=HTTPServer, handler_class=APIServer,
        addr="localhost", port=8000):
    """Run the server."""
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


def algorithm(a, b=None):
    """
    Get results from the two models.


    Machine Learning go brrrr.
    """
    if not b:
        b = 'layer1_relu'
    with torch.no_grad():
        emb = 'glove'
        if 'fasttext' in b:
            emb = 'fasttext'
        if 'word2vec' in b:
            emb = 'word2vec'
        print(a, b, models[b], emb)
        return demo(a, models[b], emb)


def get_details(data):
    """
    Converts results from algorithms into human-friendly mesages.
    """

    data = data.strip().split(r"%20123%20")
    return 200, json.dumps({
        "message": "What's up?",
        "probabilities": algorithm(' '.join(data[0].split('%20')), data[1])
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)
