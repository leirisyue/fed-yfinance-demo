# Crypto Data Science Kết hợp Federated Learning
```bash
https://github.com/leirisyue/fed-yfinance-demo.git
```

### start server
```bash
python -m pt_server -r 50
```
### Start client alice
```bash
python -m pt_client -s alice
```

### Start client alice
```bash
python -m pt_client -s bob
```


# Deploy VMS azure
### error with linux VMS
``
On Linux, run first ``sudo apt update``. Then the command would be: ``sudo apt install python3-pip``
``

pip3 install -U numpy
pip3 install -U scikit-learn scipy matplotlib
pip3 install -U torch==1.9.1
pip3 install -U torchvision==0.10.1
pip3 install -U flwr==0.17.0

<div align="center">
<img src="asset/workflow.png">
</div>