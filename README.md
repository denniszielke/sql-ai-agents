# Talking to a database with a large language model

This project demonstrates different approaches on how to implement a bot that can talk to a relational database.

![image info](./screenshot.png)

## Deploy Infrastructure

```
echo "log into azure dev cli - only once"
azd auth login

echo "provisioning all the resources with the azure dev cli"
azd up

echo "get and set the value for AZURE_ENV_NAME"
source <(azd env get-values | grep AZURE_ENV_NAME)

echo "building and deploying the streamlit user interface"
bash ./azd-hooks/deploy.sh sql-agents $AZURE_ENV_NAME
```

## Starting up


```
conda create -n sqlagents python=3.12

conda activate sqlagents

python -m pip install -r requirements.txt   

python -m streamlit run app.py --server.port=8000
```