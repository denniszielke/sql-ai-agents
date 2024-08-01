# Talking to a database with a large language model

This project demonstrates different approaches on how to implement a bot that can talk to a relational database.

![image info](./screenshot.png)

## Starting up


```
conda create -n sqlagents python=3.12

conda activate sqlagents

python -m pip install -r requirements.txt   

python -m streamlit run app.py --server.port=8000
```