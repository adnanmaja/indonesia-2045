import json
from dotenv import load_dotenv
import os
import requests
import pandas as pd

load_dotenv()

api_key = os.getenv("BPS_WebAPi")

# https://webapi.bps.go.id/v1/api/view/publikasi/{publication_id}/key/{api_key}/format/json
# https://webapi.bps.go.id/v1/api/list/domain/{domain_code}/model/{model_code}/var/0/key/{api_key}/format/json
url = f"https://webapi.bps.go.id/v1/api/view/lang/eng/domain/0000/model/publikasi/id/7290b829d2eaa972e4968d19/key/{api_key}/format/json"
url2 = f"https://webapi.bps.go.id/v1/api/list/model/th/2019/lang/ind/domain/0000/var/543/key/{api_key}"

res = requests.get(url2)
data = res.json()

with open('test.json', 'w') as file:
    json.dump(data, file)