import requests
import yaml
import os

config = yaml.safe_load(open(os.path.join('.', 'config.yaml'), 'r'))
TG_API = config.get('API_TOKEN')
whook = config.get('WEBHOOK')
print(f"WHOOK: {whook}\nTG_API: {TG_API}")

r = requests.get(f"https://api.telegram.org/bot{TG_API}/setWebhook?url=https://{whook}/")

print(r.json())