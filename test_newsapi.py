import requests
import json

URL = "https://newsapi.org/v2/everything"
API_KEY = "ceff811fa62d4758b3805083f34d75e1"

params = {
    "q": "Trump",
    "domains": "wsj.com",
    "language": "en",
    "sortBy": "relevancy",
    "pageSize": 20,
    "apiKey": API_KEY
}

try:
    response = requests.get(URL, params=params)
    print("Status Code:", response.status_code)
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print("Request failed:", e)
