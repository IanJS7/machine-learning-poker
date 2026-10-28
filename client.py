
import requests

url = "http://127.0.0.1:8000/score"

body = {
    "suit_card_1": 1,
    "rank_card_1": 1,
    "suit_card_2": 1,
    "rank_card_2": 2,
    "suit_card_3": 1,
    "rank_card_3": 3,
    "suit_card_4": 1,
    "rank_card_4": 4,
    "suit_card_5": 1,
    "rank_card_5": 5
}

try:
    response = requests.post(url, json=body, timeout=10)

    print("Codigo HTTP:", response.status_code)
    response.raise_for_status()

    print("Respuesta del modelo:")
    print(response.json())

except requests.RequestException as error:
    print("Error al consultar la API:", error)
