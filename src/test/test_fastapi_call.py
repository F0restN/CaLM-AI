import requests

def main():
    BASE_URL = "http://127.0.0.1:8000"
    body:dict = {
        "stream": True,
    }
    

    response = requests.post(
            url=f"{BASE_URL}/",
            headers={"accept": "application/json"},
            json={**body},
            stream=True
        )

    response.raise_for_status()

    if body["stream"]:
        return response.iter_lines()
    else: 
        return response.json()
    
if __name__ == "__main__":
    print(main())