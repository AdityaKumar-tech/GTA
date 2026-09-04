import requests

URL = "https://overpass-api.de/api/interpreter"

QUERY = """
[out:json][timeout:60];

(
  node(20.9,85.0,21.0,85.1)["power"];
  way(20.9,85.0,21.0,85.1)["power"];
);

out center;
"""

HEADERS = {
    "User-Agent": "SIH-Industrial-Thermal-Monitor/1.0 (research project)",
    "Accept": "application/json",
    "Content-Type": "text/plain",
}

print("=" * 70)
print("OVERPASS API TEST")
print("=" * 70)

try:

    response = requests.post(
        URL,
        data=QUERY,
        headers=HEADERS,
        timeout=120
    )

    print("HTTP status:", response.status_code)

    if response.status_code == 200:

        data = response.json()

        print("✓ CONNECTION SUCCESSFUL")
        print("Elements returned:", len(data.get("elements", [])))

    else:

        print("✗ REQUEST FAILED")
        print(response.text[:1000])

except Exception as e:

    print("✗ CONNECTION ERROR")
    print(e)