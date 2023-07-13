
import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

# URL to the directory containing the files to be downloaded
language = "en-es"
url = f"https://data.statmt.org/opus-100-corpus/v1.0/supervised/{language}/"
save_directory = f"./Datasets/{language}"

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML response
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the anchor tags in the HTML
links = soup.find_all('a')

# Extract the href attribute from each anchor tag
file_links = [link['href'] for link in links if '.' in link['href']]

# Download each file
for file_link in tqdm(file_links):
    file_url = url + file_link
    save_path = os.path.join(save_directory, file_link)
    
    print(f"Downloading {file_url}")
    
    # Send a GET request for the file
    file_response = requests.get(file_url)
    if file_response.status_code == 404:
        print(f"Could not download {file_url}")
        continue
    
    # Save the file to the specified directory
    with open(save_path, 'wb') as file:
        file.write(file_response.content)
    
    print(f"Saved {file_link}")

print("All files have been downloaded.")