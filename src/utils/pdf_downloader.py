import requests
from logger import logger


def download_pdf(url: str, save_path: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()
    
    try:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Successfully downloaded PDF to {save_path}")
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        raise e
    
if __name__ == "__main__":
    download_pdf("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10774282/pdf/nihms-1766664.pdf", "./src/utils/PMC10774282.pdf")
    