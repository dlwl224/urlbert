
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import csv
import time

OUTPUT_FILE = "/home/kong/urlbert/url_bert/urlbert2/dataset/link/severance_all_links.csv"

def crawl_all_links(start_url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    driver.get(start_url)
    time.sleep(3)  # 페이지 로딩 기다림

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(start_url, href)
        if full.startswith(("http://", "https://")):
            links.add(full)

    return list(links)

def save_to_csv(links, filename):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        for url in links:
            writer.writerow([url])

if __name__ == "__main__":
    start = "https://www.severance.healthcare/severance/"
    all_links = crawl_all_links(start)
    print(f"총 {len(all_links)}개 링크 수집됨.")
    save_to_csv(all_links, OUTPUT_FILE)
