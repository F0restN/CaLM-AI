import os
from loguru import logger
from pathlib import Path
import asyncio
import concurrent.futures
from urllib.request import urlopen, urlretrieve
import time
from bs4 import BeautifulSoup
import pandas as pd
import requests
import math
import urllib.error

# from crawler_crawl4AI import crawl_crawl4ai
# from utils.text_processor_crawl4ai import extract_abstract_discussion_conclusion

max_threads = 20
max_retries = 5
max_pages = 3
retry_delay = 2

project_root = Path(__file__).parent.parent.parent
logger.info(f"project_root: {project_root}")

output_dir = os.path.join(project_root, "data", "raw_content")
logger.info(f"output_dir: {output_dir}")

class pubmed_record:
    def __init__(self, search_term=None, start_time=None, end_time=None):


        # logger.info("project_root: ", project_root)

        # output_dir = os.path.join(project_root, "data", "raw_content")
        # logger.info("output_dir: ", output_dir)
        
        # input the search term
        self.search_term = "healthcare dementia Alzheimer's Generative AI"
        # self.search_term = input("Please enter a keyword:")

        # input the time range
        # self.start_time, self.end_time = input("Range of publication time to be searched (YYYY/MM/DD - YYYY/MM/DD): ").split("-")

        self.start_time = "2024/01/01"
        self.end_time = "2024/08/31"
        [self.start_y, self.start_m, self.start_d] = self.start_time.split('/')
        [self.end_y, self.end_m, self.end_d] = self.end_time.split('/')

    def pub_article_search_url_get(self):
        search_term_list = self.search_term.split()
        if len(search_term_list) > 1:
            search_term_url = "+".join(search_term_list)
        else:
            search_term_url = self.search_term

        # PubMed
        # url_p1 = 'https://pubmed.ncbi.nlm.nih.gov/?term=({})'.format(search_term_url)
        # url_p2 = '{}%2F{}%2F{}%5BDate+-+Publication%5D+%3A+{}%2F{}%2F{}%5BDate+-+Publication%5D%29%29&size=200'.format(start_y,start_m,start_d,end_y,end_m,end_d)

        # PubMed Central
        url_p1 = 'https://www.ncbi.nlm.nih.gov/pmc/?term=({})'.format(
            search_term_url)
        url_p2 = '{}%2F{}%2F{}%3A{}%2F{}%2F{}%5Bdp%5D'.format(
            self.start_y, self.start_m, self.start_d, self.end_y, self.end_m, self.end_d)
        url = url_p1 + url_p2
        print(url)

        '''get the total number of publications'''
        soup = BeautifulSoup(urlopen(url).read(), 'html.parser')
        n_pub = int(soup.find_all("meta", attrs={
                    'name': 'ncbi_resultcount'})[0]['content'])
        print("Number of publications:", n_pub)

        '''get number of pages'''
        total_pages = int(math.ceil(n_pub / 10))
        total_pages = min(total_pages, max_pages)
        print('Number of pages to process:', total_pages)

        url_page = []
        for i in range(1, total_pages + 1):
            url2 = url + '&page={}'.format(i)
            url_page += [url2]

        # return url_page[:10]
        return url_page

    def pmid_get(self, url):
        for attempt in range(max_retries):
            try:
                soup_page = BeautifulSoup(urlopen(url).read(), 'html.parser')
                # 查找所有包含 PMCID 的元素
                articles = soup_page.find_all("div", class_="rprt")
                pmid_1page = []
                for article in articles:
                    # 在每篇文章中查找 PMCID
                    pmcid_element = article.find("dl", class_="rprtid")
                    if pmcid_element:
                        pmcid = pmcid_element.find("dd").text.strip()
                        pmid_1page.append(pmcid)
                time.sleep(1)
                return pmid_1page
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                raise

    def pmid_all(self, url_allpage):
        if not url_allpage:
            print("No articles found")
            return []
        threads = min(max_threads, len(url_allpage))
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            pmids = executor.map(self.pmid_get, url_allpage)
        return pmids

    def pmid_get_main(self):
        pmid_all = list(self.pmid_all(self.pub_article_search_url_get()))
        pmid_url = []
        for pmid_1page in pmid_all:
            for pmid in pmid_1page:
                p_url = "https://pmc.ncbi.nlm.nih.gov/articles/" + pmid
                pmid_url += [p_url]
        for pmid in pmid_url:
            print(pmid + "\n")
        return pmid_url

    def pub_tab(self, url_1paper):
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        for attempt in range(max_retries):
            try:
                # Add longer delay between requests with exponential backoff
                if attempt > 0:
                    sleep_time = retry_delay * \
                        (2 ** attempt)  # Exponential backoff
                    time.sleep(sleep_time)

                response = requests.get(url_1paper, headers=headers)

                # Handle 429 specifically
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        continue
                    else:
                        response.raise_for_status()

                response.raise_for_status()

                pub_record = []
                soup_1paper = BeautifulSoup(response.content, 'html.parser')

                # Title
                sig_paper_title = soup_1paper.find(
                    "meta", attrs={'name': 'citation_title'}, recursive=True)['content']

                # Author
                sig_paper_authors = soup_1paper.find_all(
                    "meta", attrs={'name': 'citation_author'})
                sig_paper_authors_list = list()
                for a in sig_paper_authors:
                    sig_paper_authors_list += [a.get('content')]

                # Published Date
                try:
                    sig_paper_date = soup_1paper.find(
                        "meta", attrs={'name': 'citation_publication_date'}, recursive=True)['content']
                except TypeError:
                    sig_paper_date = soup_1paper.find(
                        "meta", attrs={'name': 'citation_online_date'}, recursive=True)['content']

                sig_paper_date_break = sig_paper_date.split("/")

                if len(sig_paper_date_break) == 1:
                    sig_paper_date2 = '/'.join(
                        [sig_paper_date_break[0], "01", "01"])
                elif len(sig_paper_date_break) == 2:
                    sig_paper_date2 = '/'.join(sig_paper_date_break +
                                               [self.start_d])
                else:
                    sig_paper_date2 = sig_paper_date

                if sig_paper_date2 > self.end_time:
                    sig_paper_date2 = self.end_time
                else:
                    sig_paper_date2 = sig_paper_date2

                # PMID
                pmid = url_1paper.split("/")[4]

                # PDF URL
                pdf_url_element = soup_1paper.find(
                    "meta", attrs={'name': 'citation_pdf_url'}, recursive=True)
                sig_paper_pdf_url = pdf_url_element['content'] if pdf_url_element else "No PDF URL scraped"

                # DOI
                doi_element = soup_1paper.find(
                    "meta", attrs={'name': 'citation_doi'}, recursive=True)
                sig_paper_doi = doi_element['content'] if doi_element else "No DOI scraped"

                # TODO: Fix this
                # Abstract, Discussion, Conclusion
                # markdown_content = loop.run_until_complete(crawl_crawl4ai(url_1paper))
                # abs_content = extract_abstract_discussion_conclusion(markdown_content)['abstract']

                # Paper Summary
                pub_summary_paper_info = {"Title": [sig_paper_title],
                                          "PMID": [pmid],
                                          "DOI": [sig_paper_doi],
                                          "Authors": [", ".join(sig_paper_authors_list)],
                                          "Published Date": [sig_paper_date2],
                                          # "Abstract": [abs_content],
                                          "URL": [url_1paper],
                                          "PDF URL": [sig_paper_pdf_url]}

                # TODO: add more information
                pub_summary_df_paper_info = pd.DataFrame(
                    pub_summary_paper_info)
                pub_record += [pub_summary_df_paper_info]
                time.sleep(0.5)

                return (pub_record)

            except (requests.exceptions.RequestException, urllib.error.HTTPError) as e:
                if attempt == max_retries - 1:
                    raise
                continue
            finally:
                loop.close()  # Clean up the loop

    def pub_tab_all(self):
        url_allpage = self.pmid_get_main()
        threads = min(max_threads, len(url_allpage))

        # Add delay between batches
        time.sleep(2)  # Add delay before starting threads

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            # Reduce number of concurrent requests
            chunk_size = 5  # Process in smaller chunks
            results = []
            for i in range(0, len(url_allpage), chunk_size):
                chunk = url_allpage[i:i + chunk_size]
                chunk_results = list(executor.map(self.pub_tab, chunk))
                results.extend(chunk_results)
                time.sleep(2)  # Add delay between chunks

        return results

    def pub_tab_all_main(self):
        
        df_paper_info = pd.DataFrame()

        t0 = time.time()
        records_all = self.pub_tab_all()

        for records in list(records_all):
            df_paper_info = pd.concat(
                [df_paper_info, records[0]], ignore_index=True)

        df_paper_info.to_csv(output_dir + "/pubmed_search_summary_" +
                             time.strftime("%Y_%m_%d_%H_%M_%S") + ".csv", index=False)

        t1 = time.time()
        print(f"{df_paper_info.shape[0]} papers were downloaded with {t1 - t0} seconds")

        return df_paper_info

if __name__ == "__main__":
    pubmed_record().pub_tab_all_main()
