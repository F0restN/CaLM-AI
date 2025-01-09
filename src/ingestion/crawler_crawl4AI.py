# import json
import asyncio
import time
import os
from pathlib import Path
from loguru import logger
from crawl4ai import AsyncWebCrawler, WebCrawler
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

project_root = Path(__file__).parent.parent.parent
logger.info(f"project_root: {project_root}")

output_dir = os.path.join(project_root, "data", "raw_content")
logger.info(f"output_dir: {output_dir}")

schema = {
    "name": "Medical Article Content",
    "baseSelector": "article",
    "fields": [
        {
            "name": "article_content",
            "selector": "section.body.main-article-body",
            "type": "nested",
            "fields": [
                {
                    "name": "sections",
                    "selector": "section",
                    "type": "nested_list",
                    "fields": [
                        {
                            "name": "title",
                            "selector": "h2",
                            "type": "text"
                        },
                        {
                            "name": "content",
                            "selector": "p",
                            "type": "list",
                            "fields": [
                                {
                                    "name": "paragraph",
                                    "type": "text"
                                }
                            ]
                        },
                        {
                            "name": "subsections",
                            "selector": "section",
                            "type": "nested_list",
                            "fields": [
                                {
                                    "name": "subtitle",
                                    "selector": "h3",
                                    "type": "text"
                                },
                                {
                                    "name": "subcontent",
                                    "selector": "p",
                                    "type": "list",
                                    "fields": [
                                        {
                                            "name": "paragraph",
                                            "type": "text"
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}

json_css_strategy = JsonCssExtractionStrategy(schema)


async def crawl_crawl4ai(url: str):
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Run the crawler on a URL
        result = await crawler.arun(
            url=url,
            # extraction_strategy=json_css_strategy,
            exclude_external_links=True,
            exclude_external_images=True,
            excluded_tag=['meta', 'style'],
            bypass_cache=True
        )

        # error handling
        if not result.success:
            print(f"Crawl failed: {result.error_message}")
            print(f"Status code: {result.status_code}")
            return None

        # Print the extracted content
        # print(result.fit_markdown)

        # with open(output_dir + "/result_" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".json", "w", encoding='utf-8') as f:
        #     json.dump(json.loads(result.extracted_content),
        #               f, indent=4, ensure_ascii=False)

        return result

if __name__ == "__main__":
    result = asyncio.run(crawl_crawl4ai("https://pmc.ncbi.nlm.nih.gov/articles/PMC11320145/"))
    # persist the result to a markdown file
    if result:
        with open(output_dir + "/web_scrape_result_" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".md", "w", encoding='utf-8') as f:
            f.write(result.markdown)
