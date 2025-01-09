## Test for Extracting Structured Data Using LLM
import os
import json
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

# class OpenAIModelFee(BaseModel):
#     model_name: str = Field(..., description="Name of the OpenAI model.")
#     input_fee: str = Field(..., description="Fee for input token for the OpenAI model.")
#     output_fee: str = Field(..., description="Fee for output token for the OpenAI model.")
    
class PubMedResearchPaper(BaseModel):
    title: str = Field(..., description="Title of the research paper.")
    abstract: str = Field(..., description="Abstract part of the research paper, the brief summary of the research paper.")
    discussion: str = Field(..., description="Discussion part of the research paper, the detailed discussion of the research paper.")
    conclusion: str = Field(..., description="Conclusion part of the research paper, the conclusion of the research paper.")


async def extract_pubmed_research_paper():
    url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10897520/'

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=url,
            word_count_threshold=1,
            extraction_strategy=LLMExtractionStrategy(
                provider="ollama/llama3.2",
                # api_token=os.getenv('OPENAI_API_KEY'),
                schema=PubMedResearchPaper.model_json_schema(),
                extraction_type="schema",
                instruction="""
                From the crawled content, extract the title, abstract, discussion, and conclusion of the research paper.
                The extracted content should be in the format of a JSON object with the following keys and corresponding values:
               
                - title: the title of the research paper
                - abstract: the abstract of the research paper
                - discussion: the discussion of the research paper
                - conclusion: the conclusion of the research paper
                
                Please strictly follow the format and do not miss any information.
                """,
                verbose=True,
                timeout=6000,
            ),
            bypass_cache=True,
        )

    model_fees = json.loads(result.extracted_content)
    print(f"Number of models extracted: {len(model_fees)}")

    with open("./pubmed_research_paper.json", "w", encoding="utf-8") as f:
        json.dump(model_fees, f, indent=2)

asyncio.run(extract_pubmed_research_paper())