from typing import List, Optional
from langchain_core.documents import Document
from .logger import logger


class ToolKits:
    def __init__(self):
        pass

    def straight_array_to_string(self, array: list, separator: str = ",") -> str:
        if not array:
            return "NAN"
        return separator.join(array)


def pretty_print_list_docs(docs: List[Optional[Document]], properties: List[str] = None):
    for i, doc in enumerate(docs):
        print(
            f"------------------------------------- || Document #{i+1} || -------------------------------------")

        if not properties:
            print(f"Document: {doc.page_content} \n")
            print(f"Metadata: {doc.metadata} \n")

        if properties:
            for prop in properties:
                print(f"{prop}: {doc[prop]} \n")


if __name__ == "__main__":
    tools = ToolKits()
    print(tools.straight_array_to_string([]))
    print(pretty_print_list_docs([], []))
