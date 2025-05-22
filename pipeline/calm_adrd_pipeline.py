import requests
import json

from typing import List, Union, Generator, Iterator, Dict
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class Pipeline:

    class Valves(BaseModel):
        calm_adrd_base_url: str = "http://localhost:8011"
        max_retries: int = 1
        threshold: float = 0.6
        model: str = "qwen2.5:32b"
        intermediate_model: str = "qwen2.5-coder:7b"
        doc_number: int = 10
        timeout: int = 30

    def __init__(self):
        self.name = "CaLM AI - ADRD"
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"===== ON STARTUP EXEC: {__name__}")
        pass

    async def on_shutdown(self):
        print(f"===== ON SHUTDOWN EXEC: {__name__}")
        pass

    async def on_valves_updated(self):
        print(f"===== ON VALVES UPDATED EXEC: {__name__}")
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the Model API request is made. 
        # You can modify the form data before it is sent to the Model API.
        
        print(f"===== INLET EXEC: {body}")
        
        # Example inlet body structure:
        # {
        #     'stream': True,
        #     'model': 'calm_adrd_pipeline', 
        #     'messages': [{
        #         'role': 'user',
        #         'content': str # User's question about Alzheimer's care
        #     }],uvicorn main_graph:fastapi_app --host 0.0.0.0 --port 8011 --reload
        #     'metadata': {
        #         'user_id': str,
        #         'chat_id': str,
        #         'message_id': str, 
        #         'session_id': str,
        #         'tool_ids': None,
        #         'files': None,
        #         'features': {
        #             'image_generation': bool,
        #             'web_search': bool
        #         }
        #     }
        # }

        body['stream'] = False
        body["current_session"] = body.get("metadata", {})
        
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        # This function is called after the Model API response is completed. 
        # You can modify the messages after they are received from the Model API.
        
        print(f"===== OUTLET EXEC: {body}")

        return body

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator[str]]:

        payload = {
            "user_query": user_message,
            "chat_session": messages,
            "body_config": body,
            **self.valves.model_dump(), # model, intermediate_model, max_retries, threshold, timeout
        }

        try:
            response = requests.post(
                url=f"{self.valves.calm_adrd_base_url}/ask-calm-adrd-agent",
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=False,
                timeout=self.valves.timeout
            )
            
            response.raise_for_status()
            
            ans = response.json()

            # Format response. 
            formatted_response = f"\n{ans['answer']}\n\n"
            
            if ans['sources'] and len(ans['sources']) > 0:
                formatted_response += "##### References\n"
                for source in ans['sources']:
                    title = source.get('title', 'Untitled Document')
                    url = source.get('url', '#')
                    formatted_response += f"- [{title}]({url})\n"
                    
            if ans['follow_up_questions'] and len(ans['follow_up_questions']) > 0:
                formatted_response += "\n##### Questions you might ask\n"
                for i, question in enumerate(ans['follow_up_questions'], 1):
                    formatted_response += f"{i}. {question}\n"

            return formatted_response

        except requests.HTTPError as e:
            # Log detailed error on server side
            error_msg = f"Service error: {e.response.text if hasattr(e, 'response') else str(e)}"
            status_code = e.response.status_code if hasattr(e, 'response') else 500
            print(f"[ERROR] HTTP Error occurred: {error_msg} (Status: {status_code})")
            
            # Return user-friendly message in markdown
            return """
            ### Sorry, we're experiencing some technical difficulties 😔

            Our service team has been notified and is working on fixing the issue. Please try again later.

            If the problem persists, please contact our technical support team.

            ---
            *Error Reference: {status_code}*
            *Error Message: {error_msg}*
            """.format(status_code=status_code, error_msg=e)
        
        except Exception as e:
            # Log detailed error on server side
            error_msg = f"Internal server error: {str(e)}"
            print(f"[ERROR] Unexpected error occurred: {error_msg}")
            
            # Return user-friendly message in markdown
            return """
            ### Sorry, something went wrong 😔
            
            We encountered an unexpected error while processing your request. Our team has been notified and is looking into it.
            
            Please try again in a few moments. If the issue continues, contact our support team.
            
            ---
            *Error Reference: {status_code}*
            *Error Message: {error_msg}*
            """.format(status_code=status_code, error_msg=e)