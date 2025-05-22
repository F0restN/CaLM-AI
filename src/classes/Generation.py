from typing import List, Dict
from pydantic import BaseModel, Field

class AIGeneration(BaseModel):
    answer: str = Field(description="answer for user's question, use your best knowledge and judgement to answer the question, say 'I'm sorry, I don't know' if you don't know the answer")
    follow_up_questions: List[str] = Field(description="possible questions that user might ask after reading the answer, if there are no follow up questions, return an empty list")

class Generation(AIGeneration):
    sources: List[Dict[str | None, str | None]] = Field(description="list of sources that we use to generate answer")
    
class ReasoningGeneration(BaseModel):
    
    thinking: str = Field(description="thinking process of reasoning model, wrapped by tag <think></think>")
    answer: str = Field(description="Main answer for user's query, markdown syntax")
    additional_kwargs: Dict = Field(default_factory=dict, description="Additional metadata like model name, token counts, etc.")
    response_metadata: Dict = Field(default_factory=dict, description="Response metadata including model, timestamps, and usage statistics")
    id: str = Field(default="", description="Unique identifier for the generation run")
    usage_metadata: Dict = Field(default_factory=dict, description="Token usage information including input, output and total tokens")
    
    
    class Config:
        from_attributes: True
        json_schema_extra = {
            "examples": [
                {
                    "thinking": "<think>First, I need to understand the user's question about Alzheimer's care.</think>",
                    "answer": "# Supporting Your Loved One with Alzheimer's\n\nIt's completely normal to feel overwhelmed when caring for someone with Alzheimer's. Remember to take breaks when needed and reach out to support groups in your area.\n\n## Practical Tips\n- Establish a regular routine\n- Use simple, clear communication\n- Create a safe environment\n\nMany caregivers find that joining a support group helps them cope with the emotional challenges.",
                    "additional_kwargs": {
                        "model_name": "deepseek-r1:14b",
                        "temperature": 0.7
                    },
                    "response_metadata": {
                        'model': 'deepseek-r1:14b', 
                        'created_at': '2025-03-28T20:31:21.611112794Z', 
                        'done': True, 
                        'done_reason': 'stop', 
                        'total_duration': 7529174511, 
                        'load_duration': 25718848, 
                        'prompt_eval_count': 521, 
                        'prompt_eval_duration': 111000000,
                        'eval_count': 512, 
                        'eval_duration': 7391000000, 
                        # 'message': 'Message'(role='assistant', content='', images=None, tool_calls=None)
                    },
                    "id": "gen_12345",
                    "usage_metadata": {
                        "input_tokens": 512,
                        "output_tokens": 256,
                        "total_tokens": 768
                    }
                }
            ]
        }
    
    
    @classmethod
    def parse_reasoning_output(cls, output_text) -> "ReasoningGeneration":
        """
        Parse the raw output text from the reasoning node and convert it into a ReasoningGeneration object.
        
        Args:
            output_text: The raw output text from the reasoning node or an AIMessage object
            
        Returns:
            ReasoningGeneration: A structured object containing the parsed content
        """
        import re
        import ast
        
        # 检查 output_text 是否为字符串，如果不是，尝试提取内容
        if not isinstance(output_text, str):
            # 处理 AIMessage 或其他对象，尝试获取其 content 属性
            try:
                # 如果是具有 content 属性的对象（如 AIMessage）
                output_text_str = str(output_text.content)
            except AttributeError:
                # 如果没有 content 属性，则尝试直接转为字符串
                output_text_str = str(output_text)
        else:
            output_text_str = output_text
        
        # Extract thinking part using regex
        thinking_match = re.search(r'<think>(.*?)</think>', output_text_str, re.DOTALL)
        thinking = thinking_match.group(0) if thinking_match else ""
        
        # Extract answer part - content between thinking and additional_kwargs
        if thinking:
            # Find the position after the thinking block
            thinking_end_pos = output_text_str.find('</think>') + len('</think>')
            # Find the position of additional_kwargs
            additional_kwargs_pos = output_text_str.find('additional_kwargs=')
            if additional_kwargs_pos > -1:
                answer = output_text_str[thinking_end_pos:additional_kwargs_pos].strip()
            else:
                answer = output_text_str[thinking_end_pos: len(output_text_str)]
        else:
            # If no thinking block found, try to extract answer directly
            answer_match = re.search(r'^(.*?)additional_kwargs=', output_text_str, re.DOTALL)
            answer = answer_match.group(1).strip() if answer_match else ""
        
        # Extract metadata sections using regex
        try:
            additional_kwargs_match = re.search(r'additional_kwargs=(\{.*?\})', output_text_str)
            additional_kwargs = ast.literal_eval(additional_kwargs_match.group(1)) if additional_kwargs_match else {}
            
            response_metadata_match = re.search(r'response_metadata=(\{.*?\})', output_text_str)
            response_metadata = ast.literal_eval(response_metadata_match.group(1)) if response_metadata_match else {}
            
            id_match = re.search(r"id='(.*?)'", output_text_str)
            id_value = id_match.group(1) if id_match else ""
            
            usage_metadata_match = re.search(r'usage_metadata=(\{.*?\})', output_text_str)
            usage_metadata = ast.literal_eval(usage_metadata_match.group(1)) if usage_metadata_match else {}
        except (SyntaxError, ValueError) as e:
            # Fallback to empty dictionaries if parsing fails
            print(f"Error parsing metadata: {e}")
            additional_kwargs = {}
            response_metadata = {}
            id_value = ""
            usage_metadata = {}
        
        # Create and return the ReasoningGeneration object
        return cls(
            thinking=thinking,
            answer=answer,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
            id=id_value,
            usage_metadata=usage_metadata
        )
            
    
    # @classmethod
    def __str__(self):
        """
        Override the default print function to provide a more readable representation
        of the ReasoningGeneration object.
        
        Returns:
            str: A string representation of the ReasoningGeneration object
        """
        output = []
        output.append("ReasoningGeneration:")
        
        # if self.thinking:
        #     output.append(f"Thinking: {self.thinking}")
        
        if self.answer:
            output.append(f"Answer: {self.answer}")
        
        # if self.additional_kwargs:
        #     output.append(f"Additional Kwargs: {self.additional_kwargs}")
        
        # if self.id:
        #     output.append(f"ID: {self.id}")
        
        # if self.response_metadata:
        #     output.append(f"Response Metadata: {self.response_metadata}")
        
        # if self.usage_metadata:
        #     output.append(f"Usage Metadata: {self.usage_metadata}")
        
        return "\n".join(output)
        