from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from utils.PROMPT import CLAUDE_EMOTIONAL_SUPPORT_PROMPT
from utils.llm_manager import _get_llm, _get_deepseek
from classes.Generation import ReasoningGeneration

def generation_with_rm(context, question, chat_session) -> ReasoningGeneration:
    
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_session"],
        template=CLAUDE_EMOTIONAL_SUPPORT_PROMPT
    )

    # llm  = _get_llm("deepseek-r1:14b", 0)
    llm = _get_deepseek("deepseek-reasoner", 0.6)
    
    # chain = prompt | llm | json_parser
    chain = prompt | llm 


    try:
        response = chain.invoke({
            'context': context,
            'question': question,
            'chat_session': []
        })
        
        # return ReasoningGeneration.parse_reasoning_output(response)
        
        return response.content
        
    except Exception as e:
        print(f"Answer generation failed: {str(e)}")
        raise
    
    
if __name__ == "__main__":
    print(generation_with_rm([], 'what is adrd ?', []))