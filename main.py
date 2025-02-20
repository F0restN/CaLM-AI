import os

from dotenv import load_dotenv
from fastapi import FastAPI

from classes.AdaptiveDecision import AdaptiveDecision
from classes.RequestBody import RequestBody
from answer_generation import generate_answer
from checkpoints.retrieval_grading import grade_retrieval
from checkpoints.query_extander import query_extander
from checkpoints.adaptive_decision import adaptive_rag_decision
from embedding.vector_store import get_connection
from embedding.embedding_models import get_nomic_embedding

app = FastAPI()

load_dotenv()

PGVECTOR_CONN = os.environ.get("PGVECTOR_CONN")
if not PGVECTOR_CONN:
    raise ValueError("PGVECTOR_CONN environment variable is not set")


r_kb = get_connection(connection=PGVECTOR_CONN,embedding_model=get_nomic_embedding(),collection_name='research_kb')
p_kb = get_connection(connection=PGVECTOR_CONN,embedding_model=get_nomic_embedding(),collection_name='peer_support_kb')


@app.post("/")
def main(requestBody: RequestBody):
    
    print(requestBody)
    print("\n")
    
    user_query = requestBody.user_query
    threshold = requestBody.threshold
    max_retries = requestBody.max_retries
    doc_number = requestBody.doc_number
    model = requestBody.model
    temperature = requestBody.temperature
    intermida_model = requestBody.intermida_model
    chat_session = requestBody.chat_session
    
    if not user_query:
        return "Please type in your question"
    
    # latest_human_message = next(
    #     (msg.content for msg in reversed(chat_session) if msg.role == "user"),
    #     user_query
    # )
    latest_human_message = user_query
    
    # ============= User Intention Detection and Routing to correspondent KB ============= 
    
    # Routing, answer directly if retrieval is not necessary
    adaptive_decision: AdaptiveDecision = adaptive_rag_decision(latest_human_message, model=intermida_model, temperature=temperature)
    
    print(adaptive_decision)
    
    if adaptive_decision.require_extra_re is False:
        print("No need to retrieve from KB")
        return generate_answer(latest_human_message, model=model, temperature=0.6)
    

    # Get relevant document, search professional kb either way 
    # relevant_doc = []    
    # TODO: Format issue, haven't been used for generation
    # relevant_doc.append(
    #     r_kb.similarity_search(latest_human_message)
    # )
    
    ## Retrieve relevant documents according to user's intention
    if adaptive_decision.knowledge_base == "peer_support":
        crs_kb = p_kb
    else:
        crs_kb = r_kb
    
    # ============= Document grading and query expand =============     
    
    retry_count = 0
    filtered_docs = []
    query_messgae = latest_human_message
    
    while True:
        retrieved_docs = crs_kb.similarity_search(query_messgae, k=doc_number)
        graded_doc = grade_retrieval(latest_human_message, retrieved_docs, model=intermida_model, temperature=temperature)   

        missing_topics = []
        for doc in graded_doc:
            if doc.relevance_score >= threshold:
                filtered_docs.append(doc)
            else:
                missing_topics.append([ mt for mt in doc.missing_topics])
        
        query_messgae = query_extander(latest_human_message, missing_topics, model=intermida_model, temperature=temperature)[0]
        retry_count = retry_count + 1
        
        if retry_count >= max_retries or len(filtered_docs) >= doc_number:
            break    
    
    filtered_docs.sort(key=lambda x: x.relevance_score, reverse=True) # Sort all documents by relevance score
    
    
    # ============= Generate =============     
    
    # Get last 3 messages from chat history, or all if less than 3
    chat_context = chat_session[-4:-1] if len(chat_session) >= 4 else chat_session[:]
    
    return generate_answer(latest_human_message, filtered_docs, chat_context, model=model, temperature=temperature)


@app.get("/health")
def health_check():
    return {"status": "healthy"}    

if __name__ == "__main__":
    payload = RequestBody(
        user_query="my mom seems forgetting thinks, what should I do ?",
        threshold=0.65,
        doc_number=4,
        max_retries=1,
        model="phi4:latest",
        intermida_model="qwen2.5:latest",
        chat_session=[
            {
                "id": "c3db2446-651b-4603-b2a8-2ba558c97ff4",
                "role": "user",
                "content": "What are 5 creative things I could do with my kids' art? I don't want to throw them away",
                "timestamp": 1738345174
            },
            {
                "id": "dcc67334-532e-49c7-9876-996de89ed416",
                "role": "assistant",
                "content": "you are a very helpful ai assistant",
                "timestamp": 1738345287
            }
        ],
    )
    
    res = main(payload)
    
    print(res)

