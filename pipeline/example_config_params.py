body = {
    'stream': True, 
    'model': 'calm_adrd_pipeline', 
    'messages': [{'role': 'user', 'content': 'test'}], 
    'user': {
        'name': 'Drake', 
        'id': '6f9e002d-2159-4e34-8263-8b89b6d3cc60', 
        'email': 'yuz211@pitt.edu', 
        'role': 'admin'
        }
    },

response = [
    {
        "id": "695195d8-2813-4cc6-b08a-0554888f71ff",
        "title": "'### ... ADRD?'",
        "updated_at": 1739312428,
        "created_at": 1739312104
    }
]


# Example initial state (include request body) for CaLM ADRD agent

## From Openweb-UI portal

# Example request parameters
init_state = {
    # Core query parameters
    'user_query': 'Can you recommend activities that are suitable for someone with dementia to engage in and enjoy?',
    'doc_number': 4,
    'threshold': 0.75,
    'temperature': 0.3,
    'max_retries': 2,
    'model': 'phi4:latest',
    'intermida_model': 'qwen2.5',
    'chat_session': [
        ChatMessage(
            id=None,
            role='user',
            content='Can you recommend activities that are suitable for someone with dementia to engage in and enjoy?',
            timestamp=None
        )
    ],
    
    # Request configuration
    'body_config': {
        'stream': True,
        'model': 'calm_adrd_pipeline',
        'messages': [{
            'role': 'user',
            'content': 'Can you recommend activities that are suitable for someone with dementia to engage in and enjoy?'
        }],
        'current_session': {
            'user_id': '6f9e002d-2159-4e34-8263-8b89b6d3cc60',
            'chat_id': '751b90af-1cf4-4634-b27d-6aa35ee78fba', 
            'message_id': 'be4a843d-ae05-4a50-8ce0-9ccc19a2f16a',
            'session_id': 'Y6uLYHkeMyD2rlTLAAAB',
            'tool_ids': None,
            'files': None,
            'features': {
                'image_generation': False,
                'web_search': False
            }
        },
        'user': {
            'name': 'Drake',
            'id': '6f9e002d-2159-4e34-8263-8b89b6d3cc60',
            'email': 'yuz211@pitt.edu', 
            'role': 'admin'
        }
    }
}