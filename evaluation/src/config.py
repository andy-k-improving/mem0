# config.py
import os

CONFIG = {
    "embedder": {
        "provider": "aws_bedrock",
        "config": {
            "model": "amazon.titan-embed-text-v2:0",
            "embedding_dims": 1024,
        },
    },
    "llm": {
        "provider": "aws_bedrock",
        "config": {
            # "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            # "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            # "model": "us.qwen.qwen3-32b-v1:0",
            # "model": "deepseek.v3-v1:0",
            # "model": "openai.gpt-oss-20b-1:0",
            # "temperature": 0.1,
            "max_tokens": 2000,
        },
    },
    "graph_store": {
        "provider": "neptune",
        "config": {
            "endpoint": f"neptune-graph://{os.getenv('GRAPH_ID', 'default-graph-id')}",
        },
    },
    "vector_store": {
        "provider": "neptune",
        "config": {
            "collection_name": "test",
            "endpoint": f"neptune-graph://{os.getenv('GRAPH_ID', 'default-graph-id')}",
        },
    },
}
