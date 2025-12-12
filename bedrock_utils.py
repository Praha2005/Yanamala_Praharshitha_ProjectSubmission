import boto3
import json

# -------------------------------------------------
# Global config
# -------------------------------------------------
REGION = "us-west-2"  # Oregon – same region as your KB


def _agent_client():
    """
    Client for Bedrock Agent Runtime (used to query Knowledge Bases).
    """
    return boto3.client("bedrock-agent-runtime", region_name=REGION)


def _runtime_client():
    """
    Client for Bedrock Runtime (used to invoke text models).
    """
    return boto3.client("bedrock-runtime", region_name=REGION)


# -------------------------------------------------
# 1. Query the Knowledge Base
# -------------------------------------------------
def query_knowledge_base(kb_id: str, query_text: str, top_k: int = 3):
    """
    Retrieve relevant chunks from the Bedrock Knowledge Base.

    :param kb_id: Knowledge base ID (e.g. 'YCXJT4XOV3')
    :param query_text: User question
    :param top_k: How many chunks to retrieve
    :return: list of retrievalResults
    """
    client = _agent_client()

    response = client.retrieve(
        knowledgeBaseId=kb_id,
        retrievalQuery={"text": query_text},
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": top_k
            }
        },
    )

    results = response.get("retrievalResults", [])
    return results


# -------------------------------------------------
# 2. Generate a response using a Bedrock text model
# -------------------------------------------------
def generate_response(
    prompt: str,
    context: str,
    model_id: str = "amazon.titan-text-lite-v1",
):
    """
    Call an LLM on Bedrock, giving it the retrieved context.

    :param prompt: Original user question
    :param context: Text snippets from the KB
    :param model_id: Text model ID
    :return: Generated answer as string
    """
    client = _runtime_client()

    # Build final prompt for the model
    full_prompt = (
        "You are a helpful assistant. Use the context to answer the question. "
        "If the context does not contain the answer, say you are not sure.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {prompt}\n\nAnswer:"
    )

    body = {
        "inputText": full_prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.2,
            "topP": 0.9,
        },
    }

    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
    )

    # Bedrock returns a streaming body; read and decode it
    raw = response["body"].read()
    data = json.loads(raw.decode("utf-8"))

    return data["results"][0]["outputText"]


# -------------------------------------------------
# 3. Simple prompt validation
# -------------------------------------------------
def valid_prompt(query: str) -> bool:
    """
    Basic validation: not empty / not just spaces.
    You can extend this to filter out unsafe prompts if needed.
    """
    if not query:
        return False

    if not query.strip():
        return False

    return True
