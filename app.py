from bedrock_utils import query_knowledge_base, generate_response, valid_prompt

# Your Knowledge Base ID (from Bedrock console / Terraform output)
KB_ID = "YCXJT4XOV3"


def main():
    print("Chatbot Ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        # Allow exit
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Validate prompt
        if not valid_prompt(user_input):
            print("Please enter a non-empty question.\n")
            continue

        try:
            # 1) Retrieve relevant context from the Knowledge Base
            retrieval_results = query_knowledge_base(KB_ID, user_input)

            if retrieval_results:
                context_chunks = [
                    item["content"]["text"]
                    for item in retrieval_results
                    if "content" in item and "text" in item["content"]
                ]
                context = "\n\n".join(context_chunks)
            else:
                context = "No relevant context was retrieved from the knowledge base."

            # 2) Generate the response using the LLM
            answer = generate_response(user_input, context)

            print("\nAssistant:", answer, "\n")
            print("-" * 60 + "\n")

        except Exception as e:
            print("\n[ERROR] Something went wrong while calling Bedrock:")
            print(e)
            print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
