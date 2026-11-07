from transformers import pipeline , AutoTokenizer , AutoModelForCausalLM

def create_simple_llm():
    # smaller version of gpt2
    model_name = "distilgpt2" 
    
    generator = pipeline('text-generation',
                         model=model_name,
                         pad_token_id=50256)
    return generator

def generate_text(generator , prompt , max_length=1000):
    
    result = generator(prompt, max_length=max_length, num_return_sequences=1, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]

def run_llm_demo():
    
    print("loading Simple LLM Model...")
    generator = create_simple_llm()
    
    print("\n ✨Simple LLM Demo ✨")
    print("This demo shows basic text generation using a small language model")
    
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Python programming is"
    ]
    
    for prompt in prompts:
        print("\n ⭐ Prompt:" , prompt)
        print("👉 Generated:" , generate_text(generator , prompt))
        input("\n Press Enter to see next example...")
        

def interactive_demo():
    """
    Allows users to interact with the model
    """
    generator = create_simple_llm()

    print("\n🤖 Interactive LLM Demo")
    print("Type your prompts (or 'quit' to exit)")

    while True:
        prompt = input("\n👉 Enter your prompt: ")
        if prompt.lower() == 'quit':
            break

        response = generate_text(generator, prompt)
        print("\n💬 Generated response:")
        print(response)

if __name__ == "__main__":
    print("Choose a demo:")
    print("1. Run basic demonstration")
    print("2. Interactive mode")

    choice = input("Enter your choice (1-2): ")

    if choice == '1':
        run_llm_demo()
    elif choice == '2':
        interactive_demo()
    else:
        print("Invalid choice!")