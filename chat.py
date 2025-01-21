import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Step 1: Load the model and tokenizer from Hugging Face
model_name = "<username>/<model>"  # Replace with your model name from Huggingface
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Set the device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Step 3: Initialize conversation history
conversation_history = []

# Step 4: Define your system prompt
system_prompt = "You are a helpful assistant that thinks through problems step by step. When responding to questions, first analyze the question, then explain your reasoning, and finally provide a conclusion.  Please be thorough but concise in your explanations."  # Replace with your system prompt

# Function to generate a response
def generate_response(user_input):
    # Validate user input
    if not user_input.strip():
        print("You: (no input provided)")
        sys.stdout.flush()  # Flush output
        return "No input provided."

    # Combine the system prompt and user input
    combined_prompt = f"{system_prompt}\nUser: {user_input}\nAssistant:"

    # Tokenize the combined prompt
    inputs = tokenizer(combined_prompt, return_tensors="pt").to(device)

    # Generate output with adjusted parameters
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.5,
                top_k=50,
                top_p=0.85,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )

        # Decode the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the assistant's response
        assistant_response = response.split("Assistant:")[-1].strip()  # Get the part after "Assistant:"

        # Check for empty response
        if not assistant_response:
            print("Assistant: (no response generated)")
            sys.stdout.flush()  # Flush output
            return "No response generated."

        # Print the user input and corresponding assistant response
        print(f"You: {user_input}")
        print(f"Assistant: {assistant_response}")
        sys.stdout.flush()  # Flush output

        return assistant_response

    except Exception as e:
        print(f"Error during response generation: {e}")
        sys.stdout.flush()  # Flush output
        return "An error occurred while generating a response."

# Main loop for conversation
while True:
    # Get user input
    user_input = input("You: ")
    
    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the conversation.")
        break

    # Generate assistant response
    assistant_response = generate_response(user_input)

    # Update conversation history
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Assistant: {assistant_response}")
