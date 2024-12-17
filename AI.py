import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from heuristics import heuristic_score


model_name = 'bigscience/bloom-560m'  # You can use 'gpt2-medium' for a larger model
tokenizer =  BloomTokenizerFast.from_pretrained(model_name)
model = BloomForCausalLM.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

def improve_cv_text(input_text, max_length=500):
    # Tokenize the input text
    print(f"Score before: {heuristic_score(input_text)}")
    
    prompt = f"Improve the following text to make it more professional and free from grammatical errors: {input_text}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    print(1)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,          # Explicit attention mask
            max_length=max_length,
            num_beams = 15,
            do_sample=True,
            temperature = 0.15,
            num_return_sequences=1,
            early_stopping = True,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id    # Set pad token explicitly
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(3)
    # Decode the generated text
    improved_text = generated_text[90:].strip()

    final_result = improved_text
    print(f"Score after: {heuristic_score(final_result)}")

    print(f"{improved_text}")
    return final_result