import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
from heuristics import heuristic_score


model_name = 'bigscience/bloom-560m' # We use this to minimize load, we tried using bigger models and it causes the laptop to crash
tokenizer =  BloomTokenizerFast.from_pretrained(model_name)
model = BloomForCausalLM.from_pretrained(model_name)
model.eval()

def improve_cv_text(input_text, max_length=500):
    # Tokenize the input text
    print(f"Score before: {heuristic_score(input_text)}")
    
    prompt = f"Improve the following text to make it more professional and free from grammatical errors: {input_text}"
    inputs = tokenizer.encode(prompt, return_tensors='pt') # Converts the text we sent into token so it can be processed
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,          
            max_length=max_length, # max length of the improved text
            num_beams = 15, # Beam search strategy. We tell the model to explore 15 difference sequence every step
            do_sample=True, # This allows the model to be somewhat creative
            temperature = 0.15, # This allows the model to be very deterministic but can be creative sometimes
            num_return_sequences=1, # Generate 1 output
            early_stopping = True,
            no_repeat_ngram_size=2, 
            pad_token_id=tokenizer.eos_token_id 
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # convert token back to text
    # Decode the generated text
    improved_text = generated_text[90:].strip()

    final_result = improved_text
    print(f"Score after: {heuristic_score(final_result)}")

    print(f"{improved_text}")
    return final_result