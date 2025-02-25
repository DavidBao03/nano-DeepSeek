import torch
import torch.nn.functional as F
from model import Transformer, DeepSeekConfig, ModelArgs
import time

import tiktoken

def generate(model, prompt, num_return_sequences=4, max_length = 32):
    model.eval()

    enc = tiktoken.get_encoding("gpt2")

    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens
    start_pos = 0
    final_gen = tokens
    while final_gen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits = DeepSeek(xgen, start_pos) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            final_gen = torch.cat((final_gen, xcol), dim=1)
            xgen = xcol
            start_pos += xgen.size(1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = final_gen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"{decoded}")

    for layer in DeepSeek.layers:
        layer.ffn.get_expert_activations()


if __name__ == '__main__':
    DeepSeek = Transformer(DeepSeekConfig)
    checkpoint = torch.load('deepseek_new.pt', map_location='cpu', weights_only=False)
    DeepSeek.load_state_dict(checkpoint['model'])

    start_time = time.time()
    num_return_sequences = 10
    max_length = 32

    generate(DeepSeek, "I'm a language model which", num_return_sequences = num_return_sequences, max_length = max_length)
    print(f"generating {num_return_sequences * max_length} tokens costs {time.time() - start_time:.2f} seconds")