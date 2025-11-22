import tiktoken
import torch

from cse599o_basics.softmax import stable_softmax

def top_p_sampling(
    logits: torch.Tensor,  # shape (vocab_size)
    p: float = 0.9,
    t: float = 1.0
):
    # handle t
    logits = logits / t
    probs = stable_softmax(logits, dim=-1)

    # sort everything to find where to do the p cutoff
    sorted_probs, sorted_idxs = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # cutoff_indicies[0] will be idx of first non zero value in (cumulative_probs > p)
    # in other words, the first idx into cumulative_probs (and by 
    # construction sorted_probs) that exceeds p
    cutoff_indices = torch.nonzero(cumulative_probs > p)

    # plus one for off-by-one when doing slice
    cutoff_index = cutoff_indices[0].item() + 1  \
                   if cutoff_indices.numel() > 0 else \
                   len(sorted_probs)

    sorted_probs = sorted_probs[:cutoff_index]
    sorted_idxs = sorted_idxs[:cutoff_index]

    # normalize, sample index, return token
    sorted_probs = sorted_probs / sorted_probs.sum()
    next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idxs[next_token_idx]

def decode(
    model: torch.nn.Module,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    model.eval()

    input_ids = torch.tensor(tokenizer.encode(prompt, allowed_special="all"))

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)  # outputs shape: (seq_len, vocab_size)

        logits = outputs[-1, :]  # logits are for last token in sequence, shape (vocab_size)
        next_token = top_p_sampling(logits, p=top_p, t=temperature)
        input_ids = torch.concat([input_ids, next_token])

        # this has to match the tokenizer's eos_token_id to cut off generation
        if next_token.item() == tokenizer.eot_token:
            break

    return tokenizer.decode(input_ids.tolist())