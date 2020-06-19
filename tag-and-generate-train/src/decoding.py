import torch as th
import math


def sample(model, src_tokens, temperature=1.0, max_len=200, device=None):
    # Either decode on the model's device or specified device
    # (in which case move the model accordingly)
    if device is None:
        device = list(model.parameters())[0].device
    else:
        model = model.to(device)
    # Go into eval mode (e.g. disable dropout)
    model.eval()
    # Encode source sentece
    src_tensor = th.LongTensor(src_tokens).to(device).view(-1, 1)
    encodings = model.encode(src_tensor)
    # Initialize decoder state
    state = model.initial_state()
    # Start decoding
    out_tokens = [model.vocab["<sos>"]]
    eos_token = model.vocab["<eos>"]
    while out_tokens[-1] != eos_token and len(out_tokens) <= max_len:
        current_token = th.LongTensor([out_tokens[-1]]).view(1, 1).to(device)
        # One step of the decoder
        log_p, state = model.decode_step(current_token, encodings, state)
        # Probabilities
        probs = th.exp(log_p / temperature).view(-1)
        # Sample
        next_token = th.multinomial(probs.view(-1), 1).item()
        # Add to the generated sentence
        out_tokens.append(next_token)
    # Return generated token (idxs) without <sos> and <eos>
    out_tokens = out_tokens[1:]
    if out_tokens[-1] == eos_token:
        out_tokens = out_tokens[:-1]
    return out_tokens


def greedy(model, src_tokens, max_len=200, device=None):
    # Either decode on the model's device or specified device
    # (in which case move the model accordingly)
    if device is None:
        device = list(model.parameters())[0].device
    else:
        model = model.to(device)
    # Go into eval mode (e.g. disable dropout)
    model.eval()
    # Encode source sentece
    src_tensor = th.LongTensor(src_tokens).to(device).view(-1, 1)
    encodings = model.encode(src_tensor)
    # Initialize decoder state
    state = model.initial_state()
    # Start decoding
    out_tokens = [model.vocab["<sos>"]]
    eos_token = model.vocab["<eos>"]
    while out_tokens[-1] != eos_token and len(out_tokens) <= max_len:
        current_token = th.LongTensor([out_tokens[-1]]).view(1, 1).to(device)
        # One step of the decoder
        log_p, state = model.decode_step(current_token, encodings, state)
        # Sample
        next_token = log_p.view(-1).argmax()
        # Add to the generated sentence
        out_tokens.append(next_token.item())
    # Return generated token (idxs) without <sos> and <eos>
    out_tokens = out_tokens[1:]
    if out_tokens[-1] == eos_token:
        out_tokens = out_tokens[:-1]
    return out_tokens


def beam_search(
    model,
    src_tokens,
    prefer_gtag,
    src_tag,
    beam_size=1,
    len_penalty=0.0,
    max_len=200,
    # style_prior=1,  # lower the better!
    device=None
):
    # assert style_prior <= 1 and style_prior > 0
    # Either decode on the model's device or specified device
    # (in which case move the model accordingly)
    if device is None:
        device = list(model.parameters())[0].device
    else:
        model = model.to(device)
    # Go into eval mode (e.g. disable dropout)
    model.eval()
    # Encode source sentece
    src_tensor = th.LongTensor(src_tokens).to(device).view(-1, 1)
    encodings = model.encode(src_tensor)
    # Initialize beams
    beams = [{
        # Tokens generated in this beam
        "tokens": [model.vocab["<sos>"]],
        # Internal decoder state
        "state": model.initial_state(),
        # log probabilityof the sequence
        "log_p": 0,
        # Whether this beam is dead
        "is_over": False,
    }]
    # Start decoding
    eos_token = model.vocab["<eos>"]
    t = 0
    while not beams[-1]["is_over"]:
        # Pass on dead beams
        beam_candidates = [beam for beam in beams if beam["is_over"]]
        # Take a step on all active beams
        active_beams = [beam for beam in beams if not beam["is_over"]]
        # Last produced tokens
        current_tokens = th.LongTensor(
            [beam["tokens"][-1] for beam in active_beams])
        current_tokens = current_tokens.view(1, -1).to(device)
        # Decoder states
        states = [
            th.cat([beam["state"][layer] for beam in active_beams], dim=1)
            if beams[0]["state"][0] is not None
            else None
            for layer in range(model.n_layers)
        ]
        # Take a step
        log_ps, new_states = model.decode_step(
            current_tokens,
            encodings.repeat(1, len(active_beams), 1),
            states,
        )
        # Topk tokens at this step
        log_ps = log_ps.view(log_ps.size(1), -1)

        # Style Prior
        # log_ps[:, model.vocab["GMASK"]] -= math.log(style_prior)

        log_p_tokens, top_tokens = log_ps.topk(beam_size, dim=-1)
        #print(log_ps.shape)
        
        # Append to candidates
        for i, beam in enumerate(active_beams):
            for token, log_p_token in zip(top_tokens[i], log_p_tokens[i]):
                # Update tokens, state and log_p
                candidate = {
                    "tokens": beam["tokens"] + [token.item()],
                    "state": [h[:, i:i+1].detach() for h in new_states],
                    "log_p": beam["log_p"] + log_p_token.item(),
                    "is_over": False,
                }
                # check whether this beam is over
                generated_eos = candidate["tokens"][-1] == eos_token
                too_long = len(candidate["tokens"]) > max_len
                candidate["is_over"] = generated_eos or too_long
                # Save candidate
                beam_candidates.append(candidate)
        t += 1
        # Now rerank and keep top beams
        beams = sorted(
            beam_candidates,
            key=lambda beam: beam["log_p"] /  # log probability
            (len(beam["tokens"]))**len_penalty,  # Length penalty
        )[-beam_size:]  # top k
    # Return generated token (idxs) without <sos> and <eos>
    
    if prefer_gtag == 1:  # prefer the hypothesis that's mostly gtags
        num_gtag_criterion = lambda x: len([x_i for x_i in x["tokens"] if src_tag in model.vocab[x_i]]) 
        beams = sorted(beams, key=num_gtag_criterion)
        
    out_tokens = beams[-1]["tokens"][1:]
    if out_tokens[-1] == eos_token:
        out_tokens = out_tokens[:-1]
    return out_tokens
