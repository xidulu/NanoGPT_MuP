# Do the coordinate check for the nano GPT model

import torch
import torch.optim as optim

from mup import MuReadout, make_base_shapes, set_base_shapes, MuAdamW
from mup.coord_check import get_coord_data, plot_coord_data, _record_coords
from nano_gpt import GPT, GPTConfig


def train_and_generate(model, sequence, optimizer, config, device="cuda:0"):
    model.to(device)
    model.train()
    epochs = 5
    inputs = torch.tensor([sequence], dtype=torch.long).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(inputs)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    input_ids = torch.tensor([[sequence[0]]], dtype=torch.long).to(device)
    generated_sequence = [sequence[0]]

    for _ in range(len(sequence) - 1):
        with torch.no_grad():
            output = model(input_ids)
            logits = output["logits"]

            predicted_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_sequence.append(predicted_token_id)

            input_ids = torch.cat(
                [
                    input_ids,
                    torch.tensor([[predicted_token_id]], dtype=torch.long).to(
                        device
                    ),
                ],
                dim=1,
            )

    return generated_sequence



config_base = GPTConfig(
    vocab_size=50257,
    n_layer = 12,  # default: 12
    n_head = 12, # default: 12
    n_embd = 192,  # default: 768
    block_size = 256,  # default: 1024
    bias = False,
    return_dict=True
)

config_delta = GPTConfig(
    vocab_size=50257,
    n_layer = 12,  # default: 12
    n_head = 12, # default: 12
    n_embd = 96,  # default: 768
    block_size = 384,  # default: 1024
    bias = False,
    return_dict=True
)

model_base = GPT(config_base)
model_delta = GPT(config_delta)
# make_base_shapes(model_base, model_delta, f'./gpt_48.bsh')
make_base_shapes(model_base, model_delta, f'./gpt_192.bsh')

model_dict = {}
for n_embd in [48, 96, 192, 384, 768]:
    config = GPTConfig(
        vocab_size=50257,
        n_layer = 12,  # default: 12
        n_head = 12, # default: 12
        n_embd = n_embd,  # default: 768
        block_size = 256,  # default: 1024
        bias = False,
        return_dict=True
    )
    model = GPT(config)
    set_base_shapes(model,  f'./gpt_192.bsh', rescale_params=True, do_assert=False)
    model_dict[str(n_embd)] = model

sequence = list(range(200))
data_loader = [
    {'idx': torch.tensor([sequence[:-1]], dtype=torch.long).to('cuda'),
     'targets': torch.tensor([sequence[1:]], dtype=torch.long).to('cuda')}
]

DF = get_coord_data(model_dict, data_loader,
                    optimizer='adam', dict_in_out=True, output_name='loss',
                    nsteps=20)
plot_coord_data(DF, legend=True,
        save_to="coord_check.png",
        face_color='xkcd:light grey')