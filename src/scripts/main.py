"""
credits: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import math
import os
from tempfile import TemporaryDirectory
import time
from typing import Optional
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator


from src.models.vanilla_transformer import generate_square_subsequent_mask, VanillaTransformer
from src.utils.data import get_batch, bptt, preprocess_data, batchify
from src.utils.datasets.minecraft import MinecraftDataset


def train(model: nn.Module,
          train_data: torch.Tensor,
          num_tokens: int,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          epoch: int,
          device: Optional[str] = "cpu") -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
            
        # move data to device
        data = data.to(device)
        targets = targets.to(device)
        
        output = model(data, src_mask)
        loss = criterion(output.view(-1, num_tokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(
                f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module,
             eval_data: torch.Tensor,
             num_tokens: int,
             criterion: nn.Module,
             device: Optional[str] = "cpu") -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
                
            # move data to device
            data = data.to(device)
            targets = targets.to(device)
        
            output = model(data, src_mask)
            output_flat = output.view(-1, num_tokens)
            print(torch.argmax(output_flat, dim=-1), targets)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


if __name__ == "__main__":
    # optimatization parameters
    epochs = 20

    # data parameters
    data_dir = "/mnt/data/home/manuel/milt/data/worlds/shmar"
    train_batch_size = 256
    eval_batch_size = 256

    # model parameters
    emsize = 1024  # embedding dimension
    d_hid = 1024  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 16  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability

    # device parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # optimization parameters
    lr = 5.0  # learning rate
    
    # load and tokenize data
    data = MinecraftDataset(data_dir)
    vocab = build_vocab_from_iterator(data, specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    num_tokens = len(vocab)  # size of vocabulary

    # load and split data
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
    
    val_size = int(0.5 * len(val_data))
    test_size = len(val_data) - val_size
    val_data, test_data = torch.utils.data.random_split(val_data, [val_size, test_size])
    
    train_data = preprocess_data(train_data, vocab)
    val_data = preprocess_data(val_data, vocab)
    test_data = preprocess_data(test_data, vocab)

    # split data into batches
    train_data = batchify(train_data, train_batch_size)  # shape [seq_len, batch_size]
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    # initialize model
    model = VanillaTransformer(num_tokens, emsize, nhead, d_hid, nlayers,
                               dropout).to(device)

    # set up optimization modules (loss, optimizer, scheduler)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    
    # start training
    best_val_loss = float('inf')

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(model, train_data, num_tokens, criterion, optimizer, scheduler, epoch, device)
            val_loss = evaluate(
                model,
                val_data,
                num_tokens,
                criterion,
                device)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()
        model.load_state_dict(
            torch.load(best_model_params_path))  # load best model states

    test_loss = evaluate(
                model,
                test_data,
                num_tokens,
                criterion,
                device)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
          f'test ppl {test_ppl:8.2f}')
    print('=' * 89)