# ruff: noqa: E501  # Ignore line length warnings
import argparse
import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import flagquantum as fq

# ==================== Core Modules ====================


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi-head attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        mask: Optional[torch.Tensor] = None,
        use_bias: bool = False,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.mask = mask
        self.attn_weights = None

        self.k_linear = None
        self.q_linear = None
        self.v_linear = None
        self.combine_heads = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Separate into multiple heads"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[nn.Dropout] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention scores and weighted sum"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        attn = torch.matmul(scores, value)
        return attn, scores

    def downstream(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Downstream processing after attention computation"""
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)

        x, self.attn_weights = self.attention(q, k, v, mask, dropout=self.dropout)
        concat = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return concat


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Classical multi-head attention"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        mask: Optional[torch.Tensor] = None,
        use_bias: bool = False,
    ):
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        assert (
            embed_dim == self.embed_dim
        ), f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"

        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        x = self.downstream(q, k, v, batch_size, mask or self.mask)
        return self.combine_heads(x)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum multi-head attention"""

    class QLayer(nn.Module):
        def __init__(self, n_wires: int = 8, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers

            func_list = [
                {"func": "ry", "wires": [i], "input_idx": [i]} for i in range(n_wires)
            ]
            self.encoder = fq.GeneralEncoder(func_list)

            base_mod = [self.encoder]
            for _ in range(n_layers):
                base_mod = (
                    base_mod
                    + [fq.CX(wires=[i, (i + 1) % n_wires]) for i in range(n_wires)]
                    + [
                        fq.RY(wires=[i], has_params=True, trainable=True)
                        for i in range(n_wires)
                    ]
                )

            self.unitary_mod = fq.InvertibleUnitary(base_mod)

        def forward(
            self, x: torch.Tensor, q_device: fq.DistributedQuantumDevice
        ) -> torch.Tensor:
            q_device.reset_states()
            self.unitary_mod(q_device, x)
            return fq.measure_allZ(q_device, shots=0, training=True)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        world_size: int = 8,
        dropout: float = 0.1,
        n_qubits: int = 8,
        n_qlayers: int = 3,
        **kwargs,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.world_size = world_size

        self.q_device = None
        self.last_bsz = None

        self.q_layer = self.QLayer(n_wires=n_qubits, n_layers=n_qlayers)
        self.proj_in = nn.Linear(embed_dim, n_qubits)
        self.proj_out = nn.Linear(n_qubits, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _get_device(self, bsz: int) -> fq.DistributedQuantumDevice:
        """Get or create a quantum device"""
        if self.q_device is None or self.last_bsz != bsz:
            if self.q_device is not None:
                del self.q_device
                torch.cuda.empty_cache()

            self.q_device = fq.DistributedQuantumDevice(
                n_wires=self.n_qubits,
                bsz=bsz,
                device="cuda",
                world_sz=self.world_size,
                invertible=True,
            )
            self.last_bsz = bsz
        else:
            self.q_device.reset_states()
        return self.q_device

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        device = x.device

        x_flat = x.reshape(-1, embed_dim)
        x_qubit = self.proj_in(x_flat)

        current_dev = self._get_device(batch_size * seq_len)

        # Parallel computation of Q, K, V
        k_flat = self.q_layer(x_qubit, current_dev)
        q_flat = self.q_layer(x_qubit, current_dev)
        v_flat = self.q_layer(x_qubit, current_dev)

        k = k_flat.to(device).view(batch_size, seq_len, self.n_qubits)
        q = q_flat.to(device).view(batch_size, seq_len, self.n_qubits)
        v = v_flat.to(device).view(batch_size, seq_len, self.n_qubits)

        q_proj = self.proj_out(q)
        k_proj = self.proj_out(k)
        v_proj = self.proj_out(v)

        x_attn = self.downstream(q_proj, k_proj, v_proj, batch_size, mask or self.mask)
        x_attn = self.norm(x_attn + x)

        x_attn_flat = x_attn.reshape(-1, embed_dim)
        x_attn_qubit = self.proj_in(x_attn_flat)
        output_flat = self.q_layer(x_attn_qubit, current_dev).to(device)

        return self.proj_out(output_flat).view(batch_size, seq_len, embed_dim)


# ==================== Feed-Forward Network Modules ====================


class FeedForwardBase(nn.Module):
    """Base class for feed-forward network"""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, ffn_dim)
        self.linear_2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Classical feed-forward network"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        return self.linear_2(x)


class FeedForwardQuantum(FeedForwardBase):
    """Quantum feed-forward network"""

    class QL(nn.Module):
        def __init__(self, n_wires: int = 8, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers

            func_list = [
                {"func": "ry", "wires": [i], "input_idx": [i]} for i in range(n_wires)
            ]
            self.encoder = fq.GeneralEncoder(func_list)

            base_mod = [self.encoder]
            for _ in range(n_layers):
                base_mod = (
                    base_mod
                    + [fq.CX(wires=[i, (i + 1) % n_wires]) for i in range(n_wires)]
                    + [
                        fq.RY(wires=[i], has_params=True, trainable=True)
                        for i in range(n_wires)
                    ]
                )

            self.unitary_mod = fq.InvertibleUnitary(base_mod)

        def forward(
            self, x: torch.Tensor, q_device: fq.DistributedQuantumDevice
        ) -> torch.Tensor:
            q_device.reset_states()
            self.unitary_mod(q_device, x)
            return fq.measure_allZ(q_device, shots=0, training=True)

    def __init__(
        self,
        embed_dim: int,
        n_qubits: int,
        world_size: int = 8,
        n_qlayers: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(embed_dim, ffn_dim=n_qubits, dropout=dropout)

        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.world_size = world_size

        self.q_device = None
        self.last_bsz = None

        self.q_l = self.QL(n_wires=n_qubits, n_layers=n_qlayers)
        self.linear_1 = nn.Linear(embed_dim, n_qubits)
        self.linear_2 = nn.Linear(n_qubits, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _get_device(self, bsz: int) -> fq.DistributedQuantumDevice:
        """Get or create a quantum device"""
        if self.q_device is None or self.last_bsz != bsz:
            if self.q_device is not None:
                del self.q_device
                torch.cuda.empty_cache()

            self.q_device = fq.DistributedQuantumDevice(
                n_wires=self.n_qubits,
                bsz=bsz,
                device="cuda",
                world_sz=self.world_size,
                invertible=True,
            )
            self.last_bsz = bsz
        else:
            self.q_device.reset_states()
        return self.q_device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        residual = x
        x = self.linear_1(x)
        x_flat = x.reshape(-1, self.n_qubits)

        q_device = self._get_device(batch_size * seq_len)
        x_flat_out = self.q_l(x_flat, q_device)

        if x_flat_out.device != x.device:
            x_flat_out = x_flat_out.to(x.device)

        x = x_flat_out.view(batch_size, seq_len, self.n_qubits)
        x = self.linear_2(x)
        return self.norm(x + residual)


# ==================== Transformer Modules ====================


class TransformerBlockBase(nn.Module):
    """Base class for transformer block"""

    def __init__(
        self,
        embed_dim: int,
        num_head: int,
        ff_dim: int,
        dropout: float = 0.1,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.attn = None
        self.ffn = None
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(x, self.mask)
        x = self.norm1(attn_output + x)
        x = self.dropout1(x)

        ff_output = self.ffn(x)
        x = self.norm2(ff_output + x)
        x = self.dropout2(x)
        return x


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(embed_dim, num_heads, ff_dim, dropout, mask)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout, mask)
        self.ffn = FeedForwardClassical(embed_dim, ff_dim, dropout)


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum transformer block"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        dropout: float = 0.1,
        mask: Optional[torch.Tensor] = None,
        q_device: str = "default.qubit",
    ):
        super().__init__(embed_dim, num_heads, ffn_dim, dropout, mask)

        self.attn = MultiHeadAttentionQuantum(
            embed_dim,
            num_heads,
            n_qubits=n_qubits_transformer,
            n_qlayers=n_qlayers,
            dropout=dropout,
            mask=mask,
        )

        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, n_qubits_ffn, n_qlayers, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)


# ==================== Positional Encoding ====================


class PositionalEncoder(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, embed_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_dim)))
                if i + 1 < embed_dim:
                    pe[pos, i + 1] = math.cos(
                        pos / (10000 ** ((2 * (i + 1)) / embed_dim))
                    )

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ==================== Text Classifier ====================


class TextClassifier(nn.Module):
    """Text classifier with optional quantum components"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        num_classes: int,
        vocab_size: int,
        ffn_dim: int = 32,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        dropout: float = 0.1,
        q_device: str = "device.qubit",
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.dropout = nn.Dropout(dropout)

        use_quantum = n_qubits_transformer > 0

        if use_quantum:
            print(
                f"Using quantum transformer with {n_qubits_transformer} qubits, {n_qlayers} layers"
            )
            if n_qubits_ffn > 0:
                print(f"Quantum feed-forward with {n_qubits_ffn} qubits")
            else:
                print("Classical feed-forward")

            transformer_blocks = [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer=n_qubits_transformer,
                    n_qubits_ffn=n_qubits_ffn,
                    n_qlayers=n_qlayers,
                    dropout=dropout,
                    q_device=q_device,
                )
                for _ in range(num_blocks)
            ]
        else:
            print(f"Using classical transformer with {num_blocks} blocks")
            transformer_blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]

        self.transformers = nn.Sequential(*transformer_blocks)

        if num_classes > 2:
            self.class_logits = nn.Linear(embed_dim, num_classes)
        else:
            self.class_logits = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.class_logits(x)


# ==================== Utility Functions ====================


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_distributed() -> Optional[int]:
    """Initialize distributed training environment"""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank
    return None


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Calculate training epoch time in minutes and seconds"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_dataset_loaders(
    tokenizer, max_seq_len: int, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Get train and test data loaders for IMDB dataset"""
    local_path = "./models/imdb_dataset"
    raw_datasets = load_dataset(local_path)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )

    tokenized_datasets = raw_datasets.map(tokenize_fn, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_loader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size
    )
    test_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

    return train_loader, test_loader


# ==================== Training and Evaluation Functions ====================


def train(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    max_batches: int = 10,
) -> Tuple[float, float]:
    """Train the model for one epoch"""
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    device = next(model.parameters()).device
    step_count = 0

    for i, batch in enumerate(iterator):
        if i >= max_batches:
            break

        step_count += 1
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device).float().view(-1, 1)

        predictions = model(input_ids)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(predictions)
        predicted_labels = (probs > 0.5).float()
        correct = (predicted_labels == labels).float().sum()
        acc = correct / labels.size(0)

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # Clean up intermediate variables
        del predictions, loss, probs, predicted_labels, correct

        if i % 2 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    return epoch_loss / step_count, epoch_acc / step_count


def evaluate(
    model: nn.Module, iterator: DataLoader, criterion: nn.Module, max_batches: int = 2
) -> Tuple[float, float]:
    """Evaluate the model on validation/test set"""
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    device = next(model.parameters()).device
    step_count = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if i >= max_batches:
                break

            step_count += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device).float().view(-1, 1)

            predictions = model(input_ids)

            if predictions.dim() == 3:
                predictions = predictions.squeeze(1)

            loss = criterion(predictions, labels)

            probs = torch.sigmoid(predictions)
            predicted_labels = (probs > 0.5).float()
            correct = (predicted_labels == labels).float().sum()
            acc = correct / labels.size(0)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / step_count, epoch_acc / step_count


# ==================== Main Function ====================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Quantum Text Classifier")

    # Data arguments
    parser.add_argument("-B", "--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument(
        "-s", "--max_seq_len", default=16, type=int, help="Maximum sequence length"
    )

    # Model arguments
    parser.add_argument(
        "-e", "--embed_dim", default=8, type=int, help="Embedding dimension"
    )
    parser.add_argument(
        "-H", "--n_heads", default=2, type=int, help="Number of attention heads"
    )
    parser.add_argument(
        "-t",
        "--n_transformer_blocks",
        default=1,
        type=int,
        help="Number of transformer blocks",
    )
    parser.add_argument("-f", "--ffn_dim", default=8, type=int, help="FFN dimension")
    parser.add_argument(
        "-d", "--dropout_rate", default=0.1, type=float, help="Dropout rate"
    )

    # Quantum arguments
    parser.add_argument(
        "-q",
        "--n_qubits_transformer",
        default=22,
        type=int,
        help="Number of qubits for transformer",
    )
    parser.add_argument(
        "-Q", "--n_qubits_ffn", default=22, type=int, help="Number of qubits for FFN"
    )
    parser.add_argument(
        "-L", "--n_qlayers", default=1, type=int, help="Number of quantum layers"
    )
    parser.add_argument(
        "-D", "--q_device", default="default.qubit", type=str, help="Quantum device"
    )

    # Training arguments
    parser.add_argument(
        "-E", "--n_epochs", default=100, type=int, help="Number of epochs"
    )
    parser.add_argument("-l", "--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument(
        "-C", "--n_classes", default=2, type=int, help="Number of classes"
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Initialize distributed environment
    local_rank = setup_distributed()

    if local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
        print(f"Distributed training on rank {local_rank}, device {device}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Single GPU training on {device}")

    # Prepare data
    print("Loading tokenizer and datasets...")
    local_model_path = "./models/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    train_iter, test_iter = get_dataset_loaders(
        tokenizer, args.max_seq_len, args.batch_size
    )

    # Initialize model
    print("Initializing model...")
    model = TextClassifier(
        embed_dim=args.embed_dim,
        num_heads=args.n_heads,
        num_blocks=args.n_transformer_blocks,
        num_classes=args.n_classes,
        vocab_size=tokenizer.vocab_size,
        ffn_dim=args.ffn_dim,
        n_qubits_transformer=args.n_qubits_transformer,
        n_qubits_ffn=args.n_qubits_ffn,
        n_qlayers=args.n_qlayers,
        dropout=args.dropout_rate,
        q_device=args.q_device,
    ).to(device)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    # Wrap with DDP
    if local_rank is not None and torch.cuda.device_count() > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        print(f"Distributed training with world size: {dist.get_world_size()}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if args.n_classes <= 2
        else torch.nn.CrossEntropyLoss()
    )

    # Training loop
    best_valid_loss = float("inf")

    for epoch in range(args.n_epochs):
        start_time = time.time()

        if local_rank is None or local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.n_epochs}")

        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_iter, criterion)

        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if local_rank is None or local_rank == 0:
                torch.save(model.state_dict(), "model.pt")

        if local_rank is None or local_rank == 0:
            print(f"Epoch: {epoch + 1:02} | Time: {mins}m {secs}s")
            print(
                f"\tTrain Loss: {train_loss:.10g} | Train Acc: {train_acc * 100:.2f}%"
            )
            print(f"\tVal. Loss: {valid_loss:.10g} |  Val. Acc: {valid_acc * 100:.2f}%")

    # Clean up
    if local_rank is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
