"""Teaching MotherBrain to see, and measuring whether it worked.

Attaching a vision tower is easy and proves nothing. The loss falls either
way, because the language half alone can learn what captions look like without
ever consulting the picture - and a model that has learned only that will
still produce a fluent, confident, wrong caption for every image you show it.

So the measurement here is forced choice over every caption the world admits:
show the model an image it has never seen, score all thirty-two possible
captions, and see whether the true one comes out lowest. Chance is one in
thirty-two. Anything materially above that is information arriving through the
tower, because nothing else about the model changed.
"""

from __future__ import annotations

import torch

from motherbrain.imagedata import COLOURS, SHAPES, caption, pairs


def all_captions() -> list[str]:
    """Every caption this world can produce - the forced-choice alternatives."""
    return [caption(shape, colour) for shape in SHAPES for colour in COLOURS]


def encode_batch(tok, captions: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise captions into (inputs, targets), padded and masked.

    Padding is masked out with -100 rather than trained on, so a short caption
    in a batch of long ones does not teach the model to emit filler.
    """
    rows = [tok.encode(c, bos=True, eos=True) for c in captions]
    width = max(len(r) for r in rows)
    inputs, targets = [], []
    for row in rows:
        padded = row + [0] * (width - len(row))
        inputs.append(padded[:-1])
        # Targets are the next token; padding contributes no loss.
        targets.append([(t if i < len(row) - 1 else -100)
                        for i, t in enumerate(padded[1:])])
    return (torch.tensor(inputs, device=device),
            torch.tensor(targets, device=device))


@torch.no_grad()
def forced_choice_accuracy(model, tok, samples, device, batch_size: int = 32
                           ) -> float:
    """Share of held-out images whose true caption scores best of all captions.

    Scoring every alternative rather than sampling one removes the decoding
    strategy from the measurement: this is what the model believes, not what it
    happened to say.
    """
    if not samples:
        return 0.0
    was_training = model.training
    model.eval()

    candidates = all_captions()
    images = torch.stack([img for img, _ in samples]).to(device)
    truth = [cap for _, cap in samples]

    # (n_images, n_candidates) of per-caption loss.
    scores = torch.zeros(len(samples), len(candidates), device=device)
    for j, candidate in enumerate(candidates):
        idx, targets = encode_batch(tok, [candidate], device)
        for start in range(0, len(samples), batch_size):
            chunk = images[start:start + batch_size]
            n = chunk.shape[0]
            rows = targets.expand(n, -1)
            logits, _ = model(idx.expand(n, -1), targets=rows, images=chunk)
            # The model returns a batch mean; the comparison is per image, so
            # the per-token loss is recomputed and averaged row by row.
            per_token = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), rows.reshape(-1),
                ignore_index=-100, reduction="none").view(n, -1)
            mask = (rows != -100).float()
            scores[start:start + n, j] = (per_token * mask).sum(1) / mask.sum(1)

    chosen = scores.argmin(dim=1)
    correct = sum(1 for i, c in enumerate(chosen) if candidates[c] == truth[i])
    if was_training:
        model.train()
    return correct / len(samples)


def train_sight(model, tok, device, steps: int = 400, batch_size: int = 16,
                lr: float = 3e-4, image_size: int = 64, n_train: int = 2048,
                n_eval: int = 128, seed: int = 1337, progress_cb=None
                ) -> dict:
    """Train an already-attached vision tower. Returns what it learned.

    Only the tower is optimised; the language model is frozen. A tower learning
    to see while the layers reading it are also moving is two moving targets,
    and the text the model already knows is the fixed thing the visual vectors
    have to land on.
    """
    torch.manual_seed(seed)
    training = pairs(n_train, size=image_size, seed=seed)
    held_out = pairs(n_eval, size=image_size, seed=seed + 9999)

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.vision.parameters():
        p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no vision parameters to train")

    before = forced_choice_accuracy(model, tok, held_out, device)

    opt = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []

    for step in range(steps):
        pick = torch.randint(len(training), (batch_size,), generator=generator)
        batch = [training[i] for i in pick.tolist()]
        images = torch.stack([img for img, _ in batch]).to(device)
        idx, targets = encode_batch(tok, [cap for _, cap in batch], device)

        _, loss = model(idx, targets=targets, images=images)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(loss.item())
        if progress_cb:
            progress_cb({"step": step + 1, "total": steps, "loss": loss.item()})

    model.eval()
    after = forced_choice_accuracy(model, tok, held_out, device)
    return {
        "loss_before": round(sum(losses[:10]) / max(len(losses[:10]), 1), 4),
        "loss_after": round(sum(losses[-10:]) / max(len(losses[-10:]), 1), 4),
        "accuracy_before": round(before, 4),
        "accuracy_after": round(after, 4),
        "chance": round(1 / len(all_captions()), 4),
        "n_train": n_train,
        "n_eval": n_eval,
        "trainable_params": sum(p.numel() for p in trainable),
    }


def create_sight_patch(run_dir, device: str = "auto", steps: int = 3000,
                       batch_size: int = 24, lr: float = 6e-4,
                       layers: int = 4, width: int = 256, heads: int = 4,
                       image_size: int = 64, patch_size: int = 16,
                       n_train: int = 4096, n_eval: int = 192,
                       note: str = "sight", progress_cb=None,
                       tower_state: dict | None = None):
    """Give the current version sight, and record it as the next one.

    Growth by experts makes the model bigger at what it already does; this
    makes it able to do something it could not do at all. It is still growth -
    the tower's parameters are new and nothing before it is touched - so it
    goes through the same lineage, with the same promise that every version is
    larger than the last.

    `tower_state` loads an already-trained tower instead of training one, so a
    long run does not have to happen inside whatever process is recording it.
    """
    import time
    import uuid

    from motherbrain.growth import add_sight
    from motherbrain.patches import PatchStore, Version, build_version
    from motherbrain.train import pick_device

    store = PatchStore(run_dir)
    dev = pick_device(device)
    model, tok, base_version = build_version(run_dir, device=device)
    model.to(dev)

    if getattr(model, "vision", None) is not None:
        raise ValueError(f"v{base_version} can already see")

    params_before = model.n_params()
    add_sight(model, layers=layers, width=width, heads=heads,
              image_size=image_size, patch_size=patch_size)

    if tower_state is not None:
        model.vision.load_state_dict(
            {k: v.float() for k, v in tower_state.items()})
        held_out = pairs(n_eval, size=image_size, seed=1337 + 9999)
        accuracy = forced_choice_accuracy(model, tok, held_out, dev)
        result = {"loss_before": 0.0, "loss_after": 0.0,
                  "accuracy_before": 0.0, "accuracy_after": round(accuracy, 4),
                  "trainable_params": sum(p.numel() for p in model.vision.parameters()),
                  "n_train": n_train, "n_eval": n_eval,
                  "chance": round(1 / len(all_captions()), 4)}
    else:
        result = train_sight(model, tok, dev, steps=steps, batch_size=batch_size,
                             lr=lr, image_size=image_size, n_train=n_train,
                             n_eval=n_eval, progress_cb=progress_cb)

    payload = {name: tensor.detach().cpu().clone()
               for name, tensor in model.state_dict().items()
               if name.startswith("vision.")}
    if not payload:
        raise RuntimeError("the vision tower produced no weights to save")

    version = Version(
        version=store.head + 1,
        patch_id=uuid.uuid4().hex[:8],
        parent=base_version,
        created_at=time.time(),
        doc_start=store.consumed_docs(),
        doc_end=store.consumed_docs(),
        n_documents=0,
        n_chars=0,
        n_tokens=0,
        steps=steps,
        rank=0,
        trainable_params=result["trainable_params"],
        loss_before=result["loss_before"],
        loss_after=result["loss_after"],
        sources=["rendered image-caption pairs"],
        note=note,
        base_fingerprint=store.base_fingerprint,
        mode="sight",
        params_before=params_before,
        params_after=model.n_params(),
        vision_layers=layers,
        vision_width=width,
        vision_heads=heads,
        image_size=image_size,
        patch_size=patch_size,
        sight_accuracy=result["accuracy_after"],
    )
    store.record(version, payload)
    return version, result
