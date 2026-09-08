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


# ---- all three senses ------------------------------------------------------


def sense_sets(image_size: int, n_each: int, seed: int) -> dict:
    """Training pairs for each sense, and the captions each is scored against.

    Kept apart rather than pooled so accuracy can be reported per sense. A
    single mixed number would hide a tower that learned pictures and heard
    nothing, which is exactly the failure worth catching.
    """
    from motherbrain.imagedata import pairs as image_pairs
    from motherbrain.mediadata import (all_sound_captions, all_video_captions,
                                       sound_pairs, video_pairs)

    return {
        "sight": {"pairs": image_pairs(n_each, size=image_size, seed=seed),
                  "captions": all_captions()},
        "sound": {"pairs": sound_pairs(n_each, size=image_size, seed=seed + 1),
                  "captions": all_sound_captions()},
        "video": {"pairs": video_pairs(n_each, size=image_size, seed=seed + 2),
                  "captions": all_video_captions()},
    }


@torch.no_grad()
def score_against(model, tok, samples, candidates, device,
                  batch_size: int = 24) -> float:
    """Forced choice over one sense's own captions.

    Scoring each sense against only its own alternatives is the honest
    comparison: asking whether a sound is "a red circle" is not a question
    about hearing.
    """
    if not samples:
        return 0.0
    was_training = model.training
    model.eval()

    images = torch.stack([t for t, _ in samples]).to(device)
    truth = [c for _, c in samples]
    scores = torch.zeros(len(samples), len(candidates), device=device)

    for j, candidate in enumerate(candidates):
        idx, targets = encode_batch(tok, [candidate], device)
        for start in range(0, len(samples), batch_size):
            chunk = images[start:start + batch_size]
            n = chunk.shape[0]
            rows = targets.expand(n, -1)
            logits, _ = model(idx.expand(n, -1), targets=rows, images=chunk)
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


def measure_senses(model, tok, device, image_size: int, n_eval: int,
                   seed: int) -> dict:
    """Held-out accuracy for each sense, beside its own chance rate."""
    held = sense_sets(image_size, n_eval, seed + 9999)
    out = {}
    for name, data in held.items():
        out[name] = {
            "accuracy": round(score_against(model, tok, data["pairs"],
                                            data["captions"], device), 4),
            "chance": round(1 / len(data["captions"]), 4),
            "n": len(data["pairs"]),
        }
    return out


def train_senses(model, tok, device, steps: int = 3000, batch_size: int = 18,
                 lr: float = 4e-4, image_size: int = 64, n_each: int = 1400,
                 n_eval: int = 96, seed: int = 4242, progress_cb=None,
                 eval_every: int = 400, on_eval=None) -> dict:
    """Teach one tower all three senses at once.

    Interleaved rather than one sense after another: trained in sequence, the
    tower forgets pictures while learning sound, which is the ordinary result
    and not an interesting one. Every batch mixes all three.
    """
    torch.manual_seed(seed)
    training = sense_sets(image_size, n_each, seed)
    pool = [(t, c) for data in training.values() for t, c in data["pairs"]]

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.vision.parameters():
        p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no perception parameters to train")

    before = measure_senses(model, tok, device, image_size, n_eval, seed)
    opt = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    losses: list[float] = []
    best = {"score": -1.0, "state": None, "senses": before}

    for step in range(1, steps + 1):
        pick = torch.randint(len(pool), (batch_size,), generator=generator)
        batch = [pool[i] for i in pick.tolist()]
        images = torch.stack([t for t, _ in batch]).to(device)
        idx, targets = encode_batch(tok, [c for _, c in batch], device)

        _, loss = model(idx, targets=targets, images=images)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        losses.append(loss.item())
        if progress_cb:
            progress_cb({"step": step, "total": steps, "loss": loss.item()})

        if step % eval_every == 0 or step == steps:
            senses = measure_senses(model, tok, device, image_size, n_eval, seed)
            model.train()
            # Above-chance margin summed across senses: the thing being
            # trained for, rather than any one sense at the others' expense.
            score = sum(s["accuracy"] - s["chance"] for s in senses.values())
            if score > best["score"]:
                best = {"score": score, "senses": senses,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.vision.state_dict().items()}}
            if on_eval:
                on_eval(step, senses, score, score >= best["score"])

    if best["state"] is not None:
        model.vision.load_state_dict(best["state"])
    model.eval()
    return {
        "before": before,
        "after": best["senses"],
        "loss_before": round(sum(losses[:20]) / max(len(losses[:20]), 1), 4),
        "loss_after": round(sum(losses[-20:]) / max(len(losses[-20:]), 1), 4),
        "trainable_params": sum(p.numel() for p in trainable),
        "steps": steps,
    }


def create_hearing_patch(run_dir, device: str = "auto", steps: int = 4000,
                         batch_size: int = 18, lr: float = 4e-4,
                         extra_layers: int = 2, n_each: int = 1400,
                         n_eval: int = 96, note: str = "hearing",
                         progress_cb=None, on_eval=None,
                         tower_state: dict | None = None):
    """Teach the tower sound and video as well, and record it as the next version.

    The tower is deepened rather than retrained in place. A version that
    learns more should be a larger model, not the same one rearranged, and the
    lineage refuses any patch that does not add parameters. New blocks start as
    exact identities, so nothing the model could already see is thrown away in
    order to hear.
    """
    import time
    import uuid

    from motherbrain.growth import deepen_sight
    from motherbrain.patches import PatchStore, Version, build_version
    from motherbrain.train import pick_device

    store = PatchStore(run_dir)
    dev = pick_device(device)
    model, tok, base_version = build_version(run_dir, device=device)
    model.to(dev)

    if getattr(model, "vision", None) is None:
        raise ValueError(
            f"v{base_version} has no perception tower to deepen. `mb sight` "
            f"builds one first.")

    params_before = model.n_params()
    image_size = model.cfg.image_size
    deepen_sight(model, extra_layers=extra_layers)

    if tower_state is not None:
        model.vision.load_state_dict({k: v.float()
                                      for k, v in tower_state.items()})
        senses = measure_senses(model, tok, dev, image_size, n_eval, 4242)
        result = {"before": senses, "after": senses, "loss_before": 0.0,
                  "loss_after": 0.0, "steps": steps,
                  "trainable_params": sum(p.numel()
                                          for p in model.vision.parameters())}
    else:
        result = train_senses(model, tok, dev, steps=steps,
                              batch_size=batch_size, lr=lr,
                              image_size=image_size, n_each=n_each,
                              n_eval=n_eval, progress_cb=progress_cb,
                              on_eval=on_eval)

    payload = {name: tensor.detach().cpu().clone()
               for name, tensor in model.state_dict().items()
               if name.startswith("vision.")}

    after = result["after"]
    version = Version(
        version=store.head + 1,
        patch_id=uuid.uuid4().hex[:8],
        parent=base_version,
        created_at=time.time(),
        doc_start=store.consumed_docs(),
        doc_end=store.consumed_docs(),
        n_documents=0, n_chars=0, n_tokens=0,
        steps=result["steps"],
        rank=0,
        trainable_params=result["trainable_params"],
        loss_before=result["loss_before"],
        loss_after=result["loss_after"],
        sources=["rendered sounds and clips"],
        note=note,
        base_fingerprint=store.base_fingerprint,
        mode="hearing",
        params_before=params_before,
        params_after=model.n_params(),
        vision_layers=model.cfg.vision_layers,
        vision_width=model.cfg.vision_width,
        vision_heads=model.cfg.vision_heads,
        image_size=image_size,
        patch_size=model.cfg.patch_size,
        extra_vision_layers=extra_layers,
        sight_accuracy=after["sight"]["accuracy"],
        sound_accuracy=after["sound"]["accuracy"],
        video_accuracy=after["video"]["accuracy"],
    )
    store.record(version, payload)
    return version, result
