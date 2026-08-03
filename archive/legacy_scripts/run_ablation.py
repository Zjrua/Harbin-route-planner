"""消融实验 — 精简版（抑制tqdm，快速输出结果）.

用法:
    uv run python scripts/run_ablation.py 2>&1 | grep -E "===|结果|完成|val_loss|score|实验|SKIP|DONE"
"""

import yaml, torch, numpy as np, pandas as pd, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.dataset import create_dataloaders
from src.models.transformer import ItineraryTransformer
from src.models.losses import RouteLoss
from src.evaluate import route_distance, route_time, satisfaction_score, diversity_score, composite_score
from src.inference import generate_with_constraints, filter_route, optimize_route_order


def deep_merge(base, override):
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def train_quick(config, name, device):
    """快速训练（无tqdm输出）."""
    model = ItineraryTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader, val_loader, _ = create_dataloaders("data/processed", config)

    shared = train_loader.dataset.get_shared_data(device)

    # Run encoder once
    model.eval()
    with torch.no_grad():
        encoder_output = model.encode(
            shared["poi_features"].unsqueeze(0),
            shared["adjacency"].unsqueeze(0),
        )

    criterion = RouteLoss(
        ce_weight=config["loss"]["ce_weight"],
        distance_weight=config["loss"]["distance_weight"],
        mhc_weight=config["loss"]["mhc_weight"],
    )

    opt_cfg = config["optimizer"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if "attn" in n and p.requires_grad], "lr": opt_cfg["lr_attn"]},
        {"params": [p for n, p in model.named_parameters() if "ffn" in n and p.requires_grad], "lr": opt_cfg["lr_ffn"]},
        {"params": [p for n, p in model.named_parameters() if "attn" not in n and "ffn" not in n and p.requires_grad], "lr": (opt_cfg["lr_attn"] + opt_cfg["lr_ffn"]) / 2},
    ]
    optimizer = torch.optim.AdamW(grouped, weight_decay=opt_cfg["weight_decay"])

    total_epochs = config["training"]["epochs"]
    patience_limit = config["training"]["patience"]
    grad_clip = opt_cfg.get("grad_clip", 1.0)
    base_tf = config["training"]["teacher_forcing_ratio"]

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(total_epochs):
        model.train()
        model.encoder.eval()  # encoder output is shared, no dropout
        tf_ratio = base_tf * max(0.0, 1.0 - epoch / total_epochs)
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch_device = _build_batch(batch, device, shared, encoder_output)
            optimizer.zero_grad()
            output = model(batch_device)
            logits, target, distances = _align_logits(output, batch_device)
            loss = criterion(logits, target, distances, output.get("embeddings"))
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_device = _build_batch(batch, device, shared, encoder_output)
                output = model(batch_device)
                logits, target, distances = _align_logits(output, batch_device)
                loss = criterion(logits, target, distances, output.get("embeddings"))
                val_loss += loss.item()
                n_val += 1
        val_loss /= max(n_val, 1)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_val_loss": best_val_loss,
            }, Path(config["experiment"]["save_dir"]) / f"ablation_{name}.pt")
        else:
            patience_counter += 1

        if epoch % 10 == 0 or improved:
            marker = " *" if improved else ""
            sys.stdout.write(f"\r  {name}: epoch {epoch+1}/{total_epochs} loss={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f}{marker}\n")
            sys.stdout.flush()

        if patience_counter >= patience_limit:
            break

    return {"name": name, "epochs": best_epoch, "best_val_loss": float(best_val_loss), "params": n_params}, model


def _build_batch(batch, device, shared, encoder_output):
    """Build batch dict with pre-computed encoder output."""
    route_seq, scores, route_activity = batch
    return {
        "poi_features": shared["poi_features"].unsqueeze(0),
        "adjacency": shared["adjacency"].unsqueeze(0),
        "route_sequence": route_seq.to(device),
        "distances": shared["distances"],
        "scores": scores.to(device),
        "activity_types": route_activity.to(device),
        "_encoder_output": encoder_output,
    }


def _align_logits(output, batch_device):
    logits = output["logits"]
    target = batch_device["route_sequence"][:, 1:]
    min_len = min(logits.size(1), target.size(1))
    logits = logits[:, :min_len]
    target = target[:, :min_len]
    distances = batch_device["distances"]
    if distances.dim() == 2:
        distances = distances.unsqueeze(0).expand(logits.size(0), -1, -1)
    return logits, target, distances


def evaluate_route(model, device, config):
    """生成路线并评估."""
    data_dir = Path("data/processed")
    poi_features = torch.from_numpy(np.load(data_dir / "poi_features.npy")).float()
    adjacency = torch.from_numpy(np.load(data_dir / "adjacency.npy")).float()
    dist_matrix = np.load(data_dir / "distance_matrix.npy")
    time_matrix = np.load(data_dir / "time_matrix.npy")
    pois = pd.read_csv(data_dir / "poi_metadata.csv", encoding="utf-8")
    ratings = pois["rating"].values
    categories = pois["category"].values

    poi_atypes = None
    if (data_dir / "poi_activity_types.npy").exists():
        poi_atypes = torch.from_numpy(np.load(data_dir / "poi_activity_types.npy")).long().to(device)

    cluster_id = None
    if (data_dir / "cluster_id.npy").exists():
        cluster_id = np.load(data_dir / "cluster_id.npy")

    model.eval()
    poi_feat_dev = poi_features.unsqueeze(0).to(device)
    adj_dev = adjacency.unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_output = model.encode(poi_feat_dev, adj_dev)

    # Pick diverse scenic starts (top-rated + spread across indices)
    scenic_idx = pois[pois["category"] == "景点"].sort_values("rating", ascending=False).index.tolist()
    starts = scenic_idx[:5] if len(scenic_idx) >= 5 else scenic_idx[:3]

    all_metrics = {"distance": [], "time": [], "satisfaction": [], "diversity": []}

    for start_id in starts:
        raw = generate_with_constraints(model, encoder_output, start_id, 5, 10, device, poi_atypes, cluster_id)
        filtered = filter_route(raw, 10)
        if len(filtered) < 2:
            continue
        types_np = poi_atypes.cpu().numpy() if poi_atypes is not None else None
        optimized = optimize_route_order(filtered, pois, dist_matrix, types_np)
        d = route_distance(optimized, dist_matrix)
        t = route_time(optimized, time_matrix)
        s = satisfaction_score(optimized, ratings)
        div = diversity_score(optimized, categories)
        for k, v in zip(all_metrics.keys(), [d, t, s, div]):
            all_metrics[k].append(v)

    if not all_metrics["distance"]:
        return None

    avg = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    avg["max_distance"] = float(dist_matrix.max() * 10)
    avg["max_time"] = float(time_matrix.max() * 10)
    weights = {"distance": 0.30, "time": 0.25, "satisfaction": 0.25, "diversity": 0.20}
    avg["composite"] = composite_score(avg, weights)

    return {"avg_distance_km": round(avg["distance"], 1), "avg_satisfaction": round(avg["satisfaction"], 2),
            "avg_diversity": round(avg["diversity"], 2), "composite_score": round(avg["composite"], 4)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    base_config = yaml.safe_load(open("configs/default.yaml", "r", encoding="utf-8"))
    ablation = yaml.safe_load(open("configs/ablation.yaml", "r", encoding="utf-8"))

    experiments = [
        ("full_model", "Full Model"),
        ("no_engram", "-Engram"),
        ("no_mhc", "-MHC"),
        ("no_engram_mhc", "-Engram-MHC"),
        ("baseline", "Baseline"),
        ("engram_k3", "K=3"),
        ("engram_k10", "K=10"),
    ]

    results = []
    for key, name in experiments:
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")

        if key not in ablation:
            print(f"  SKIP: no config")
            continue

        override = ablation[key]
        if isinstance(override, str):
            continue
        config = deep_merge(base_config, override)

        train_result, model = train_quick(config, name, device)
        eval_result = evaluate_route(model, device, config)

        if eval_result is None:
            print(f"  WARNING: evaluation failed")
            continue

        row = {**train_result, **eval_result}
        results.append(row)
        print(f"  => val_loss={row['best_val_loss']:.4f} score={row['composite_score']:.4f}")

    # Final table
    print(f"\n{'='*70}")
    print(f"  消融实验结果")
    print(f"{'='*70}")
    print(f"  {'实验':<18s} {'val_loss':>8s} {'距离':>6s} {'满意度':>6s} {'多样性':>6s} {'综合':>6s} {'Delta':>6s}")
    print(f"  {'─'*18} {'─'*8} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")

    full_score = results[0]["composite_score"] if results else 0
    for r in results:
        delta = r["composite_score"] - full_score
        print(f"  {r['name']:<18s} {r['best_val_loss']:>8.4f} {r['avg_distance_km']:>6.1f} "
              f"{r['avg_satisfaction']:>6.2f} {r['avg_diversity']:>6.2f} "
              f"{r['composite_score']:>6.4f} {delta:>+6.4f}")

    # Save
    with open("output/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv("output/ablation_results.csv", index=False, encoding="utf-8")
    print(f"\n  结果已保存: output/ablation_results.json")


if __name__ == "__main__":
    main()
