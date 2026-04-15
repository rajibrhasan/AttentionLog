import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from utils import open_config, create_model
from detector.utils import process_attn, process_attn_prefill


def collect_heatmaps(model, instruction, data_list, prefill=False, desc="Processing"):
    heatmaps = []
    for data in tqdm(data_list, desc=desc):
        if prefill:
            attention_maps, input_range = model.prefill_inference(instruction, data)
            heatmap = process_attn_prefill(attention_maps, input_range)
        else:
            _, _, attention_maps, _, input_range, _ = model.inference(instruction, data)
            heatmap = process_attn(attention_maps[0], input_range, "normalize_sum")
        heatmaps.append(heatmap)
    return np.array(heatmaps)


def plot_divergence_heatmap(normal_heatmaps, anomaly_heatmaps, save_path, model_name, dataset):
    """Plot 1: Side-by-side heatmap of normal vs anomaly attention."""
    normal_mean = np.mean(normal_heatmaps, axis=0)
    anomaly_mean = np.mean(anomaly_heatmaps, axis=0)

    n_layers, n_heads = normal_mean.shape
    vmin = 0
    vmax = max(normal_mean.max(), anomaly_mean.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, data, title in [(axes[0], normal_mean, 'Normal Data'),
                             (axes[1], anomaly_mean, 'Anomaly Data')]:
        im = ax.imshow(data, aspect='auto', cmap='YlGnBu', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Heads', fontsize=13)
        ax.set_ylabel('Layers', fontsize=13)
        ax.set_xticks(range(0, n_heads, max(1, n_heads // 16)))
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 16)))
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    plt.savefig(f'{save_path}/divergence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}/divergence_heatmap.png")


def plot_difference_heatmap(normal_heatmaps, anomaly_heatmaps, save_path, model_name, dataset, important_heads=None):
    """Plot std-penalized divergence heatmap (matches head-selection score)
    with selected heads highlighted."""
    normal_mean = np.mean(normal_heatmaps, axis=0)
    anomaly_mean = np.mean(anomaly_heatmaps, axis=0)
    normal_std = np.std(normal_heatmaps, axis=0)
    anomaly_std = np.std(anomaly_heatmaps, axis=0)
    diff = (normal_mean - anomaly_mean) - (normal_std + anomaly_std)

    n_layers, n_heads = diff.shape
    vmax = max(abs(diff.min()), abs(diff.max()))

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(f'Divergence Score: (Normal − Anomaly) − (σ_N + σ_A)\n{model_name} on {dataset}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Heads', fontsize=13)
    ax.set_ylabel('Layers', fontsize=13)
    ax.set_xticks(range(0, n_heads, max(1, n_heads // 16)))
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 16)))
    ax.invert_yaxis()
    plt.colorbar(im, ax=ax, shrink=0.6, label='Std-penalized divergence')

    # Highlight selected heads with rectangles
    if important_heads and important_heads != "all":
        for layer, head in important_heads:
            rect = plt.Rectangle((head - 0.5, layer - 0.5), 1, 1,
                                 linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(f'{save_path}/difference_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}/difference_heatmap.png")


def plot_top_heads_distribution(normal_heatmaps, anomaly_heatmaps, save_path, model_name, dataset, top_k=6):
    """Plot 2: Score distribution for top divergent heads."""
    normal_mean = np.mean(normal_heatmaps, axis=0)
    anomaly_mean = np.mean(anomaly_heatmaps, axis=0)
    normal_std = np.std(normal_heatmaps, axis=0)
    anomaly_std = np.std(anomaly_heatmaps, axis=0)

    divergence = normal_mean - anomaly_mean
    diff_signal = divergence - (normal_std + anomaly_std)

    # Find top-k heads by divergence signal
    flat_indices = np.argsort(diff_signal.flatten())[-top_k:]
    top_heads = [np.unravel_index(idx, divergence.shape) for idx in flat_indices]
    top_heads = sorted(top_heads, key=lambda x: diff_signal[x[0], x[1]], reverse=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (layer, head) in enumerate(top_heads):
        ax = axes[i]
        normal_scores = normal_heatmaps[:, layer, head]
        anomaly_scores = anomaly_heatmaps[:, layer, head]

        ax.hist(normal_scores, bins=30, alpha=0.6, color='blue', label='Normal', density=True)
        ax.hist(anomaly_scores, bins=30, alpha=0.6, color='red', label='Anomaly', density=True)
        ax.axvline(np.mean(normal_scores), color='blue', linestyle='--', linewidth=1.5)
        ax.axvline(np.mean(anomaly_scores), color='red', linestyle='--', linewidth=1.5)
        ax.set_title(f'Layer {layer}, Head {head}\n'
                     f'μ_N−μ_A={divergence[layer, head]:.4f}, '
                     f'd={diff_signal[layer, head]:.4f}')
        ax.legend(fontsize=8)

    fig.suptitle(f'Top {top_k} Divergent Heads - {model_name} on {dataset}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_path}/top_heads_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}/top_heads_distribution.png")


def plot_selected_heads_scores(normal_heatmaps, anomaly_heatmaps, important_heads, save_path, model_name, dataset):
    """Plot 3: Per-sample focus score FS(L) = mean over selected heads."""
    normal_scores = np.array([np.mean([h[l, hd] for l, hd in important_heads]) for h in normal_heatmaps])
    anomaly_scores = np.array([np.mean([h[l, hd] for l, hd in important_heads]) for h in anomaly_heatmaps])

    # Methodology assumes normal logs yield HIGHER focus scores than anomalous
    # (instruction-following stays intact). Warn if the data shows the opposite.
    if np.mean(normal_scores) <= np.mean(anomaly_scores):
        print(f"WARNING: anomaly mean focus ({np.mean(anomaly_scores):.4f}) >= "
              f"normal mean focus ({np.mean(normal_scores):.4f}). "
              f"Detector will need flip=True for this (model, dataset).")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(normal_scores, bins=40, alpha=0.6, color='blue', label='Normal', density=True)
    axes[0].hist(anomaly_scores, bins=40, alpha=0.6, color='red', label='Anomaly', density=True)
    axes[0].axvline(np.mean(normal_scores), color='blue', linestyle='--', linewidth=2, label=f'Normal mean={np.mean(normal_scores):.4f}')
    axes[0].axvline(np.mean(anomaly_scores), color='red', linestyle='--', linewidth=2, label=f'Anomaly mean={np.mean(anomaly_scores):.4f}')
    axes[0].set_xlabel('Attention Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Score Distribution (Selected Heads)')
    axes[0].legend()

    # Scatter
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    colors = ['blue' if l == 0 else 'red' for l in all_labels]
    axes[1].scatter(range(len(all_scores)), all_scores, c=colors, alpha=0.4, s=10)
    axes[1].axhline((np.mean(normal_scores) + np.mean(anomaly_scores)) / 2, color='green', linestyle='--', label='Threshold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Attention Score')
    axes[1].set_title('Per-Sample Scores')
    axes[1].legend()

    fig.suptitle(f'Selected Heads ({len(important_heads)} heads) - {model_name} on {dataset}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_path}/selected_heads_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}/selected_heads_scores.png")


def main(args):
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(config_path=model_config_path)
    model_config["params"]["max_output_tokens"] = 1
    important_heads = model_config["params"]["important_heads"]
    model = create_model(config=model_config)

    if args.windowed:
        from data.windowed import load_windowed_sampled
        normal_samples, anomaly_samples = load_windowed_sampled(
            args.train_csv, n_normal=args.num_data, n_anomaly=args.num_data, seed=42
        )
    elif args.dataset == "bgl":
        from data.bgl import load_bgl_sampled
        normal_samples, anomaly_samples = load_bgl_sampled(
            args.bgl_path, n_normal=args.num_data, n_anomaly=args.num_data, seed=42
        )
    elif args.dataset == "spirit":
        from data.spirit import load_spirit_sampled
        normal_samples, anomaly_samples = load_spirit_sampled(
            args.spirit_path, n_normal=args.num_data, n_anomaly=args.num_data, seed=42
        )
    elif args.dataset == "thunderbird":
        from data.thunderbird import load_thunderbird_sampled
        normal_samples, anomaly_samples = load_thunderbird_sampled(
            args.thunderbird_path, n_normal=args.num_data, n_anomaly=args.num_data, seed=42
        )
    elif args.dataset == "hdfs":
        from data.hdfs import load_hdfs_sampled
        normal_samples, anomaly_samples = load_hdfs_sampled(
            args.hdfs_log_path, args.hdfs_label_path,
            n_normal=args.num_data, n_anomaly=args.num_data, seed=42
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    normal_data = [s['text'] for s in normal_samples]
    anomaly_data = [s['text'] for s in anomaly_samples]

    print(f"Collecting heatmaps: {len(normal_data)} normal, {len(anomaly_data)} anomaly")
    normal_heatmaps = collect_heatmaps(model, args.instruction, normal_data, prefill=args.prefill, desc="Normal")
    anomaly_heatmaps = collect_heatmaps(model, args.instruction, anomaly_data, prefill=args.prefill, desc="Anomaly")

    print("\nGenerating plots...")
    plot_divergence_heatmap(normal_heatmaps, anomaly_heatmaps, args.save_dir, args.model_name, args.dataset)
    plot_difference_heatmap(normal_heatmaps, anomaly_heatmaps, args.save_dir, args.model_name, args.dataset,
                            important_heads=important_heads if important_heads != "all" else None)
    plot_top_heads_distribution(normal_heatmaps, anomaly_heatmaps, args.save_dir, args.model_name, args.dataset)

    if important_heads != "all":
        plot_selected_heads_scores(normal_heatmaps, anomaly_heatmaps, important_heads, args.save_dir, args.model_name, args.dataset)

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize attention heads')
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--num_data', default=50, type=int, help='Samples per class')
    parser.add_argument('--instruction', default='Summarize this standard server log entry', type=str)
    parser.add_argument('--save_dir', default='./plots', type=str)
    parser.add_argument('--windowed', action='store_true')
    parser.add_argument('--train_csv', type=str, default=None)
    parser.add_argument('--bgl_path', default='./data/bgl/BGL.log', type=str)
    parser.add_argument('--spirit_path', default='./data/spirit/spirit2.log', type=str)
    parser.add_argument('--thunderbird_path', default='./data/thunderbird/Thunderbird.log', type=str)
    parser.add_argument('--hdfs_log_path', default='./data/hdfs/HDFS.log', type=str)
    parser.add_argument('--hdfs_label_path', default='./data/hdfs/anomaly_label.csv', type=str)
    parser.add_argument('--prefill', action='store_true',
                        help='Use prefill-only attention (last data token → data tokens)')
    args = parser.parse_args()

    main(args)
