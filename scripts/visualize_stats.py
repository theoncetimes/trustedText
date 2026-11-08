import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load training statistics
# data directory is at project root, not in scripts/
data_dir = Path(__file__).resolve().parent.parent / "data"
stats_path = data_dir / ".cache" / "train_stats.json"

if not stats_path.exists():
    print(f"No training statistics found at {stats_path}")
    print("Please run 'python -m trustedtext.train' first to generate statistics.")
    exit(1)

with open(stats_path, "r") as f:
    stats = json.load(f)

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"Training Statistics - {stats['timestamp']}", fontsize=16, fontweight="bold")

# 1. Dataset Distribution
ax1 = plt.subplot(2, 3, 1)
labels_dist = ["Human", "AI"]
counts = [stats["dataset"]["human_count"], stats["dataset"]["ai_count"]]
colors = ["#4CAF50", "#FF5722"]
bars = ax1.bar(labels_dist, counts, color=colors, alpha=0.7, edgecolor="black")
ax1.set_title("Dataset Distribution", fontweight="bold")
ax1.set_ylabel("Number of Samples")
ax1.grid(axis="y", alpha=0.3)
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f"{count}\n({count/sum(counts)*100:.1f}%)",
             ha="center", va="bottom", fontweight="bold")

# 2. Train/Test Split
ax2 = plt.subplot(2, 3, 2)
split_labels = ["Training", "Test"]
split_counts = [stats["dataset"]["train_size"], stats["dataset"]["test_size"]]
colors_split = ["#2196F3", "#FFC107"]
bars = ax2.bar(split_labels, split_counts, color=colors_split, alpha=0.7, edgecolor="black")
ax2.set_title("Train/Test Split", fontweight="bold")
ax2.set_ylabel("Number of Samples")
ax2.grid(axis="y", alpha=0.3)
for bar, count in zip(bars, split_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f"{count}\n({count/stats['dataset']['total_samples']*100:.1f}%)",
             ha="center", va="bottom", fontweight="bold")

# 3. Cross-Validation Scores
ax3 = plt.subplot(2, 3, 3)
if stats["cross_validation"]["scores"]:
    cv_scores = stats["cross_validation"]["scores"]
    x = range(1, len(cv_scores) + 1)
    ax3.plot(x, cv_scores, marker="o", linewidth=2, markersize=8, color="#9C27B0")
    ax3.axhline(y=stats["cross_validation"]["mean_f1"], color="red", 
                linestyle="--", label=f"Mean: {stats['cross_validation']['mean_f1']:.3f}")
    ax3.fill_between(x, 
                     stats["cross_validation"]["mean_f1"] - stats["cross_validation"]["std_f1"],
                     stats["cross_validation"]["mean_f1"] + stats["cross_validation"]["std_f1"],
                     alpha=0.2, color="red")
    ax3.set_title("Cross-Validation F1-Macro Scores", fontweight="bold")
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("F1-Macro Score")
    ax3.set_xticks(x)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, 1])
else:
    ax3.text(0.5, 0.5, "No CV data available", ha="center", va="center", fontsize=12)
    ax3.set_title("Cross-Validation F1-Macro Scores", fontweight="bold")

# 4. Confusion Matrix
ax4 = plt.subplot(2, 3, 4)
if "confusion_matrix" in stats:
    cm = np.array(stats["confusion_matrix"])
    im = ax4.imshow(cm, cmap="Blues", alpha=0.7)
    ax4.set_title("Confusion Matrix", fontweight="bold")
    ax4.set_xlabel("Predicted Label")
    ax4.set_ylabel("True Label")
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Human", "AI"])
    ax4.set_yticklabels(["Human", "AI"])
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax4.text(j, i, cm[i, j],
                           ha="center", va="center", color="black", fontsize=20, fontweight="bold")
    
    # Add colorbar
    plt.colorbar(im, ax=ax4)
else:
    ax4.text(0.5, 0.5, "No confusion matrix", ha="center", va="center", fontsize=12)
    ax4.set_title("Confusion Matrix", fontweight="bold")

# 5. Precision, Recall, F1-Score Comparison
ax5 = plt.subplot(2, 3, 5)
if "test_metrics" in stats:
    metrics = ["Precision", "Recall", "F1-Score"]
    human_scores = [
        stats["test_metrics"]["precision_human"],
        stats["test_metrics"]["recall_human"],
        stats["test_metrics"]["f1_human"]
    ]
    ai_scores = [
        stats["test_metrics"]["precision_ai"],
        stats["test_metrics"]["recall_ai"],
        stats["test_metrics"]["f1_ai"]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, human_scores, width, label="Human", color="#4CAF50", alpha=0.7)
    bars2 = ax5.bar(x + width/2, ai_scores, width, label="AI", color="#FF5722", alpha=0.7)
    
    ax5.set_title("Test Metrics by Class", fontweight="bold")
    ax5.set_ylabel("Score")
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)
    ax5.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.3f}",
                    ha="center", va="bottom", fontsize=9)
else:
    ax5.text(0.5, 0.5, "No test metrics", ha="center", va="center", fontsize=12)
    ax5.set_title("Test Metrics by Class", fontweight="bold")

# 6. Test Accuracy
ax6 = plt.subplot(2, 3, 6)
if "test_metrics" in stats:
    accuracy = stats["test_metrics"]["accuracy"]
    ax6.text(0.5, 0.6, f"{accuracy:.1%}", ha="center", va="center", 
             fontsize=48, fontweight="bold", color="#2196F3")
    ax6.text(0.5, 0.3, "Test Accuracy", ha="center", va="center", 
             fontsize=16, fontweight="bold")
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.axis("off")
    
    # Add a circle background
    circle = plt.Circle((0.5, 0.6), 0.25, color="#2196F3", alpha=0.1)
    ax6.add_patch(circle)
else:
    ax6.text(0.5, 0.5, "No test metrics", ha="center", va="center", fontsize=12)
    ax6.set_title("Test Accuracy", fontweight="bold")
    ax6.axis("off")

plt.tight_layout()

# Save figure
output_path = data_dir / ".cache" / "train_stats.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Visualization saved to {output_path}")

# Display the plot
plt.show()
