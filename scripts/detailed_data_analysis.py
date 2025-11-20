"""
Comprehensive data analysis for training data and model performance.
This script provides deeper insights into:
- Data distribution and bias
- Class imbalance
- Control combination patterns
- Sample visualization
- Per-class metrics

Run after training: python scripts/detailed_data_analysis.py
"""

import numpy as np
import pickle
import lzma
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def load_data():
    """Load training data from record files"""
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    all_images = []
    all_labels = []
    
    records = sorted(Path("./").glob("record_*.npz"))
    
    for record in records:
        with lzma.open(record, "rb") as file:
            import sys
            
            # Temporarily remove rallyrobopilot to avoid Ursina import
            rallyrobopilot_modules = {k: v for k, v in sys.modules.items() 
                                     if k.startswith('rallyrobopilot')}
            for module_name in rallyrobopilot_modules:
                del sys.modules[module_name]
            
            try:
                class SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if name == 'SensingSnapshot':
                            class SensingSnapshot:
                                def __init__(self):
                                    self.image = None
                                    self.current_controls = None
                            return SensingSnapshot
                        return super().find_class(module, name)
                
                data = SafeUnpickler(file).load()
                
                valid_count = 0
                for snapshot in data:
                    if hasattr(snapshot, 'image') and hasattr(snapshot, 'current_controls'):
                        if snapshot.image is not None and snapshot.current_controls is not None:
                            all_images.append(snapshot.image)
                            all_labels.append(snapshot.current_controls)
                            valid_count += 1
                
                print(f"  {record.name}: {valid_count} valid snapshots")
                
            finally:
                sys.modules.update(rallyrobopilot_modules)
    
    X = np.array(all_images)
    y = np.array(all_labels, dtype=int)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Image shape: {X.shape[1:]}")
    
    return X, y


def analyze_class_distribution(y, save_dir):
    """Analyze distribution of each control and combinations"""
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*70)
    
    control_names = ['Forward', 'Back', 'Left', 'Right']
    
    # Individual control distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Individual Control Distribution', fontsize=16)
    
    for i, (ax, name) in enumerate(zip(axes.flat, control_names)):
        counts = np.bincount(y[:, i])
        percentages = counts / len(y) * 100
        
        bars = ax.bar(['Off', 'On'], counts, color=['#e74c3c', '#2ecc71'])
        ax.set_ylabel('Count')
        ax.set_title(f'{name}')
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%\n({int(height):,})',
                   ha='center', va='bottom')
        
        print(f"{name:8s}: Off={counts[0]:5d} ({percentages[0]:5.1f}%), "
              f"On={counts[1]:5d} ({percentages[1]:5.1f}%)")
    
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png', dpi=150)
    print(f"\n✓ Saved: {save_dir / 'class_distribution.png'}")
    
    # Control combinations
    print("\n" + "-"*70)
    print("CONTROL COMBINATION ANALYSIS")
    print("-"*70)
    
    # Convert to tuples for counting
    combinations = [tuple(row) for row in y]
    combination_counts = Counter(combinations)
    
    # Get top 15 most common combinations
    top_combinations = combination_counts.most_common(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    combo_labels = []
    combo_counts = []
    
    for combo, count in top_combinations:
        label_parts = []
        if combo[0]: label_parts.append('Fwd')
        if combo[1]: label_parts.append('Back')
        if combo[2]: label_parts.append('Left')
        if combo[3]: label_parts.append('Right')
        
        label = '+'.join(label_parts) if label_parts else 'None'
        combo_labels.append(label)
        combo_counts.append(count)
        
        pct = count / len(y) * 100
        print(f"  {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    bars = ax.barh(combo_labels, combo_counts, color='steelblue')
    ax.set_xlabel('Count')
    ax.set_title('Top 15 Control Combinations')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
               f' {int(width):,}',
               ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'control_combinations.png', dpi=150)
    print(f"\n✓ Saved: {save_dir / 'control_combinations.png'}")


def analyze_steering_bias(y, save_dir):
    """Analyze left vs right steering bias"""
    print("\n" + "="*70)
    print("STEERING BIAS ANALYSIS")
    print("="*70)
    
    left_count = np.sum(y[:, 2])
    right_count = np.sum(y[:, 3])
    
    total_steering = left_count + right_count
    
    if total_steering == 0:
        print("⚠️  No steering data found!")
        return
    
    left_pct = left_count / total_steering * 100
    right_pct = right_count / total_steering * 100
    
    print(f"\nSteering distribution (when steering):")
    print(f"  Left:  {left_count:5d} ({left_pct:5.1f}%)")
    print(f"  Right: {right_count:5d} ({right_pct:5.1f}%)")
    
    # Bias score (-100 to +100, 0 = balanced)
    bias_score = (right_count - left_count) / total_steering * 100
    
    print(f"\nBias score: {bias_score:+.1f}")
    if abs(bias_score) < 10:
        print("  ✓ Well balanced!")
    elif abs(bias_score) < 30:
        print(f"  ⚠️  Slight bias toward {'right' if bias_score > 0 else 'left'}")
    else:
        print(f"  ❌ Strong bias toward {'right' if bias_score > 0 else 'left'}")
        print("     Consider data augmentation (mirroring)")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    wedges, texts, autotexts = ax.pie(
        [left_count, right_count],
        labels=['Left', 'Right'],
        autopct='%1.1f%%',
        colors=['#3498db', '#e74c3c'],
        startangle=90
    )
    
    # Make percentage text bold and larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
    
    ax.set_title(f'Steering Balance\nBias Score: {bias_score:+.1f}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'steering_bias.png', dpi=150)
    print(f"\n✓ Saved: {save_dir / 'steering_bias.png'}")


def analyze_image_samples(X, y, save_dir):
    """Visualize sample images for each class"""
    print("\n" + "="*70)
    print("SAMPLE IMAGE VISUALIZATION")
    print("="*70)
    
    control_names = ['Forward', 'Back', 'Left', 'Right']
    
    # For each control, show 5 samples where it's ON
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle('Sample Images per Control (when active)', fontsize=16)
    
    for i, name in enumerate(control_names):
        # Find indices where this control is active
        active_indices = np.where(y[:, i] == 1)[0]
        
        if len(active_indices) == 0:
            print(f"  {name}: No active samples found")
            continue
        
        # Sample 5 random images
        sample_indices = np.random.choice(active_indices, size=min(5, len(active_indices)), replace=False)
        
        for j, idx in enumerate(sample_indices):
            ax = axes[i, j]
            img = X[idx]
            ax.imshow(img)
            ax.axis('off')
            
            if j == 0:
                ax.set_ylabel(name, fontsize=12, weight='bold')
        
        print(f"  {name}: {len(active_indices)} active samples")
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_images.png', dpi=150)
    print(f"\n✓ Saved: {save_dir / 'sample_images.png'}")


def analyze_image_statistics(X, save_dir):
    """Analyze image brightness, contrast, and color distribution"""
    print("\n" + "="*70)
    print("IMAGE STATISTICS ANALYSIS")
    print("="*70)
    
    # Sample 1000 images for statistics
    sample_size = min(1000, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_indices]
    
    # Calculate statistics
    mean_brightness = np.mean(X_sample, axis=(1, 2, 3))
    std_brightness = np.std(X_sample, axis=(1, 2, 3))
    
    print(f"\nBrightness (0-255 scale):")
    print(f"  Mean: {np.mean(mean_brightness):.1f} ± {np.std(mean_brightness):.1f}")
    print(f"  Range: [{np.min(mean_brightness):.1f}, {np.max(mean_brightness):.1f}]")
    
    print(f"\nContrast (std deviation):")
    print(f"  Mean: {np.mean(std_brightness):.1f} ± {np.std(std_brightness):.1f}")
    
    # Color channel distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Brightness distribution
    axes[0, 0].hist(mean_brightness, bins=50, color='gray', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Mean Brightness')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Brightness Distribution')
    axes[0, 0].grid(alpha=0.3)
    
    # Contrast distribution
    axes[0, 1].hist(std_brightness, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Contrast (Std Dev)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Contrast Distribution')
    axes[0, 1].grid(alpha=0.3)
    
    # RGB channel means
    r_means = np.mean(X_sample[:, :, :, 0], axis=(1, 2))
    g_means = np.mean(X_sample[:, :, :, 1], axis=(1, 2))
    b_means = np.mean(X_sample[:, :, :, 2], axis=(1, 2))
    
    axes[1, 0].hist(r_means, bins=50, color='red', alpha=0.5, label='Red', edgecolor='black')
    axes[1, 0].hist(g_means, bins=50, color='green', alpha=0.5, label='Green', edgecolor='black')
    axes[1, 0].hist(b_means, bins=50, color='blue', alpha=0.5, label='Blue', edgecolor='black')
    axes[1, 0].set_xlabel('Mean Channel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('RGB Channel Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Show sample mean image
    mean_image = np.mean(X_sample, axis=0).astype(np.uint8)
    axes[1, 1].imshow(mean_image)
    axes[1, 1].set_title('Mean Image (averaged over samples)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'image_statistics.png', dpi=150)
    print(f"\n✓ Saved: {save_dir / 'image_statistics.png'}")


def generate_summary_report(X, y, save_dir):
    """Generate a text summary report"""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    report_path = save_dir / 'data_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING DATA ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Dataset size
        f.write(f"Dataset Size: {len(X):,} samples\n")
        f.write(f"Image Shape: {X.shape[1:]}\n")
        f.write(f"Total Pixels per Image: {np.prod(X.shape[1:]):,}\n\n")
        
        # Control distribution
        f.write("-"*70 + "\n")
        f.write("CONTROL DISTRIBUTION\n")
        f.write("-"*70 + "\n")
        
        control_names = ['Forward', 'Back', 'Left', 'Right']
        for i, name in enumerate(control_names):
            count = np.sum(y[:, i])
            pct = count / len(y) * 100
            f.write(f"{name:8s}: {count:6d} / {len(y):6d} ({pct:5.1f}%)\n")
        
        # Steering bias
        f.write("\n" + "-"*70 + "\n")
        f.write("STEERING BIAS\n")
        f.write("-"*70 + "\n")
        
        left_count = np.sum(y[:, 2])
        right_count = np.sum(y[:, 3])
        total_steering = left_count + right_count
        
        if total_steering > 0:
            bias_score = (right_count - left_count) / total_steering * 100
            f.write(f"Left:  {left_count:6d} ({left_count/total_steering*100:5.1f}%)\n")
            f.write(f"Right: {right_count:6d} ({right_count/total_steering*100:5.1f}%)\n")
            f.write(f"Bias Score: {bias_score:+.2f}\n")
            
            if abs(bias_score) < 10:
                f.write("Status: ✓ Well balanced\n")
            elif abs(bias_score) < 30:
                f.write(f"Status: ⚠️  Slight bias toward {'right' if bias_score > 0 else 'left'}\n")
            else:
                f.write(f"Status: ❌ Strong bias toward {'right' if bias_score > 0 else 'left'}\n")
        
        # Image statistics
        f.write("\n" + "-"*70 + "\n")
        f.write("IMAGE STATISTICS\n")
        f.write("-"*70 + "\n")
        
        sample_size = min(1000, len(X))
        X_sample = X[np.random.choice(len(X), sample_size, replace=False)]
        
        mean_brightness = np.mean(X_sample)
        std_brightness = np.std(X_sample)
        
        f.write(f"Mean Brightness: {mean_brightness:.2f}\n")
        f.write(f"Std Brightness: {std_brightness:.2f}\n")
        f.write(f"Min Pixel Value: {np.min(X_sample)}\n")
        f.write(f"Max Pixel Value: {np.max(X_sample)}\n")
        
        # Recommendations
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        recommendations = []
        
        # Check data size
        if len(X) < 1000:
            recommendations.append("⚠️  Dataset is very small (<1000). Collect more data (aim for 5000+).")
        elif len(X) < 5000:
            recommendations.append("⚠️  Dataset is small (<5000). More data would improve training.")
        else:
            recommendations.append("✓ Dataset size is good.")
        
        # Check forward motion
        forward_pct = np.sum(y[:, 0]) / len(y) * 100
        if forward_pct < 30:
            recommendations.append(f"⚠️  Low forward motion ({forward_pct:.1f}%). Car may not learn to drive.")
        else:
            recommendations.append("✓ Forward motion looks good.")
        
        # Check steering
        left_pct = np.sum(y[:, 2]) / len(y) * 100
        right_pct = np.sum(y[:, 3]) / len(y) * 100
        if left_pct < 5 and right_pct < 5:
            recommendations.append("⚠️  Almost no steering data. Include more turns.")
        else:
            recommendations.append("✓ Steering data present.")
        
        # Check steering bias
        if total_steering > 0 and abs(bias_score) > 30:
            recommendations.append("⚠️  Strong steering bias detected. Use data augmentation (mirroring).")
        
        for rec in recommendations:
            f.write(f"{rec}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved: {report_path}")


def main():
    """Run complete data analysis"""
    print("\n" + "="*70)
    print("COMPREHENSIVE TRAINING DATA ANALYSIS")
    print("="*70)
    
    # Create output directory
    output_dir = Path("data_analysis")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load data
    X, y = load_data()
    
    if len(X) == 0:
        print("\n❌ No data found! Collect training data first.")
        return
    
    # Run analyses
    analyze_class_distribution(y, output_dir)
    analyze_steering_bias(y, output_dir)
    analyze_image_samples(X, y, output_dir)
    analyze_image_statistics(X, output_dir)
    generate_summary_report(X, y, output_dir)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - class_distribution.png")
    print("  - control_combinations.png")
    print("  - steering_bias.png")
    print("  - sample_images.png")
    print("  - image_statistics.png")
    print("  - data_analysis_report.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
