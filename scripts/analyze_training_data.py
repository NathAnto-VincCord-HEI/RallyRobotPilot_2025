"""
Quick script to analyze training data quality.
Run: python scripts/analyze_training_data.py
"""

import pickle
import lzma
from pathlib import Path
import numpy as np

def analyze_data():
    print("="*70)
    print("TRAINING DATA ANALYSIS")
    print("="*70)
    
    # Load all record files
    records = sorted(Path("./").glob("record_*.npz"))
    
    if len(records) == 0:
        print("\n❌ No record_*.npz files found!")
        print("Collect data first: python scripts/main.py (press R to record)")
        return
    
    all_snapshots = []
    print(f"\nFound {len(records)} record files:")
    
    for record in records:
        try:
            with lzma.open(record, "rb") as file:
                data = pickle.load(file)
                print(f"  {record.name}: {len(data)} snapshots")
                all_snapshots.extend(data)
        except Exception as e:
            print(f"  {record.name}: ERROR - {e}")
    
    if len(all_snapshots) == 0:
        print("\n❌ No snapshots found!")
        return
    
    print(f"\nTotal snapshots: {len(all_snapshots)}")
    
    # Filter valid snapshots
    valid = [s for s in all_snapshots if s.image is not None and s.current_controls is not None]
    print(f"Valid snapshots (with image & controls): {len(valid)}")
    
    if len(valid) == 0:
        print("\n❌ No valid snapshots!")
        return
    
    # Analyze controls distribution
    print("\n" + "="*70)
    print("CONTROL DISTRIBUTION")
    print("="*70)
    
    controls = np.array([s.current_controls for s in valid])
    control_names = ['Forward (W)', 'Backward (S)', 'Left (A)', 'Right (D)']
    
    print("\nHow often each control was pressed:")
    for i, name in enumerate(control_names):
        count = np.sum(controls[:, i])
        percentage = (count / len(valid)) * 100
        bar_length = int(percentage / 2)
        bar = "█" * bar_length
        print(f"  {name:15s}: {count:5d} ({percentage:5.1f}%) {bar}")
    
    # Check for issues
    print("\n" + "="*70)
    print("QUALITY CHECKS")
    print("="*70)
    
    forward_pct = (np.sum(controls[:, 0]) / len(valid)) * 100
    backward_pct = (np.sum(controls[:, 1]) / len(valid)) * 100
    left_pct = (np.sum(controls[:, 2]) / len(valid)) * 100
    right_pct = (np.sum(controls[:, 3]) / len(valid)) * 100
    
    issues = []
    
    if forward_pct < 30:
        issues.append(f"⚠️  Low forward motion ({forward_pct:.1f}%) - Car won't learn to move")
    else:
        print(f"✓ Forward motion: {forward_pct:.1f}% (good)")
    
    if forward_pct < 10:
        issues.append(f"❌ CRITICAL: Almost no forward driving! Model can't learn to move")
    
    if left_pct < 5 and right_pct < 5:
        issues.append(f"⚠️  Almost no steering ({left_pct:.1f}% left, {right_pct:.1f}% right)")
    else:
        print(f"✓ Steering: {left_pct:.1f}% left, {right_pct:.1f}% right")
    
    if backward_pct > 50:
        issues.append(f"⚠️  Too much backward ({backward_pct:.1f}%) - mostly reversing?")
    
    if len(valid) < 1000:
        issues.append(f"⚠️  Only {len(valid)} snapshots - recommend at least 5000 for good training")
    else:
        print(f"✓ Data volume: {len(valid)} snapshots (good)")
    
    # Check image variation
    images = np.array([s.image for s in valid[:100]])  # Sample first 100
    image_std = np.std(images)
    if image_std < 10:
        issues.append(f"⚠️  Low image variation - all scenes look similar?")
    else:
        print(f"✓ Image variation: {image_std:.1f} (good)")
    
    # Summary
    print("\n" + "="*70)
    if len(issues) == 0:
        print("✓ DATA LOOKS GOOD! Ready to train.")
        print("\nNext step:")
        print("  python scripts/train_improved_cnn.py --model improved --epochs 50")
    else:
        print("❌ DATA QUALITY ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRecommendation:")
        print("  1. Collect more/better data (python scripts/main.py, press R)")
        print("  2. Drive smoothly, include forward motion and turns")
        print("  3. Aim for 5-10 minutes of diverse driving")
    
    print("="*70)

if __name__ == "__main__":
    analyze_data()
