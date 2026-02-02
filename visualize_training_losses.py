import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob


def find_event_files(log_dir):
    """Find all event files recursively in the log directory"""
    # Try multiple search patterns
    patterns = [
        os.path.join(log_dir, 'events.out.tfevents.*'),
        os.path.join(log_dir, '**', 'events.out.tfevents.*'),
    ]
    
    event_files = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        event_files.extend(found)
    
    # Remove duplicates
    event_files = list(set(event_files))
    
    return event_files


def extract_scalars_from_tfevents(log_dir, tags=['train_loss_epoch', 'val_loss']):
    """Extract scalar data from TensorBoard event files"""
    
    # Find all event files
    event_files = find_event_files(log_dir)
    
    if not event_files:
        print(f"❌ No event files found in {log_dir}")
        print("\nSearching in parent directory...")
        parent_dir = os.path.dirname(log_dir.rstrip('/'))
        if parent_dir and parent_dir != log_dir:
            event_files = find_event_files(parent_dir)
            if event_files:
                print(f"Found {len(event_files)} event file(s) in parent directory")
        
        if not event_files:
            # Try to suggest alternatives
            print("\nLooking for log directories...")
            possible_dirs = []
            search_dirs = ['.', './log_seizure_detection', './logs', './log']
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    for item in os.listdir(search_dir):
                        item_path = os.path.join(search_dir, item)
                        if os.path.isdir(item_path) and ('log' in item.lower() or 'CHB' in item):
                            possible_dirs.append(item_path)
            
            if possible_dirs:
                print("\nPossible log directories found:")
                for i, d in enumerate(possible_dirs[:10], 1):
                    print(f"  {i}. {d}")
                print("\nTry running with one of these directories using --log_dir")
            
            return None
    
    print(f"✓ Found {len(event_files)} event file(s)")
    
    # Read from ALL event files and merge data
    all_data = {}
    all_available_tags = set()
    
    for event_file in sorted(event_files):
        print(f"Reading from: {event_file}")
        
        # Load the event file
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        # Get available tags
        available_tags = ea.Tags()['scalars']
        all_available_tags.update(available_tags)
        
        # Extract data for each tag
        for tag in available_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            if tag not in all_data:
                all_data[tag] = {'steps': [], 'values': []}
            
            all_data[tag]['steps'].extend(steps)
            all_data[tag]['values'].extend(values)
    
    print(f"\nAvailable tags across all files ({len(all_available_tags)}):")
    for tag in sorted(all_available_tags):
        count = len(all_data[tag]['values']) if tag in all_data else 0
        print(f"  - {tag} ({count} values)")
    
    # Extract requested tags
    data = {}
    for tag in tags:
        if tag in all_data and len(all_data[tag]['values']) > 0:
            # Sort by steps to ensure correct order
            combined = list(zip(all_data[tag]['steps'], all_data[tag]['values']))
            combined.sort(key=lambda x: x[0])
            steps, values = zip(*combined)
            
            data[tag] = {'steps': list(steps), 'values': list(values)}
            print(f"\n✓ Extracted {len(values)} values for '{tag}'")
        else:
            print(f"\n⚠️  Warning: Tag '{tag}' not found in logs")
    
    # If no specific tags found, return all data
    if not data and all_data:
        print("\n⚠️  Requested tags not found, returning all available data")
        return all_data
    
    return data if data else all_data


def plot_losses(data, save_path='training_losses.png'):
    """Plot training and validation losses"""
    
    if not data:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Both losses on same plot
    ax1 = axes[0]
    if 'train_loss_epoch' in data:
        # Use actual epoch numbers if available
        steps = data['train_loss_epoch']['steps']
        # For epoch-level metrics, steps often represent epochs directly
        # If there's an 'epoch' tag, we could use that, otherwise assume steps are epochs
        epochs = list(range(len(data['train_loss_epoch']['values'])))
        ax1.plot(epochs, 
                data['train_loss_epoch']['values'], 
                label='Training Loss', marker='o', markersize=3, linewidth=2)
    if 'val_loss' in data:
        epochs = list(range(len(data['val_loss']['values'])))
        ax1.plot(epochs, 
                data['val_loss']['values'], 
                label='Validation Loss', marker='s', markersize=3, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (BCE)', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss difference
    ax2 = axes[1]
    if 'train_loss_epoch' in data and 'val_loss' in data:
        train_losses = data['train_loss_epoch']['values']
        val_losses = data['val_loss']['values']
        
        # Calculate difference
        min_len = min(len(train_losses), len(val_losses))
        diff = [val_losses[i] - train_losses[i] for i in range(min_len)]
        epochs = list(range(min_len))
        
        ax2.plot(epochs, diff, 
                label='Val Loss - Train Loss', 
                marker='d', markersize=3, linewidth=2, color='purple')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss Difference', fontsize=12)
        ax2.set_title('Validation - Training Loss Gap', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {save_path}")
    plt.close()


def plot_all_metrics(all_data, save_dir='./plots'):
    """Plot all available metrics from already extracted data"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    if not all_data:
        print("No data to plot")
        return
    
    # Separate metrics by type
    loss_tags = [t for t in all_data.keys() if 'loss' in t.lower()]
    metric_tags = [t for t in all_data.keys() if t not in loss_tags and t != 'epoch']
    
    # Plot losses (epoch-level only)
    if loss_tags:
        # Filter to only epoch-level metrics (exclude step-level metrics)
        epoch_loss_tags = [t for t in loss_tags if 'epoch' in t.lower() or 'val' in t.lower()]

        if epoch_loss_tags:
            fig, ax = plt.subplots(figsize=(10, 6))
            for tag in epoch_loss_tags:
                values = all_data[tag]['values']
                # Use sequential epoch numbers
                epochs = list(range(len(values)))
                ax.plot(epochs, values, label=tag, marker='o', markersize=3, linewidth=2)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training and Validation Loss (by Epoch)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            loss_path = os.path.join(save_dir, 'all_losses.png')
            plt.savefig(loss_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {loss_path}")
            plt.close()
        else:
            print("⚠ No epoch-level loss metrics found to plot")
    
    # Plot other metrics
    if metric_tags:
        n_metrics = len(metric_tags)
        n_cols = 2
        n_rows = (n_metrics + 1) // 2
        
        # Handle single metric case
        if n_metrics == 1:
            fig, ax = plt.subplots(figsize=(7, 5))
            axes = [ax]
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten()
        
        for idx, tag in enumerate(metric_tags):
            values = all_data[tag]['values']
            # Use sequential epoch numbers for validation/test metrics
            epochs = list(range(len(values)))
            
            axes[idx].plot(epochs, values, marker='o', markersize=3, linewidth=2)
            axes[idx].set_xlabel('Epoch', fontsize=10)
            axes[idx].set_ylabel(tag, fontsize=10)
            axes[idx].set_title(tag, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(metric_tags), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        metrics_path = os.path.join(save_dir, 'all_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {metrics_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize training losses from TensorBoard logs")
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default=None,
        help="Path to TensorBoard log directory (e.g., ./log_seizure_detection/CHB_MIT_8patients-CNNTransformer-...)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically find the most recent log directory"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("TensorBoard Loss Visualization")
    print("="*80)
    
    # Auto-detect log directory if not provided
    if args.log_dir is None or args.auto:
        print("\nSearching for log directories...")
        log_dirs = glob.glob('./log_seizure_detection/*')
        log_dirs.extend(glob.glob('./log/*'))
        log_dirs = [d for d in log_dirs if os.path.isdir(d)]
        
        if log_dirs:
            # Use most recent
            args.log_dir = max(log_dirs, key=os.path.getmtime)
            print(f"✓ Found log directory: {args.log_dir}")
        else:
            print("❌ No log directories found. Please specify with --log_dir")
            exit(1)
    
    print(f"\nLog directory: {args.log_dir}")
    print(f"Save directory: {args.save_dir}")
    print()
    
    # Extract and plot losses - try common tag names
    data = extract_scalars_from_tfevents(
        args.log_dir, 
        tags=['train_loss_epoch', 'train_loss', 'val_loss', 'train_loss_step']
    )
    
    if data:
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Plot if we have any loss data
        has_loss_data = any('loss' in tag.lower() for tag in data.keys())
        if has_loss_data:
            plot_losses(data, save_path=os.path.join(args.save_dir, 'training_validation_losses.png'))
        
        # Pass the already extracted data instead of re-reading
        plot_all_metrics(data, save_dir=args.save_dir)
        print("\n" + "="*80)
        print("✓ Visualization complete!")
        print(f"Plots saved to: {args.save_dir}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ No data to visualize")
        print("="*80)
