import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

def plot_confusion_matrix(confusion_matrix, classes):
    """Plot a confusion matrix with seaborn when available."""
    if confusion_matrix is None:
        print("Confusion Matrix is not supported for true Multi-Label evaluation.")
        return
        
    plt.figure(figsize=(10, 8))

    if sns is not None:
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
        )
        return

    plt.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.yticks(range(len(classes)), classes)

    max_value = max(max(row) for row in confusion_matrix) if len(confusion_matrix) else 0
    threshold = max_value / 2
    for row_index, row in enumerate(confusion_matrix):
        for col_index, value in enumerate(row):
            plt.text(
                col_index,
                row_index,
                f'{value:d}',
                ha='center',
                va='center',
                color='white' if value > threshold else 'black',
            )

def display_academic_metrics(results_dict, title="Model Evaluation"):
    """Formats and displays the relevant evaluation metrics for report"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    report = results_dict['classification_report']
    task_mode = results_dict.get('task_mode', 'single_label')
    
    # Print high-level metrics
    acc_key = 'accuracy' if 'accuracy' in report else 'micro avg'
    acc_val = report.get('accuracy', report.get('micro avg', {}).get('precision', 0.0))
    
    print(f"Task Mode:        {task_mode.replace('_', ' ').title()}")
    print(f"Overall Accuracy: {acc_val:.4f}")
    print(f"Macro F1-Score:   {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1-Score:{report['weighted avg']['f1-score']:.4f}\n")
    
    # Print Per-Class Metrics
    df_report = pd.DataFrame(report).transpose()
    # Drop the average rows to just show classes
    drop_keys = ['accuracy', 'macro avg', 'weighted avg', 'micro avg', 'samples avg']
    df_class_only = df_report.drop(columns=[], index=[k for k in drop_keys if k in df_report.index])
    
    print("--- Per-Class Performance ---")
    print(df_class_only[['precision', 'recall', 'f1-score']].round(4))
    
    # Plot Confusion Matrix (Only if Single Label)
    if task_mode == "single_label" and results_dict['confusion_matrix'] is not None:
        plot_confusion_matrix(results_dict['confusion_matrix'], results_dict['classes'])
        plt.ylabel('Actual Instrument')
        plt.xlabel('Predicted Instrument')
        plt.title(f'{title} - Confusion Matrix')
        plt.tight_layout()
        plt.show()