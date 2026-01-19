"""
MLOps Metrics Utilities
Professional metric calculation functions with validation and flexibility
"""

def calculate_metrics(*metric_names, predictions=None, actuals=None, **options):
    """
    Calculate multiple ML metrics flexibly and safely

    Why this design?
    - *metric_names: Can calculate any number of metrics in one call
    - **options: Configurable (rounding, validation levels, etc.)
    - Input validation: Prevents pipeline failures
    - Single function: Consistent calculations across team

    Args:
        *metric_names: Names of metrics to calculate ('accuracy', 'precision', etc.)
        predictions: Model predictions
        actuals: True labels  
        **options: Configuration options (round=2, validate=True, etc.)

    Returns:
        dict: Calculated metrics

    Example:
        metrics = calculate_metrics('accuracy', 'precision', 
                                  predictions=[1,0,1], actuals=[1,1,1], 
                                  round=3)
    """

    # Input validation - crucial for MLOps pipelines!
    if predictions is None or actuals is None:
        raise ValueError("Both predictions and actuals are required")

    if len(predictions) != len(actuals):
        raise ValueError(f"Length mismatch: predictions({len(predictions)}) vs actuals({len(actuals)})")

    if len(predictions) == 0:
        raise ValueError("Cannot calculate metrics on empty data")

    # Configuration with defaults
    round_digits = options.get('round', None)
    verbose = options.get('verbose', False)

    if verbose:
        print(f"Calculating {len(metric_names)} metrics on {len(predictions)} samples")

    results = {}

    # Calculate each requested metric
    for metric in metric_names:
        if metric == 'accuracy':
            correct = sum(p == a for p, a in zip(predictions, actuals))
            results[metric] = correct / len(predictions)

        elif metric == 'precision':
            tp = sum(p == 1 and a == 1 for p, a in zip(predictions, actuals))
            fp = sum(p == 1 and a == 0 for p, a in zip(predictions, actuals))
            results[metric] = tp / (tp + fp) if (tp + fp) > 0 else 0

        elif metric == 'recall':
            tp = sum(p == 1 and a == 1 for p, a in zip(predictions, actuals))
            fn = sum(p == 0 and a == 1 for p, a in zip(predictions, actuals))
            results[metric] = tp / (tp + fn) if (tp + fn) > 0 else 0

        elif metric == 'confusion_matrix':
            tp = sum(p == 1 and a == 1 for p, a in zip(predictions, actuals))
            tn = sum(p == 0 and a == 0 for p, a in zip(predictions, actuals))
            fp = sum(p == 1 and a == 0 for p, a in zip(predictions, actuals))
            fn = sum(p == 0 and a == 1 for p, a in zip(predictions, actuals))
            results[metric] = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

        else:
            print(f"Warning: Unknown metric '{metric}' - skipping")

    # Apply rounding if requested
    if round_digits is not None:
        for key, value in results.items():
            if isinstance(value, float):
                results[key] = round(value, round_digits)

    return results


def transform_predictions(*transformations, predictions=None, **options):
    """
    Apply transformations to predictions using lambda functions

    Why this pattern?
    - Flexible: Can apply any sequence of transformations
    - Reusable: Same function for different transformation pipelines
    - Functional: Uses lambda for quick operations
    """

    if predictions is None:
        raise ValueError("Predictions are required")

    result = predictions.copy() if hasattr(predictions, 'copy') else list(predictions)

    transform_map = {
        'round': lambda x: round(x, options.get('decimals', 2)),
        'threshold': lambda x: 1 if x > options.get('threshold', 0.5) else 0,
        'normalize': lambda x: x / options.get('max_value', 1.0),
        'clip': lambda x: max(options.get('min_val', 0), min(x, options.get('max_val', 1)))
    }

    for transform in transformations:
        if transform in transform_map:
            result = [transform_map[transform](x) for x in result]
        else:
            print(f"Warning: Unknown transformation '{transform}'")

    return result

print("✅ MLOps metrics utilities created!")
