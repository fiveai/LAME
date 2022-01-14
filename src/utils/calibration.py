import numpy as np


def expected_calibration_error(gt, y_probs, num_bins=20):
    N = y_probs.shape[0]
    gt = gt.squeeze()
    # assert len(gt.shape) == 1, (gt.shape, gt)
    # assert len(y_probs.shape) == 2 or len(y_probs.shape) == 3, y_probs.shape
    if len(y_probs.shape) == 3:
        print("Ensemble detected.")
        y_probs = y_probs.mean(axis=1)
    logits_y = np.argmax(y_probs, axis=-1)
    correct = (logits_y == gt).astype(np.float32)
    prob_y = np.max(y_probs, axis=-1)

    bins = np.linspace(start=0, stop=1.0, num=num_bins + 1)  # First bin is actually bin 1
    binned_probas = np.digitize(prob_y, bins=bins, right=True) - 1

    o = 0
    histo = np.zeros(num_bins)
    density = [0] * num_bins
    for i, b in enumerate(range(num_bins)):
        mask = (binned_probas == b)  
        density[i] = mask.sum() / N
        if np.any(mask):
            histo[i] = np.mean(correct[mask])
            o += np.abs(np.sum(correct[mask] - prob_y[mask]))
    return bins, histo, density, o / N
