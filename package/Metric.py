import numpy as np

def mea_acc(ground_truth, inferred):
    
    # Ensure both matrices are numpy arrays
    ground_truth = np.array(ground_truth)
    inferred = np.array(inferred)
    
    # Check if both matrices are of the same shape
    if ground_truth.shape != inferred.shape:
        raise ValueError("The ground truth and inferred matrices must have the same shape.")
    
    # Convert matrices to binary (presence/absence of edges)
    ground_truth_binary = (ground_truth != 0).astype(int)
    inferred_binary = (inferred != 0).astype(int)
    
    # True Positive (TP): Edge exists in both ground truth and inferred matrices
    TP = np.sum((ground_truth_binary == 1) & (inferred_binary == 1))
    
    # False Positive (FP): Edge exists in inferred but not in ground truth
    FP = np.sum((ground_truth_binary == 0) & (inferred_binary == 1))
    
    # False Negative (FN): Edge exists in ground truth but not in inferred
    FN = np.sum((ground_truth_binary == 1) & (inferred_binary == 0))
    
    # Compute SHD (Structural Hamming Distance)
    SHD = FP + FN
    
    # Compute True Positive Rate (TPR)
    if (TP + FN) > 0:
        TPR = TP / (TP + FN)
    else:
        TPR = 0  # Avoid division by zero, assuming no true edges are present

    # Compute False Discovery Rate (FDR)
    if (TP + FP) > 0:
        FDR = FP / (TP + FP)
    else:
        FDR = 0  # Avoid division by zero, assuming no inferred edges are present

    return SHD, FDR, TPR



