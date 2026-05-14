def mask_normalized_mean(values, mask):
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom

