import torchvision.transforms as T

def get_train_transforms():
    """
    Augmentation pipeline for training
    apply mild transformations to generalize the network against the 
    irregular spacing and background noise,without destroying the core structure
    """
    return T.Compose([
        T.ToPILImage(),
        # Mild rotation and translation to mimic the bounding-box jitter
        T.RandomAffine(degrees=3, translate=(0.02, 0.02)),
        # Convert to tensor and scale pixel values to [0.0, 1.0]
        T.ToTensor(),
        # Normalize to mean=0.5, std=0.5,shift pixel values in [-1.0, 1.0] range,
        # for faster CNN gradients convergence
        T.Normalize(mean=[0.5], std=[0.5])
    ])

def get_val_transforms():
    """
    Validation and Inference pipeline.
    Strictly deterministic. No random augmentations, just scaling and normalization.
    """
    return T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])