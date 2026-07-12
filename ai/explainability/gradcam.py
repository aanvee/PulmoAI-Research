import torch
class GradCAM:

    # ==========================================================
    # Initialization
    # ==========================================================

    def __init__(self, model, target_layer):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        self.forward_handle = None

        self.backward_handle = None

        self._register_hooks()

    # ==========================================================
    # Register Hooks
    # ==========================================================

    def _register_hooks(self):

        def forward_hook(module, input, output):

            self.activations = output
            output.retain_grad()

        self.forward_handle = self.target_layer.register_forward_hook(
            forward_hook
        )

    # ==========================================================
    # Remove Hooks
    # ==========================================================

    def remove_hooks(self):

        if self.forward_handle is not None:

            self.forward_handle.remove()

    # ==========================================================
    # Forward Pass
    # ==========================================================

    def forward(self, image, metadata):

        return self.model(
            image,
            metadata
        )
    # ==========================================================
    # Generate Grad-CAM Heatmap
    # ==========================================================

    def generate_heatmap(
        self,
        image,
        metadata,
        target_class
    ):

        # Forward Pass
        outputs = self.forward(
            image,
            metadata
        )

        # Clear Previous Gradients
        self.model.zero_grad()

        # Target Score
        score = outputs[:, target_class]

        # Backward Pass
        score.backward(torch.ones_like(score))

        # ======================================================
        # Gradient Extraction
        # ======================================================

        gradients = self.activations.grad.detach()

        activations = self.activations.detach()

        # ======================================================
        # Channel Importance Weights
        # ======================================================

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ======================================================
        # Heatmap Generation
        # ======================================================

        heatmap = torch.sum(
            weights * activations,
            dim=1
        )

        heatmap = torch.relu(
            heatmap
        )

        # ======================================================
        # Normalize Heatmap
        # ======================================================

        heatmap -= heatmap.min()

        heatmap /= (
            heatmap.max() + 1e-8
        )

        return heatmap.squeeze().cpu().numpy()
    # ==========================================================
    # Overlay Heatmap on Original Image
    # ==========================================================

    def overlay_heatmap(
        self,
        image_path,
        heatmap,
        alpha=0.4
    ):

        import cv2
        import numpy as np

        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        original = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        # Resize Heatmap
        heatmap = cv2.resize(
            heatmap,
            (
                original.shape[1],
                original.shape[0]
            )
        )

        # Convert Heatmap to Color
        heatmap = np.uint8(
            255 * heatmap
        )

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # Overlay
        overlay = cv2.addWeighted(
            original,
            1 - alpha,
            heatmap,
            alpha,
            0
        )

        return overlay

    # ==========================================================
    # Save Visualization
    # ==========================================================

    def save_visualization(
        self,
        overlay,
        output_path
    ):

        import cv2

        overlay = cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            output_path,
            overlay
        )

        print(f"Grad-CAM saved to: {output_path}")

if __name__ == "__main__":

    import torch
    from PIL import Image

    from ai.models.multimodal_model import PulmoAIModel
    from ai.data.preprocessing.transforms import valid_transform

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    CHECKPOINT = "ai/checkpoints/best_model.pth"

    IMAGE_PATH = r"D:\archive\images_001\images\00000003_000.png"

    OUTPUT_PATH = "gradcam_output.png"

    metadata = torch.tensor(
    [[0.0, 0.0, 0.1937046004842615]],
    dtype=torch.float32).to(DEVICE)

    model = PulmoAIModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image = valid_transform(
        image
    ).unsqueeze(0)

    image = image.to(DEVICE)

    # Last DenseNet Convolution Layer
    target_layer = model.image_encoder.backbone.features

    gradcam = GradCAM(
        model,
        target_layer
    )

    heatmap = gradcam.generate_heatmap(
        image,
        metadata,
        target_class=13    # No Finding
    )

    overlay = gradcam.overlay_heatmap(
        IMAGE_PATH,
        heatmap
    )

    gradcam.save_visualization(
        overlay,
        OUTPUT_PATH
    )

    gradcam.remove_hooks()

    print("Grad-CAM Completed Successfully.")