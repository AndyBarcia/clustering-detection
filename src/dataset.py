import math
import torch
from torch.utils.data import IterableDataset
import matplotlib.pyplot as plt


DEFAULT_INSTANCE_PALETTE_RGB = (
    (224, 64, 64),
    (64, 160, 224),
    (96, 192, 96),
    (240, 192, 64),
)


def get_default_instance_palette(device="cpu") -> torch.Tensor:
    return torch.tensor(DEFAULT_INSTANCE_PALETTE_RGB, dtype=torch.uint8, device=device)


class BatchedSyntheticIterableDataset(IterableDataset):
    def __init__(
        self,
        generator,
        total_samples,
        batch_size,
        drop_last=False,
    ):
        super().__init__()
        self.generator = generator
        self.total_samples = int(total_samples)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def __len__(self):
        if self.drop_last:
            return self.total_samples // self.batch_size
        return math.ceil(self.total_samples / self.batch_size)

    def __iter__(self):
        produced = 0
        next_image_id = 0

        while produced < self.total_samples:
            remaining = self.total_samples - produced

            if remaining < self.batch_size and self.drop_last:
                break

            current_bs = min(self.batch_size, remaining)

            images, targets = self.generator.generate_batch(
                batch_size=current_bs,
                start_idx=next_image_id,
            )

            yield images, targets

            produced += current_bs
            next_image_id += current_bs


class SyntheticPanopticBatchGenerator:
    def __init__(
        self,
        height=256,
        width=256,
        max_objects=10,
        device="cuda",
    ):
        self.height = height
        self.width = width
        self.max_objects = max_objects
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        self.classes = ["Background", "Square", "Triangle"]

        self.margin = 20
        self.min_size = 10
        self.max_size = 40

        if self.width <= 2 * self.margin or self.height <= 2 * self.margin:
            raise ValueError("height/width are too small for margin=20.")

        # Cached coordinate grids on device
        self.yy = torch.arange(self.height, device=self.device, dtype=torch.int32).view(
            1, 1, self.height, 1
        )
        self.xx = torch.arange(self.width, device=self.device, dtype=torch.int32).view(
            1, 1, 1, self.width
        )
        self.instance_palette = self._build_instance_palette()

    def _build_instance_palette(self) -> torch.Tensor:
        return get_default_instance_palette(device=self.device)

    def _boxes_from_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        masks: [N, H, W] bool
        returns: [N, 4] float32 in xyxy
        """
        if masks.numel() == 0:
            return torch.zeros((0, 4), dtype=torch.float32, device=self.device)

        rows = masks.any(dim=2)  # [N, H]
        cols = masks.any(dim=1)  # [N, W]

        y_min = rows.float().argmax(dim=1)
        y_max = self.height - 1 - rows.flip(1).float().argmax(dim=1)
        x_min = cols.float().argmax(dim=1)
        x_max = self.width - 1 - cols.flip(1).float().argmax(dim=1)

        return torch.stack([x_min, y_min, x_max, y_max], dim=1).to(torch.float32)

    @torch.no_grad()
    def generate_batch(self, batch_size, start_idx=0):
        """
        Returns:
          images: [B, 3, H, W] float32
          targets: list[dict]
        """
        B = batch_size
        M = self.max_objects
        H = self.height
        W = self.width
        device = self.device

        # Number of objects per image
        num_objs = torch.randint(
            low=1,
            high=M + 1,
            size=(B,),
            device=device,
            dtype=torch.int32,
        )  # [B]

        # Active object slots per image
        obj_idx0 = torch.arange(M, device=device, dtype=torch.int32).view(1, M)  # [1, M]
        active = obj_idx0 < num_objs.view(B, 1)  # [B, M] bool

        # Instance ids are 1..M within each image
        instance_ids = torch.arange(1, M + 1, device=device, dtype=torch.int32).view(
            1, M, 1, 1
        )  # [1, M, 1, 1]

        # Random attributes for all object slots
        class_ids = torch.randint(
            low=1,
            high=3,
            size=(B, M),
            device=device,
            dtype=torch.int64,
        )  # 1=square, 2=triangle

        color_indices = torch.randint(
            low=0,
            high=self.instance_palette.shape[0],
            size=(B, M),
            device=device,
            dtype=torch.int64,
        )
        colors = self.instance_palette[color_indices]  # [B, M, 3]

        cx = torch.randint(
            low=self.margin,
            high=W - self.margin,
            size=(B, M),
            device=device,
            dtype=torch.int32,
        )
        cy = torch.randint(
            low=self.margin,
            high=H - self.margin,
            size=(B, M),
            device=device,
            dtype=torch.int32,
        )
        size = torch.randint(
            low=self.min_size,
            high=self.max_size + 1,
            size=(B, M),
            device=device,
            dtype=torch.int32,
        )

        # Expand attrs to [B, M, 1, 1]
        cx_e = cx.unsqueeze(-1).unsqueeze(-1)
        cy_e = cy.unsqueeze(-1).unsqueeze(-1)
        size_e = size.unsqueeze(-1).unsqueeze(-1)

        # Broadcasted rasterization
        dx = (self.xx - cx_e).abs()
        dy = (self.yy - cy_e).abs()

        square_masks = (dx <= size_e) & (dy <= size_e)

        # Triangle:
        # top    = (cx, cy - size)
        # left   = (cx - size, cy + size)
        # right  = (cx + size, cy + size)
        #
        # y_rel = y - (cy - size), valid in [0, 2*size]
        # horizontal half-width = y_rel / 2
        y_rel = self.yy - (cy_e - size_e)
        triangle_masks = (
            (y_rel >= 0)
            & (y_rel <= 2 * size_e)
            & ((2 * (self.xx - cx_e).abs()) <= y_rel)
        )

        obj_masks = torch.where(
            class_ids.unsqueeze(-1).unsqueeze(-1) == 1,
            square_masks,
            triangle_masks,
        )  # [B, M, H, W]

        # Remove inactive slots
        obj_masks = obj_masks & active.unsqueeze(-1).unsqueeze(-1)

        # Painter's algorithm:
        # later object slots overwrite earlier ones, so max instance_id wins
        instance_map = torch.where(
            obj_masks,
            instance_ids.expand(B, -1, H, W),
            torch.zeros((1,), dtype=torch.int32, device=device),
        ).amax(dim=1)  # [B, H, W]

        # Visible masks for each slot
        visible_masks = instance_map.unsqueeze(1) == instance_ids  # [B, M, H, W]
        keep = active & visible_masks.flatten(2).any(dim=2)  # [B, M]

        # Boxes for all candidate visible masks in one vectorized pass
        visible_masks_flat = visible_masks.view(B * M, H, W)
        boxes_flat = self._boxes_from_masks(visible_masks_flat).view(B, M, 4)

        # Filter tiny artifacts
        valid_box = (
            ((boxes_flat[..., 2] - boxes_flat[..., 0]) >= 1)
            & ((boxes_flat[..., 3] - boxes_flat[..., 1]) >= 1)
        )
        keep = keep & valid_box

        # Build RGB image by per-image palette lookup
        palette = torch.zeros((B, M + 1, 3), dtype=torch.uint8, device=device)
        palette[:, 1:] = colors

        batch_idx = torch.arange(B, device=device).view(B, 1, 1)
        images = palette[batch_idx, instance_map.long()]  # [B, H, W, 3]
        images = images.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)  # [B, 3, H, W]

        background_masks = (instance_map == 0)
        background_boxes = self._boxes_from_masks(background_masks)

        # Package ragged targets per image
        targets = []
        for b in range(B):
            kb = keep[b]
            fg_masks_b = visible_masks[b][kb].to(torch.uint8)
            fg_labels_b = class_ids[b][kb].to(torch.int64)
            fg_boxes_b = boxes_flat[b][kb].to(torch.float32)
            fg_colors_b = colors[b][kb].to(torch.uint8)

            bg_masks_b = background_masks[b].unsqueeze(0).to(torch.uint8)
            bg_labels_b = torch.zeros((1,), dtype=torch.int64, device=device)
            bg_boxes_b = background_boxes[b].unsqueeze(0).to(torch.float32)
            bg_colors_b = torch.zeros((1, 3), dtype=torch.uint8, device=device)

            masks_b = torch.cat([bg_masks_b, fg_masks_b], dim=0)
            labels_b = torch.cat([bg_labels_b, fg_labels_b], dim=0)
            boxes_b = torch.cat([bg_boxes_b, fg_boxes_b], dim=0)
            colors_b = torch.cat([bg_colors_b, fg_colors_b], dim=0)

            area_b = masks_b.flatten(1).sum(dim=1).to(torch.float32)
            iscrowd_b = torch.zeros((labels_b.numel(),), dtype=torch.int64, device=device)
            image_id_b = torch.tensor([start_idx + b], dtype=torch.int64, device=device)

            targets.append(
                {
                    "boxes": boxes_b,
                    "labels": labels_b,
                    "masks": masks_b,
                    "color": colors_b,
                    "image_id": image_id_b,
                    "area": area_b,
                    "iscrowd": iscrowd_b,
                }
            )

        return images, targets


def visualize_sample(image, target):
    image = image.detach().cpu()
    target = {
        k: (v.detach().cpu() if torch.is_tensor(v) else v)
        for k, v in target.items()
    }

    img_np = image.permute(1, 2, 0).numpy()
    masks = target["masks"].numpy()
    labels = target["labels"].numpy()
    boxes = target["boxes"].numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(img_np)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    composite_mask = torch.zeros((img_np.shape[0], img_np.shape[1]), dtype=torch.float32).numpy()

    for i, mask in enumerate(masks):
        composite_mask[mask == 1] = i + 1
        box = boxes[i]
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            edgecolor="white",
            linewidth=1.5,
        )
        ax[1].add_patch(rect)
        ax[1].text(
            box[0],
            max(box[1] - 5, 0),
            f"Cls: {labels[i]}",
            color="yellow",
            fontsize=10,
            weight="bold",
        )

    ax[1].imshow(composite_mask, cmap="nipy_spectral")
    ax[1].set_title("Instances & Labels")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = SyntheticPanopticBatchGenerator(
        height=256,
        width=256,
        max_objects=10,
        device=device,
    )

    batch_size = 16

    # Dense batch tensor + ragged target list
    images, targets = generator.generate_batch(batch_size=batch_size, start_idx=0)

    print(images.shape)  # [B, 3, H, W]
    print(len(targets))  # B

    # Visualize first sample
    visualize_sample(images[0], targets[0])
