import torch
import torch.utils.data
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt


class SyntheticPanopticDataset(torch.utils.data.Dataset):
    def __init__(self, length=100, height=256, width=256, max_objects=10):
        self.length = length
        self.height = height
        self.width = width
        self.max_objects = max_objects
        
        # Classes: 0=Background, 1=Square, 2=Triangle
        self.classes = ['Background', 'Square', 'Triangle']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Init empty canvas
        # Image: H, W, 3 (RGB)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Instance Map: H, W (stores instance IDs)
        instance_map = np.zeros((self.height, self.width), dtype=np.int32)
        
        obj_ids = []
        obj_classes = []
        
        num_objs = random.randint(1, self.max_objects)
        
        # 2. Draw objects (Painter's algorithm: later objects overwrite earlier ones)
        for i in range(num_objs):
            instance_id = i + 1
            class_id = random.randint(1, 2) # 1=Square, 2=Triangle
            
            # Random color for the image
            color = np.random.randint(0, 255, (3,)).tolist()
                        
            cx = random.randint(20, self.width - 20)
            cy = random.randint(20, self.height - 20)
            size = random.randint(10, 40)
            
            if class_id == 1: # Square
                x1, y1 = cx - size, cy - size
                x2, y2 = cx + size, cy + size
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(instance_map, (x1, y1), (x2, y2), instance_id, -1)
                
            elif class_id == 2: # Triangle
                pt1 = (cx, cy - size)
                pt2 = (cx - size, cy + size)
                pt3 = (cx + size, cy + size)
                points = np.array([pt1, pt2, pt3])
                cv2.fillPoly(img, [points], color)
                cv2.fillPoly(instance_map, [points], instance_id)

            # Store what we attempted to draw, but we verify existence later
            obj_ids.append(instance_id)
            obj_classes.append(class_id)

        # 3. Process masks based on what is actually visible
        # In panoptic segmentation, masks must not overlap. 
        # Since we used painter's algo on 'instance_map', we just extract unique IDs.
        present_ids = np.unique(instance_map)
        present_ids = present_ids[present_ids > 0] # Exclude background (0)

        masks = []
        labels = []
        boxes = []

        for inst_id in present_ids:
            # Create binary mask for this instance
            mask = (instance_map == inst_id).astype(np.uint8)
            
            # Find which class this instance originally was
            # (We look up the index in our creation lists)
            original_idx = obj_ids.index(inst_id)
            lbl = obj_classes[original_idx]
            
            # Bounding box
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            # Filter small artifacts
            if (xmax - xmin) < 1 or (ymax - ymin) < 1:
                continue

            masks.append(mask)
            labels.append(lbl)
            boxes.append([xmin, ymin, xmax, ymax])

        # Convert to PyTorch Tensors
        if len(masks) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            # Handle rare empty image case
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, self.height, self.width), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # Normalize image to 0-1 and channels first (C, H, W)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.as_tensor(img, dtype=torch.float32)

        return img, target

def visualize_sample(image, target):
    """
    Displays the image and overlays the ground truth masks.
    """
    img_np = image.numpy().transpose(1, 2, 0) # Back to H,W,C
    masks = target['masks'].numpy()
    labels = target['labels'].numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: The Raw Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Synthetic Input Image")
    plt.axis('off')
    
    # Plot 2: Semantic/Instance Map
    plt.subplot(1, 2, 2)
    # Create a composite mask for visualization
    composite_mask = np.zeros((img_np.shape[0], img_np.shape[1]))
    
    for i, mask in enumerate(masks):
        # Assign a unique value per instance to visualize separation
        composite_mask[mask == 1] = labels[i] * 10 + i 
        
        # Draw bounding box on the map
        box = target['boxes'][i].numpy()
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                             fill=False, edgecolor='white', linewidth=1)
        plt.gca().add_patch(rect)
        plt.text(box[0], box[1], f"Cls: {labels[i]}", color='white', fontsize=8, backgroundcolor='black')

    plt.imshow(composite_mask, cmap='nipy_spectral', interpolation='nearest')
    plt.title("Ground Truth Instances & Labels")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_sample(image, target):
    """
    Creates a popup window showing the image and ground truth.
    """
    # Convert from (C, H, W) to (H, W, C)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    masks = target['masks'].cpu().numpy()
    labels = target['labels'].cpu().numpy()
    boxes = target['boxes'].cpu().numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title('Dataset Sample Viewer')

    # Left Plot: Raw Image
    ax[0].imshow(img_np)
    ax[0].set_title("Input Image")
    ax[0].axis('off')
    
    # Right Plot: Instance Segmentation
    composite_mask = np.zeros((img_np.shape[0], img_np.shape[1]))
    for i, mask in enumerate(masks):
        # Create a unique value for each instance for visual contrast
        composite_mask[mask == 1] = i + 1 
        
        # Overlay Bounding Box
        box = boxes[i]
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                             fill=False, edgecolor='white', linewidth=1.5)
        ax[1].add_patch(rect)
        ax[1].text(box[0], box[1]-5, f"Cls: {labels[i]}", color='yellow', fontsize=10, weight='bold')

    ax[1].imshow(composite_mask, cmap='nipy_spectral')
    ax[1].set_title("Instances & Labels")
    ax[1].axis('off')
    
    plt.tight_layout()
    print("Close the popup window to continue...")
    plt.show()

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    dataset = SyntheticPanopticDataset(length=10)    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    for images, targets in data_loader:
        visualize_sample(images[0], targets[0])
        break