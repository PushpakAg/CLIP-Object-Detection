import torch
import clip
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

image_path = '/home/pushpak/Downloads/clip_test3.jpeg'
image = Image.open(image_path).convert("RGB")

transformed_image = F.to_tensor(image).unsqueeze(0)

with torch.no_grad():
    model_output = faster_rcnn_model(transformed_image)

threshold = 0.9  
boxes = model_output[0]['boxes']
scores = model_output[0]['scores']

filtered_boxes = [box for box, score in zip(boxes, scores) if score > threshold]

filtered_images = []
for box in filtered_boxes:
    x_min, y_min, x_max, y_max = map(int, box)
    cropped_img = image.crop((x_min, y_min, x_max, y_max))
    filtered_images.append(preprocess(cropped_img).unsqueeze(0))

filtered_images_tensor = torch.cat(filtered_images).to(device)

x = input("enter text: ")
text_inputs = clip.tokenize([x]).to(device)

with torch.no_grad():
    image_features = clip_model.encode_image(filtered_images_tensor)
    text_features = clip_model.encode_text(text_inputs)

similarity_scores = torch.cosine_similarity(image_features, text_features, dim=1)

clip_threshold = 0.2 # Set according to your application needs
final_boxes = [box for box, score in zip(filtered_boxes, similarity_scores) if score > clip_threshold]

fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

for box in final_boxes:
    x_min, y_min, x_max, y_max = box
    width, height = x_max - x_min, y_max - y_min
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='b', facecolor='none')
    ax.add_patch(rect)

plt.show()