from matplotlib import pyplot as plt
import cv2

def draw_image_on_plt(image, title=""):
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    plt.show()