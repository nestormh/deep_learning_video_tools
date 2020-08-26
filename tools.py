from matplotlib import pyplot as plt

def draw_image_on_plt(image, title=""):
    plt.figure(figsize=(15, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title(title)
    plt.show()