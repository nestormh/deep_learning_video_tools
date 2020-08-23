from inpainting import GenerativeInpainting
from segmentation import DeepLabSegmentation
import neuralgym.neuralgym as ng
import cv2
import labels

if __name__ == "__main__":
    # TODO: Pass as parameter
    config_file = "config/config.yml"
    config = ng.Config(config_file)

    segmentation = DeepLabSegmentation(config)
    inpainting = GenerativeInpainting(config)

    # cases = ["bike", "dogs", "personinroom", "wallstreet"]
    # for case in cases:
    #     image = cv2.imread(f'data/{case}.jpg')
    #
    #     resized_image, seg_map = segmentation.run(image)
    #     mask = labels.mask_from_labels(seg_map, config.classes_to_remove)
    #
    #     output = inpainting.inpaint_image(resized_image, mask)
    #     cv2.imwrite(f'/tmp/{case}_inpainted.png', output)

    # NEXT: Pass a video as input

    cap = cv2.VideoCapture(f'data/people_walking.mp4')

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    while (cap.isOpened()):
        ret, frame = cap.read()

        resized_image, seg_map = segmentation.run(frame)
        mask = labels.mask_from_labels(seg_map, config.classes_to_remove)

        new_frame = inpainting.inpaint_image(resized_image, mask)

        if not out:
            out = cv2.VideoWriter(f'data/wo_people_walking.avi', fourcc, 25.0, new_frame.shape[:-1])

        out.write(new_frame)

    cap.release()
    out.release()