from inpainting import GenerativeInpainting
from segmentation import DeepLabSegmentation
from registration import ImageRegistration
import neuralgym.neuralgym as ng
import cv2
import labels

# Just inpainting, no registration
def test1():
    # TODO: Pass as parameter
    config_file = "config/config.yml"
    config = ng.Config(config_file)

    segmentation = DeepLabSegmentation(config)
    inpainting = GenerativeInpainting(config)

    video_in = cv2.VideoCapture(f'data/people_walking.mp4')

    frame_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_in.get(cv2.CAP_PROP_FPS))

    video_out = None

    count = 0
    while (video_in.isOpened()):
        print(f"==== Processing image {count} ====")
        count += 1

        ret, frame = video_in.read()

        if not ret:
            break

        resized_image, seg_map = segmentation.run(frame)
        mask = labels.mask_from_labels(seg_map, config.classes_to_remove)

        if config.dilate_mask.apply:
            if config.dilate_mask.type == "ellipse":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (config.dilate_mask.kernel_size, config.dilate_mask.kernel_size))
                mask = cv2.dilate(mask, kernel, iterations=1)
            elif config.dilate_mask.type == "rect":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                   (config.dilate_mask.kernel_size, config.dilate_mask.kernel_size))
                mask = cv2.dilate(mask, kernel, iterations=1)
            elif config.dilate_mask.type == "cross":
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                                   (config.dilate_mask.kernel_size, config.dilate_mask.kernel_size))
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                raise Exception("Unsupported dilation type. Exiting...")

        new_frame = inpainting.inpaint_image(resized_image, mask)
        resized_new_frame = cv2.resize(new_frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

        if not video_out:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_out = cv2.VideoWriter(f'data/wo_people_walking_no_registration.avi', fourcc, fps, (frame_width, frame_height))

        video_out.write(resized_new_frame)

    video_in.release()
    video_out.release()

# Inpainting + registration
def test2():
    # TODO: Pass as parameter
    config_file = "config/config.yml"
    config = ng.Config(config_file)

    segmentation = DeepLabSegmentation(config)
    registration = ImageRegistration(config)
    inpainting = GenerativeInpainting(config)

    video_in = cv2.VideoCapture(f'data/people_walking.mp4')

    frame_width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_in.get(cv2.CAP_PROP_FPS))

    video_out = None

    count = 0
    while (video_in.isOpened()):
        print(f"==== Processing image {count} ====")
        count += 1

        ret, frame = video_in.read()

        if not ret:
            break

        resized_image, seg_map = segmentation.run(frame)
        mask = labels.mask_from_labels(seg_map, config.classes_to_remove)
        bg_mask = labels.mask_from_labels(seg_map, config.background_classes)

        if config.dilate_mask.apply:
            if config.dilate_mask.type == "ellipse":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (config.dilate_mask.kernel_size, config.dilate_mask.kernel_size))
                mask = cv2.dilate(mask, kernel, iterations=1)
            elif config.dilate_mask.type == "rect":
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                   (config.dilate_mask.kernel_size, config.dilate_mask.kernel_size))
                mask = cv2.dilate(mask, kernel, iterations=1)
            elif config.dilate_mask.type == "cross":
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,
                                                   (config.dilate_mask.kernel_size, config.dilate_mask.kernel_size))
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                raise Exception("Unsupported dilation type. Exiting...")

        new_frame = registration.updateBuffer(resized_image, mask, bg_mask)
        resized_new_frame = cv2.resize(new_frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

        if not video_out:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_out = cv2.VideoWriter(f'data/wo_people_walking_registration.avi', fourcc, fps,
                                        (frame_width, frame_height))

        video_out.write(resized_new_frame)

    video_in.release()
    video_out.release()

if __name__ == "__main__":
    # test1()
    test2()