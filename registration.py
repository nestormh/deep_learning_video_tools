import tools
import cv2
import numpy as np
import random
import time
from timeit import default_timer as timer

class ImageItem:
    def __init__(self, image=None, mask=None, bg_mask = None, filled=None, inpainted=None):
        self.image = image
        self.mask = mask
        self.bg_mask = bg_mask
        self.filled = filled
        self.inpainted = inpainted
        self.used_mask = np.zeros(mask.shape, np.uint8) if np.any(mask) else None
        self.inpaint_mask = cv2.bitwise_not(mask) if np.any(mask) else None
        self.final_image = image
        self.homography = None

        self.warped = None
        self.keypoints = None
        self.descriptors = None

class ImageRegistration:
    def __init__(self, config):
        self.config = config

        self.orb_detector = cv2.ORB_create(config.num_features)

        self.buffer = []

        self.inpainting = GenerativeInpainting(config)

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)


    def initializeLastItem(self, lastItem):
        img_gray = cv2.cvtColor(lastItem.image, cv2.COLOR_BGR2GRAY)
        bg_mask = cv2.cvtColor(lastItem.bg_mask, cv2.COLOR_BGR2GRAY)

        lastItem.keypoints, lastItem.descriptors = self.orb_detector.detectAndCompute(img_gray, bg_mask)

    def warpImages(self, item, lastItem, width, height):
        matches = self.flann.knnMatch(lastItem.descriptors, item.descriptors, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                good.append(m)

        src_pts = []
        dst_pts = []
        if len(good) > self.config.min_match_count:
            src_pts = np.float32([lastItem.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([item.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        if len(src_pts) > self.config.min_match_count:
            item.homography, mask_pairs = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

        item.warped.image = cv2.warpPerspective(src=item.image,
                                                M=item.homography,
                                                dsize=(width, height),
                                                flags=cv2.INTER_CUBIC)
        item.warped.mask = cv2.warpPerspective(src=item.mask,
                                               M=item.homography,
                                               dsize=(width, height),
                                               flags=cv2.INTER_CUBIC)

    def fillImages(self, item, lastItem):
        orig_mask = lastItem.mask - item.warped.mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.erode_filled_image_by_px, config.erode_filled_image_by_px))
        mask = cv2.erode(orig_mask, kernel, iterations=1)
        ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, lastItem.mask)
        # We don't want to use again data we already have
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(lastItem.used_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.dilate_inpainting_mask_by_px, config.dilate_inpainting_mask_by_px))
        inpaint_mask = cv2.erode(mask, kernel, cv2.BORDER_CONSTANT, iterations=1)
        lastItem.inpaint_mask = cv2.bitwise_or(lastItem.inpaint_mask, inpaint_mask)

        warped = cv2.bitwise_and(item.warped.image, mask)
        neg = cv2.bitwise_and(lastItem.final_image, cv2.bitwise_not(mask))
        lastItem.final_image = cv2.max(neg, warped)

        colormask = np.zeros(lastItem.image.shape, np.uint8)
        colormask[:, :] = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
        lastItem.used_mask += mask

    def fillFrame(self, lastItem):
        start_time = timer()
        print("COMPUTING...")
        self.initializeLastItem(lastItem)

        step = self.config.registration_step
        if len(self.buffer) / step < self.config.min_images_to_process:
            step = max(int(len(self.buffer) / self.config.min_images_to_process), 1)

        for idx in range(len(self.buffer) - 1, 0, -step):
            print(f"idx: {idx}")

            item = self.buffer[idx]
            item.warped = ImageItem()
            height, width = item.image.shape[:-1]

            self.warpImages(item, lastItem, width, height)

            self.fillImages(item, lastItem)

        lastItem.inpaint_mask = cv2.bitwise_not(lastItem.inpaint_mask)

        # Repeat the process, this time using older inpainted images for keeping as much consistency as possible

        for idx in range(len(self.buffer) - 1, 0, -step):
            print(f"idx: {idx}")
            item = self.buffer[idx]

            height, width = item.image.shape[:-1]

            item.warped.inpainted = cv2.warpPerspective(src=item.inpainted,
                                                    M=item.homography,
                                                    dsize=(width, height),
                                                    flags=cv2.INTER_CUBIC)

            orig_mask = item.inpaint_mask - lastItem.used_mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                               (config.erode_filled_image_by_px, config.erode_filled_image_by_px))
            mask = cv2.erode(orig_mask, kernel, iterations=1)
            ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, lastItem.mask)
            # We don't want to use again data we already have
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(lastItem.used_mask))

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (
            config.dilate_inpainting_mask_by_px, config.dilate_inpainting_mask_by_px))
            inpaint_mask = cv2.erode(mask, kernel, cv2.BORDER_CONSTANT, iterations=1)
            lastItem.inpaint_mask = cv2.bitwise_and(lastItem.inpaint_mask, cv2.bitwise_not(inpaint_mask))

            warped = cv2.bitwise_and(item.warped.inpainted, mask)
            neg = cv2.bitwise_and(lastItem.final_image, cv2.bitwise_not(mask))
            lastItem.final_image = cv2.max(neg, warped)

            colormask = np.zeros(lastItem.image.shape, np.uint8)
            colormask[:, :] = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
            lastItem.used_mask += mask

        # TODO: Inpainting is external?
        # return lastItem.final_image

        print(f"Registration time: {timer() - start_time}")

        lastItem.inpainted = self.inpainting.inpaint_image(lastItem.final_image, lastItem.inpaint_mask)


        return lastItem.inpainted

    def updateBuffer(self, image, mask, bg_mask):
        lastItem = ImageItem(image, mask, bg_mask)

        return_img = image
        if len(self.buffer) > -1:
            return_img = self.fillFrame(lastItem)
        else:
            self.initializeLastItem(lastItem)

        self.buffer.insert(0, lastItem)
        if len(self.buffer) > self.config.max_images_in_buffer:
            self.buffer.pop()

        print(f"{len(self.buffer)} elements in buffer")

        # return imageFilled, mask

        return return_img

    def updateInpainted(self, inpainted):
        self.buffer[0].inpainted = inpainted

if __name__ == "__main__":
    import neuralgym.neuralgym as ng
    from inpainting import GenerativeInpainting
    from segmentation import DeepLabSegmentation
    import labels

    # config_file = "config/config.yml"
    # config = ng.Config(config_file)

    # registration = ImageRegistration(config)

    # for count in range(15):
    #     resized_image = cv2.imread(f"data/buffer/image_{count}.png")
    #     mask = cv2.imread(f"data/buffer/mask_{count}.png")
    #     bg_mask = cv2.imread(f"data/buffer/bg_mask_{count}.png")
    #     # new_frame = cv2.imread(f"data/buffer/inpainting_{count}.png")
    #
    #     registration.updateBuffer(resized_image, mask, bg_mask)

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
        ret, frame = video_in.read()

        print(f"Processing frame: {count}")

        # if count % 10 != 0:
        #     count += 1
        #     continue
        # count += 1


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

        # if count == 2:
        #     break

        if not video_out:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_out = cv2.VideoWriter(f'data/wo_people_walking_registration_fixed.avi', fourcc, fps, (frame_width, frame_height))

        video_out.write(resized_new_frame)

        # TODO: Remove.
        time.sleep(0.1)

    video_in.release()
    video_out.release()

