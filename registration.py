import tools
import cv2
import numpy as np
import random
from inpainting import GenerativeInpainting
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
        else:
            item.warped.image = None
            item.warped.mask = None
            return

        if len(src_pts) > self.config.min_match_count:
            item.homography, mask_pairs = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
        else:
            item.warped.image = None
            item.warped.mask = None
            return

        item.warped.image = cv2.warpPerspective(src=item.image,
                                                M=item.homography,
                                                dsize=(width, height),
                                                flags=cv2.INTER_CUBIC)
        item.warped.mask = cv2.warpPerspective(src=item.mask,
                                               M=item.homography,
                                               dsize=(width, height),
                                               flags=cv2.INTER_CUBIC)

    def warpInpaintedImage(self, item, width, height):
        item.warped.inpainted = cv2.warpPerspective(src=item.inpainted,
                                                    M=item.homography,
                                                    dsize=(width, height),
                                                    flags=cv2.INTER_CUBIC)

    def fillImagesRegistration(self, item, lastItem):
        if not isinstance(lastItem.mask, np.ndarray):
            return

        if not isinstance(lastItem.used_mask, np.ndarray):
            lastItem.used_mask = np.zeros(item.mask.shape, np.uint8)

        orig_mask = lastItem.mask - item.warped.mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.config.erode_filled_image_by_px, self.config.erode_filled_image_by_px))
        mask = cv2.erode(orig_mask, kernel, iterations=1)
        ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, lastItem.mask)
        # We don't want to use again data we already have
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(lastItem.used_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.config.dilate_inpainting_mask_by_px, self.config.dilate_inpainting_mask_by_px))
        inpaint_mask = cv2.erode(mask, kernel, cv2.BORDER_CONSTANT, iterations=1)
        if isinstance(lastItem.inpaint_mask, np.ndarray):
            lastItem.inpaint_mask = cv2.bitwise_or(lastItem.inpaint_mask, inpaint_mask)
        else:
            lastItem.inpaint_mask = inpaint_mask

        warped = cv2.bitwise_and(item.warped.image, mask)
        neg = cv2.bitwise_and(lastItem.final_image, cv2.bitwise_not(mask))
        lastItem.final_image = cv2.max(neg, warped)

        colormask = np.zeros(lastItem.image.shape, np.uint8)
        colormask[:, :] = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
        lastItem.used_mask += mask

    # It looks similar to fillImagesRegistration, but it comes with minor but relevant differences.
    # In another function for clarity
    def fillImagesInpainting(self, item, lastItem):
        orig_mask = item.inpaint_mask - lastItem.used_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.config.erode_filled_image_by_px, self.config.erode_filled_image_by_px))
        mask = cv2.erode(orig_mask, kernel, iterations=1)
        ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, lastItem.mask)
        # We don't want to use again data we already have
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(lastItem.used_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (self.config.dilate_inpainting_mask_by_px, self.config.dilate_inpainting_mask_by_px))
        inpaint_mask = cv2.erode(mask, kernel, cv2.BORDER_CONSTANT, iterations=1)
        lastItem.inpaint_mask = cv2.bitwise_and(lastItem.inpaint_mask, cv2.bitwise_not(inpaint_mask))

        warped = cv2.bitwise_and(item.warped.inpainted, mask)
        neg = cv2.bitwise_and(lastItem.final_image, cv2.bitwise_not(mask))
        lastItem.final_image = cv2.max(neg, warped)

        colormask = np.zeros(lastItem.image.shape, np.uint8)
        colormask[:, :] = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
        lastItem.used_mask += mask

    def fillFrame(self, lastItem):
        start_time = timer()
        self.initializeLastItem(lastItem)

        step = self.config.registration_step
        if len(self.buffer) / step < self.config.min_images_to_process:
            step = max(int(len(self.buffer) / self.config.min_images_to_process), 1)

        # First, we want to fill the image with the real information in previous images,
        # so we don't need to "imagine" the content.
        for idx in range(len(self.buffer) - 1, 0, -step):
            item = self.buffer[idx]
            item.warped = ImageItem()
            height, width = item.image.shape[:-1]

            # Step 1. Images and masks are registered
            self.warpImages(item, lastItem, width, height)

            # Step 2. Derivated masks are created and content is filled
            self.fillImagesRegistration(item, lastItem)

        lastItem.inpaint_mask = cv2.bitwise_not(lastItem.inpaint_mask)

        # Repeat the process, this time using older inpainted images for keeping as much consistency as possible
        # The idea is using the previously "imagined" information, so there is coherence between frames.
        for idx in range(len(self.buffer) - 1, 0, -step):
            item = self.buffer[idx]
            height, width = item.image.shape[:-1]

            # Step 1. Inpainted image is also warped
            self.warpInpaintedImage(item, width, height)

            # Step 2. Derivated masks are created and content is filled
            self.fillImagesInpainting(item, lastItem)

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

        return return_img

    def updateInpainted(self, inpainted):
        self.buffer[0].inpainted = inpainted
