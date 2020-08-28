import tools
import cv2
import numpy as np
import random

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

        self.warped = None
        self.keypoints = None
        self.descriptors = None

class ImageRegistration:
    def __init__(self, config):
        self.config = config

        self.orb_detector = cv2.ORB_create(config.num_features)

        self.buffer = []

    def initializeLastItem(self, lastItem):
        img_gray = cv2.cvtColor(lastItem.image, cv2.COLOR_BGR2GRAY)
        bg_mask = cv2.cvtColor(lastItem.bg_mask, cv2.COLOR_BGR2GRAY)

        lastItem.keypoints, lastItem.descriptors = self.orb_detector.detectAndCompute(img_gray, bg_mask)

        # im_with_keypoints = cv2.drawKeypoints(img_gray,
        #                   lastItem.keypoints,
        #                   np.array([]),
        #                   (0, 0, 255),
        #                   cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #
        # tools.draw_image_on_plt(im_with_keypoints, "im_with_keypoints")

    def computeTransformations(self, lastItem):
        self.initializeLastItem(lastItem)

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # TODO: For warping, iterate in reverse order to get more static scenes among frames
        for idx, item in enumerate(reversed(self.buffer)):
            # if idx != 9:
            #     continue

            item.warped = ImageItem()

            height, width = item.image.shape[:-1]

            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            # print(lastItem.descriptors)
            # print(item.descriptors)

            matches = flann.knnMatch(lastItem.descriptors, item.descriptors, k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]

            # ratio test as per Lowe's paper
            good = []
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]
                    good.append(m)

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)

            # img_with_matches = cv2.drawMatchesKnn(
            #     cv2.cvtColor(lastItem.image, cv2.COLOR_BGR2GRAY),
            #     lastItem.keypoints,
            #     cv2.cvtColor(item.image, cv2.COLOR_BGR2GRAY),
            #     item.keypoints,
            #     matches,
            #     None, **draw_params)

            # tools.draw_image_on_plt(img_with_matches, "img_with_matches")

            # TODO: Do not continue if there are not enough points. Clear the remaining images from the buffer in this case
            # if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([lastItem.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([item.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # last_points = np.zeros((len(matches), 2))
            # item_points = np.zeros((len(matches), 2))
            #
            # for i in range(len(matches)):
            #     print(matches[i])
            #     last_points[i, :] = lastItem.keypoints[matches[i].queryIdx].pt
            #     item_points[i, :] = item.keypoints[matches[i].trainIdx].pt

            # TODO: Again, do not continue if there are not enough points. Clear the remaining images from the buffer in this case
            # if len(good)>MIN_MATCH_COUNT:
            homography, mask_pairs = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

            item.warped.image = cv2.warpPerspective(src=item.image,
                                                    M=homography,
                                                    dsize=(width, height),
                                                    flags=cv2.INTER_CUBIC)
            item.warped.mask = cv2.warpPerspective(src=item.mask,
                                                    M=homography,
                                                    dsize=(width, height),
                                                    flags=cv2.INTER_CUBIC)

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

            # tools.draw_image_on_plt(orig_mask, "orig_mask")
            # tools.draw_image_on_plt(lastItem.mask, "lastItem.mask")
            # tools.draw_image_on_plt(item.warped.mask, "item.warped.mask")
            # tools.draw_image_on_plt(mask, "mask")
            # tools.draw_image_on_plt(lastItem.mask - mask, "diff")

            warped = cv2.bitwise_and(item.warped.image, mask)
            neg = cv2.bitwise_and(lastItem.final_image, cv2.bitwise_not(mask))
            # dst = cv2.max(neg, warped)
            lastItem.final_image = cv2.max(neg, warped)

            # tools.draw_image_on_plt(dst, "dst")

            # lastItem.final_mask = cv2.bitwise_and(cv2.bitwise_or(lastItem.final_mask, orig_mask), mask)
            colormask = np.zeros(lastItem.image.shape, np.uint8)
            colormask[:, :] = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))
            colormask = cv2.bitwise_and(colormask, mask)
            # lastItem.final_mask = cv2.bitwise_or(lastItem.final_mask, colormask)
            lastItem.used_mask += mask
            # lastItem.final_image = cv2.bitwise_and(dst, cv2.bitwise_not(lastItem.final_mask))


            # tools.draw_image_on_plt(lastItem.used_mask, "lastItem.final_mask")
            # tools.draw_image_on_plt(lastItem.final_image, "lastItem.final_image")
            # tools.draw_image_on_plt(lastItem.inpaint_mask, "lastItem.inpaint_mask")

        # tools.draw_image_on_plt(lastItem.used_mask, "lastItem.final_mask (final)")
        # tools.draw_image_on_plt(lastItem.final_image, "lastItem.final_image (final)")
        # tools.draw_image_on_plt(lastItem.inpaint_mask, "lastItem.inpaint_mask (final)")

        # return lastItem.final_image

        # TODO: Repeat, this time using older inpainted images for keeping as much consistency as possible

        from inpainting import GenerativeInpainting
        inpainting = GenerativeInpainting(config)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (config.dilate_inpainting_mask_by_px, config.dilate_inpainting_mask_by_px))
        # mask_inpaint = cv2.dilate(lastItem.final_mask, kernel, cv2.BORDER_CONSTANT, iterations=1)
        # # tools.draw_image_on_plt(mask_inpaint, "mask_inpaint")
        # _, mask_inpaint = cv2.threshold(mask_inpaint, 1, 255, cv2.THRESH_BINARY)
        lastItem.inpaint_mask = cv2.bitwise_not(lastItem.inpaint_mask)
        # tools.draw_image_on_plt(lastItem.inpaint_mask, "mask_inpaint")


        inpainted = inpainting.inpaint_image(lastItem.final_image, lastItem.inpaint_mask)


        # tools.draw_image_on_plt(inpainted, "inpainted")
        #
        # compare_inpainted = inpainting.inpaint_image(lastItem.image, lastItem.mask)
        # tools.draw_image_on_plt(lastItem.mask, "lastItem.mask")
        # tools.draw_image_on_plt(compare_inpainted, "compare_inpainted")

        return inpainted

    def fillImage(self, lastItem):
        print(lastItem)

    def fillImageWithInpainted(self, lastItem):
        print(lastItem)

    def generateMask(self, lastItem):
        print(lastItem)

    def updateBuffer(self, image, mask, bg_mask):
        lastItem = ImageItem(image, mask, bg_mask)

        return_img = image
        if len(self.buffer) > 0:
            return_img = self.computeTransformations(lastItem)
            # self.fillImage(lastItem)
            # self.fillImageWithInpainted(lastItem)
            # self.generateMask(lastItem)
        else:
            self.initializeLastItem(lastItem)

        # lastItem.image = np.zeros(lastItem.image.shape, np.uint8)
        # import random
        # lastItem.image[:, :] = (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))

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

        # if count % 5 != 0:
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

    video_in.release()
    # video_out.release()

