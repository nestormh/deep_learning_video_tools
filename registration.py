import tools
import cv2
import numpy as np

class ImageItem:
    def __init__(self, image=None, mask=None, bg_mask = None, filled=None, inpainted=None):
        self.image = image
        self.mask = mask
        self.bg_mask = bg_mask
        self.filled = filled
        self.inpainted = inpainted

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
        for idx, item in enumerate(self.buffer):
            item.warped = ImageItem()

            height, width = item.image.shape[:-1]

            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,  # 12
                                key_size=12,  # 20
                                multi_probe_level=1)  # 2
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            print(lastItem.descriptors)
            print(item.descriptors)

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

            img_with_matches = cv2.drawMatchesKnn(
                cv2.cvtColor(lastItem.image, cv2.COLOR_BGR2GRAY),
                lastItem.keypoints,
                cv2.cvtColor(item.image, cv2.COLOR_BGR2GRAY),
                item.keypoints,
                matches,
                None, **draw_params)

            # tools.draw_image_on_plt(img_with_matches, "img_with_matches")

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

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

            item.warped.image = cv2.warpPerspective(src=item.image,
                                                    M=homography,
                                                    dsize=(width, height),
                                                    flags=cv2.INTER_CUBIC)

            tools.draw_image_on_plt(item.warped.image, "item.warped.image")
            tools.draw_image_on_plt(cv2.cvtColor(item.warped.image, cv2.COLOR_BGR2GRAY) - cv2.cvtColor(lastItem.image, cv2.COLOR_BGR2GRAY), "sub")
            tools.draw_image_on_plt(mask, "mask")


            exit(0)


    def fillImage(self, lastItem):
        print(lastItem)

    def fillImageWithInpainted(self, lastItem):
        print(lastItem)

    def generateMask(self, lastItem):
        print(lastItem)

    def updateBuffer(self, image, mask, bg_mask):
        lastItem = ImageItem(image, mask, bg_mask)

        # TODO: Remove this condition once finished development
        if len(self.buffer) == 10:
            self.computeTransformations(lastItem)
            # self.fillImage(lastItem)
            # self.fillImageWithInpainted(lastItem)
            # self.generateMask(lastItem)
            exit(0)
        else:
            self.initializeLastItem(lastItem)

        self.buffer.insert(0, lastItem)
        if len(self.buffer) > self.config.max_images_in_buffer:
            self.buffer.pop()

        print(f"{len(self.buffer)} elements in buffer")

        # return imageFilled, mask

    def updateInpainted(self, inpainted):
        self.buffer[0].inpainted = inpainted

if __name__ == "__main__":
    import neuralgym.neuralgym as ng

    config_file = "config/config.yml"
    config = ng.Config(config_file)

    registration = ImageRegistration(config)

    for count in range(15):
        resized_image = cv2.imread(f"data/buffer/image_{count}.png")
        mask = cv2.imread(f"data/buffer/mask_{count}.png")
        bg_mask = cv2.imread(f"data/buffer/bg_mask_{count}.png")
        # new_frame = cv2.imread(f"data/buffer/inpainting_{count}.png")

        registration.updateBuffer(resized_image, mask, bg_mask)

