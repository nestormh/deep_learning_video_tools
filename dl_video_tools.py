from inpainting import GenerativeInpainting
import cv2

if __name__ == "__main__":
    # TODO: Parameterize
    CHECKPOINT_DIR = 'models/inpainting/release_places2_256_deepfill_v2'

    cases = ["bike", "bike_square", "dogs", "dogs_squared", "personinroom", "wallstreet"]
    for case in cases:
        image = cv2.imread(f'data/{case}_input.png')
        mask = cv2.imread(f'data/{case}_mask.png')

        gi = GenerativeInpainting("inpaint.yml", CHECKPOINT_DIR)
        output = gi.inpaint_image(image, mask)
        cv2.imwrite(f'data/{case}_inpainted2.png', output)

        break

