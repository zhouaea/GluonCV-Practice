import time

import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import mxnet as mx
import matplotlib.pyplot as plt

model = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)

# Compile the model for faster speed
model.hybridize()
print("Model hybridized")

# Load the webcam handler
cap = cv2.VideoCapture(0)
# Let the camera autofocus.
time.sleep(1)
print("Camera loaded")

NUM_FRAMES = 200 # you can change this
for i in range(NUM_FRAMES):
    # Load frame from the camera
    ret, frame = cap.read()

    # Image pre-processing
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    transformed_image, hwc_image = gcv.data.transforms.presets.yolo.transform_test(frame, short=512)

    # Run frame through network
    class_IDs, scores, bounding_boxes = model(transformed_image)

    # Display the result
    img = gcv.utils.viz.cv_plot_bbox(hwc_image, bounding_boxes[0], scores[0], class_IDs[0], class_names=model.classes)
    gcv.utils.viz.cv_plot_image(img)

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()