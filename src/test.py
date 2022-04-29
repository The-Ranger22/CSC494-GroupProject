import numpy as np
import cv2
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from image_cnn import ASLClassifier
from hand_tracking import HandFinder


asl_clf64 = ASLClassifier()
asl_clf64.compile_model(seq_model=load_model('asl_clf_64x64_10e.h5'))
#asl_clf64.model.summary()
cap = cv2.VideoCapture(0)



left_fat = [i for i in range(80)]
right_fat = [i + 560 for i in range(80)]
total_fat = [*left_fat, *right_fat]
img_counter = 0
num_frames_to_wait = 0
hand_finder = HandFinder()
last_pred_val = "NONE"

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #frame = np.delete(frame, total_fat, axis=1)
    img_counter += 1
    # Recopying the frame due to cv2 being finicky
    frame = frame.copy()
    try:
        hand_img, frame = hand_finder.detect(frame, skeleton=True)
        if img_counter > num_frames_to_wait:
            hand_img = resize(hand_img, (64, 64), method='nearest').numpy()
            pred_cat, confidence, scores = asl_clf64.ident(hand_img.reshape(1, 64, 64, 3))
            print(confidence)
            last_pred_val = f'Sign: \'{pred_cat}\'| Confidence: {confidence*100}%'
            img_counter = 0
    except:
        #import traceback
        #traceback.print_exc()
        pass

    frame = cv2.putText(
        frame,
        last_pred_val,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()