import sys
import cv2 as cv

file_path = sys.argv[1]
video = cv.VideoCapture(file_path)

if (video.isOpened()== False):
    print('erro ao abrir o arquivo de video')

centers = []

while(video.isOpened()):
    # altura e largura do video
    height = video.get(cv.CAP_PROP_FRAME_HEIGHT)
    width = video.get(cv.CAP_PROP_FRAME_WIDTH)
    ret, frame = video.read()

    if ret == True:
        # time.sleep(0.05)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (11, 11), 0)

        # basket = 50
        # bouncing_ball = 120
        thresh, bin = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV)

        erode = cv.erode(bin, (5,5), iterations=2)
        dilate = cv.dilate(erode, (5,5), iterations=2)

        contours, img = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # center = 0

        # cv.drawContours(frame, contours, -1, (0,255,0), 3)

        for cnt in contours:
            area = cv.contourArea(cnt)

            # basket = (8000, 12000)
            # bouncing_ball = (50, 3000)
            if 500 < area < 3000:
                # cv.drawContours(frame, [cnt], -1, (0,255,0), 3)

                M = cv.moments(cnt)
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                centers.append((cX, cY))

                for center_x, center_y in centers[int(len(centers)*.3):-1]:
                    cv.circle(frame, (center_x, center_y), 3, (0, 255, 0), -1)

                # último círculo
                cv.circle(frame, (cX, cY), 7, (0, 255, 0), -1)
            
        cv.imshow('binary', dilate)
        cv.imshow('original', frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

video.release()
cv.destroyAllWindows()