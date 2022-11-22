import argparse
import cv2 as cv

parser = argparse.ArgumentParser(
    prog = 'object_tracker',
    description = 'Tracing objects with OpenCV'
)

parser.add_argument('-f', '--file')
parser.add_argument('-a', '--area')
parser.add_argument('-t', '--tail', default=.7)    
parser.add_argument('-c', '--contours', default=False)

args = parser.parse_args()

file_path = args.file
tail_start_thresh = args.tail
draw_contours = args.contours

area_inf = float(args.area.split(',')[0][1:])
area_sup = float(args.area.split(',')[1].strip()[0:-1])

video = cv.VideoCapture(file_path)

if (video.isOpened()== False):
    print('erro ao abrir o arquivo de video')

# subtração do fundo das imagens
subtractor = cv.bgsegm.createBackgroundSubtractorMOG()

# centros dos contornos dos objetos
# em movimento
centers = []

if __name__ == '__main__':
    while(video.isOpened()):
        height = video.get(cv.CAP_PROP_FRAME_HEIGHT)
        width = video.get(cv.CAP_PROP_FRAME_WIDTH)

        ret, frame = video.read()

        if ret == True:
            # time.sleep(0.05)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (11, 11), 0)
            
            # thresh, bin = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV)
            img_bin = subtractor.apply(blurred)

            erode = cv.erode(img_bin, (5,5), iterations=2)
            dilate = cv.dilate(erode, (5,5), iterations=2)

            contours, img = cv.findContours(
                dilate,
                cv.RETR_TREE,
                cv.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv.contourArea(cnt)

                if area_inf < area < area_sup:

                    if draw_contours:
                        cv.drawContours(frame, [cnt], -1, (0,255,0), 1)

                    M = cv.moments(cnt)
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    centers.append((cX, cY))

                for i, (center_x, center_y) in \
                    enumerate(centers[int(len(centers)*tail_start_thresh):-1]):
                    
                    # círculos da trajetória
                    cv.circle(
                        frame,
                        (center_x, center_y),
                        3,
                        (0, 255, 0),
                        -1
                    )

                    # linhas da trajetória
                    for i in range(int(len(centers)*tail_start_thresh), len(centers)-1):
                        center = centers[i]
                        next_center = centers[i+1]

                        cv.line(
                            img=frame,
                            pt1=(center[0], center[1]),
                            pt2=(next_center[0], next_center[1]),
                            color=(0, 255, 0),
                            thickness=1
                        )

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