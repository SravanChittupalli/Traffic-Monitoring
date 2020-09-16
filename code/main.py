from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#plt.ion()
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
from sort import *
tracker = Sort()
memory = {}
counter = 0
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def animate( ):
    global plot_x , plot_y , count_inc , time_interval
    plot_y.append(time_interval/1000)
    plot_x.append(count_inc)
    plt.cla()
    plt.plot(plot_x , plot_y)


def support_func_V85(arr , int_part , dec_part):
    print(arr , int_part , dec_part)
    if len(arr) != 0:
        if int_part == len(arr):
            return arr[int(int_part)-1]
        if dec_part == 0.0 or int_part == len(arr) - 1 or int_part == len(arr):
            return arr[int(int_part)]
        else:
            return (arr[int(int_part)] + arr[int(int_part)+1])/2
    else:
        return 0


def calcV85speed(bike , car , truck , bus):
    bike.sort()
    bus.sort()
    car.sort()
    truck.sort()

    bus_dec , bus_int ,  = math.modf((len(bus)*0.85)+0.5)
    car_dec , car_int = math.modf((len(car)*0.85)+0.5)
    bike_dec , bike_int = math.modf((len(bike)*0.85)+0.5)
    truck_dec , truck_int = math.modf((len(truck)*0.85)+0.5)

    truck = support_func_V85(truck , truck_int , truck_dec)
    car = support_func_V85(car , car_int , car_dec)
    bike = support_func_V85(bike , bike_int , bike_dec)
    bus = support_func_V85(bus , bus_int , bus_dec)

    return bike , car , truck , bus




def calc_speed(cur_time , time_rec):
    #print("time_rec",time_rec)
    try:
        A = time_rec['A']
        try:
            B = time_rec['B']
            C = time_rec['C']
        except:
            avg_speed = 5000/(cur_time-time_rec['A'])
            return round(avg_speed*18/5 , 1)
    except:
        return None
    speed_1 = 1.6666/(time_rec['B']-time_rec['A'])
    #print(speed_1*18/5)
    speed_2 = 1.6666/(time_rec['C']-time_rec['B'])
    #print(speed_2*18/5)
    speed_3 = 1.6666/(cur_time-time_rec['C'])
    #print(speed_3*18/5)
    avg_speed = (speed_1 + speed_2 + speed_3)*1000/3
    print("avg_speed",avg_speed)
    #avg_speed = 5000/(cur_time-time_rec['A'])
    return round(avg_speed*18/5 , 1)


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if (detection[1] * 100) > 60:
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None
car_cnt = 0
truck_cnt = 0
motorbike_cnt = 0
bus_cnt = 0
Vcar = []
Vtruck = []
Vmotorbike = []
Vbus = []
entry = {}
speed = 0
time_last = 0
last_count = 0
count_inc = 0
time_interval = 0
plot_y = [0]
plot_x = [0]
plt.style.use('fivethirtyeight')
plt.plot(plot_x , plot_y)

############################
file_handler = open('track_speed_log.csv' , 'w' , newline='')
writer = csv.writer(file_handler)
writer.writerow(['Date' , 'Speed' , 'Vehicle' , 'Obj ID'])
############################

def YOLO():

    global metaMain, netMain, altNames , car_cnt , truck_cnt , motorbike_cnt , bus_cnt , Vcar , Vtruck , Vmotorbike , Vbus , entry , speed , writer , file_handler , plot_y , plot_x , time_last , last_count , count_inc , time_interval ,  memory , counter
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("test_videos/test.mp4")
    print(cap.get(cv2.CAP_PROP_FPS))
    cap.set(3, 1280)
    cap.set(4, 720)
    out = cv2.VideoWriter(
        "axelCount_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        # print("current time",timestamp)
        time_interval = timestamp - time_last

        #print(timestamp)
        prev_time = time.time()
        ret, frame_read = cap.read()

        if ret:
            ROI = np.copy(frame_read)
            ROI[: , : , :] = 0
            region = frame_read[240:790 , 350:1400]
            ROI[240:790 , 350:1400] = region
            cv2.rectangle(frame_read , (400 , 240) , (1400 , 790) , (173 , 50, 200) , 2)
            #cv2.line(frame_read , (450 , 61) , (450 , 456) , (255 , 0 , 0) , 5 )
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)
            ROI_resized = cv2.resize(ROI,
                                    (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                    interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image , ROI_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            warp_pts = ((241 , 160) , (434 , 170) , (402 , 398) , (189 , 336))
            (tl , tr , br , bl) = warp_pts
            # now that we have our rectangle of points, let's compute
            # the width of our new image
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            # ...and now for the height of our new image
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

            maxWidth = max(int(widthA) , int(widthB))
            maxHeight = max(int(heightA) , int(heightB))

            rect = np.float32(np.array(warp_pts[:4]))

            dst = np.array([[0 , 0] , 
                            [maxWidth , 0],
                            [maxWidth , maxHeight],
                            [0 , maxHeight]] , dtype="float32")
            # calculate the perspective transform matrix and warp
            # the perspective to grab the screen
            M = cv2.getPerspectiveTransform(np.asarray(rect), dst)
            warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
            dets = []
            full_dets=[]
            if len(detections) > 0:
            # loop over the indexes we are keeping
                for i in range (0,len(detections)):
                    if(detections[i][0].decode() != "person"):
                    #print(len(detections))
                        (x, y) = (detections[i][2][0], detections[i][2][1])
                        (w, h) = (detections[i][2][2] , detections[i][2][3] )
                        dets.append([float(x-w/2), float(y-h/2), float(x+w/2), float(y+h/2), float(detections[i][1])])
                        full_dets.append([int((float(x-w/2)+float(x+w/2))/2) , int((float(y-h/2)+float(y+h/2))/2), detections[i][0].decode()])
                        #print(np.shape(dets))
                        #print(full_dets)
            dets = np.asarray(dets)
            tracks = tracker.update(dets)
            #print(tracks)

            boxes = []
            indexIDs = []
            c = []
            previous = memory.copy()
            memory = {}

            for track in tracks:
                # As boxes co-ordinates and indexes are appended one by one we are storing them
                # in a dictionary
                #print(track)
                for i , det_cent in enumerate(full_dets):
                    euclidean_distance = math.sqrt( (det_cent[0]-(track[0]+track[2])/2)**2 + (det_cent[1]-(track[1]+track[3])/2)**2 )
                    if euclidean_distance <= 4:
                        indexIDs.append(int(track[4]))
                        pnts = np.array([[[track[0] , track[1]] , [track[2] , track[3]]]] , dtype="float32")
                        warped_track = cv2.perspectiveTransform(pnts, M)[0]
                        boxes.append([warped_track[0][0], warped_track[0][1], warped_track[1][0], warped_track[1][1], det_cent[2] , track[0], track[1], track[2], track[3]])
                        memory[indexIDs[-1]] = boxes[-1]
                        break
            
            if len(boxes) > 0:
                i = int(0)
                for box in boxes:
                    # extract the bounding box coordinates
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))
                    #(x_box , y_box) = (int(box[5]), int(box[6]))
                    #(w_box , h_box) = (int(box[8]), int(box[7]))
                    #cv2.rectangle(image, (int(x_box), int(y_box)), (int((x_box+w_box)/2), int((y_box+h_box)/2)), (255 , 0 , 0), 2)

                    if indexIDs[i] in previous:
                        previous_box = previous[indexIDs[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        # p0 = prevCentroid
                        # p1 = current centroid
                        # condition for counter to increment is if line between prev centroid and current centroid intersect then increment counter
                        p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                        p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                        cv2.putText(warp, str(indexIDs[i]), p0 , cv2.FONT_HERSHEY_SIMPLEX , 0.5,(0, 255, 0) , 2)
                        cv2.line(warp , p0, p1, (0 , 255 , 0), 2)
                        if intersect(p0, p1, (55 , 220) , (55 , 0)):
                            if indexIDs[i] not in entry:
                                #print('############################')
                                entry[indexIDs[i]] = {}
                                entry[indexIDs[i]]['A'] = timestamp
                        if intersect(p0, p1, (96 , 220) , (96 , 0)):
                            try:
                                entry[indexIDs[i]]['B'] = timestamp
                            except:
                                entry[indexIDs[i]] = {}
                                continue
                        if intersect(p0, p1, (138 , 220) , (138 , 0)):
                            try:
                                entry[indexIDs[i]]['C'] = timestamp
                            except:
                                entry[indexIDs[i]] = {}
                                continue
                        #print(entry)
                        #print(p0 , p1)
                        # If diagonal intersects with line then increment counter
                        if intersect(p0, p1, (182 , 220) , (182 , 0)):
                            try:
                                entry[indexIDs[i]]
                            except KeyError:
                                entry[indexIDs[i]] = {}
                            seconds = time.time()
                            local_time = time.ctime(seconds)
                            if box[4] == 'car':
                                speed = calc_speed(timestamp , entry[indexIDs[i]])
                                if speed != None:
                                    car_cnt+=1
                                    Vcar.append(speed)
                                    writer.writerow([local_time , speed , box[4] , indexIDs[i]])
                            elif box[4] == 'truck':
                                speed = calc_speed(timestamp , entry[indexIDs[i]])
                                if speed != None:
                                    truck_cnt+=1
                                    Vtruck.append(speed)
                                    writer.writerow([local_time , speed , box[4] , indexIDs[i]])
                            elif box[4] == 'motorbike':
                                speed = calc_speed(timestamp , entry[indexIDs[i]])
                                if speed != None:
                                    motorbike_cnt+=1
                                    Vmotorbike.append(speed)
                                    writer.writerow([local_time , speed , box[4] , indexIDs[i]])
                            elif box[4] == 'bus':
                                speed = calc_speed(timestamp , entry[indexIDs[i]])
                                if speed != None:
                                    bus_cnt+=1
                                    Vbus.append(speed)
                                    writer.writerow([local_time , speed , box[4] , indexIDs[i]])
                            if speed != None:
                                print("Obj Id: {0} Speed: {1} ".format(indexIDs[i] , speed))
                                counter += 1
                    i+=1
            cv2.putText(image,
                    "Total Count = {}".format(counter),
                    (10 , 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "Car Count = {}".format(car_cnt),
                    (10 , 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "Truck Count = {}".format(truck_cnt),
                    (10 , 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "Motorbike Count = {}".format(motorbike_cnt),
                    (10 , 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            cv2.putText(image,
                    "Bus Count = {}".format(bus_cnt),
                    (10 , 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
            if speed != None:
                cv2.putText(image,
                        "Last Recorded Speed: {} km/hr".format(round(speed , 1)),
                        (160 , 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        [0, 255, 255], 2)    
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
            cv2.line(image , (231 , 344) , (280 , 207) , (255 , 0 , 0) , 2 )
            cv2.line(image , (367 , 381) , (401 , 230) , (255 , 0 , 0) , 2 )
            cv2.line(warp , (55 , 220) , (55, 0) , (255 , 0 , 0) , 2 )
            cv2.line(warp , (106 , 220) , (106 , 0) , (255 , 0 , 0) , 2 )
            cv2.line(warp , (148 , 220) , (148 , 0) , (255 , 0 , 0) , 2 )
            cv2.line(warp , (192 , 220) , (192 , 0) , (255 , 0 , 0) , 2 )
            #print(1/(time.time()-prev_time))
            out.write(image)
            cv2.imshow('Demo', image)
            #cv2.imshow('ROI' , ROI)
            cv2.imshow("transformed" , warp)
            k = cv2.waitKey(3)
            if k&0xFF ==  ord('q'):
                break
            #if time_interval > 5000:
            #   time_last = timestamp
            #  count_inc = counter - last_count
            #  plot_y.append(time_interval/1000)
            #  plot_x.append(count_inc)
            #  plt.cla()
            #  plt.plot(plot_x , plot_y)
                #ani = animation.FuncAnimation(plt.gcf() , animate , fargs=(count_inc))
            #  last_count = counter
        #plt.tight_layout()
        #plt.draw()  
        else:
            print("ERROR")    
            break
    cap.release()
    out.release()
    V85motorbike , V85car , V85truck , V85bus = calcV85speed(Vmotorbike , Vcar , Vtruck , Vbus)

    writer.writerow(['' , '' ,'' ,''])
    writer.writerow(['' , '' ,'' ,''])
    writer.writerow(['' , 'Count' , 'Vd[km/hr]' , 'Vmax[km/hr]' , 'V85[km/hr]'])
    writer.writerow(['Two-wheelers' , motorbike_cnt , np.sum(Vmotorbike)/len(Vmotorbike) , max(Vmotorbike)  , V85motorbike])
    writer.writerow(['Car' , car_cnt , np.sum(Vcar)/len(Vcar) , max(Vcar)  , V85car])
    writer.writerow(['Bus' , bus_cnt , np.sum(Vbus)/len(Vbus) , max(Vbus)  , V85bus])
    writer.writerow(['Truck' , truck_cnt , np.sum(Vtruck)/len(Vtruck) , max(Vtruck)  , V85truck])
    writer.writerow(['' , '' ,'' ,''])
    max_array = [max(Vmotorbike) , max(Vcar) ,max(Vbus) , max(Vtruck)]
    avg_speed = [np.sum(Vmotorbike)/len(Vmotorbike) , np.sum(Vcar)/len(Vcar) , np.sum(Vbus)/len(Vbus) , np.sum(Vtruck)/len(Vtruck) ]
    writer.writerow(['Total' , counter , np.sum(avg_speed)/4 , max(max_array) , (V85motorbike + V85car + V85truck + V85bus)/4 ])
    
    file_handler.close()

if __name__ == "__main__":
    YOLO()
