import udp_stream
import cv2

udp_stream = udp_stream.UDPStream("UMI_GRIPPER_LEFT", "192.168.0.141", 11001, "recv")

udp_stream.start_recv_thread()

while True:
    try:
        frame = udp_stream.frame
        if frame is not None:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)
        break

cv2.destroyAllWindows()
udp_stream.stop_recv_thread()
udp_stream.close_camera()
        
        
        
        


