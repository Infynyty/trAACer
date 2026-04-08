import sys
import threading

import numpy as np
import uvicorn

from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import WebserverDeviceLayerReceiver, WebserverDeviceLayerSender
from traacer.network_stack.layer.payload_layer.LinkLayer import NoopLinkLayerReceiver
from traacer.network_stack.layer.physical_layer.OOKPhysicalLayer import OOKPhysicalLayerReceiver, OOKPhysicalLayerSender
from traacer.network_stack.packet import LinkLayerPacket
from traacer.receiver.server import app

link_layer_receiver = NoopLinkLayerReceiver()
physical_layer_receiver = OOKPhysicalLayerReceiver(link_layer_receiver)
device_layer_receiver = WebserverDeviceLayerReceiver(physical_layer_receiver, recordings_dir="/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/recordings")

device_layer_sender = WebserverDeviceLayerSender()
physical_layer_sender = OOKPhysicalLayerSender(device_layer_sender, 2)


def run_webserver():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def start_receiver():
    device_layer_receiver.send_up()

def start_sender():
    device_layer_sender.start()
    packet = LinkLayerPacket(np.array([150002, 0, 700031, 0]))
    physical_layer_sender.send_down(packet)
    device_layer_sender.stop()

webserver_thread = threading.Thread(target=run_webserver, daemon=True)
webserver_thread.start()

receiver_thread = threading.Thread(target=start_receiver, daemon=True)
receiver_thread.start()

sender_thread = threading.Thread(target=start_sender, daemon=True)
sender_thread.start()

receiver_thread.join()