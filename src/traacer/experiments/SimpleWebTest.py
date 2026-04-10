import queue
import sys
import threading

import numpy as np
import uvicorn
from matplotlib import pyplot as plt

from traacer.metrics.plot import compare_signals, wait_for_signals
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import WebserverDeviceLayerReceiver, WebserverDeviceLayerSender
from traacer.network_stack.layer.payload_layer.LinkLayer import NoopLinkLayerReceiver
from traacer.network_stack.layer.physical_layer.FSKPhysicalLayer import FSKPhysicalLayerSender
from traacer.network_stack.layer.physical_layer.OOKPhysicalLayer import OOKPhysicalLayerReceiver, OOKPhysicalLayerSender
from traacer.network_stack.layer.physical_layer.PSKPhysicalLayer import PSKPhysicalLayerSender
from traacer.network_stack.packet import LinkLayerPacket
from traacer.receiver.server import app, cert_path, key_path

link_layer_receiver = NoopLinkLayerReceiver()
physical_layer_receiver = OOKPhysicalLayerReceiver(link_layer_receiver, carrier_freq=1000, bit_duration=0.005)
#physical_layer_receiver = FSKPhysicalLayerReceiver(link_layer_receiver)
device_layer_receiver = WebserverDeviceLayerReceiver(physical_layer_receiver, recordings_dir="/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/recordings")

device_layer_sender = WebserverDeviceLayerSender()
#physical_layer_sender = PSKPhysicalLayerSender(device_layer_sender)
physical_layer_sender = OOKPhysicalLayerSender(device_layer_sender, 2, bit_duration=0.005, carrier_freq=1000)
#physical_layer_sender = FSKPhysicalLayerSender(device_layer_sender, 2, bit_duration=0.2)


def run_webserver():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
        log_level="critical"
    )

def start_receiver():
    device_layer_receiver.send_up()

def start_sender():
    device_layer_sender.start()
    packet = LinkLayerPacket(np.array([0xff55ff55], dtype=np.uint32))
    physical_layer_sender.send_down(packet)
    device_layer_sender.stop()


webserver_thread = threading.Thread(target=run_webserver, daemon=True)
webserver_thread.start()

receiver_thread = threading.Thread(target=start_receiver, daemon=True)
receiver_thread.start()

sender_thread = threading.Thread(target=start_sender, daemon=True)
sender_thread.start()


#wait_for_signals()


receiver_thread.join()