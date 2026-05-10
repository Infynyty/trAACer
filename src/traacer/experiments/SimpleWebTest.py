import queue
import sys
import threading
from time import sleep

import numpy as np
import uvicorn
from matplotlib import pyplot as plt

from traacer.metrics.bit_comparator import bit_queue, compare_bits
from traacer.metrics.plot import compare_signals, wait_for_signals
from traacer.network_stack.layer.device_layer.WAVDeviceLayer import WAVDeviceLayerReceiver, WAVDeviceLayerSender
from traacer.network_stack.layer.device_layer.WebserverDeviceLayer import WebserverDeviceLayerReceiver, WebserverDeviceLayerSender
from traacer.network_stack.layer.payload_layer.LinkLayer import NoopLinkLayerReceiver
from traacer.network_stack.layer.physical_layer.CSSPhysicalLayer import CSSPhysicalLayerReceiver, CSSPhysicalLayerSender
from traacer.network_stack.layer.physical_layer.FSKPhysicalLayer import FSKPhysicalLayerSender, FSKPhysicalLayerReceiver
from traacer.network_stack.layer.physical_layer.OFDMPhysicalLayer import OFDMPhysicalLayerSender, \
    OFDMPhysicalLayerReceiver
from traacer.network_stack.layer.physical_layer.OOKPhysicalLayer import OOKPhysicalLayerReceiver, OOKPhysicalLayerSender
from traacer.network_stack.layer.physical_layer.PSKPhysicalLayer import PSKPhysicalLayerSender
from traacer.network_stack.layer.physical_layer.PhysicalLayerUtil import bytes_to_bits
from traacer.network_stack.packet import LinkLayerPacket
from traacer.receiver.server import app, cert_path, key_path

link_layer_receiver = NoopLinkLayerReceiver()
#physical_layer_receiver = OOKPhysicalLayerReceiver(link_layer_receiver, carrier_freq=20000, bit_duration=0.05)
#physical_layer_receiver = FSKPhysicalLayerReceiver(link_layer_receiver, bit_duration=0.05)
#physical_layer_receiver = CSSPhysicalLayerReceiver()
physical_layer_receiver = OFDMPhysicalLayerReceiver()
device_layer_receiver = WebserverDeviceLayerReceiver(physical_layer_receiver, recordings_dir="/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/recordings")
#device_layer_receiver = WAVDeviceLayerReceiver(physical_layer_receiver, "/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/recordings/test.wav")

device_layer_sender = WebserverDeviceLayerSender()
#device_layer_sender = WAVDeviceLayerSender("/Users/kasimirstadie/PycharmProjects/BT/trAACer/src/traacer/receiver/recordings/test.wav")
#physical_layer_sender = PSKPhysicalLayerSender(device_layer_sender)
#physical_layer_sender = OOKPhysicalLayerSender(device_layer_sender, 2, bit_duration=0.05, carrier_freq=20000)
physical_layer_sender = OFDMPhysicalLayerSender(device_layer_sender)
#physical_layer_sender = FSKPhysicalLayerSender(device_layer_sender, 2, bit_duration=0.05)
#physical_layer_sender = CSSPhysicalLayerSender(device_layer_sender)


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
    rng = np.random.default_rng()
    data = rng.integers(0, 2, size=192, dtype=np.uint8)
    bit_queue.put(data)
    packet = LinkLayerPacket(np.frombuffer(data, dtype=np.uint8))
    physical_layer_sender.send_down(packet)
    device_layer_sender.stop()


webserver_thread = threading.Thread(target=run_webserver, daemon=True)
webserver_thread.start()



sender_thread = threading.Thread(target=start_sender, daemon=True)
sender_thread.start()


receiver_thread = threading.Thread(target=start_receiver, daemon=True)
receiver_thread.start()


compare_bits()


receiver_thread.join()