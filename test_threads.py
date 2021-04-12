from threading import *
from time import sleep
import numpy as np

class RealSenseCamera(Thread):
    def run(self):
        for i in range(5):
            sleep(1)
            print("Hello")

class PupilTracker(Thread):
    def run(self):
        for i in range(5):
            sleep(1)
            print("Hi")

t1 = RealSenseCamera()
t2 = PupilTracker()

# Runs in the main thread
t1.start()
sleep(0.2)
t2.start()

t1.join()
t2.join()