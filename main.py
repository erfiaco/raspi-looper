print('LOADING...')

import pyaudio
import numpy as np
import time
import os
from gpiozero import Button
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device
Device.pin_factory = LGPIOFactory()
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import ImageFont, ImageDraw
import sys

debounce_length = 0.03

PLAYBUTTONS = (Button(26, bounce_time=debounce_length),
               Button(5,  bounce_time=debounce_length))
RECBUTTONS  = (Button(6,  bounce_time=debounce_length),
               Button(13, bounce_time=debounce_length))

# OLED setup
serial = i2c(port=1, address=0x3C)
oled = ssd1306(serial, width=128, height=64)

def oled_message(msg):
    '''Show a centered message on the OLED'''
    with canvas(oled) as draw:
        draw.text((0, 24), msg, fill='white')

# get configuration (audio settings etc.) from file
settings_file = open('Config/settings.prt', 'r')
parameters = settings_file.readlines()
settings_file.close()

RATE     = int(parameters[0])
CHUNK    = int(parameters[1])
FORMAT   = pyaudio.paInt16
CHANNELS = 1
latency_in_milliseconds  = int(parameters[2])
LATENCY  = round((latency_in_milliseconds/1000) * (RATE/CHUNK))
INDEVICE  = int(parameters[3])
OUTDEVICE = int(parameters[4])
overshoot_in_milliseconds = int(parameters[5])
OVERSHOOT = round((overshoot_in_milliseconds/1000) * (RATE/CHUNK))
MAXLENGTH = int(12582912 / CHUNK)
SAMPLEMAX = 0.9 * (2**15)
LENGTH = 0

print(str(RATE) + ' ' + str(CHUNK))
print('latency correction (buffers): ' + str(LATENCY))
print('looking for devices ' + str(INDEVICE) + ' and ' + str(OUTDEVICE))

silence = np.zeros([CHUNK], dtype=np.int16)
output_volume = 1.0
down_ramp = np.linspace(1, 0, CHUNK)
up_ramp   = np.linspace(0, 1, CHUNK)

def fade_in(buffer):
    np.multiply(buffer, up_ramp, out=buffer, casting='unsafe')

def fade_out(buffer):
    np.multiply(buffer, down_ramp, out=buffer, casting='unsafe')

pa = pyaudio.PyAudio()

class audioloop:
    def __init__(self):
        self.initialized   = False
        self.length_factor = 1
        self.length        = 0
        self.main_audio    = np.zeros([MAXLENGTH, CHUNK], dtype=np.int16)
        self.dub_audio     = np.zeros([MAXLENGTH, CHUNK], dtype=np.int16)
        self.readp         = 0
        self.writep        = 0
        self.is_recording  = False
        self.is_playing    = False
        self.is_waiting    = False
        self.last_buffer_recorded = 0
        self.preceding_buffer     = np.zeros([CHUNK], dtype=np.int16)
        self.dub_ratio     = 1.0

    def increment_pointers(self):
        if self.readp == self.length - 1:
            self.readp = 0
            if self.is_recording:
                self.dub_ratio = self.dub_ratio * 0.9
        else:
            self.readp = self.readp + 1
        self.writep = (self.writep + 1) % self.length

    def initialize(self):
        print('initialize called')
        if self.initialized:
            return
        self.writep = self.length - 1
        self.last_buffer_recorded = self.writep
        self.length_factor = (int((self.length - OVERSHOOT) / LENGTH) + 1)
        self.length = self.length_factor * LENGTH
        fade_out(self.main_audio[self.last_buffer_recorded])
        preceding_buffer_copy = np.copy(self.preceding_buffer)
        fade_in(preceding_buffer_copy)
        self.main_audio[self.length - 1, :] += preceding_buffer_copy[:]
        self.readp = (self.writep + LATENCY) % self.length
        self.initialized = True
        self.is_playing   = True
        self.increment_pointers()

    def add_buffer(self, data):
        if self.length >= (MAXLENGTH - 1):
            self.length = 0
            return
        self.main_audio[self.length, :] = np.copy(data)
        self.length = self.length + 1

    def toggle_mute(self):
        self.is_playing = not self.is_playing

    def is_restarting(self):
        if not self.initialized:
            return False
        return self.readp == 0

    def read(self):
        if not self.initialized:
            return silence
        if not self.is_playing:
            self.increment_pointers()
            return silence
        tmp = self.readp
        self.increment_pointers()
        return self.main_audio[tmp, :] + self.dub_audio[tmp, :]

    def dub(self, data):
        if not self.initialized:
            return
        datadump = np.copy(data)
        self.main_audio[self.writep, :] = (self.main_audio[self.writep, :] * 0.9
                                          + self.dub_audio[self.writep, :] * self.dub_ratio)
        self.dub_audio[self.writep, :] = datadump[:]

    def clear(self):
        self.main_audio    = np.zeros([MAXLENGTH, CHUNK], dtype=np.int16)
        self.dub_audio     = np.zeros([MAXLENGTH, CHUNK], dtype=np.int16)
        self.initialized   = False
        self.is_playing    = False
        self.is_recording  = False
        self.is_waiting    = False
        self.length_factor = 1
        self.length        = 0
        self.readp         = 0
        self.writep        = 0
        self.last_buffer_recorded     = 0
        self.preceding_buffer         = np.zeros([CHUNK], dtype=np.int16)

    def undo(self):
        self.dub_audio     = np.zeros([MAXLENGTH, CHUNK], dtype=np.int16)
        self.is_recording  = False
        self.is_waiting    = False

    def clear_or_undo(self):
        if self.is_playing:
            self.undo()
        else:
            self.clear()

    def start_recording(self, previous_buffer):
        self.is_recording     = True
        self.is_waiting       = False
        self.preceding_buffer = np.copy(previous_buffer)

    def set_recording(self):
        print('set_recording called')
        already_recording = self.is_recording
        if self.is_recording and not self.initialized:
            self.initialize()
        self.is_recording = False
        self.is_waiting   = False
        if not already_recording:
            self.is_waiting = True

loops = (audioloop(), audioloop())
prev_rec_buffer = np.zeros([CHUNK], dtype=np.int16)

def update_volume():
    global output_volume
    peak = np.max(np.abs(
        loops[0].main_audio.astype(np.int32) + loops[1].main_audio.astype(np.int32)
      + loops[0].dub_audio.astype(np.int32)  + loops[1].dub_audio.astype(np.int32)
    ))
    output_volume = SAMPLEMAX / peak if peak > SAMPLEMAX else 1

def show_status():
    '''Muestra el estado de cada pista en la OLED'''
    with canvas(oled) as draw:
        for i, loop in enumerate(loops):
            y = i * 15

            if loop.is_recording:
                estado = 'REC '
            elif loop.is_waiting:
                estado = 'WAIT'
            elif not loop.initialized:
                estado = '----'
            elif loop.is_playing:
                estado = 'PLAY'
            else:
                estado = 'MUTE'
            draw.text((0, y), f'T{i+1}: {estado}', fill='white')

setup_is_recording  = False
setup_donerecording = False
play_buffer = np.zeros([CHUNK], dtype=np.int16)

def looping_callback(in_data, frame_count, time_info, status):
    global play_buffer, prev_rec_buffer, setup_donerecording, setup_is_recording, LENGTH
    if status:
        print(f'[{time.strftime("%H:%M:%S")}] callback status: {status}', flush=True)
    current_rec_buffer = np.right_shift(np.frombuffer(in_data, dtype=np.int16), 2)
    if not setup_donerecording:
        if setup_is_recording:
            if LENGTH >= MAXLENGTH:
                setup_donerecording = True
                setup_is_recording  = False
                return (silence, pyaudio.paContinue)
            loops[0].add_buffer(current_rec_buffer)
            LENGTH = LENGTH + 1
            return (silence, pyaudio.paContinue)
        else:
            return (silence, pyaudio.paContinue)
    if loops[0].is_restarting():
        for loop in loops:
            if loop.is_waiting:
                loop.start_recording(prev_rec_buffer)
    if loops[0].is_waiting and not loops[0].initialized:
        loops[0].start_recording(prev_rec_buffer)
    for loop in loops:
        if loop.is_recording:
            if loop.initialized:
                loop.dub(current_rec_buffer)
            else:
                loop.add_buffer(current_rec_buffer)
    play_buffer[:] = np.multiply((
          loops[0].read().astype(np.int32)
        + loops[1].read().astype(np.int32)
    ), output_volume, casting='unsafe').astype(np.int16)
    prev_rec_buffer = np.copy(current_rec_buffer)
    return (play_buffer, pyaudio.paContinue)

looping_stream = pa.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    input_device_index=INDEVICE,
    output_device_index=OUTDEVICE,
    frames_per_buffer=CHUNK,
    start=True,
    stream_callback=looping_callback
)

time.sleep(3)
print('ready')
oled_message('READY - REC T1')

# esperar primer botón de grabación (pista 1)
RECBUTTONS[0].wait_for_press()
setup_is_recording = True
loops[0].start_recording(prev_rec_buffer)
oled_message('RECORDING T1...')

time.sleep(0.5)

# esperar a parar la grabación
RECBUTTONS[0].wait_for_press()
setup_is_recording  = False
setup_donerecording = True
loops[0].initialize()
loops[0].set_recording()
show_status()
time.sleep(0.5)

# --- sesión en vivo ---

finished = False

def finish():
    global finished
    finished = True

def restart_looper():
    oled_message('RESTARTING...')
    pa.terminate()
    os.execlp(sys.executable, sys.executable, 'main.py')
    #os.execlp('python3', 'python3', 'main.py')

for i in range(2):
    RECBUTTONS[i].when_held    = loops[i].clear_or_undo
    RECBUTTONS[i].when_pressed = loops[i].set_recording
    RECBUTTONS[i].when_released = update_volume
    PLAYBUTTONS[i].when_pressed = loops[i].toggle_mute

PLAYBUTTONS[1].when_held = finish
PLAYBUTTONS[0].when_held = restart_looper

while not finished:
    show_status()
    time.sleep(0.25)

pa.terminate()
oled_message('BYE')
for button in RECBUTTONS:
    button.close()
for button in PLAYBUTTONS:
    button.close()
# Volver al boot menu
import subprocess
subprocess.Popen(["/usr/bin/python3", "/home/Javo/Proyects/boot_menu/boot_menu.py"])
print('Done...')
