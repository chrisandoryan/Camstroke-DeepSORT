from pynput import keyboard
import pandas as pd
import time
import uuid

"""
Metrics collected:
1. Time between keydown events (keydelay)
2. Time between keydown and keyup (keyhold)
3. Time between keyup events (?)
"""

keystroke_data = list()


def get_related_keystroke(keystroke_data, char):
    last = -1
    for i, k in enumerate(keystroke_data):
        if k['keytext'] == char:
            if k['keyrelease'] == 0 and k['keyhold'] == 0 and k['keydelay'] == 0:
                last = i
    # return last element that satisfy the criterion
    return last


def store_keystroke(event_type, data):
    global keystroke_data
    char, stroke_time = data
    if event_type == "press":
        # print("{} pressed at {}".format(char, stroke_time))
        if len(keystroke_data) > 0:
            last_data = keystroke_data[-1]
            print("Last data", last_data)
            last_data.update({
                'keydelay': abs(last_data['keypress'] - stroke_time)
            })
        keystroke_data.append({
            'id': uuid.uuid4(),
            'keypress': stroke_time,
            'keyrelease': 0,
            'keyhold': 0,
            'keydelay': 0,
            'keytext': char,
        })
    elif event_type == "release":
        # print("{} released at {}".format(char, stroke_time))
        data = keystroke_data[get_related_keystroke(keystroke_data, char)]
        data.update({
            'keyrelease': stroke_time,
            'keyhold': abs(data['keypress'] - stroke_time),
        })
    print(keystroke_data)
    
def on_press(key):
    stroke_time = time.time()
    try:
        char = key.char
    except AttributeError:
        # special key (space, tab, etc.)
        char = key
    finally:
        store_keystroke("press", (char, stroke_time))


def on_release(key):
    stroke_time = time.time()
    char = key
    store_keystroke("release", (char, stroke_time))
    if key == keyboard.Key.esc:
        print("Stopping...")
        return False


def run(asynchronous=False):
    if asynchronous:
        print("Running in asynchronous mode")
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()
    else:
        print("Running in synchronous mode")
        with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
            listener.join()

run()