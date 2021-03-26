from pynput import keyboard
import pandas as pd
import time
import uuid
import csv
from helpers import keylog as keylogutils
from helpers.utils import epoch_to_millis, print_info

"""
Metrics collected:
1. Time between keydown events (keydelay) DDT
2. Time between keydown and keyup (keyhold) DUT
3. Time between keyup events (?)

According to *, best features for keystroke dynamic is DDT and DUT

*: Farhi, N., Nissim, N., & Elovici, Y. (2019). Malboard: A novel user keystroke impersonation attack and trusted detection framework based on side-channel analysis. Computers & Security, 85, 240-269.
"""

DATA_PATH = "../results/experiments/test_3/keylog_data.csv"

def read_data(data_path):
    result = list()
    with open(data_path, 'r') as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                result.append(dict(row))
                line_count += 1
        return result


keystroke_data = list()


def write_result(output_path):
    columns = [x for x in keystroke_data[0].keys()]
    try:
        with open(output_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            for data in keystroke_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")


def get_related_keystroke(char):
    global keystroke_data
    last = None
    for i, k in enumerate(keystroke_data):
        if k['keytext'] == char:
            if k['keyrelease'] == 0:
                last = i
    # return last element that satisfy the criterion
    return last


def store_keystroke(event_type, data):
    global keystroke_data
    char, stroke_time = data
    char = str(char).replace("'", "")
    if event_type == "press":
        if len(keystroke_data) > 0:
            last_data = keystroke_data[-1]
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
        index = get_related_keystroke(char)
        if index is not None:
            data = keystroke_data[index]
            data.update({
                'keyrelease': stroke_time,
                'keyhold': abs(data['keypress'] - stroke_time),
            })
        else:
            raise ValueError


def on_press(key):
    stroke_time = epoch_to_millis(time.time())
    try:
        char = key.char
    except AttributeError:
        # special key (space, tab, etc.)
        char = key
    finally:
        store_keystroke("press", (char, stroke_time))


def on_release(key):
    stroke_time = epoch_to_millis(time.time())
    char = key
    store_keystroke("release", (char, stroke_time))
    if key == keyboard.Key.esc:
        print_info("Keylogger has been stopped")
        write_result(DATA_PATH)
        return False


def run(save_path, asynchronous=False):
    global DATA_PATH
    DATA_PATH = save_path

    print_info("Keylogger running in %s mode" % ("Asynchronous" if asynchronous else "Synchronous"))
    print_info("Press ESC to stop Keylogger")

    if asynchronous:
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()
    else:
        with keyboard.Listener(
                on_press=on_press,
                on_release=on_release) as listener:
            listener.join()
    
    return keystroke_data, DATA_PATH