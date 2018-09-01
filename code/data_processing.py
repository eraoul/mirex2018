
from collections import namedtuple
import copy
from enum import Enum
import json
import os

from keras.utils import to_categorical
import numpy as np
import pandas as pd


# # Utilities

# ## Constants

# In[2]:


SIG_DIGITS = 4
QUANTIZATION = 12  # smallest unit is 1/12 of a beat
MAX_EVENT_BEATS = 4

MIDI_MIN = 21
MIDI_MAX = 108

MAX_EVENT_SUBBEATS = QUANTIZATION * MAX_EVENT_BEATS

MIDI_LEN = MIDI_MAX - MIDI_MIN + 1  # 88 keys

# One-hot vector lengths
NUM_COMMAND_CLASSES = 3
NUM_MIDI_CLASSES = MIDI_LEN + 1                # + 1 for "0" case
NUM_DURATION_CLASSES = MAX_EVENT_SUBBEATS + 1  # + 1 for "0" case

# Start of range is inclusive, end of range is exclusive. 
COMMAND_VEC_RANGE = (0, NUM_COMMAND_CLASSES)
MIDI_VEC_RANGE = (COMMAND_VEC_RANGE[1], COMMAND_VEC_RANGE[1] + NUM_MIDI_CLASSES)
DURATION_VEC_RANGE = (MIDI_VEC_RANGE[1], MIDI_VEC_RANGE[1] + NUM_DURATION_CLASSES)
VEC_LENGTH = DURATION_VEC_RANGE[1]

INPUT_NOTES = 30
OUTPUT_NOTES = 10
SLIDING_WINDOW_NOTES = 5

INPUT_TIMESTEPS = 4 * INPUT_NOTES
OUTPUT_TIMESTEPS = 4 * OUTPUT_NOTES
SLIDING_WINDOW_TIMESTEPS = 4 * SLIDING_WINDOW_NOTES


## Key


class KeyQuality(Enum):
    Major = 0,
    Minor = 1


STR_TO_PITCH_CLASS = {
    'C': 0,
    'B#': 0,
    'C#': 1,
    'DB': 1,
    'D': 2,
    'D#': 3,
    'EB': 3,
    'E': 4,
    'FB': 4,
    'F': 5,
    'F#': 6,
    'GB': 6,
    'G': 7,
    'G#': 8,
    'AB': 8,
    'A': 9,
    'A#': 10,
    'BB': 10,
    'B': 11,
    'CB': 11
}

STR_TO_MODE = {
    'MAJOR': KeyQuality.Major,
    'MINOR': KeyQuality.Minor
}

def parse_key_string(str_key):
    """Returns a tuple of root pitch class (int, 0=C to 11=B) and KeyQuality."""
    tokens = str_key.strip().upper().split(' ')
    if len(tokens) != 2:
        raise Exception("Can't parse key: %s" % str_key)
    pitch_str, mode_str = tokens
    pc = STR_TO_PITCH_CLASS[pitch_str]
    mode = STR_TO_MODE[mode_str]
    return pc, mode    


# Time

def get_timeshift_set(score):
    s = set()
    for _, event_type, _, time_delta in score:
        if event_type == ScoreEventType.TimeShift:
            s.add(time_delta)
    return s


class Duration():
    def __init__(self, beats, subbeats, quantization=QUANTIZATION):
        self.beats = beats
        self.subbeats = subbeats
        self.quantization = quantization

    def total_beats(self):
        return self.beats + self.subbeats / self.quantization

    def total_subbeats(self):
        return self.beats * self.quantization + self.subbeats
    
    def subtract_subbeats(self, subbeats):
        """Modify the duration by subtracting the given # of subbeats."""
        # Find the total # of full beats to subtract.
        beats_to_subtract = subbeats // QUANTIZATION    # int part
        subbeats_to_subtract = subbeats % QUANTIZATION  # fractional part
        
        self.beats -= beats_to_subtract
        self.subbeats -= subbeats_to_subtract         # might be negative now
        
        # Borrow 1 from beats, just like in by-hand subtraction
        if self.subbeats < 0:
            self.subbeats += QUANTIZATION
            self.beats -= 1
        
    @staticmethod
    def MakeDuration(float_duration, quantization=QUANTIZATION):
        beats = int(float_duration)
        subbeats = round((float_duration % 1) * quantization)
        return Duration(beats, subbeats, quantization)
    
    def __repr__(self):
        return 'Duration(%d,%d,%d)' % (self.beats, self.subbeats, self.quantization)


# Score Types

class ScoreEventType(Enum):
    Undefined = -1
    NoteOff = 0
    NoteOn = 1
    TimeShift = 2
    
# In neural net, representation will be: [COMMAND, OPT_MIDI, OPT_TIME_DELTA]
# That is, 3 separate one-hot-encoding blocks, with N/A value possible for MIDI.
# ex:
# [NoteOn, MIDI, 0], [NoteOff, MIDI, 0], [TimeShift, N/A, time_delta]

# MIDI is an int, 0 for N/A or 1-88 representing piano keys 21 to 108
# time_delta will be quantized 1/12 of quarter notes. Duration can be from 0-48 12/th notes.
#    Longer durations will be represented with multiple successive time_shift commands.
    
# time is a float, mostly for output use; not fed into neural net.
# time_delta is a quantized Duration object.

ScoreEvent = namedtuple('ScoreEvent', ['time', 'type', 'midi', 'time_delta'])


def convert_df_to_score(df): # Monophonic
    """The df is an input csv with time, midi, and duration.
    Output is a list of ScoreEvents."""
    
    # Fix durations that are too long and overlap later notes. This is supposed to be
    # monophonic data, but it isn't always.
    for i, row in df.iterrows():
        if i >= len(df) - 1: # skip last row
            break
        time = row['time']
        next_time = df.loc[i+1, 'time']
        if time + row['duration'] > next_time:
            df.loc[i, 'duration']= next_time - time
        
    events = []
    prev_time = None
    for _, row in df.iterrows():
        time = round(row['time'], SIG_DIGITS)
        midi = int(row['midi'])
        duration = round(row['duration'], SIG_DIGITS)
        
        if prev_time is None:
            prev_time = time
        
        delta_time = round(time - prev_time, SIG_DIGITS)
        
        # Quantize time to beats and subbeats.
        time_quantized = Duration.MakeDuration(time)
        duration_quantized = Duration.MakeDuration(duration)
        delta_time_quantized = Duration.MakeDuration(delta_time)
        
        # Create events.
        #if delta_time_quantized.total_subbeats() > 0:
        # TimeShift.
        events.append(ScoreEvent(prev_time, ScoreEventType.TimeShift, None, delta_time_quantized))
        
        # NoteOn
        events.append(ScoreEvent(time, ScoreEventType.NoteOn, midi, None))
        
        # TimeShift for duration of note
        # DOESN'T WORK if duration is too long for next note. 
        # Preprocessing above fixes durations.
        events.append(ScoreEvent(time, ScoreEventType.TimeShift, None, duration_quantized))
        time += duration
        
        # NoteOff
        events.append(ScoreEvent(time, ScoreEventType.NoteOff, midi, None))
        
        prev_time = time
    
    return events            


# In[10]:


def score_event_to_nnet_input_list(ev):
    # Output is a list of numpy arrays: 3 one-hot encodings, concatenated.
    # List will usually be 1-element long, but must be multiple for long durations.
    # numpy array has 3 one-hot sections (command, midi, duration)
    # All 3 are concatenated together into ndarray: [command...midi...duration]
    
    _, event_type, midi, time_delta_orig = ev
    time_delta = copy.copy(time_delta_orig)
    #print(ev)
    result = []
    
    command_vec = to_categorical(event_type.value, num_classes=NUM_COMMAND_CLASSES)
    
    if not midi:
        midi = 0
    else:
        midi = midi - MIDI_MIN + 1
    
    midi_vec = to_categorical(midi, num_classes=NUM_MIDI_CLASSES)  # +1 for the 0 case
    
    if not time_delta:
        time_delta = Duration(0, 0)
    
    # Handle long duration events: repeat them.
    while time_delta.total_subbeats() > MAX_EVENT_SUBBEATS:
        duration_vec = to_categorical(MAX_EVENT_SUBBEATS, num_classes=NUM_DURATION_CLASSES)
        total_vec = np.concatenate((command_vec, midi_vec, duration_vec))
        result.append(total_vec)
        time_delta.subtract_subbeats(MAX_EVENT_SUBBEATS)
        
    # Regular duration event.
    duration_vec = to_categorical(time_delta.total_subbeats(), num_classes=NUM_DURATION_CLASSES)
    result.append(np.concatenate((command_vec, midi_vec, duration_vec)))
    
    return result


def score_to_array(score):
    arrays_list = []
    for ev in score:
        arrays_list.append(np.array(score_event_to_nnet_input_list(ev)))
    return np.vstack(arrays_list)

# MIDI

def get_midi_set(score):
    s = set()
    for _, event_type, midi, _ in score:
        if event_type == ScoreEventType.NoteOn:
            s.add(midi)
    return s


# File parsing

size = 'large'   # 'small', medium', 'large'
DATA_PATH = 'data/PPDD-Jul2018_sym_mono_%s/PPDD-Jul2018_sym_mono_%s' % (size, size)


DESCRIPTOR_PATH = os.path.join(DATA_PATH, 'descriptor')       # .json files
PRIME_CSV_PATH = os.path.join(DATA_PATH, 'prime_csv')         # .csv
CONT_FOIL_CSV_PATH = os.path.join(DATA_PATH, 'cont_foil_csv') # .csv
CONT_TRUE_CSV_PATH = os.path.join(DATA_PATH, 'cont_true_csv') # .csv


def get_file_list(path, ext):
    """Returns a list of all files in the given path with the given string ending 'ext'."""
    return [f for f in os.listdir(path) if 
            os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]

#descriptor_files = get_file_list(DESCRIPTOR_PATH, 'json')


# Read Dataset

all_scores = []

def get_score_end_time(score):
    """Return the timestamp just after the final note in the score ends."""
    if not score:
        return 0
    last_event = score[-1]

    if last_event.time_delta is None:
        return last_event.time

    return last_event.time + last_event.time_delta.total_beats()

def read_input_file(filepath):
    """Reads a CSV of note events.

    Output is a tuple of
    1) matrix ready as input for neural net
    2) end time of score (float)
    3) channel #
    """
    df = pd.read_csv(filepath,
                     header=None,
                     names=['time', 'midi', 'mpn', 'duration', 'channel'])
    score = convert_df_to_score(df)
    end_time = get_score_end_time(score)
    channel = int(df['channel'][0])
    return score_to_array(score), end_time, channel


def preprocess_data(data, desired_length):
    """Forces the data to be the desired length.

    Input is a 2D matrix of [timestep, features].
    desired_length is in the first (time) dimension.

    Output is another 2D matrix.
    """
    if len(data) < 1:
        raise Exception('Input score is length 0.')
    if len(data) == desired_length:
        return data

    # Copy/paste to left to pad short input sequences if necessary
    # First, make extra copies of content until long enough.
    x = data
    while len(x) < desired_length:
        x = np.vstack((x, data))

    # Cut extra length at start if necessary.
    if len(x) > desired_length:
        x = x[len(x) - desired_length:]

    return x


def read_dataset():
    # all_durations = set()
    #all_midi = set()

    for i, json_file in enumerate(descriptor_files):
        if i % 1000 == 0:
            print(i)
        guid = os.path.splitext(json_file)[0]
        #print('Processing file %d: %s' % (i, guid))
        prime_csv_file = os.path.join(PRIME_CSV_PATH, guid) + '.csv'

        df_prime = pd.read_csv(prime_csv_file, header=None, 
                               names=['time', 'midi', 'mpn', 'duration', 'channel'])
        score = convert_df_to_score(df_prime)
        all_scores.append(score)

        # Used this code to analyze dataset.
        # Result: quantization is to the 1/12 of a beat.
        #all_durations.update(get_timeshift_set(score))

        # Result: MIDI notes from 21-108 in use
        #all_midi.update(get_midi_set(score))

        with open(os.path.join(DESCRIPTOR_PATH, descriptor_files[i])) as f:
            j = json.load(f)
            id_lakh = j['idLakh']
            bpm = j['bpm']
            time_sig_numerator, time_sig_denominator = j.get('timeSignature', [4, 4])
            key_estimate = j['keyEstimate']

            #print (guid, id_lakh, bpm, time_sig_numerator, time_sig_denominator, key_estimate)
    print('Done!')
    return [score_to_array(s) for s in all_scores]

# Training Examples


def get_example(matrix, start_row, input_timesteps=INPUT_TIMESTEPS, output_timesteps=OUTPUT_TIMESTEPS):
    """Returns a pair of input, output ndarrays. Input starts at start_row and has the given input length.
    Output starts at next timestep and has the given output length."""
    # Make sure there are enough time steps remaining.
    if len(matrix) < start_row + input_timesteps + output_timesteps:
        raise Exception('Not enough rows to get example.')
    input_ex = matrix[start_row : start_row + input_timesteps]
    output_ex = matrix[start_row + input_timesteps : start_row+input_timesteps+output_timesteps]
    return (input_ex, output_ex)


def get_examples_for_song(matrix, input_timesteps=INPUT_TIMESTEPS, output_timesteps=OUTPUT_TIMESTEPS, 
                          sliding_window_hop_length=SLIDING_WINDOW_TIMESTEPS):
    """Returns tuple of (array of input examples, array of output examples), generated via a sliding window on
    the input matrix. For example: output might be two ndarrays of shapes (277, 120, 141), (277, 40, 141).
    Dimensions are (examples, timesteps, features)
    """
    input_examples = []
    output_examples = []
    for i in range(len(matrix) - input_timesteps - output_timesteps + 1):
        x, y = get_example(matrix, i, input_timesteps, output_timesteps)
        input_examples.append(x)
        output_examples.append(y)
    if not input_examples:
        return None
    return np.stack(input_examples), np.stack(output_examples)
