import argparse
from functools import partial
import glob
import os

import keras
from keras.layers import Input
from keras.models import Model
import numpy as np

from data_processing import read_input_file, preprocess_data


NUM_LSTM_NODES = 1024             # Num of intermediate LSTM nodes

SIG_FIGS = 5
SIG_DIGITS = 4
QUANTIZATION = 12  # smallest unit is 1/12 of a beat
MAX_EVENT_BEATS = 4

MIDI_MIN = 21
MIDI_MAX = 108

MAX_EVENT_SUBBEATS = QUANTIZATION * MAX_EVENT_BEATS

MIDI_LEN = MIDI_MAX - MIDI_MIN + 1  # 88 keys

# Adds a start command node for the Y data in timestep 0, to start off decoder. Also expands feature vector length.
ADD_START = True  # Set to False for Attention, True for basic seq2seq
USE_TEACHER_FORCING = True  # False for Attention, True for basic seq2seq

# One-hot vector lengths
NUM_COMMAND_CLASSES = 3
if ADD_START:
    NUM_COMMAND_CLASSES += 1

NUM_MIDI_CLASSES = MIDI_LEN + 1  # + 1 for "0" case
NUM_DURATION_CLASSES = MAX_EVENT_SUBBEATS + 1  # + 1 for "0" case

# Start of range is inclusive, end of range is exclusive.
COMMAND_VEC_RANGE = (0, NUM_COMMAND_CLASSES)
MIDI_VEC_RANGE = (
COMMAND_VEC_RANGE[1], COMMAND_VEC_RANGE[1] + NUM_MIDI_CLASSES)
DURATION_VEC_RANGE = (
MIDI_VEC_RANGE[1], MIDI_VEC_RANGE[1] + NUM_DURATION_CLASSES)
VEC_LENGTH = DURATION_VEC_RANGE[1]

# Y values have an extra "start" command option. X values don't.
VEC_LENGTH_X = VEC_LENGTH
if ADD_START:
    VEC_LENGTH_X -= 1

INPUT_NOTES = 30
OUTPUT_NOTES = 10   # Used for training

MAX_OUTPUT_STEPS = 100
OUTPUT_BEATS = 10  # Used for generation


INPUT_TIMESTEPS = 4 * INPUT_NOTES
OUTPUT_TIMESTEPS = 4 * OUTPUT_NOTES

# y values have an extra timestep if adding start symbol
if ADD_START:
    OUTPUT_TIMESTEPS += 1

pc_to_degree_flat_key = [0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6]
pc_to_degree_sharp_key = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]



# Load from disk; reconstruct inference model
# Returns a function that performs inference on a given sequence.
def load_model():
    print('Loading model...')
    model = keras.models.load_model('s2s_mono_continuation.h5')
    print('Reconstructing model architecture:')
    encoder_inputs = model.get_layer(name='encoder_input').input
    encoder_lstm_1 = model.get_layer(name='encoder_lstm_1')
    encoder_states1 = encoder_lstm_1.output[1:3]
    encoder_lstm_2 = model.get_layer(name='encoder_lstm_2')
    encoder_states2 = encoder_lstm_2.output[1:3]

    decoder_inputs = model.get_layer(name='decoder_lstm_1').input[0]

    decoder_lstm1 = model.get_layer(name='decoder_lstm_1')
    decoder_lstm2 = model.get_layer(name='decoder_lstm_2')
    decoder_dense1 = model.get_layer(name='decoder_output_command')
    decoder_dense2 = model.get_layer(name='decoder_output_midi')
    decoder_dense3 = model.get_layer(name='decoder_output_duration')

    # Run Model

    # Define sampling models
    encoder_model1 = Model(encoder_inputs, encoder_states1)

    encoder_model2 = Model(encoder_inputs, encoder_states2)
    encoder_model2.summary()

    decoder_state_input_h1 = Input(shape=(NUM_LSTM_NODES,),
                                   name='inference_decoder_h1')
    decoder_state_input_c1 = Input(shape=(NUM_LSTM_NODES,),
                                   name='inference_decoder_c1')
    decoder_state_input_h2 = Input(shape=(NUM_LSTM_NODES,),
                                   name='inference_decoder_h2')
    decoder_state_input_c2 = Input(shape=(NUM_LSTM_NODES,),
                                   name='inference_decoder_c2')
    decoder_states_inputs1 = [decoder_state_input_h1, decoder_state_input_c1]
    decoder_states_inputs2 = [decoder_state_input_h2, decoder_state_input_c2]

    zz1 = decoder_lstm1(decoder_inputs, initial_state=decoder_states_inputs1)
    decoder_outputs1new, decoder_state_h1, decoder_state_c1 = zz1

    zz2 = decoder_lstm2(decoder_outputs1new, initial_state=decoder_states_inputs2)

    decoder_outputs2new, decoder_state_h2, decoder_state_c2 = zz2

    decoder_states1 = [decoder_state_h1, decoder_state_c1]
    decoder_states2 = [decoder_state_h2, decoder_state_c2]
    decoder_outputs_final1 = decoder_dense1(decoder_outputs2new)
    decoder_outputs_final2 = decoder_dense2(decoder_outputs2new)
    decoder_outputs_final3 = decoder_dense3(decoder_outputs2new)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs1 + decoder_states_inputs2,
        [decoder_outputs_final1, decoder_outputs_final2,
         decoder_outputs_final3] + decoder_states1 + decoder_states2)

    decoder_model.summary()

    def seq2seq_from_models(encoder_model1, encoder_model2, decoder_model,
                            input_seq):
        # Encode the input as state vectors.
        h1, c1 = encoder_model1.predict(input_seq)
        states_value1 = [h1, c1]
        h2, c2 = encoder_model2.predict(input_seq)
        states_value2 = [h2, c2]

        # Generate first input: Start vector.
        target_seq = np.zeros((1, VEC_LENGTH))
        target_seq[0, 0] = 1  # first element is "1" to indicate "start"

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        output_sequence = []
        step = 0
        #while step < OUTPUT_TIMESTEPS:
        total_time = 0
        while step < MAX_OUTPUT_STEPS and total_time < OUTPUT_BEATS * QUANTIZATION:
            z = decoder_model.predict(
                [np.expand_dims(target_seq, 0)] + states_value1 + states_value2)
            out_vec1, out_vec2, out_vec3, h1, c1, h2, c2 = z

            sampled_command = np.argmax(out_vec1[0, 0, :])
            sampled_midi = np.argmax(out_vec2[0, 0, :])
            sampled_dur = np.argmax(out_vec3[0, 0, :])

            # print(sampled_command, sampled_midi, sampled_dur)
            output_sequence.append((sampled_command, sampled_midi, sampled_dur))
            step += 1
            total_time += sampled_dur

            # Exit condition: either hit max length
            # or find stop character.
            # if (sampled_word == '</S>' or step > max_output_seq_len):
            #    stop_condition = True

            # Update the target sequence (of length 1).

            target_seq = np.zeros((1, VEC_LENGTH))
            target_seq[0, sampled_command] = 1
            target_seq[0, MIDI_VEC_RANGE[0] + sampled_midi] = 1
            target_seq[0, DURATION_VEC_RANGE[0] + sampled_dur] = 1

            # Update states
            states_value1 = [h1, c1]
            states_value2 = [h2, c2]

        return output_sequence

    return partial(seq2seq_from_models, encoder_model1, encoder_model2,
                   decoder_model)


# Convert Output to CSV and MIDI

def midi_to_mnn(midi, flat_key=True):
    octave = 3 + midi // 12
    pc = midi % 12
    pc_to_degree = pc_to_degree_flat_key if flat_key else pc_to_degree_sharp_key
    degree = pc_to_degree[pc]
    return octave * 7 + degree + 4


def seq_to_tuples(seq, start_time=100, channel=0, flat_key=True):
    t = round(start_time, SIG_DIGITS)  # time in beats
    subbeat = 0  # curent subbeat in beat for t, range is 0 to QUANTIZATION-1

    notes = []

    cur_note_start = 0
    cur_note = None
    cur_dur = None

    for command, midi, dur in seq:
        mnn = midi_to_mnn(midi, flat_key)
        # Time-shift
        if command == 3:
            # Record note/rest start data.
            cur_dur = dur
            cur_note_start = round(t + subbeat / QUANTIZATION, SIG_DIGITS)

            # Update current time.
            subbeat += dur
            if subbeat > QUANTIZATION:
                subbeat = dur % QUANTIZATION
                t += dur // QUANTIZATION + 1

        # Note on.
        elif command == 2:
            if cur_note:
                notes.append((cur_note_start, midi, mnn,
                              round(cur_dur / QUANTIZATION, SIG_DIGITS),
                              channel))
            cur_note = midi + MIDI_MIN - 1  # -1 for the 0 case

        # Note off.
        elif command == 1:
            if cur_note:
                notes.append((cur_note_start, midi, mnn,
                              round(cur_dur / QUANTIZATION, SIG_DIGITS),
                              channel))
            cur_note = 0
    return notes


def print_tuples(tuples):
    for t in tuples:
        print(t)


def target_outputs_to_seq(N, target1, target2, target3):
    return list(zip(np.argmax(target1[N], axis=1),
                    np.argmax(target2[N], axis=1),
                    np.argmax(target3[N], axis=1)))


def target_outputs_to_tuples(N, target1, target2, target3):
    return seq_to_tuples(target_outputs_to_seq(N, target1, target2, target3))


def seq_to_csv(seq, filename='tst.csv', start_time=100, channel=0):
    with open(filename, 'w') as f:
        f.writelines(
            ','.join(str(x) for x in tup) + '\n'
            for tup in seq_to_tuples(seq,
                                     start_time=start_time,
                                     channel=channel))


def predict_for_files(seq2seq, input_generator):
    ex = 0
    for [x, y], [target1, target2, target3] in validation_generator:
        for i in range(len(x)):
            if ex % 100 == 0:
                print(ex)
            seq = seq2seq(np.expand_dims(x[i], axis=0))
            seq_to_csv(seq,
                       filename='outputs/validation_output_%08d.csv' % ex,
                       start_time=100)
            ex += 1


if __name__ == '__main__':
    print('--------------------------------------------------')
    print('MIREX 2018: Patterns for Prediction (Eric Nichols)')
    print('--------------------------------------------------')

    parser = argparse.ArgumentParser(
        description='Generate continuations for symbolic monophonic music.')
    parser.add_argument('--input', '-i', metavar='INPUT_PATH', nargs=1,
                        help='Path to the input file directory')
    parser.add_argument('--output', '-o', metavar='OUTPUT_PATH', nargs=1,
                        help='Path to the output file directory')

    args = parser.parse_args()

    input_files = glob.glob(os.path.join(args.input[0], '*.csv'))
    output_path = args.output[0]

    num_input_files = len(input_files)
    print('Found %d files' % num_input_files)
    print('Writing output to %s\n' % output_path)

    seq2seq = load_model()

    # Process each file.
    print('Processing...')
    num_success = 0
    for i, filename in enumerate(input_files):
        if i and i % 10 == 0:
            print('Processed %d of %d' % (i, num_input_files))

        try:
            # Load data
            data, end_time, channel = read_input_file(filename)
            data_processed = preprocess_data(data, INPUT_TIMESTEPS)

            # Predict.
            seq = seq2seq(np.expand_dims(data_processed, axis=0))

            # Write to disk.
            filebase, _ = os.path.splitext(os.path.basename(filename))
            seq_to_csv(seq,
                       os.path.join(output_path, filebase + '_continued.csv'),
                       start_time=end_time,
                       channel=channel)
            num_success += 1

        except Exception as e:
            print('ERROR: Problem processing file %s.' % filename)
            print('Exception: %s' % str(e))
            print('Continuing...\n')
    print('Done! Wrote %d files to %s' % (num_success, output_path))



