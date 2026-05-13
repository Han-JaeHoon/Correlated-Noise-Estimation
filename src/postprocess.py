# src/postprocess.py

import numpy as np


def reshape_mid_measure_syndrome(raw, n_rounds, shots=1, n_stabilizers=8):
    """
    Normalize QNode output from mid_measure=True mode.

    Expected logical output:
        shape = (shots, n_rounds, n_stabilizers)

    PennyLane can return, for shots=1:
        [array(0), array(1), ...]

    For shots>1:
        [array([...]), array([...]), ...]

    This function handles both.
    """

    if shots == 1:
        flat = np.array([int(np.asarray(x)) for x in raw], dtype=int)
        return flat.reshape(1, n_rounds, n_stabilizers)

    raw_array = np.array(raw)
    raw_array = raw_array.reshape(n_rounds, n_stabilizers, shots)
    raw_array = np.moveaxis(raw_array, 2, 0)

    return raw_array.astype(int)


def compute_detection_events(syndrome):
    """
    Compute detection events.

    detection[t] = syndrome[t+1] XOR syndrome[t]

    Input:
        syndrome shape = (shots, n_rounds, n_stabilizers)

    Output:
        shape = (shots, n_rounds - 1, n_stabilizers)
    """

    return syndrome[:, 1:, :] ^ syndrome[:, :-1, :]


def normalize_final_sample(raw):
    """
    Normalize QNode output from mid_measure=False, return_probs=False mode.

    For shots=1:
        output is usually shape (n_anc,)

    For shots>1:
        output is usually shape (shots, n_anc)
    """

    return np.array(raw).astype(int)


def normalize_final_probs(raw):
    """
    Normalize QNode output from mid_measure=False, return_probs=True mode.
    """

    return np.array(raw, dtype=float)


def flatten_bits(array):
    """
    Flatten a binary array and convert to a string.

    Example:
        [[1,0,0], [0,1,0]]
        -> "100010"
    """

    arr = np.array(array).astype(int).flatten()
    return "".join(str(int(x)) for x in arr)


def comma_join(xs):
    """
    Convert a list-like object to comma-separated string.
    """

    return ",".join(str(x) for x in xs)


def print_single_shot_sequence(syndrome, detection_events, shot_idx=0):
    """
    Print one shot's syndrome and detection sequence in readable form.
    """

    print("Syndrome sequence")
    print("-----------------")

    for t in range(syndrome.shape[1]):
        z_part = syndrome[shot_idx, t, :4]
        x_part = syndrome[shot_idx, t, 4:]

        print(f"round {t}: Z={z_part}  X={x_part}")

    print()
    print("Detection events")
    print("----------------")

    for t in range(detection_events.shape[1]):
        z_part = detection_events[shot_idx, t, :4]
        x_part = detection_events[shot_idx, t, 4:]

        print(f"between round {t} and {t + 1}: Z={z_part}  X={x_part}")