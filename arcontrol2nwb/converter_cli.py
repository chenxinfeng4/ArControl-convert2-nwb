# python BF_arc2nwb.py 2022-1113-224711.txt
import sys
import os
import re
import warnings

import numpy as np
from datetime import datetime
try:
    from ndx_beadl import (
        Task,
        TaskProgram,
        TaskSchema,
        EventTypesTable,
        EventsTable,
        StateTypesTable,
        StatesTable,
        # TrialsTable,
        ActionTypesTable,
        ActionsTable,
        TaskArgumentsTable)
    NDX_BEADL_AVAILABLE = True
except ImportError:
    NDX_BEADL_AVAILABLE = False
    warnings.warn("ndx-beadl not installed.")
from pynwb.epoch import TimeIntervals
from pynwb import (
    NWBFile,
    TimeSeries,
    NWBHDF5IO)
from pynwb.behavior import BehavioralEvents

"""
Convert arcontrol_data.TXT to arcontrol_data.MAT. Just like "BF_arc2mat.m".
Xinfeng Chen, 2020-3-2
$ pip install mat4py
$ pyinstaller BF_arcmat.py
"""


def parse(arc_data_filename: str):
    """
    Convert arcontrol_data.TXT to MAT-style dictionary. Just like "BF_arc2nwb.m"

    :param arc_data_filename: Name of the arcontrol output txt data file

    :return: Dict containing: 1) 'info' key with a dict of all the states and events
             and their corresponding description, 2) keys of all the events an states
             with the timing arrays .
    """

    # parse the header with the state and event definitions
    expression_header = re.compile('^@(IN\d+|OUT\d+|C\d+|C\d+S\d+):(.*)$')
    expression_taskname = re.compile('^-----(\w+)-----$')
    expression_arcbg = re.compile(r'^ArC-bg$')
    MAT = {}
    MAT['info'] = {}
    isokfile = False
    for str in open(arc_data_filename):
        res_header = re.findall(expression_header, str)
        res_taskname = re.findall(expression_taskname, str)
        res_arcbg = re.findall(expression_arcbg, str)
        if res_header:
            style, comment = res_header[0]
            MAT['info'][style] = comment
        elif res_taskname:
            MAT['info']['task'] = res_taskname[0]
        elif res_arcbg:
            isokfile = True
            break
    assert isokfile,  "It's NOT a data file from ArControl!"

    # parse the data with the event timings
    expression = re.compile('^(IN\d+|OUT\d+|C\d+S\d+):(\w.*)$')
    for str in open(arc_data_filename):
        res_expression = re.findall(expression, str)
        if res_expression:
            style, nums = res_expression[0]
            nums_list = eval('[' + nums.replace(' ', ', ') + ']')
            MAT.setdefault(style, []).append(nums_list)

    return MAT


def arc2dict(arc_data_filename: str,
             arc_taskprogram_filename: str = None):
    """
    Convert arcontrol_data.TXT to a data dictionary.

    In contrast to the parse function, the dict is further processed via the
    __cs_append_duration and __c_create functions and the session_start_time is added.

    :param arc_data_filename: Name of the arcontrol output txt data file
    :param arc_taskprogram_filename: Path to the ARControl task program (Optional)

    :return: Dict containing: 1) 'info' key with a dict of all the states and events
             and their corresponding description, 2) keys of all the events an states
             with the timing arrays, 3) 'session_start_time' with the datetime of the session start.
    """
    # read raw from txt
    MAT = parse(arc_data_filename)

    # session start time
    MAT['session_start_time'] = get_session_start_time(arc_data_filename=arc_data_filename)

    # Add the task program
    MAT['info']['task_program'] = None
    if arc_taskprogram_filename is not None:
        with open(arc_taskprogram_filename) as f:
            MAT['info']['task_program'] = f.read()

    # Add the task schema
    MAT['info']['task_schema'] = None   # TODO add the task schema

    # TODO parse trial structure and add to the MAT dict if possible

    # append duration to CxSx, create component records
    __cs_append_duration(MAT)
    __c_create(MAT)

    return MAT


def get_session_start_time(arc_data_filename: str):
    """
    Get the session start time based on the name of arccontrol data file

    :param arc_data_filename:
    :return: datatime object with the session_start_time
    """
    filetime_str = os.path.splitext(os.path.basename(arc_data_filename))[0]
    return datetime.strptime(filetime_str, "%Y-%m%d-%H%M%S").astimezone()


def convert(
        arc_data_filename: str,
        arc_taskprogram_filename: str = None,
        nwb_filename: str = None,
        append_to_nwb_file: bool = False,
        use_behavioral_time_series: bool = not NDX_BEADL_AVAILABLE,
        use_ndx_beadl: bool = NDX_BEADL_AVAILABLE):
    """
    Convert an ARControl recording to NWB

    :param arc_data_filename: Path to the  arcontrol_data.TXT data file
    :param arc_taskprogram_filename: Path to the ARControl task program
    :param nwb_filename: Name of the NWB file to write. If None, then the NWB file will be named
                     according to the arc_data_filename, i.e, arc_data_filename.replace('txt', 'nwb').
    :param append_to_nwb_file: If the NWB file exists, should we append to it (True) or overwrite the file (False)
    :param use_behavioral_time_series: Boolean indicating whether to use behavioral timeseries to store the data
    :param use_ndx_beadl: Boolean indicating whether to use the ndx-beadl extension in NWB

    :return: Dictionary generated by the parse function
    """
    MAT = arc2dict(arc_data_filename=arc_data_filename,
                   arc_taskprogram_filename=arc_taskprogram_filename)

    # save to file #
    savenwb(nwb_filename=nwb_filename if nwb_filename is not None else os.path.splitext(arc_data_filename)[0] + '.nwb',
            append_to_nwb_file=append_to_nwb_file,
            MAT=MAT,
            use_behavioral_time_series=use_behavioral_time_series,
            use_ndx_beadl=use_ndx_beadl)

    return MAT


def __cs_append_duration(MAT: dict):
    """
    Process the MAT dict produced by the parse function to append control state durations

    :param MAT: Data dictionary produced by the parse function
    :return: MAT dictionary with the updated data
    """
    CS_pattern = re.compile('^C\d+S\d+$')
    CS_event_mm = {CS: np.array(v) for CS, v in MAT.items()
                   if CS_pattern.match(CS)}

    T_seq = np.squeeze(np.concatenate([v for v in CS_event_mm.values()]))
    assert T_seq.ndim == 1
    T_seq.sort()
    T_seq_ext = np.append(T_seq, T_seq[-1])
    CS_end_mm = {e: T_seq_ext[np.searchsorted(T_seq, v.flatten(), 'right')]
                 for e, v in CS_event_mm.items()}
    csdata_dict = dict()
    for e in CS_event_mm:
        t_bg = CS_event_mm[e][:, 0]
        t_end = CS_end_mm[e]
        t_dur = t_end - t_bg
        csdata_dict[e] = np.stack([t_bg, t_dur]).T

    MAT.update(csdata_dict)
    MAT['info'].setdefault('C0S0', 'End session')


def __c_create(MAT: dict):
    """

    :param MAT: Data dictionary produced by the parse function
    :return: MAT dictionary with the updated data
    """
    CS_pattern = re.compile('^C\d+S\d+$')
    CS_event_mm = {CS: np.array(v) for CS, v in MAT.items()
                   if CS_pattern.match(CS)}

    T_seq = []
    c_seq = []
    for e, v in CS_event_mm.items():
        T_seq.extend(v[:, 0].tolist())
        c_seq.extend([e]*len(v[:, 0]))
    T_seq = np.array(T_seq)
    c_seq = np.array(c_seq)
    argsort_ind = np.argsort(T_seq)
    T_seq = T_seq[argsort_ind]
    c_seq = c_seq[argsort_ind]
    comp_name_l, comp_switch_l = [], []
    comp_name_now = ''
    for cs1_name, T in zip(c_seq, T_seq):
        if cs1_name.split('S')[1] != '1' and cs1_name != 'C0S0':
            continue
        if cs1_name != comp_name_now:
            comp_name_l.append(cs1_name)
            comp_switch_l.append(T)
            comp_name_now = cs1_name

    comp_name_l = np.array(comp_name_l)
    comp_bg_l = comp_switch_l
    comp_end_l = [*comp_switch_l[1:], T_seq[-1]]
    comp_dur_l = [end-bg for bg, end in zip(comp_bg_l, comp_end_l)]
    comp_bg_dur_l = np.stack([comp_bg_l, comp_dur_l]).T

    cdata_dict = dict()
    for cs1_name in comp_name_l:
        comp_name = cs1_name.split('S')[0]
        comp_bg_dur = comp_bg_dur_l[comp_name_l == cs1_name]
        cdata_dict[comp_name] = comp_bg_dur

    MAT.update(cdata_dict)
    MAT['info'].setdefault('C0', 'End session')


def add_arc_to_nwbfile(
        MAT: dict,
        nwbfile: NWBFile,
        use_behavioral_time_series: bool = not NDX_BEADL_AVAILABLE,
        use_ndx_beadl: bool = NDX_BEADL_AVAILABLE
):
    """
    Add data to an in-memory NWBFile object. The input nwbfile object is modified but
    writing the data to disk is left (if desired) is left to the caller.

    :param MAT: Dictionary generated by arc2dict function
    :param nwbfile: The NWBFile object to add the data to
    :param use_behavioral_time_series: Boolean indicating whether to use behavioral timeseries to store the data
    :param use_ndx_beadl: Boolean indicating whether to use the ndx-beadl extension

    :return: The modified nwbfile object
    """
    if use_ndx_beadl and not NDX_BEADL_AVAILABLE:
        raise ValueError("The ndx-beadl extensions is not available. "
                         "Install the extension and try again or set use_ndx_beadl=False")
    if not use_ndx_beadl and not use_behavioral_time_series:
        raise ValueError("Either use_ndx_beadl and/or use_behavioral_time_series must be set to True")

    session_start_time = MAT['session_start_time']
    task_name = MAT['info']['task']  # noqa  TODO where to save task_name if NWBFile exists (append to session description?)
    task_program = MAT['info']['task_program']
    task_schema = MAT['info']['task_schema']

    io_in_events = {IO: np.array(v) / 1000 for IO, v in MAT.items()
                    if 'IN' in IO}  # process input IO time as sec
    io_out_actions = {IO: np.array(v) / 1000 for IO, v in MAT.items()
                      if 'OUT' in IO}  # process output IO time as sec
    CS_pattern = re.compile('(^C\d+S\d+$)|(^C\d+$)')
    cs_events = {CS: np.array(v) / 1000 for CS, v in MAT.items()
                 if CS_pattern.match(CS)}  # process CS time as sec
    state_types = {k: v for k, v in MAT['info'].items() if CS_pattern.match(k)}
    event_types = {k: v for k, v in MAT['info'].items() if 'IN' in k}
    action_types = {k: v for k, v in MAT['info'].items() if 'OUT' in k}

    # Determine the time_offset to use relative to the timestamp_reference time of the NWBFile
    # If the reference time of the NWB file is different than the session start time of
    # the ARControl file then we need to adjust timestamps accordingly
    time_offset = (session_start_time - nwbfile.timestamps_reference_time).total_seconds()

    if use_behavioral_time_series:
        __add_arc_to_nwbfile_behavioral_series(
            nwbfile=nwbfile,
            io_in_events=io_in_events,
            io_out_actions=io_out_actions,
            cs_events=cs_events,
            info=MAT['info'],
            time_offset=time_offset,
        )

    if use_ndx_beadl:
        __add_arc_to_nwbfile_ndx_beadl(
            nwbfile=nwbfile,
            task_schema=task_schema,
            task_program=task_program,
            state_types=state_types,
            event_types=event_types,
            action_types=action_types,
            io_in_events=io_in_events,
            io_out_actions=io_out_actions,
            cs_events=cs_events,
            time_offset=time_offset
        )

    return nwbfile


def __add_arc_to_nwbfile_behavioral_series(nwbfile: NWBFile,
                                           io_in_events: dict,
                                           io_out_actions: dict,
                                           cs_events: dict,
                                           info: dict,
                                           time_offset: float):
    """
    Add the data to the NWBFile using BehavioralTimeSeries and TimeIntervals
    """
    world_event = (io_in_events | io_out_actions | cs_events)
    assert world_event.keys() <= info.keys()

    time_len = max([v[-1, 0] + v[-1, 1] for v in world_event.values()])

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Raw ArControl event"
    )
    behavioral_events = BehavioralEvents(name="BehavioralEvents")

    s1 = TimeIntervals(
        name="ArControl_Events",
        description="intervals for each event",
    )
    s1.add_column(name="event", description="I/O and State events")
    for e, v in world_event.items():
        for start_t, dur_t in v:
            s1.add_row(start_time=start_t + time_offset,
                       stop_time=start_t + time_offset + dur_t,
                       event=e)
    _ = nwbfile.add_time_intervals(s1)

    for e, v in world_event.items():
        tbg_pre, tend = v[:, 0], v[:, 0] + v[:, 1]
        ddt = 0.0001
        tend[tend == tbg_pre] += ddt
        tbg = tbg_pre + ddt
        tend_post = tend + ddt
        seq_t_1 = np.concatenate((tbg, tend))
        seq_t_0 = np.concatenate((tbg_pre, tend_post))
        seq_v_1 = np.ones(seq_t_1.shape)
        seq_v_0 = np.zeros(seq_t_0.shape)
        seq_t = np.concatenate((seq_t_0, seq_t_1, [time_len + ddt]))
        seq_v = np.concatenate((seq_v_0, seq_v_1, [0]))
        if not np.any(seq_t == 0.0):
            seq_t = np.append(seq_t, 0.0)
            seq_v = np.append(seq_v, 0)

        argind = np.argsort(seq_t)
        timestamps = seq_t[argind] + time_offset
        data = seq_v[argind]

        time_series = TimeSeries(
            name=e,
            data=data,
            timestamps=timestamps,
            comments=info[e],
            description=info[e],
            unit="TTL",
        )
        behavioral_events.add_timeseries(time_series)

    behavior_module.add(behavioral_events)


def __add_arc_to_nwbfile_ndx_beadl(nwbfile: NWBFile,
                                   task_schema: str,
                                   task_program: str,
                                   state_types: dict,
                                   event_types: dict,
                                   action_types: dict,
                                   io_in_events: dict,
                                   io_out_actions: dict,
                                   cs_events: dict,
                                   time_offset: float):
    """
    Add the data to the NWBFile using the NDX BEADL extension
    """
    # Define the task schema
    task_schema = TaskSchema(
        name='task_schema',
        data=task_schema if task_schema is not None else "",
        version="0.1.0",
        language="XSD"
    )

    # Define the task program
    task_program = TaskProgram(
        name='task_program',
        data=task_program if task_program is not None else "",
        schema=task_schema,
        language="XML"
    )

    # Define task arguments
    task_arg_table = TaskArgumentsTable()  # TODO: Populate the TaskArgumentsTable

    # define the state types table
    state_types_table = StateTypesTable(description="ARControl control states")
    state_types_table.add_column(name='state_label', description='ARControl control state label')
    for state_name, state_label in state_types.items():
        state_types_table.add_row(state_name=state_name, state_label=state_label)

    # define the events types table
    event_types_table = EventTypesTable(description="ARControl input events")
    event_types_table.add_column(name='event_label', description='ARControl event label')
    for event_name, event_label in event_types.items():
        event_types_table.add_row(event_name=event_name, event_label=event_label)

    # define the actions types table
    action_types_table = ActionTypesTable(description="ARControl output actions")
    action_types_table.add_column(name='action_label', description='ARControl ouput actions label')
    for action_name, action_label in action_types.items():
        action_types_table.add_row(action_name=action_name, action_label=action_label)

    # define the task
    task = Task(
        task_program=task_program,
        task_schema=task_schema,
        event_types=event_types_table,
        state_types=state_types_table,
        action_types=action_types_table,
        task_arguments=task_arg_table
    )
    nwbfile.add_lab_meta_data(task)

    # TODO Check if additional timestamp transformation from (lines 314 - 344) need to be applied here as well
    # TODO sort events_table, actions_table, and states_table by timestamps
    # define the events table
    events_table = EventsTable(
        description="ARControl input events acquired during the experiment",
        event_types_table=event_types_table)
    events_table.add_column(name='duration', description='ARControl input event duration')
    event_name_index = {e: i for i, e in enumerate(event_types_table['event_name'])}
    for event_name, event_times in io_in_events.items():
        event_index = event_name_index[event_name]
        for timerange in event_times:
            events_table.add_row(event_type=event_index, value="",
                                 timestamp=timerange[0] + time_offset,
                                 duration=timerange[1])
    nwbfile.add_acquisition(events_table)

    # define the actions table
    actions_table = ActionsTable(
        description="ARControl output actions acquired during the experiment",
        action_types_table=action_types_table)
    actions_table.add_column(name='duration', description='ARControl output actions duration')
    action_name_index = {a: i for i, a in enumerate(action_types_table['action_name'])}
    for action_name, action_times in io_out_actions.items():
        action_index = action_name_index[action_name]
        for timerange in action_times:
            actions_table.add_row(action_type=action_index,
                                  value="",  # TODO does ARControl define values for output actions?
                                  timestamp=timerange[0] + time_offset,
                                  duration=timerange[1])
    nwbfile.add_acquisition(actions_table)

    # define the states table
    states_table = StatesTable(
        description="ARControl states  acquired during the experiment",
        state_types_table=state_types_table)
    state_name_index = {e: i for i, e in enumerate(state_types_table['state_name'])}
    for state_name, state_times in cs_events.items():
        state_index = state_name_index[state_name]
        for timerange in state_times:
            states_table.add_row(state_type=state_index,
                                 start_time=timerange[0] + time_offset,
                                 stop_time=timerange[0] + timerange[1])
    nwbfile.add_acquisition(states_table)

    # noqa TODO define and populate the TrialsTable. Trials should be defined by transitions between components indicated by the prefix of the name of the state
    # trials = TrialsTable(description="ARControl behavioral trials ",
    #                      states_table=states_table,
    #                      events_table=events_table,
    #                      actions_table=actions_table)
    # nwbfile.trials = trials  # TODO check that there is no existing trials table that we should append to


def savenwb(MAT: dict,
            nwb_filename: str,
            append_to_nwb_file: bool = False,
            use_behavioral_time_series: bool = not NDX_BEADL_AVAILABLE,
            use_ndx_beadl: bool = NDX_BEADL_AVAILABLE):
    """
    Save the MAT data dict generated by the

    Typically, only one of use_ndx_beadl or use_behavioral_time_series should be set to True.
    While using both options is valid, it results in duplication of the data in the NWBFile in
    that the data will be stored both using the ndx-beadl data types and as standard time series
    and time intervals.

    :param nwb_filename: Name of the NWB file to write
    :param append_to_nwb_file: If the NWB file exists, should we append to it (True) or overwrite the file (False)
    :param MAT: Dictionary generated by arc2dict function
    :param use_behavioral_time_series: Boolean indicating whether to use behavioral timeseries to store the data
    :param use_ndx_beadl: Boolean indicating whether to use the ndx-beadl extension
    """
    session_start_time = MAT['session_start_time']
    task_name = MAT['info']['task']

    # Open the existing NWB file or create a new NWB file for write
    # Append to an existing NWB file
    if os.path.isfile(nwb_filename) and append_to_nwb_file:
        nwb_io = NWBHDF5IO(nwb_filename, "a")
        nwbfile = nwb_io.read()
    # Create a new NWBFile to write to
    else:
        nwb_io = NWBHDF5IO(nwb_filename, "w")
        nwbfile = NWBFile(
            session_description=task_name,  # required
            identifier=task_name+"."+str(session_start_time),  # required
            session_start_time=session_start_time,  # required
            experimenter="ArControl behavior recorder",  # optional
        )

    # add the arc data to the NWBFile object in memory
    add_arc_to_nwbfile(
        MAT=MAT,
        nwbfile=nwbfile,
        use_behavioral_time_series=use_behavioral_time_series,
        use_ndx_beadl=use_ndx_beadl)

    # write the NWBFile object to disk
    nwb_io.write(nwbfile)
    nwb_io.close()


if __name__ == '__main__':
    filetxts = sys.argv[1:]
    for filetxt in filetxts:
        convert(filetxt)
