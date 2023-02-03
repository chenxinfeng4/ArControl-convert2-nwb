# python BF_arc2nwb.py 2022-1113-224711.txt
import re
import sys
import os.path as osp
import re
import numpy as np
from datetime import datetime
from pynwb.epoch import TimeIntervals
from pynwb import NWBFile, TimeSeries, NWBHDF5IO
from pynwb.behavior import (
    BehavioralEvents,
)  # TODO:  TEST:  Check for behavior files that have been deprecated.

"""
Convert arcontrol_data.TXT to arcontrol_data.MAT. Just like "BF_arc2mat.m".
Xinfeng Chen, 2020-3-2
$ pip install mat4py
$ pyinstaller BF_arcmat.py
"""

def txt2MATdict(filetxt):
    """
    Convert arcontrol_data.TXT to arcontrol_data.nwb. Just like "BF_arc2nwb.m"
    :param filetxt:
    :return: None
    """

    # header #
    expression_header = re.compile('^@(IN\d+|OUT\d+|C\d+|C\d+S\d+):(.*)$')
    expression_taskname = re.compile('^-----(\w+)-----$')
    expression_arcbg  = re.compile(r'^ArC-bg$')
    MAT = {}
    MAT['info'] = {}
    isokfile = False
    for str in open(filetxt):
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

    # data #
    expression = re.compile('^(IN\d+|OUT\d+|C\d+S\d+):(\w.*)$')
    for str in open(filetxt):
        res_expression = re.findall(expression, str)
        if res_expression:
            style, nums = res_expression[0]
            nums_list = eval('[' + nums.replace(' ', ', ') + ']')
            MAT.setdefault(style, []).append(nums_list)

    return MAT


def convert(filetxt):
    # read raw from txt
    MAT = txt2MATdict(filetxt)
    filenyb = osp.splitext(filetxt)[0] + '.nwb'

    # append duration to CxSx, create component records
    cs_append_duration(MAT)
    c_create(MAT)

    # save to file #
    savenwb(filenyb, MAT)


def cs_append_duration(MAT):
    CS_pattern = re.compile('^C\d+S\d+$')
    CS_event_mm = {CS: np.array(v) for CS, v in MAT.items()
                    if CS_pattern.match(CS)}
    
    T_seq = np.squeeze(np.concatenate([v for v in CS_event_mm.values()]))
    assert T_seq.ndim==1
    T_seq.sort()
    T_seq_ext = np.append(T_seq, T_seq[-1])
    CS_end_mm={e: T_seq_ext[np.searchsorted(T_seq, v.flatten(), 'right')] 
                            for e, v in CS_event_mm.items()}
    csdata_dict = dict()
    for e in CS_event_mm:
        t_bg = CS_event_mm[e][:,0]
        t_end = CS_end_mm[e]
        t_dur = t_end - t_bg
        csdata_dict[e]= np.stack([t_bg, t_dur]).T
    
    MAT.update(csdata_dict)
    MAT['info'].setdefault('C0S0', 'End session')


def c_create(MAT):
    CS_pattern = re.compile('^C\d+S\d+$')
    CS_event_mm = {CS: np.array(v) for CS, v in MAT.items()
                    if CS_pattern.match(CS)}
    
    T_seq = []
    c_seq = []
    for e, v in CS_event_mm.items():
        T_seq.extend(v[:,0].tolist())
        c_seq.extend([e]*len(v[:,0]))
    T_seq = np.array(T_seq)
    c_seq = np.array(c_seq)
    argsort_ind = np.argsort(T_seq)
    T_seq = T_seq[argsort_ind]
    c_seq = c_seq[argsort_ind]
    comp_name_l, comp_switch_l = [], []
    comp_name_now = ''
    for cs1_name, T in zip(c_seq, T_seq):
        if cs1_name.split('S')[1]!='1' and cs1_name!='C0S0':
            continue
        if cs1_name!=comp_name_now:
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
        comp_bg_dur = comp_bg_dur_l[comp_name_l==cs1_name]
        cdata_dict[comp_name] = comp_bg_dur

    MAT.update(cdata_dict)
    MAT['info'].setdefault('C0', 'End session')


def savenwb(filenyb, MAT):
    filetime_str = osp.splitext(osp.basename(filenyb))[0]
    session_start_time = datetime.strptime(filetime_str, "%Y-%m%d-%H%M%S")

    task_name = MAT['info']['task']

    nwbfile = NWBFile(
        session_description=task_name,  # required
        identifier=task_name+"."+filetime_str,  # required
        session_start_time=session_start_time,  # required
        experimenter="ArControl behavior recorder",  # optional
    )
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="Raw ArControl event"
    )
    behavioral_events = BehavioralEvents(name="BehavioralEvents")
    IO_event = {IO: np.array(v)/1000 for IO, v in MAT.items() 
                    if 'IN' in IO or 'OUT' in IO}  # process IO time as sec
    CS_pattern = re.compile('(^C\d+S\d+$)|(^C\d+$)')
    CS_event = {CS: np.array(v)/1000 for CS, v in MAT.items()
                    if CS_pattern.match(CS)}  # process CS time as sec
    world_event = (IO_event|CS_event)
    assert world_event.keys() <= MAT['info'].keys()

    time_len = max([v[-1,0]+v[-1,1] for v in world_event.values()])

    s1 = TimeIntervals(
            name="ArControl_Events",
            description="intervals for each event",
        )
    s1.add_column(name="event", description="I/O and State events")
    for e, v in world_event.items():
        for start_t, dur_t in v:
            s1.add_row(start_time=start_t, 
                        stop_time=start_t+dur_t, 
                        event=e)
    _ = nwbfile.add_time_intervals(s1)

    for e, v in world_event.items():
        tbg_pre, tend=v[:,0], v[:,0]+v[:,1]
        ddt = 0.0001
        tend[tend==tbg_pre] += ddt
        tbg =  tbg_pre + ddt
        tend_post = tend + ddt
        seq_t_1 = np.concatenate((tbg, tend))
        seq_t_0 = np.concatenate((tbg_pre, tend_post))
        seq_v_1 = np.ones(seq_t_1.shape)
        seq_v_0 = np.zeros(seq_t_0.shape)
        seq_t = np.concatenate((seq_t_0, seq_t_1, [time_len+ddt]))
        seq_v = np.concatenate((seq_v_0, seq_v_1, [0]))
        if not np.any(seq_t==0.0):
            seq_t = np.append(seq_t, 0.0)
            seq_v = np.append(seq_v, 0)

        argind = np.argsort(seq_t)
        timestamps = seq_t[argind]
        data = seq_v[argind]

        time_series = TimeSeries(
            name=e,
            data=data,
            timestamps=timestamps,
            comments=MAT['info'][e],
            description=MAT['info'][e],
            unit = "TTL",
        )
        behavioral_events.add_timeseries(time_series)
    
    behavior_module.add(behavioral_events)

    with NWBHDF5IO(filenyb, "w") as io:
        io.write(nwbfile)


if __name__ == '__main__':
    filetxts = sys.argv[1:]
    for filetxt in filetxts:
        convert(filetxt)
