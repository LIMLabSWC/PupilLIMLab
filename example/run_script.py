import os

import pandas as pd
import numpy as np

from PupilProcessing.pupilpipeline import Pipeline, utils
from xdetectioncore.paths import posix_from_win
from pathlib import Path
import yaml
from loguru import logger
from rich.logging import RichHandler
from pyinspect import install_traceback
import argparse
import platform
from datetime import datetime

if __name__ == "__main__":
    install_traceback()

    logger.configure(
        handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    )
    logger.info('started and loading config')
    today_str = datetime.today().date().strftime('%y%m%d')
    logger_path = Path.cwd()/f'log'/f'log_{today_str}.txt'
    logger_path = utils.unique_file_path(logger_path)
    # logger.add(logger_path,level='INFO')

    install_traceback()
    logger.configure(
        handlers=[{"sink": RichHandler(markup=True), "format": "{message}"}]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('-date', default=None)
    parser.add_argument('--sess_top_query', default=None)
    parser.add_argument('--sess_td_df_query', default=None)
    parser.add_argument('--run_single', action='store_true')
    parser.add_argument('--pkl_sffx', default=None)

    args = parser.parse_args()
    os = platform.system().lower()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    logger.info('loaded config')

    tdatadir = Path(config[f'tdatadir_{os}'])
    pdatadir = Path(config[f'pdatadir_{os}'])
    assert tdatadir.is_dir(), f'{tdatadir} can not be found. check'
    assert pdatadir.is_dir(), f'{pdatadir} can not be found. check'


    pkl_prefix = pdatadir.parts[-1]
    # dirstyle = 'N_D_it'
    dirstyle = config['dirstyle']

    if dirstyle == 'N_D_it':
        list_aligned = list(pdatadir.iterdir())
        # list_aligned = list(pdatadir.rglob('*.h5'))
        aligneddir = ''
    else:
        list_aligned = list(pdatadir.iterdir())
        aligneddir = ''


    # get animals and dates from session topology
    ceph_dir = Path(config[f'ceph_dir_{os}'])
    if config.get('session_topology_path'):
        use_session_topology = True
    else:
        use_session_topology = False
    
    if use_session_topology:
        sess_top_path = ceph_dir/posix_from_win(config['session_topology_path'])
        session_topology = pd.read_csv(sess_top_path)
        session_topology['videos_dir'] = session_topology['videos_dir'].apply(lambda x: ceph_dir/posix_from_win(x))
        animals2process = session_topology['name'].unique().tolist()
        dates2process = session_topology['date'].unique().astype(str).tolist()

    else:
        session_topology = None
        animals2process = config['animals2process']
        dates2process = config['dates2process']

    do_zscore = config['do_zscore']
    bandpass_met = config['bandpass_met']
    han_size = config['han_size']
    if han_size:
        lowtype = 'hanning'
    else:
        lowtype = 'filter'
    fs = config['fs']
    pdata_topic = config['pdata_topic']
    han_size_str = f'hanning{str(han_size).replace(".","")}'*bool(han_size)

    if args.pkl_sffx is not None:
        preprocess_pkl= f'{args.pkl_sffx}.pkl'
        pkl_stem = args.pkl_sffx
    else:
        preprocess_pkl = f'{config["pklname_suffix"]}.pkl' 
        pkl_stem = config["pklname_suffix"]

    pklname = f'{pkl_stem}_{int(fs)}Hz_hpass{str(bandpass_met[0]).replace(".", "")}_lpass{str(bandpass_met[1]).replace(".", "")}' \
              f'{han_size_str}{"_rawsize" * (not do_zscore)}.pkl'

    if args.pkl_sffx is not None:
        preprocess_pkl= f'{args.pkl_sffx}.pkl'
    else:
        preprocess_pkl = f'{config["pklname_suffix"]}.pkl'
    to_redo = config.get('sess_to_redo',[])
    if to_redo:
        _to_redo = [[e] if len(e.split('_')) == 2  else
                    [f'{ee}_{e}' for ee in session_topology.query(f'date=="{e}"')['name']] if e.isnumeric() else
                    [f'{e}_{ee}' for ee in session_topology.query(f'name=="{e}"')['date']] for e in to_redo]
        to_redo = sum(_to_redo, [])

    if args.sess_top_query:
        session_topology['date_str'] = session_topology['date'].astype(str)
        session_topology = session_topology.query(args.sess_top_query)
        animals2process = session_topology['name'].unique().tolist()
        dates2process = session_topology['date'].unique().astype(str).tolist()

    if args.sess_td_df_query:
        if not use_session_topology:
            raise NotImplementedError('Need to to provide sess topology df')
        all_td_df = pd.concat([pd.read_csv(Path(td_path)) for td_path in session_topology['tdata'].values
                               if Path(td_path).is_file()])

    if config.get('use_pupilsense', False):
        # find all pupilsense sessions
        dirs2search = [sessdir for sessdir in pdatadir.iterdir()
                       if sessdir.stem.split('_')[0] in animals2process
                       and sessdir.stem.split('_')[1] in dates2process]
        all_ps_sessdirs = [sessdir for sessdir in dirs2search if list(sessdir.glob('*eye0_eye_ellipse.csv'))]
        all_ps_sess = [sessdir.stem.split('_')[:2] for sessdir in all_ps_sessdirs]
        all_ps_sessnames = [f'{e[0]}_{e[1]}' for e in all_ps_sess]
        # filter
        animals2process,dates2process = np.unique(all_ps_sess,axis=0).T.tolist()
        animals2process = sorted(animals2process)
        dates2process = sorted(dates2process)
        session_topology = session_topology[session_topology.apply(lambda row: f"{row['name']}_{row['date']}"
                                                                               in all_ps_sessnames, axis=1)]

    run = Pipeline(animals2process, dates2process,(Path(config[f'pkl_dir_{os}'])/ pklname), tdatadir,
               pdatadir, pdata_topic, fs, han_size=han_size, passband=bandpass_met, aligneddir=aligneddir,
               subjecttype=config.get('subject_type','mouse'), dlc_snapshot=[2450000, 1300000], 
               overwrite= config.get('ow_flag',False), do_zscore=do_zscore,
               lowtype=lowtype, dirstyle=dirstyle, dlc_filtflag=True, redo=to_redo,
               preprocess_pklname=Path(config[f'pkl_dir_{os}'])/preprocess_pkl,use_ttl=config['use_TTL'],
               protocol=config['protocol'],use_canny_ell=config['use_canny_ell'],
               session_topology=session_topology, use_pupilsense=config.get('use_pupilsense',False), 
               use_dlc=config.get('use_dlc',True),
               run_multiprocess=(False if args.run_single else True))
    logger.info('Main class initialised')
    run.load_pdata()