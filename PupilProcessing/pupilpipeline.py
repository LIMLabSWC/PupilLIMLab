import joblib
import pandas as pd

from .psychophysicsUtils import *
from . import utils
from xdetectioncore.behaviour import load_aggregate_td_df, add_datetimecol
from datetime import datetime, timedelta,timezone
import numpy as np
from copy import deepcopy as copy
import pathlib
from pathlib import Path
from loguru import logger
from pyinspect import install_traceback
import multiprocessing
from tqdm import tqdm
import sys


# script for building trial data and pupil data dict
# will generate pickle of dict


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def remove_missed_ttls(ts: np.ndarray) -> np.ndarray:
    # get average dt
    ts = pd.Series(ts)
    ts_diff = ts.diff()
    mean_dt = ts_diff.mean()
    good_idx = np.where(ts_diff.round('0.01S') > mean_dt.round('0.01S'))[0]
    return ts[good_idx]


def get_dlc_est_path(recdir, filt_flag, non_plabs_str,name):
    dlc_estimates_files = []
    if filt_flag:
        dlc_estimates_files_gen = Path(recdir).glob(f'{non_plabs_str}'
                                                 f'eye0DLC_resnet50_mice_pupilJul4shuffle1_'
                                                 f'*_filtered.h5')
        dlc_estimates_files = list(dlc_estimates_files_gen)
    if not len(dlc_estimates_files):
        dlc_estimates_files_gen = Path(recdir).glob(f'{non_plabs_str}'
                                                 f'eye0DLC_resnet50_mice_pupilJul4shuffle1_'
                                                 f'*.h5')
        dlc_estimates_files = list(dlc_estimates_files_gen)
    # dlc_estimates_files = list(dlc_estimates_files_gen)
    dlc_snapshot_nos = [int(str(estimate_file).replace('_filtered', '').split('_')[-1].split('.')[0])
                        for estimate_file in dlc_estimates_files]
    if not len(dlc_estimates_files):
        logger.warning(f'missing dlc for {name}')
        return None,None
    else:
        snapshot2use_idx = np.argmax(dlc_snapshot_nos)
        dlc_pathfile = dlc_estimates_files[snapshot2use_idx]
        dlc_snapshot = dlc_snapshot_nos[snapshot2use_idx]
        return dlc_pathfile,dlc_snapshot


def get_dlc_est_files(name,dlc_filtflag,non_plabs_str,plabs,rec,pupil_df):
    dlc_pathfile, snapshot2use = get_dlc_est_path(rec, dlc_filtflag, non_plabs_str, name)
    _dlc_df = pd.read_hdf(dlc_pathfile)
    if not plabs:
        estimates = _dlc_df.head(pupil_df.shape[0])
    else:
        estimates = _dlc_df.head(pupil_df.shape[0])

    return estimates


class Pipeline:
    def __init__(self,names, date_list, pkl_filename, tdatadir, pupil_dir,
                 pupil_file_tag, pupil_samplerate=60.0,outlier_params=(4, 4), overwrite=False, do_zscore=True,
                 han_size=0.2,passband=(0.1,3),aligneddir='aligned2',subjecttype='humans', dlc_snapshot=None,
                 lowtype='filter',dirstyle=r'Y_m_d\it',preprocess_pklname='',dlc_filtflag=True,redo=None,
                 protocol='default',use_ttl=False,use_canny_ell=False,session_topology=None, **kwargs):

        # load trial data
        self.existing_sessions = None
        self.pool_results = None
        daterange = [sorted(date_list)[0], sorted(date_list)[-1]]

        self.trial_data = load_aggregate_td_df(session_topology,tdatadir,)
        # for col in self.trial_data.keys():
        #     if 'Time' in col or 'Start' in col or 'End' in col:
        #         if 'Wait' not in col and 'dt' not in col and col.find('Harp') == -1 and col.find(
        #                 'Bonsai') == -1 and 'Lick' not in col:
        #             self.trial_data[f'{col}_scalar'] = [scalarTime(t) for t in self.trial_data[col]]

        # format trial data df
        try:
            self.trial_data = self.trial_data.drop(columns=['RewardCross_Time','WhiteCross_Time'])
        except KeyError:
            pass
        # add datetime cols
        # self.trial_data['Reaction_time'] = self.trial_data['Trial_End_dt']-(self.trial_data['Trial_Start_dt'] +
        #                                                                     self.trial_data['Stim1_Duration'])
        self.animals = names
        self.anon_animals = [f'Subject {i}' for i in range(len(self.animals))]  # give animals anon labels
        self.dates = date_list

        # init pupil loading vars
        self.samplerate = round(1/pupil_samplerate,3)
        self.pdir = Path(pupil_dir)
        self.pupil_file_tag = pupil_file_tag
        self.aligneddir = Path(aligneddir)
        self.pklname = Path(pkl_filename)
        self.outlier_params = outlier_params
        self.data = {}
        self.preprocessed_pklname = preprocess_pklname
        if self.preprocessed_pklname == '':
            self.preprocessed_pklname = r'pickles\generic_name_plschange.pkl'
        self.preprocessed_pklname = Path(self.preprocessed_pklname)
        self.preprocessed = self.load_pre_processed(self.preprocessed_pklname)
        self.han_size = han_size
        self.passband = passband
        self.overwrite = overwrite
        self.subjecttype = subjecttype
        self.lowtype = lowtype
        self.zscore= do_zscore
        self.dlc_filtflag = dlc_filtflag
        self.redo=redo
        self.protocol = protocol
        self.use_ttl = use_ttl
        self.sessions = {}
        self.use_dlc = kwargs.get('use_dlc',False)
        self.use_canny_ell =use_canny_ell
        self.use_pupilsense = kwargs.get('use_pupilsense',False)
        logger.debug(dirstyle)
        year_lim = datetime.strptime(daterange[0],'%y%m%d').year

        self.run_multiprocess = kwargs.get('run_multiprocess',True)

        if isinstance(session_topology,pd.DataFrame):
            # assert isinstance(session_topology,pd.DataFrame)
            all_sessnames = set(list(zip(session_topology['name'],session_topology['date'])))
            self.paireddirs = {f'{sessname[0]}_{sessname[1]}':
                                   session_topology.query('name == @sessname[0] and date == @sessname[1]')['videos_dir'].tolist()
                               for sessname in all_sessnames}
            self.paired_sessinfo = {f'{sessname[0]}_{sessname[1]}':
                                        session_topology.query('name == @sessname[0] and date == @sessname[1]')
                                    for sessname in all_sessnames}
        else:
            raise ValueError('session_topology must be a DataFrame')
        self.dlc_snapshot = dlc_snapshot

        today = datetime.strftime(datetime.now(),'%y%m%d')
        self.figdir = Path(r'figures',today)

    def load_pre_processed(self,pre_pklname:pathlib.Path):
        if self.preprocessed_pklname.exists():
            with open(pre_pklname,'rb') as pklfile:
                preprocessed_data = pickle.load(pklfile)
        else:
            'print pre processed pkl not found'
            preprocessed_data = {}
        return preprocessed_data

    def get_outliers(self,rawx,rawy,rawsize,rawdiameter,confidence=None) -> tuple[np.ndarray,np.ndarray]:

        # outlier xy
        x_pos, x_isout = removeouts(rawx,n_speed=10000,n_size=5)
        y_pos, y_isout = removeouts(rawy,n_speed=10000,n_size=5)

        # outlier speed/size
        size, size_isout = removeouts(rawsize,n_speed=10000,n_size=5)
        diameter, diameter_isout = removeouts(rawdiameter,n_speed=10000,n_size=5)

        outliers_list = [x_isout,y_isout,size_isout,diameter_isout]
        # outlier confidence
        if confidence is not None:
            outliers_list.append((confidence < 0.6).astype(int))

        outs_arr = np.array(outliers_list)
        return outs_arr.any(axis=0).astype(int),outs_arr[-1,:],outs_arr

    # @logger.catch
    def process_pupil(self,pclass,name,pdf,pdf_colname,filt_params=None):
        if filt_params == None:
            filt_params = self.passband
        logger.info(f'<< {pdf_colname.upper()} >>')

        pclass.rawPupilDiams = np.array(pdf[pdf_colname])
        # pclass[name].rawTimes = np.array(pdf.index.asi8)

        pclass.uniformSample(self.samplerate)
        pupil_diams_uni = copy(pclass.pupilDiams)
        # remove linear trend
        sess_mean = pclass.pupilDiams.mean()
        # try:
        #     pclass.pupilDiams = signal.detrend(np.ma.masked_invalid(pclass.pupilDiams))+sess_mean
        # except ValueError:
        #     logger.warning(f'ValueError detrend {name} {pdf_colname}')

        with HiddenPrints():
            pclass.removeOutliers(n_speed=4, n_size=5)
            pupil_diams_no_outs = copy(pclass.pupilDiams)
            pclass.interpolate(gapExtension=0.1)
            pupil_diams_int = copy(pclass.pupilDiams)

            pclass.interpolate(gapExtension=0.1)

            # filter blocks of nan dat
            nan_ix = copy(np.isnan(pclass.pupilDiams))
            nan_ix = np.pad(nan_ix,1)
            # nonnan_ix = np.logical_not(nan_ix)
            nan_ix_diff = np.diff(np.nonzero(nan_ix))
            nan_ix_start_end = nan_ix_diff[1:-1].reshape(-1,2)

            logger.info(f'nans before interpolation:{np.isnan(pclass.pupilDiams).sum()}')
            pclass.pupilDiams = pd.Series(pclass.pupilDiams).interpolate(limit_direction='both').to_numpy()  # interpolate over nans

            pupil_diams_nozscore = copy(pclass.pupilDiams)

            # pclass.pupilDiams = utils.smooth(pclass.pupilDiams,int(self.han_size/self.samplerate))
            if self.lowtype == 'filter':
                if filt_params[0] > 0 and filt_params[1] > 0:
                    pclass.pupilDiams = utils.butter_filter(pclass.pupilDiams, filt_params, 1 / self.samplerate, filtype='band',)
                elif filt_params[1] > 0:
                    pclass.pupilDiams = utils.butter_filter(pclass.pupilDiams, filt_params[1], 1 / self.samplerate, filtype='low')
                else:
                    pass
                
            elif self.lowtype == 'hanning':
                if filt_params[0] > 0:
                    pclass.pupilDiams = utils.butter_filter(utils.smooth(pclass.pupilDiams.copy(), int(self.han_size / self.samplerate)),
                                                            filt_params[0], 1 / self. samplerate, filtype='high')
                else:
                    pclass.pupilDiams = utils.smooth(pclass.pupilDiams.copy(), int(self.han_size / self.samplerate))

            if self.zscore:
                pclass.zScore()

        # pclass.plot(self.figdir,saveName=f'{name}_{pdf_colname}',)

        return pclass.pupilDiams, pclass.isOutlier, pupil_diams_nozscore,pupil_diams_uni, pupil_diams_no_outs, pupil_diams_int

    # @logger.catch
    def load_pdata(self):
        if not self.figdir.is_dir():
            self.figdir.mkdir()
        if self.pklname.exists() and self.overwrite is False:
            self.data = dict()
            self.data = joblib.load(self.pklname.with_suffix('.joblib'))
            logger.info(f'Loaded existing pickle {self.pklname}')
            
            existing_sessions = list(self.data.keys())
            for name in existing_sessions:  # delete empty objects
                if not hasattr(self.data[name],'pupildf'):
                    self.data.pop(name)
                elif not isinstance(self.data[name].pupildf, pd.DataFrame):  # check if None
                    self.data.pop(name)
            existing_sessions = list(self.data.keys())
            self.data = dict()
            self.preprocessed = dict()

        # elif self.pklname.exists() is False or self.overwrite is True:
        else:
            self.data = dict()
            existing_sessions = []
        self.existing_sessions = existing_sessions

        if self.redo:
            for sessname in self.redo:
                if sessname in existing_sessions:
                    existing_sessions.remove(sessname)
                if sessname in self.preprocessed.keys():
                    self.preprocessed.pop(sessname)

        for name in self.paireddirs:
            animal = name.split('_')[0]
            date = name.split('_')[1]
            if 'Human19' in name:
                continue
            if name in existing_sessions:
                continue
            if date not in self.trial_data.loc[animal].index.get_level_values('date'):
                continue
            session_TD = self.trial_data.loc[animal, date].copy().dropna(axis=1)
            if 'RewardProb' in session_TD.columns and 'prob' not in self.protocol:
                if session_TD['RewardProb'].sum() > 0:
                    continue
                if session_TD['Stage'][-1]< 3:
                    continue
            if 'Time_dt' in session_TD.columns:
                session_TD.set_index('Time_dt', append=True, inplace=True)
            else:
                session_TD.set_index('Trial_Start_dt', append=True, inplace=True)
            self.sessions[name] = session_TD
            self.data[name] = pupilDataClass(animal)
        # manager = multiprocessing.Manager()
        # self.data = manager.dict(self.data)
        sess2run = [sess for sess in self.sessions if sess not in existing_sessions]
        if self.run_multiprocess:
            with multiprocessing.Pool(16) as pool:
                logger.debug('running multi')
                # self.pool_results = list(tqdm(pool.imap(self.read_and_proccess,self.sessions.keys()),
                #                               total=len(self.sessions)))

                self.pool_results = pool.map(self.read_and_process,sess2run)
                # for session in self.sessions:
                # self.read_and_proccess(session,self.sessions[session])
        else:
            self.pool_results = [self.read_and_process(sess) for sess in tqdm(sess2run, desc='Processing session', 
                                                                               total=len(sess2run))]
        
        bad_sess = []
        for sess_name, result in zip(self.sessions, self.pool_results):
            if result[0] is None:
                bad_sess.append(sess_name)
                continue
            self.data[sess_name].pupildf = result[0]
            self.preprocessed[sess_name] = result[1]
            self.data[sess_name].trialData = self.sessions[sess_name]
        pd.Series(bad_sess).to_csv(f'{self.pklname.stem}_bad_sessions.csv',index=False,header=False)

        # logger.debug(f' data keys{self.data.keys()}')
        # logger.debug(f'pdf  = {self.data[list(self.data.keys())[0]].pupildf.shape}')
        # with open(self.pklname, 'ab') as pklfile:
        logger.info(f'Saving {self.pklname}')
        joblib.dump(self.data, self.pklname.with_suffix('.joblib'))
        # with open(self.preprocessed_pklname, 'ab') as pklfile:
        joblib.dump(self.preprocessed, self.preprocessed_pklname.with_suffix('.joblib'))

        
    def read_and_process(self, name: str):
        """Main entrypoint: orchestrates reading + processing pupil data for a session."""

        if name == "DO57_221215":  # special-case exclusion
            return None, None

        logger.info(f"Checking {name}")
        animal, date = name.split("_")
        session_TD = self.sessions[name]

        # Ensure data container exists
        self.data[name] = pupilDataClass(animal)

        if self.preprocessed.get(name):  # Already processed
            return self.finalize(name)

        # ---- Get session directory ----
        sess_recdirs = self._get_session_dirs(name)
        if sess_recdirs is None:
            return None, None

        # ---- Load pupil data ----
        animal_pupil_dfs,sess_recdirs_mask = self._load_pupil_data(name, sess_recdirs, session_TD)
        sess_recdirs = [s for s, m in zip(sess_recdirs, sess_recdirs_mask) if m]
        if not animal_pupil_dfs:
            return None, None

        # ---- DLC data ----
        dlc_dfs = self._load_dlc_data(name, sess_recdirs, animal_pupil_dfs)

        # ---- Pupilsense ----
        pupilsense_dfs = self._load_pupilsense_data(name, sess_recdirs, animal_pupil_dfs)

        # ---- Merge extractors ----
        all_extractors_df = self._merge_extractors(name, animal_pupil_dfs, dlc_dfs, pupilsense_dfs)
        if not all_extractors_df:
            return None, None

        # Save preprocessed result
        self.preprocessed[name] = all_extractors_df

        # ---- Downstream processing ----
        return self.finalize(name)
    
    # ==========================================================
    # Helpers
    # ==========================================================

    def _get_session_dirs(self, name: str):
        """Find and validate recording directories for a session."""
        try:
            sess_recdirs = self.paireddirs[name]
        except KeyError:
            logger.error(f"No recdir entry for {name}")
            return None

        if not sess_recdirs:
            logger.error(f"No session dir for {name}")
            return None

        if isinstance(sess_recdirs, str):
            sess_recdirs = [sess_recdirs]

        return sess_recdirs

    def _find_pupil_file(self, name: str):
        """Search for pupil CSV file in multiple fallback locations."""
        candidates = [
            self.pdir / f"{name}_{self.pupil_file_tag}a.csv",
            self.pdir / f"{name}_{self.pupil_file_tag}.csv",
            self.pdir / self.aligneddir / f"{name}_{self.pupil_file_tag}a.csv",
        ]
        for f in candidates:
            if f.is_file():
                return f
        return None

    def _load_pupil_data(self, name, sess_recdirs, session_TD, prefer_ttl=True):
        """
        Load pupil data from either (1) TTL-aligned recs or (2) direct CSV.
        By default, TTL data is preferred if both exist.
        """
        ttl_data = None
        csv_data = None

        # ---- Direct pupil CSVs ----
        pupil_filepath = self._find_pupil_file(name)
        if pupil_filepath and pupil_filepath.is_file():
            try:
                csv_data = pd.read_csv(pupil_filepath).dropna()
                logger.info(f"{name}: Loaded pupil file {pupil_filepath} ({len(csv_data)} rows)")
            except Exception as e:
                logger.error(f"{name}: Failed reading {pupil_filepath} - {e}")

        # ---- Recdirs + TTL alignment ----
        recs_list, event92_df_list = self._load_ttl_and_recs(name, sess_recdirs,)
        if event92_df_list is not None:
            sess_recdirs = [e is not None for e in event92_df_list]
        if recs_list is not None and event92_df_list is not None:
            try:
                ttl_data = self._align_pupil_with_ttl(name, recs_list, event92_df_list, session_TD,)
                if ttl_data is not None:
                    logger.info(f"{name}: Loaded TTL-aligned pupil recs ({len(ttl_data[0]) if isinstance(ttl_data, list) else len(ttl_data)} rows)")
            except Exception as e:
                logger.error(f"{name}: TTL alignment failed - {e}")

        # ---- Selection logic ----
        if ttl_data is not None:
            return ttl_data,sess_recdirs if isinstance(ttl_data, list) else [ttl_data]

        if csv_data is not None:
            return [csv_data],sess_recdirs

        # if ttl_data is not None:  # prefer_ttl=False but TTL was available
            # return ttl_data if isinstance(ttl_data, list) else [ttl_data]

        logger.warning(f"{name}: No pupil data found (CSV or TTL)")
        return None,None

    def _load_ttl_and_recs(self, name, sess_recdirs:list):
        """Load TTL files and recording timestamp CSVs."""
        harp_bin_dir = self.pdir.parent.parent / "harpbins"

        # Handle paired_sessinfo override
        if not self.paired_sessinfo.get(name, pd.DataFrame).empty:
            event92_files = []
            for _,sess_info in self.paired_sessinfo[name].iterrows():
                beh_bin_path = Path(sess_info["beh_bin"])
                path_overlap = min([list(self.pdir.parts).index(e) for e in beh_bin_path.parts if e in self.pdir.parts])
                _overlap = list(beh_bin_path.parts).index(self.pdir.parts[path_overlap])
                abs_bin_path = Path(*self.pdir.parts[:path_overlap]).joinpath(*beh_bin_path.parts[_overlap:])
                event92_file = abs_bin_path.parent / f"{abs_bin_path.stem}_event_data_92.csv"
                event92_files.append(event92_file)
            event92_files = [f if f.is_file() else None for f in event92_files]
            assert(len(event92_files) == len(sess_recdirs))
        else:
            animal, date = name.split("_")
            event92_files = sorted(harp_bin_dir.glob(f"{animal}_HitData_{date}*_event_data_92.csv"))
            

        # Read TTLs if requested
        if event92_files and self.use_ttl:
            event92_df_list = [pd.read_csv(event_file) if event_file is not None else None for event_file in event92_files]
        else:
            event92_df_list = None

        # Read rec CSVs
        recdir_list = sess_recdirs
        first_file = Path(recdir_list[0], f"{name}_eye0_timestamps.csv")
        if not first_file.is_file():
            logger.error(f"No file for {name} in {recdir_list[0]}")
            return None, None

        try:
            recs_list = [
                pd.read_csv(Path(rec, f"{name}_eye0_timestamps.csv"))
                for rec in recdir_list
            ]
        except pd.errors.EmptyDataError:
            logger.error(f"Issue with {' '.join(map(str, recdir_list))}")
            return None, None
        logger.debug(f'{name} len recs: {len(recs_list)}. len event92 dfs: {len(event92_df_list) if event92_df_list else "N/A"}')
        return recs_list, event92_df_list

    def _align_pupil_with_ttl(self, name, recs_list, event92_df_list, session_TD):
        """Align pupil timestamps with TTL events, handling mismatches."""
        aligned_dfs = []
        animal, date = name.split("_")
        for ri, (rec,ttl_df) in enumerate(zip(recs_list,event92_df_list)):
            if rec.empty or ttl_df is None or ttl_df.empty:
                logger.error(f"Recording for {name} empty")
                aligned_dfs.append(pd.DataFrame())
                continue

            cam_ttls = ttl_df['Timestamp']
            rec.rename(columns={"timestamp": "Timestamp"}, inplace=True)

            if self.subjecttype == "mouse":
                rec["date"] = np.full_like(rec["Timestamp"], date).astype(str).copy()
                rec.index = rec["date"]
                try:
                    add_datetimecol(rec, "Bonsai_Time")
                except Exception as e:
                    logger.error(f"Datetime col failed for {name}: {e}")

                # Align timestamps
                
                rec["Timestamp_adj"] = rec["Timestamp"] - rec["Timestamp"].iloc[0]
                harp_sync_ttl_offest_secs = cam_ttls[0] - session_TD['Harp_time'].iloc[0]
                new_times = rec['Timestamp_adj'].apply(lambda e:session_TD['Bonsai_time_dt'].iloc[0]
                                                       +timedelta(seconds=float(e)/1e9+harp_sync_ttl_offest_secs))
                bonsai0 = rec.get("Bonsai_Time_dt", pd.Series([pd.NaT]))[0]
                df = pd.DataFrame()
                df["frametime"] = new_times
                df["timestamp"] = (rec["Timestamp_adj"].values/1e9)+cam_ttls[0]
                aligned_dfs.append(df)
            else:
                aligned_dfs.append(rec)

        return aligned_dfs

    def _load_dlc_data(self, name, sess_recdir, animal_pupil_dfs):
        """Load DLC estimates and align with pupil data."""
        if not self.use_dlc:
            return [pd.DataFrame()] * len(sess_recdir)

        logger.info("Loading DLC")
        dlc_dfs = []
        for rec_ix, rec in enumerate(sess_recdir):
            try:
                dlc_pathfile, snapshot2use = get_dlc_est_path(rec, self.dlc_filtflag, name)
                if dlc_pathfile is None:
                    dlc_dfs.append(pd.DataFrame())
                    continue
                dlc_df = get_dlc_est_files(
                    name, self.dlc_filtflag, name, True, rec, animal_pupil_dfs[rec_ix]
                )
                dlc_dfs.append(dlc_df)
            except Exception as e:
                logger.error(f"DLC load failed for {name}: {e}")
                dlc_dfs.append(pd.DataFrame())
        return dlc_dfs

    def _load_pupilsense_data(self, name, sess_recdir, animal_pupil_dfs):
        """Load pupilsense outputs if available."""
        if not self.use_pupilsense:
            return [pd.DataFrame()] * len(sess_recdir)

        logger.info("Loading pupilsense")
        dfs = []
        for rec_ix, rec in enumerate(sess_recdir):
            try:
                path = Path(rec) / f"{name}_eye0_eye_ellipse.csv"
                ps_df = pd.read_csv(path)
                min_len = min(len(animal_pupil_dfs[rec_ix]), len(ps_df))
                ps_df = ps_df.iloc[:min_len]
                ps_df.index = animal_pupil_dfs[rec_ix].head(min_len).index
                animal_pupil_dfs[rec_ix] = animal_pupil_dfs[rec_ix].iloc[:min_len]
                ps_df = ps_df.rename(columns={"radius": "pupilsense_raddi_a", "height": "pupilsense_raddi_b"})
                dfs.append(ps_df[["pupilsense_raddi_a", "pupilsense_raddi_b"]])
            except Exception as e:
                logger.error(f"Pupilsense missing for {name}: {e}")
                dfs.append(pd.DataFrame())
        return dfs

    def _merge_extractors(self, name, animal_pupil_dfs, dlc_dfs, pupilsense_dfs):
        """Combine all extractor outputs into a unified list of dfs.

        Always carry over 'timestamp' from animal_pupil_dfs.
        Only add DLC/pupilsense data if enabled.
        """
        merged_list = []

        for idx, pupil_df in enumerate(animal_pupil_dfs):
            if pupil_df is None or pupil_df.empty:
                logger.warning(f"No pupil data for {name} rec {idx}")
                continue

            dfs_to_merge = [pupil_df]  # start with main pupil DF (has timestamp)

            if self.use_dlc and idx < len(dlc_dfs):
                dfs_to_merge.append(dlc_dfs[idx])
            if self.use_pupilsense and idx < len(pupilsense_dfs):
                dfs_to_merge.append(pupilsense_dfs[idx])
            if any(df.empty for df in dfs_to_merge):
                logger.warning(f"Some extractor data missing for {name} rec {idx}")
                continue

            merged = pd.concat(dfs_to_merge, axis=1)

            # safeguard: make sure timestamp column exists
            if "timestamp" not in merged.columns and "timestamp" in pupil_df.columns:
                merged["timestamp"] = pupil_df["timestamp"]

            merged_list.append(merged)

        if not merged_list:
            logger.error(f"No valid data at all for {name}")
            return None

        return merged_list

    
    def finalize(self, name:str):
        """Finalize pupil processing for one animal/day."""
        animal_pupil_processed_dfs = []
        cols2process = self._build_cols_to_process()

        for animal_pupil_subset in self.preprocessed[name]:
            animal, date = name.split("_")
            pupilclass = self._init_pupil_class(name, animal_pupil_subset)
            if pupilclass is None:
                continue

            pupil_uni = self._process_single_session(
                name, animal_pupil_subset, pupilclass, cols2process
            )
            if pupil_uni is None:
                continue

            # filter columns for saving
            try:
                df_cols = pupil_uni.columns
                cols2use_ix = [
                    "timestamp" in e
                    or "zscored" in e
                    or "out" in e
                    or "processed" in e
                    or "normed" in e
                    for e in df_cols
                ]
                cols2use = df_cols[cols2use_ix]

                # # save session dump
                # outdir = Path(r"/ceph/akrami/Dammy/mouse_pupillometry") / "full_pupil_dump"
                # outdir.mkdir(parents=True, exist_ok=True)
                # try:
                #     pupil_uni.to_csv(outdir / f"{name}_full_pupil.csv")
                # except OSError:
                #     logger.warning(f"Can't write {name}_full_pupil.csv")

                animal_pupil_processed_dfs.append(pupil_uni[cols2use])
            except KeyError:
                logger.warning(f"KeyError in finalize for {name}")
                continue

        if not animal_pupil_processed_dfs:
            logger.critical(f"No processed dfs for {name}")
            return None, None

        try:
            self.data[name].pupildf = pd.concat(animal_pupil_processed_dfs, axis=0)
        except Exception:
            logger.critical(f"<NO DFs FOR {name}>")
            return None, None

        self.data[name].trialData = self.trial_data.loc[animal, date]

        return self.data[name].pupildf, self.preprocessed[name]

    def _build_cols_to_process(self):
        """Decide which pupil columns to process based on config flags."""
        cols = []
        if self.use_dlc:
            cols.append("dlc_radii_a")
        if self.use_canny_ell:
            cols += ["canny_centre_x", "canny_centre_y", "canny_raddi_a"]
        if self.use_pupilsense:
            cols += ["pupilsense_raddi_a", "pupilsense_raddi_b"]
        if self.subjecttype == "human":
            cols += ["diameter_2d"]
        return cols


    def _init_pupil_class(self, name, animal_pupil_subset):
        """Initialize pupilDataClass and set rawTimes correctly."""
        pupilclass = pupilDataClass(f"{name}")
        if self.subjecttype == "human":
            try:
                times_as_dt = animal_pupil_subset.index.to_pydatetime()
                times_w_tzone = [dt.astimezone() for dt in times_as_dt]
                timestamps = [dt.replace(tzinfo=timezone.utc).timestamp() for dt in times_w_tzone]
                pupilclass.rawTimes = np.array(timestamps)
            except Exception:
                logger.error(f"No timestamps for {name}. Not processing.")
                return None
        else:
            if "timestamp" not in animal_pupil_subset.columns:
                logger.error(f"Missing timestamp column for {name}")
                return None
            pupilclass.rawTimes = animal_pupil_subset["timestamp"].values
        return pupilclass


    def _process_single_session(self, name, animal_pupil_subset, pupilclass, cols2process):
        """Process one session subset and return processed DataFrame."""
        with HiddenPrints():
            unitimes = uniformSample(
                pupilclass.rawTimes, pupilclass.rawTimes, new_dt=self.samplerate
            )[1]
        pupil_uni = pd.DataFrame([], index=unitimes)

        outs_list = []

        for col2norm in cols2process:
            try:
                pupil_processed = self.process_pupil(
                    pupilclass, f"{name}", animal_pupil_subset, col2norm
                )
                pupil_uni[f"{col2norm}_zscored"] = pupil_processed[0][:pupil_uni.shape[0]]
                pupil_uni[f"{col2norm}_isout"] = pupil_processed[1][:pupil_uni.shape[0]]
                pupil_uni[f"{col2norm}_processed"] = pupil_processed[2][:pupil_uni.shape[0]]
                pupil_uni[f"{col2norm}_raw"] = pupil_processed[3][:pupil_uni.shape[0]]
                pupil_uni[f"{col2norm}_no_outs"] = pupil_processed[4][:pupil_uni.shape[0]]
                pupil_uni[f"{col2norm}_int"] = pupil_processed[5][:pupil_uni.shape[0]]
                outs_list.append(pupil_processed[1])
            except Exception as e:
                logger.error(f"Can't process pupil for {name} - {e}")
                return None

        pupil_uni["isout"] = pupil_uni[f"{cols2process[0]}_isout"]
        if "dlc_EW" in cols2process and "dlc_LR" in cols2process:
            pupil_uni["dlc_EW_normed"] = (
                pupil_uni["dlc_EW_processed"] / pupil_uni["dlc_LR_processed"]
            )
            pupil_uni["isout_EW"] = pupil_uni["dlc_EW_isout"]

        return pupil_uni


if __name__ == "__main__":
    pass
    # Legacy code for running the script directly - this should be refactored to a separate runner script or notebook
    # logger_path = Path.cwd()/'log'/'logfile.txt'
    # logger_path = utils.unique_file_path(logger_path)
    # logger.add(str(logger_path),level='TRACE')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('config_file')
    # parser.add_argument('date',default=None)
    # args = parser.parse_args()

    # with open(args.config_file,'r') as file:
    #     config = yaml.safe_load(file)

    # tdatadir = r'C:\bonsai\data\Hilde'
    # # fam task
    # humans = [f'Human{i}' for i in range(20,28)]
    # humandates = [#'220209','220210','220215',
    #               '220311','220316','220405','220407','220407','220408','220422','220425']
    # task = 'fam'

    # # # humans dev norm task
    # # humans = [f'Human{i}' for i in range(28,33)]
    # # humandates = ['220518', '220523', '220524','220530','220627']
    # # task = 'normdev'

    # # with suppress(ValueError):
    # #     humans.remove('Human29')
    # #     humandates.remove('220523')

    # # han_size = 1
    # bandpass_met = (0.125, 2)
    # han_size = 0.15
    # if han_size: lowtype = 'hanning'
    # else: lowtype = 'filter'
    # do_zscore = True
    # fs = 90.0
    # pdata_topic = 'pupildata_3d'
    # han_size_str = f'hanning{str(han_size).replace(".","")}'*bool(han_size)
    # pklname = f'human_{task}_{pdata_topic.split("_")[1]}_{int(fs)}Hz_driftcorr_lpass_detrend{str(bandpass_met[1]).replace(".","")}' \
    #           f'_hpass{str(bandpass_met[0]).replace(".","")}_flipped_TOM_{han_size_str}{"_rawsize" * (not do_zscore)}.pkl'
    # aligned_dir = f'aligned_{task}'
    # run = Main(humans,humandates,os.path.join(r'c:\bonsai\gd_analysis\pickles',pklname),tdatadir,r'W:\humanpsychophysics\HumanXDetection\Data',
    #            pdata_topic,fs,han_size=1,passband=bandpass_met,aligneddir=aligned_dir,subjecttype='human',
    #            overwrite=True,dlc_snapshot=[1750000,1300000],lowtype=lowtype,do_zscore=do_zscore,redo=config['sess_to_redo'],
    #            preprocess_pklname=os.path.join(r'c:\bonsai\gd_analysis\pickles', f'human_fam.pkl'))
    # run.load_pdata()
    # # plt.plot(run.data['Human21_220316'].pupildf['rawarea_zscored'])
    # # plt.plot(run.data['Human25_220408'].pupildf['rawarea_zscored'])


