import streamlit as st
import numpy as np
import pandas as pd
import time
import cv2
import requests
import tempfile
import os

# === CONFIGURATION ===
CLIP_VIDEO_MAPPING =[{'Shrek': 
                        {
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/shr.png?raw=true',
                            'Low': {
                                    'name': 'SHREK_10a',
                                    'clip_index': 22,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530570/SHREK_10a_c_nxfopw.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443658/heatmapoverlay_SHREK_10a_s0qq7j.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538895/Shrekpath_10a_overall_male6n.mp4',
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538896/Shrekpath_2086ty_jlrs9z.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538894/Shrekpath_2081lo_oxe5yo.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538894/Shrekpath_1744an_k5zetx.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538894/Shrekpath_2024er_hszqk5.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538892/Shrekpath_2157ss_coocfg.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538890/Shrekpath_2700hm_vx5b2d.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538889/Shrekpath_2401er_qrbtk2.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538887/Shrekpath_2160rz_whol9e.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538886/Shrekpath_2168hd_rjsd0u.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538885/Shrekpath_2133lr_uh6uyx.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538884/Shrekpath_1864nw_j7dksr.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538883/Shrekpath_2775ar_ri1mf0.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538882/Shrekpath_2278te_sy9n4w.mp4'
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'SHREK_3b',
                                    'clip_index': 28,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530570/SHREK_3b_c_qg81tt.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443668/heatmapoverlay_SHREK_3b_lqcyyc.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538956/Shrekpath_3b_overall_xn1zah.mp4',
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538955/Shrekpath_2464ys_zcenzz.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538955/Shrekpath_2789ly_sqlwer.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538954/Shrekpath_2764hn_absksb.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538951/Shrekpath_1744an_nye1tu.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538951/Shrekpath_2024er_x0fqxk.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538950/Shrekpath_2700hm_opilq8.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538948/Shrekpath_2702SZ_mczmw4.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538945/Shrekpath_1776yy_ya6a1f.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538943/Shrekpath_2168hd_ra4sxw.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538942/Shrekpath_2237ns_fqfpnt.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538941/Shrekpath_2152so_zvzp0l.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744538939/Shrekpath_2367oo_yxnlfz.mp4'
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'SHREK_3a',
                                    'clip_index': 27,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530573/SHREK_3a_c_ibscr9.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443692/heatmapoverlay_SHREK_3a_dgrkux.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471343/shrek_overall_hukdzt.mp4',
                                                        'participant': ['https://res.cloudinary.com/dlggzzrag/video/upload/v1744471361/Shrekpath_2802ln_deeap9.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471360/Shrekpath_2792sl_k2tcpc.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471359/Shrekpath_2719sn_dqzmmw.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471357/Shrekpath_2103en_gvjqqe.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471357/Shrekpath_2086ty_bxmsh5.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471355/Shrekpath_2141po_po8v5g.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471350/Shrekpath_2039nn_s6jalp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471347/Shrekpath_2024er_kxryht.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471346/Shrekpath_0811ne_aqvgby.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471345/Shrekpath_2017ae_uy4nhk.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471345/Shrekpath_2033kt_fw8pw7.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744471342/Shrekpath_0724er_necb29.mp4']
                                                    }
                                                }
                        }
                    },
                    {'Deep Blue': 
                        {
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/deep.png?raw=true',
                            'Low': {
                                    'name': 'DEEPB_9c',
                                    'clip_index': 49,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530559/DEEPB_9c_c_rhhepr.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443688/heatmapoverlay_DEEPB_9c_uwtbxs.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539079/DEEPBPATH_overall_z4q7ew.mp4',
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539092/Deepbpath_2029ny_vdk6vp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539091/Deepbpath_2753nt_e6wgf5.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539091/Deepbpath_2789ly_ybig5p.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539083/Deepbpath_1875td_afapbp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539083/Deepbpath_2112el_fa8inc.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539079/Deepbpath_1776yy_dcej1t.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539077/Deepbpath_2025es_r3ya2w.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539068/Deepbpath_2278te_foq3cl.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539065/Deepbpath_2144dl_tlcaxh.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539064/Deepbpath_2180yr_urgpoe.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539064/Deepbpath_2141po_xgnnha.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539062/Deepbpath_2030nr_vlnu0c.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539052/Deepbpath_1947ne_kukdhi.mp4'
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'DEEPB_5b',
                                    'clip_index': 45,
                                    'frame_count': 719,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530556/DEEPB_5b_c_nyvoao.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443488/heatmapoverlay_DEEPB_5b_p1erof.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539111/DEEPBPATH_overall_oqnxvy.mp4',
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539119/Deepbpath_1744an_hmfaxi.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539118/Deepbpath_2133lr_xajn5u.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539117/Deepbpath_1613ey_uuy1an.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539117/Deepbpath_1848ne_feczbn.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539113/Deepbpath_2176as_adzdzr.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539110/Deepbpath_1864nw_k1uek4.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539107/Deepbpath_2719sn_yi1os0.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539107/Deepbpath_2005dh_bryzfv.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539103/Deepbpath_2327wn_qonaxx.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539102/Deepbpath_2006nh_kvlcmt.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539098/Deepbpath_2152so_j127gp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539092/Deepbpath_2072ds_k07vcc.mp4'
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'DEEPB_11a',
                                    'clip_index': 41,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530582/DEEPB_11a_c_kw5icz.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443720/heatmapoverlay_DEEPB_11a_packhk.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539091/DEEPBPATH_overall_swaldc.mp4',
                                                        'participant': [
                                                                'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539137/Deepbpath_1613ey_xuqdom.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539133/Deepbpath_2160rz_hqgjqc.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539126/Deepbpath_2024er_zyli9f.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539124/Deepbpath_2133lr_iory93.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539122/Deepbpath_2081lo_pknwls.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539104/Deepbpath_2719sn_osjjzw.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539088/Deepbpath_2270er_uq1ldr.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539080/Deepbpath_2152so_mwkabv.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539071/Deepbpath_2781yh_qlqrzj.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539070/Deepbpath_2144dl_wn9im5.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539070/Deepbpath_2072ds_q3nfaf.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539031/Deepbpath_1947ne_qb3lyk.mp4'
                                                                        ]
                                                    }
                                    }
                        }
                    },
                    {'The Squid and the Whale': 
                        {
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/squid.png?raw=true',
                            'Low': {
                                    'name': 'SQUID_8a',
                                    'clip_index': 196,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530575/SQUID_8a_c_pq5bxy.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443689/heatmapoverlay_SQUID_8a_vehssz.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539158/Squidpath_overall_z157xl.mp4',
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539166/Squidpath_2086ty_grshxp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539164/Squidpath_2139ha_rze4z2.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539163/Squidpath_2792sl_bcfkcw.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539162/Squidpath_2081lo_gtpf6w.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539161/Squidpath_2700hm_vg9cvs.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539161/Squidpath_2112el_tgitru.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539159/Squidpath_2017ae_kafcqr.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539156/Squidpath_2039nn_wah3ly.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539154/Squidpath_2006nh_a5sf5t.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539153/Squidpath_0972rs_pvu9wd.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539153/Squidpath_2327wn_qnjw40.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539152/Squidpath_2278te_wchk5b.mp4'
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'SQUID_12c',
                                    'clip_index': 192,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530575/SQUID_12c_c_xjxogc.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443695/heatmapoverlay_SQUID_12c_x7zaou.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539178/Squidpath_overall_gs113d.mp4',
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539181/Squidpath_2139ha_y2wk9f.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539180/Squidpath_1613ey_rblxlp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539179/Squidpath_2133lr_iuafad.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539178/Squidpath_2081lo_c7jyhf.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539178/Squidpath_0724er_w5z5kx.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539177/Squidpath_2160rz_n1gbnl.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539174/Squidpath_2017ae_jxsozp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539173/Squidpath_1875td_ygsktz.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539172/Squidpath_2103en_npd7bm.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539172/Squidpath_2006nh_egzpu6.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539170/Squidpath_2072ds_suq1vf.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539170/Squidpath_2367oo_nei41a.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539169/Squidpath_2065ey_b63nju.mp4'
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'SQUID_6a',
                                    'clip_index': 195,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530577/SQUID_6a_c_vpq0nm.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443691/heatmapoverlay_SQUID_6a_wpmqec.mp4',
                                    'Gaze Sequence': {
                                                        'overall': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539133/Squidpath_overall_iokfmm.mp4',
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539151/Squidpath_2033kt_e5nxj2.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539148/Squidpath_0811ne_g6aj9c.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539147/Squidpath_2009dl_mg2i3z.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539147/Squidpath_1744an_wjkt2g.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539146/Squidpath_2720dn_f7n93p.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539145/Squidpath_2753nt_jdlu2f.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539142/Squidpath_2005dh_cxdxup.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539139/Squidpath_2327wn_dgtvz2.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539137/Squidpath_2094ne_mweysk.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539134/Squidpath_2702SZ_bkzwbe.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539131/Squidpath_2153ng_g4rwa9.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539129/Squidpath_2180yr_gcaef7.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744539120/Squidpath_2130oa_bqjzle.mp4'
                                                                        ]
                                                    }
                                    }
                        }
                    }]
CLIP_NAMES = ['Shrek', 'Deep Blue', 'The Squid and the Whale']
DATA_PUPIL_URL = 'https://media.githubusercontent.com/media/Tassanai5/material-miniproject2/refs/heads/main/pupil_size.csv'

# === INITIALIZE SESSION STATE ===
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'selected_clip' not in st.session_state:
    st.session_state['selected_clip'] = None
if 'video_frame_idx' not in st.session_state:
    st.session_state['video_frame_idx'] = 0
if 'is_playing' not in st.session_state:
    st.session_state['is_playing'] = False

# === HELPER FUNCTIONS ===
def add_summ_navigation():
    """Adds a floating button at the bottom right corner for summary page navigation."""
    # Create container for the button with custom CSS
    button_container = st.container()
    
    # Apply CSS for positioning at bottom right
    button_css = """
    <style>
    .floating-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 100;
    }
    </style>
    """
    st.markdown(button_css, unsafe_allow_html=True)
    
    # Create the floating button with HTML
    with button_container:
        st.markdown(
            """
            <div class="floating-button">
                <form method="post" action="">
                    <input type="hidden" name="summ_nav" value="true">
                    <button type="submit" 
                        style="background-color: #4CAF50; color: white; padding: 10px 20px; 
                        border: none; border-radius: 5px; cursor: pointer;">
                        View Summary
                    </button>
                </form>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Check if the button was clicked (will be handled by session state)
    if st.session_state.get('summ_nav_clicked', False):
        st.session_state['summ_nav_clicked'] = False
        st.session_state['page'] = 'summ'
        st.rerun()

def display_frame(video_url, frame_count, selected_video, tag, gaze_data = None):
    # Store the video capture object in session state to avoid reloading it
    if 'video_cap' not in st.session_state or st.session_state.current_video != video_url:
        st.session_state.video_cap = cv2.VideoCapture(video_url)
        st.session_state.current_video = video_url
        # Get fps for playback
        st.session_state.fps = st.session_state.video_cap.get(cv2.CAP_PROP_FPS)
        if st.session_state.fps <= 0:
            # Default to a standard fps if we couldn't read it correctly
            st.session_state.fps = 30.0
            st.warning("Could not detect video frame rate. Using 30 FPS as default.")
        # Get total duration in seconds
        st.session_state.duration = frame_count / st.session_state.fps

    st.write(f"Now playing: {selected_video}🎥")

    # Calculate total duration and create time markers
    duration = st.session_state.duration
    fps = st.session_state.fps
    
    # Create time markers every second, plus the end time
    time_markers = [i for i in range(int(duration) + 1)] + ([round(duration, 0)] if duration % 1 > 0 else [])
    
    # Create a unique session state key for this tag
    time_state_key = f"current_time_{tag}"
    
    # Create a session state to store the current time if it doesn't exist
    if time_state_key not in st.session_state:
        st.session_state[time_state_key] = 0

    col1, col2, col3 = st.columns([6, 2, 2])

    # Define button callbacks
    def move_backward():
        current_index = time_markers.index(st.session_state[time_state_key]) if st.session_state[time_state_key] in time_markers else 0
        if current_index > 0:
            st.session_state[time_state_key] = time_markers[current_index - 1]

    def move_forward():
        current_index = time_markers.index(st.session_state[time_state_key]) if st.session_state[time_state_key] in time_markers else 0
        if current_index < len(time_markers) - 1:
            st.session_state[time_state_key] = time_markers[current_index + 1]

    with col1:
        # Create a slider for time selection (in seconds)
        time_sec = st.select_slider(
            f"Time (seconds){tag}", 
            options=time_markers,
            value=st.session_state[time_state_key],
            key=f"time_slider_{tag}"
        )
        # Update session state when slider changes
        st.session_state[time_state_key] = time_sec

    with col2:
        if st.button('◀️', use_container_width=True, key=f'last_second_{tag}'):
            move_backward()
            st.rerun()

    with col3:
        if st.button('▶️', use_container_width=True, key=f'next_second_{tag}'):
            move_forward()
            st.rerun()
    
    # Convert time to frame index
    frame_idx = int(time_sec * fps)
    frame_idx = min(frame_idx, frame_count - 1)  # Ensure frame index doesn't exceed total frames
    
    # Get the selected frame
    st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = st.session_state.video_cap.read()
    
    # Create a container for frame display that will be used for both static display and playback
    frame_container = st.empty()
    
    if 'PupilSize' in tag:
        if ret and frame is not None:
            # Use the container for initial frame display
            frame_container.image(frame, channels="BGR", 
                                caption=f"Time: {time_sec:.2f}s (Frame {frame_idx})", 
                                use_container_width=True)

            # Round the current time to match with data points
            rounded_time = round(time_sec)
            
            if 'elapsed' in gaze_data.columns and rounded_time == 0:
                avg_pupil_area = 0
            elif 'elapsed' in gaze_data.columns:
                # Get data points with elapsed time matching the rounded time
                time_data = gaze_data[(gaze_data['elapsed'] >= rounded_time - 0.5) & 
                                    (gaze_data['elapsed'] < rounded_time + 0.5)]
                avg_pupil_area = time_data['pupil_size'].mean() if not time_data.empty else 505
            else:
                avg_pupil_area = 404
                
            st.metric("Average Pupil Area", f"{avg_pupil_area:.2f}")
        else:
            st.warning("❌ Could not load this frame. It may not exist at this index.")

    elif 'GazeSequence' in tag:
        if ret and frame is not None:
            # Use the container for initial frame display
            frame_container.image(frame, channels="BGR", 
                                 caption=f"Time: {time_sec:.2f}s (Frame {frame_idx})", 
                                 use_container_width=True)
        else:
            st.warning("❌ Could not load this frame. It may not exist at this index.")
            
        # Add controls for playing a short sequence
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            play_seconds_2 = st.button("Play 2 Seconds", key=f"play_2s_{tag}")
        with col3:
            play_seconds_5 = st.button("Play 5 Seconds", key=f"play_5s_{tag}")
        with col1:
            play_speed = st.select_slider("Speed", options=["0.5x", "1x", "2x"], value="1x", key=f"speed_{tag}")

        # Play a sequence of frames when the button is clicked
        if play_seconds_2 or play_seconds_5:
            speed_factor = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[play_speed]
            seconds_to_play = 2 if play_seconds_2 else 5
            
            # Calculate the starting and ending time in seconds
            start_time = time_sec
            end_time = min(start_time + seconds_to_play, max(time_markers))
            
            # Calculate how many frames we need to show for smooth playback
            total_frames_to_play = int(seconds_to_play * fps / speed_factor)
            time_increment = seconds_to_play / total_frames_to_play
            
            # Play the frames using the same container
            current_time = start_time
            for _ in range(total_frames_to_play):
                if current_time > end_time:
                    break
                    
                # Calculate the frame index from the current time
                current_frame_idx = int(current_time * fps)
                
                # Seek to the frame
                st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, current_frame = st.session_state.video_cap.read()
                
                if ret and current_frame is not None:
                    # Update the same container with each new frame
                    frame_container.image(current_frame, channels="BGR", 
                                        caption=f"Time: {current_time:.2f}s (Frame {current_frame_idx})", 
                                        use_container_width=True)
                    
                    # If PupilSize is in tag, calculate and display average pupil size for this time
                    if 'PupilSize' in tag and 'elapsed' in gaze_data.columns:
                        # Round the current time to match with data points
                        rounded_time = round(current_time)
                        
                        # Get data points with elapsed time matching the rounded time
                        time_data = gaze_data[(gaze_data['elapsed'] >= rounded_time - 0.5) & 
                                            (gaze_data['elapsed'] < rounded_time + 0.5)]
                        avg_pupil_area = time_data['pupil_size'].mean() if not time_data.empty else 505
                        
                        st.metric("Average Pupil Area", f"{avg_pupil_area:.2f}")
                        
                # Increment time based on the speed factor
                current_time += time_increment * speed_factor
                
                # Add a small delay to control the playback speed
                time.sleep(0.05 / speed_factor)
            
            # After playback, update the slider to the end position
            # This requires rerunning the app, so we store the end time in session state
            st.session_state[time_state_key] = end_time
            st.rerun()
    else:
        # Default case for other tag types
        if ret and frame is not None:
            # Use the container for initial frame display
            frame_container.image(frame, channels="BGR", 
                                 caption=f"Time: {time_sec:.2f}s (Frame {frame_idx})", 
                                 use_container_width=True)
        else:
            st.warning("❌ Could not load this frame. It may not exist at this index.")

@st.cache_data
def load_data(url, index):
    df = pd.read_csv(url)
    df = df[df['movie_index'] == index]

    return df

# === PAGE 1: Home (Clip Selection) ===
# In your home_page function, when a movie is selected:
def home_page():
    """Displays the clip selection page."""
    st.title("Seeing Stories: A Visual Analysis of Eye-Tracking Data across movie genres")
    st.write("Our eyes don't just watch -- they reveal. In this project, we explore how people visually engage with movies across different genres using eye-tracking data. From action-packed thrillers to emotion-rich dramas, we tracked every gaze, and every flicker of attention.")
    
    # Add a randomized prefix or timestamp to ensure keys are unique on each rerun
    session_id = st.session_state.get('session_id', str(int(time.time())))
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = session_id
    
    cols = st.columns(3)
    for i, clip_name in enumerate(CLIP_NAMES):
        # Use unique keys with session_id to prevent duplication
        img_path = CLIP_VIDEO_MAPPING[i][clip_name]['Thumbnail']

        with cols[i % 3]:
            # Your image display code...
            st.image(img_path, use_container_width=True)
            if st.button(clip_name, key=f"{session_id}_clip_{i}", use_container_width=True):
                st.session_state['page'] = 'detail'
                st.session_state['selected_clip'] = clip_name
                st.session_state['video_frame_idx'] = 0
                st.session_state['is_playing'] = False
                st.session_state['light_condition'] = None
                st.rerun()
                st.session_state['page'] = 'detail'
                st.session_state['selected_clip'] = clip_name
                st.session_state['video_frame_idx'] = 0
                st.session_state['is_playing'] = False
                st.rerun()


# === PAGE 2: Clip Detail ===
def detail_page():
    """Displays the detail page for the selected clip with different visualization options."""
    selected_clip = st.session_state.get('selected_clip', '')
    if selected_clip == 'Shrek':
        i = 0
    elif selected_clip == 'Deep Blue':
        i = 1
    elif selected_clip == 'The Squid and the Whale':
        i = 2
    
    # Initialize light condition if not already set
    if 'light_condition' not in st.session_state:
        st.session_state['light_condition'] = None
    
    # Header with clip title and back button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title(f"{selected_clip}")
    # with col2:
    #     if st.button("Back to Home", key="back_main"):
    #         st.session_state['page'] = 'home'
    #         st.rerun()
    
    # Light condition selection using segmented control
    light_condition = st.segmented_control(
        label = 'Light Conditions',
        options=["Low", "Medium", "High"],
        key="light_condition_segmented"
    )

    # Process the selection
    if light_condition and light_condition != st.session_state.get('light_condition', ''):
        st.session_state['light_condition'] = light_condition
        st.rerun()

    # Display current light condition
    if st.session_state['light_condition']:
        st.info(f"Currently viewing: {selected_clip} under {st.session_state['light_condition']} light conditions")
    else:
        st.warning("Please select a light condition to view the analysis")
        
        # Add Back to Home button right after the warning
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:  # Center the button
            if st.button("Back to Home", key="back_btn", use_container_width=True):
                st.session_state['page'] = 'home'
                st.rerun()
        return  # Don't display tabs until a light condition is selected
    
    # Tabs for different visualizations based on light condition
    tab1, tab2, tab3 = st.tabs(["Pupil Size", "Heatmap", "Gaze Sequence"])
    
    light_condition = st.session_state['light_condition']
    
    with tab1:
        st.header(f"Pupil Size Analysis - {light_condition} Light Condition🔦")
        target_clip_index = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['clip_index']

        video_path = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Original']
        frame_count = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['frame_count']
        selected_video = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']
        gaze_data = load_data(DATA_PUPIL_URL, target_clip_index)

        display_frame(video_path, frame_count, selected_video, tag = 'PupilSize', gaze_data = gaze_data)
    
    with tab2:
        st.header(f"Gaze Heatmap - {light_condition.upper()} Light Condition🔦")

        # Import heatmap for light conditions
        clip_name = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']
        heatmap_url = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Heatmap']

        st.write(f"Now playing: {clip_name}🎥")

        try:
            st.video(heatmap_url)    
        except FileNotFoundError:
            st.error(f"Video file not found: {heatmap_url}")
        except Exception as e:
            st.error(f"Error displaying video: {e}")
    
    with tab3:
        st.header(f"Gaze Sequence - {light_condition} Light Condition🔦")
        
        if 'path_type' not in st.session_state:
            st.session_state['path_type'] = None

        # Path type selection using segmented control
        path_type = st.segmented_control(
            label = 'Path Type',
            options=["Overall", "Individual"],
            key="path_type_segmented"
        )

        # Process the selection
        if path_type and path_type != st.session_state.get('path_type', ''):
            st.session_state['path_type'] = path_type
            st.rerun()

        # Display current light condition
        if not st.session_state['path_type']:
            st.warning("Please select a type to view the analysis")

        if path_type == 'Overall':

            tag = 'GazeSequenceOverall'
            video_path = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Gaze Sequence']['overall']
            frame_count = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['frame_count']
            selected_video = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']
            # gaze_data = load_data(DATA_PUPIL_URL, target_clip_index)

            display_frame(video_path, frame_count, selected_video, tag,  gaze_data)


        elif path_type == 'Individual':
            gaze_path_list = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Gaze Sequence']['participant']
            n = len(gaze_path_list)
            abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
            participant_lst = [f"Participant {abc[i]}" for i in range(n)]
            participant = st.selectbox(
                                        "Who do you want to watch?",
                                        (participant_lst),
                                        index=None,
                                        placeholder="Select Participant..."
                                    )
            if participant:
                st.write("You selected:", participant)
                parti_no = abc.index(participant[-1])

                tag = 'GazeSequenceIndividual'
                video_path = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Gaze Sequence']['participant'][parti_no]
                frame_count = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['frame_count']
                selected_video = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']
                # gaze_data = load_data(DATA_PUPIL_URL, target_clip_index)

                display_frame(video_path, frame_count, selected_video, tag, gaze_data)
            
            else:
                st.warning("Please select participant")

    # Create a footer container
    footer_container = st.container()
        
    with footer_container:
        # Create three columns: empty space on left, buttons in middle, empty space on right
        left_space, button_col1, button_col2, right_space = st.columns([2, 2, 2, 2])
            
        # Place the Back to Home button in the first button column
        with left_space:
            if st.button("Back to Home", key="back_btn", use_container_width=True):
                st.session_state['page'] = 'home'
                st.rerun()
            
        # Place the View Summary button in the second button column
        with right_space:
            if st.button("View Summary", key="summary_btn", use_container_width=True):
                st.session_state['page'] = 'summ'
                st.rerun()
        
# === PAGE 3: Clip Detail ===
def summ_page():
    """Displays a summary page with embedded Tableau dashboards."""
    
    st.title("⏰Wait for Gaze Analysis Summary Dashboard...")
    
    # Description
    st.write("This page shows comprehensive analysis dashboards created in Tableau.")
    st.write("But we are still waiting for the data to fill here🙌🏻😘🫱🏻‍🫲🏾")
    
    # # Create tabs for different dashboards
    # tab1, tab2 = st.tabs(["Overall Summary", "Detailed Metrics"])
    
    # with tab1:
    #     st.header("Overall Gaze Analysis")
    #     st.write("This dashboard shows aggregated data across all videos and light conditions.")
        
    #     # Replace the URL with your actual Tableau Public dashboard URL
    #     tableau_url = "https://public.tableau.com/views/YourOverallDashboard/Dashboard1?:embed=true&:showVizHome=no&:toolbar=no"
        
    #     # Embed the Tableau dashboard
    #     st.components.v1.iframe(
    #         src=tableau_url,
    #         width=1000,
    #         height=700,
    #         scrolling=False
    #     )
    
    # with tab2:
    #     st.header("Detailed Metrics by Movie and Light Condition")
        
    #     # Filter controls
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         # These would be your actual clip names
    #         selected_clip = st.selectbox(
    #             "Select Movie", 
    #             options=CLIP_NAMES,
    #             key="tableau_clip_select"
    #         )
        
    #     with col2:
    #         light_condition = st.selectbox(
    #             "Select Light Condition",
    #             options=["Low", "Medium", "High"],
    #             key="tableau_light_select" 
    #         )
        
    #     st.write(f"Showing detailed metrics for {selected_clip} under {light_condition} light conditions")
        
    #     # For a real implementation, you would dynamically construct a URL with filters
    #     # based on the selections above, something like:
    #     tableau_detail_url = f"https://public.tableau.com/views/YourDetailedDashboard/Dashboard2?:embed=true&:showVizHome=no&:toolbar=no&Movie={selected_clip}&Light={light_condition}"
        
    #     # Embed the filtered Tableau dashboard
    #     st.components.v1.iframe(
    #         src=tableau_detail_url,
    #         width=1000,
    #         height=700,
    #         scrolling=False
    #     )
        
    #     # Alternative embedding with HTML/JS API for more control
    #     # tableau_html = f"""
    #     # <div class='tableauPlaceholder' style='width: 1000px; height: 700px;'>
    #     #     <object class='tableauViz' width='1000' height='700' style='display:none;'>
    #     #         <param name='host_url' value='https://public.tableau.com/' />
    #     #         <param name='embed_code_version' value='3' />
    #     #         <param name='site_root' value='' />
    #     #         <param name='name' value='YourDetailedDashboard/Dashboard2' />
    #     #         <param name='tabs' value='no' />
    #     #         <param name='toolbar' value='yes' />
    #     #         <param name='filter' value='Movie={selected_clip}&Light={light_condition}' />
    #     #     </object>
    #     # </div>
    #     # """
    #     # st.components.v1.html(tableau_html, width=1020, height=720)

    # Create a footer container
    footer_container = st.container()
        
    with footer_container:
        # Create three columns: empty space on left, buttons in middle, empty space on right
        left_space, button_col1, button_col2, right_space = st.columns([2, 2, 2, 2])
            
    # Place the Back to Home button in the first button column
    with left_space:
        if st.button("Back to Home", key="back_btn", use_container_width=True):
            st.session_state['page'] = 'home'
            st.rerun()
            
    # Place the View Summary button in the second button column
    # with right_space:
    #     if st.button("View Summary", key="summary_btn", use_container_width=True):
    #         st.session_state['page'] = 'summ'
    #         st.rerun()


# === MAIN APP LOGIC ===
def main():
    # Initialize session state if needed
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    
    # Route to the correct page
    if st.session_state['page'] == 'home':
        home_page()
    elif st.session_state['page'] == 'detail':
        detail_page()
    elif st.session_state['page'] == 'summ':
        summ_page()

if __name__ == "__main__":
    main()
