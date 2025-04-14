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
                            'Thumbnail': 'https://res.cloudinary.com/dlggzzrag/image/upload/v1744641406/Thumbnail_Shrek_bvhhjd.png',
                            'Low': {
                                    'name': 'SHREK_10a',
                                    'clip_index': 22,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530570/SHREK_10a_c_nxfopw.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626407/SHREK_10a_output_jmy3fm.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647915/resized_heatmapoverlay_22_SHREK_10a_lpbt8d.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744598488/ScanpathOnly_2168hd_til8fy.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744598322/ScanpathOnly_2775ar_c62ocq.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744598240/ScanpathOnly_1864nw_kwg4jo.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744598178/ScanpathOnly_2278te_qe4n65.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744598151/ScanpathOnly_2024er_odjecm.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744598148/ScanpathOnly_2086ty_bynyvo.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744598075/ScanpathOnly_1744an_bonsrm.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597991/ScanpathOnly_2157ss_z9umwy.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597961/ScanpathOnly_2081lo_igyryh.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597936/ScanpathOnly_2133lr_kon6sf.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597923/ScanpathOnly_2160rz_k0oki8.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597920/ScanpathOnly_2700hm_bnheap.mp4'
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'SHREK_3b',
                                    'clip_index': 28,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530570/SHREK_3b_c_qg81tt.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626405/SHREK_3b_output_f3vtqs.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647914/resized_heatmapoverlay_28_SHREK_3b_vbhksm.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606444/ScanpathOnly_2168hd_slapvv.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606443/ScanpathOnly_1776yy_j8kk2q.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606443/ScanpathOnly_2702SZ_a80cgc.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606442/ScanpathOnly_2237ns_qmgtmc.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606441/ScanpathOnly_2367oo_z4tj9m.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606440/ScanpathOnly_2764hn_gryuhv.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606440/ScanpathOnly_2152so_nuhau2.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606439/ScanpathOnly_2024er_uomxmk.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606439/ScanpathOnly_2789ly_ieqvvn.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606438/ScanpathOnly_1744an_lmpfwq.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744606437/ScanpathOnly_2700hm_l9kg1x.mp4'
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'SHREK_3a',
                                    'clip_index': 27,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530573/SHREK_3a_c_ibscr9.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626403/SHREK_3a_output_mqd0rx.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647910/resized_heatmapoverlay_27_SHREK_3a_sgf2ab.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597917/ScanpathOnly_2039nn_xebliu.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597915/ScanpathOnly_2103en_o9qo2k.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597912/ScanpathOnly_2719sn_p1y1al.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597910/ScanpathOnly_2141po_spqzkz.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597896/ScanpathOnly_2024er_cp84z2.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597892/ScanpathOnly_2086ty_ckfllx.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597890/ScanpathOnly_2033kt_aismls.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597887/ScanpathOnly_0811ne_ojc8ha.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597885/ScanpathOnly_0724er_mgud6i.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597885/ScanpathOnly_2792sl_xcoljd.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597883/ScanpathOnly_2802ln_pbmege.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744597881/ScanpathOnly_2017ae_e7pl0g.mp4'
                                                                        ]
                                                    }
                                                }
                        }
                    },
                    {'Deep Blue': 
                        {
                            'Thumbnail': 'https://res.cloudinary.com/dlggzzrag/image/upload/v1744641392/Thumbnail_DeepBlue_vlxdgf.png',
                            'Low': {
                                    'name': 'DEEPB_13a',
                                    'clip_index': 42,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626292/DEEPB_13a_c_zn64s5.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744634415/DEEPB_13a_output_rz1uys.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647944/resized_heatmapoverlay_42_DEEPB_13a_asoi5w.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643766/ScanpathOnly_2764hn_jjnyiz.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643763/ScanpathOnly_2278te_o9hsew.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643760/ScanpathOnly_2221ee_t3bz2v.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643757/ScanpathOnly_2176as_vgjems.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643754/ScanpathOnly_2172lt_wlbzr7.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643751/ScanpathOnly_2160rz_bcc5lh.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643748/ScanpathOnly_2157ss_ii6cac.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643745/ScanpathOnly_2139ha_h58ugz.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643742/ScanpathOnly_2103en_v3a9b5.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643739/ScanpathOnly_2029ny_h1d1cj.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643739/ScanpathOnly_2025es_yv0zbz.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643736/ScanpathOnly_0043er_zhw9x8.mp4'
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'DEEPB_5b',
                                    'clip_index': 45,
                                    'frame_count': 719,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530556/DEEPB_5b_c_nyvoao.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626392/DEEPB_5b_output_baqkav.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647948/resized_heatmapoverlay_45_DEEPB_5b_lr5mff.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610490/ScanpathOnly_2005dh_dn7win.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610488/ScanpathOnly_2006nh_vqdne4.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610487/ScanpathOnly_1864nw_xmnbcb.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610487/ScanpathOnly_2719sn_uobvlj.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610485/ScanpathOnly_2072ds_uxirlb.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610485/ScanpathOnly_1613ey_hozqqv.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610484/ScanpathOnly_2152so_mszg61.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610482/ScanpathOnly_1744an_ru64oe.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610482/ScanpathOnly_2176as_yiiogw.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610480/ScanpathOnly_1848ne_ix9yq0.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610480/ScanpathOnly_2133lr_j54yoa.mp4'
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'DEEPB_11a',
                                    'clip_index': 41,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530582/DEEPB_11a_c_kw5icz.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626395/DEEPB_11a_output_o58jwj.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647940/resized_heatmapoverlay_41_DEEPB_11a_qvscoi.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610519/ScanpathOnly_2144dl_k5szzs.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610518/ScanpathOnly_2270er_cgopaj.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610517/ScanpathOnly_2719sn_rp5god.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610516/ScanpathOnly_1947ne_ibq5bl.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610515/ScanpathOnly_2072ds_qynycp.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610514/ScanpathOnly_2781yh_u6rvct.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610513/ScanpathOnly_2152so_eajxot.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610512/ScanpathOnly_2024er_ag44o6.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610510/ScanpathOnly_1613ey_gkfrqi.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610509/ScanpathOnly_2081lo_qiltyg.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610508/ScanpathOnly_2133lr_qvsu1m.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610507/ScanpathOnly_2160rz_n1mivb.mp4'
                                                                        ]
                                                    }
                                    }
                        }
                    },
                    {'The Squid and the Whale': 
                        {
                            'Thumbnail': 'https://res.cloudinary.com/dlggzzrag/image/upload/v1744641426/Thumbnail_Squid_gv1clx.png',
                            'Low': {
                                    'name': 'SQUID_8a',
                                    'clip_index': 196,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530575/SQUID_8a_c_pq5bxy.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626423/SQUID_8a_output_hhs0sc.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647980/resized_heatmapoverlay_196_SQUID_8a_pgnxhm.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610534/ScanpathOnly_2006nh_sbvg09.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610534/ScanpathOnly_2039nn_vmq0lz.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610531/ScanpathOnly_0972rs_c6p4cn.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610529/ScanpathOnly_2278te_ngtqlh.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610528/ScanpathOnly_2086ty_umernh.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610527/ScanpathOnly_2139ha_heqknr.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610526/ScanpathOnly_2792sl_hpl1bc.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610525/ScanpathOnly_2081lo_asnxii.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610524/ScanpathOnly_2700hm_khwzfg.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610523/ScanpathOnly_2112el_gstylx.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610522/ScanpathOnly_2017ae_atwtbu.mp4'
                                                                        ]
                                                    }
                                    },
                            'Medium': {
                                    'name': 'SQUID_4a',
                                    'clip_index': 194,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626312/SQUID_4a_c_vay8ny.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744634405/SQUID_4a_output_jg3z0f.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647975/resized_heatmapoverlay_194_SQUID_4a_msf6nc.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744644469/194_Squid_4aScanpathOnly_2802ln_kapt4n.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744644460/194_Squid_4aScanpathOnly_2792sl_lgjwia.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744644451/194_Squid_4aScanpathOnly_2720dn_rp5lt0.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744644436/194_Squid_4aScanpathOnly_2160rz_eb0c0a.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744644426/194_Squid_4aScanpathOnly_2152so_kw3opj.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744644409/194_Squid_4aScanpathOnly_2139ha_z8lgky.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643823/ScanpathOnly_2130oa_xo0uml.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643820/ScanpathOnly_2094ne_ce2amb.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643817/ScanpathOnly_2081lo_csqe2w.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643814/ScanpathOnly_2065ey_dcztsu.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643811/ScanpathOnly_2037sn_lqexh1.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643810/ScanpathOnly_1864nw_qal0gc.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744643807/ScanpathOnly_0724er_oxodkr.mp4'
                                                                        ]
                                                    }
                                    },
                            'High': {
                                    'name': 'SQUID_6a',
                                    'clip_index': 195,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530577/SQUID_6a_c_vpq0nm.mov',
                                    'PupilSize': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744626421/SQUID_6a_output_tcdsky.mp4',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744647982/resized_heatmapoverlay_195_SQUID_6a_oy8zq6.mov',
                                    'Gaze Sequence': {
                                                        'participant': [
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610633/ScanpathOnly_2005dh_jjxyk2.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610633/ScanpathOnly_2094ne_ae4txk.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610631/ScanpathOnly_2702SZ_t18drt.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610629/ScanpathOnly_2153ng_kxwkgs.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610628/ScanpathOnly_2180yr_tlf2zt.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610627/ScanpathOnly_2130oa_i4hswr.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610625/ScanpathOnly_2033kt_ycrur7.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610624/ScanpathOnly_0811ne_slwo3x.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610623/ScanpathOnly_1744an_itjxld.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610622/ScanpathOnly_2720dn_knfgce.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610620/ScanpathOnly_2009dl_v4focr.mp4',
                                                                        'https://res.cloudinary.com/dlggzzrag/video/upload/v1744610619/ScanpathOnly_2753nt_cyhotz.mp4'
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

def display_frame(video_url, frame_count, selected_video, tag, gaze_data=None, light_condition=None, selected_clip=None):
    # Create a unique video key that includes the tag (which now contains light condition)
    video_key = f"video_cap_{tag}"
    current_video_key = f"current_video_{tag}"
    fps_key = f"fps_{tag}"
    duration_key = f"duration_{tag}"
    
    if selected_clip == 'Shrek':
        i = 0
    elif selected_clip == 'Deep Blue':
        i = 1
    elif selected_clip == 'The Squid and the Whale':
        i = 2
            
    if 'GazeSequence' in tag:
        # Don't display the initial frame - we'll only show the overlay with gaze data
        
        gaze_path_list = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Gaze Sequence']['participant']
        n = len(gaze_path_list)
        abc = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
        participant_lst = [f"Participant {abc[i]}" for i in range(n)]

        selected_participants = st.multiselect("Select Participants", participant_lst, default=None, key=f"participants_{tag}")
    
    
    
    # Store the video capture object in session state to avoid reloading it
    if video_key not in st.session_state or st.session_state.get(current_video_key) != video_url:
        st.session_state[video_key] = cv2.VideoCapture(video_url)
        st.session_state[current_video_key] = video_url
        # Get fps for playback
        st.session_state[fps_key] = st.session_state[video_key].get(cv2.CAP_PROP_FPS)
        if st.session_state[fps_key] <= 0:
            # Default to a standard fps if we couldn't read it correctly
            st.session_state[fps_key] = 30.0
            st.warning("Could not detect video frame rate. Using 30 FPS as default.")
        # Get total duration in seconds
        st.session_state[duration_key] = frame_count / st.session_state[fps_key]

    st.write(f"Now playing: {selected_video}🎥")

    # Calculate total duration and create time markers
    duration = st.session_state[duration_key]
    fps = st.session_state[fps_key]
    
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
            f"Time (seconds)", 
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
    st.session_state[video_key].set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret_main, frame_main = st.session_state[video_key].read()
    
    # Create a container for frame display that will be used for both static display and playback
    frame_container = st.empty()
            
    if 'GazeSequence' in tag:
        # Don't display the initial frame - we'll only show the overlay with gaze data
        if ret_main:
            frame_overlay = frame_main.copy()

            # Load scanpath per selected participant
            for p in selected_participants:
                parti_no = abc.index(p[-1])
                cap_sp = cv2.VideoCapture(gaze_path_list[parti_no])
                cap_sp.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret_sp, frame_sp = cap_sp.read()
                cap_sp.release()

                if ret_sp:
                    gray = cv2.cvtColor(frame_sp, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                    scanpath_only = cv2.bitwise_and(frame_sp, frame_sp, mask=mask)

                    # Resize to match original video dimensions
                    if scanpath_only.shape[:2] != frame_overlay.shape[:2]:
                        scanpath_only = cv2.resize(scanpath_only, (frame_overlay.shape[1], frame_overlay.shape[0]))

                    frame_overlay = cv2.add(frame_overlay, scanpath_only)

            # Only display the overlay with gaze data
            frame_container.image(frame_overlay, channels="BGR", 
                                caption=f"Time: {time_sec:.2f}s (Frame {frame_idx})",
                                use_container_width=True)
        else:
            st.warning("❌ Could not load this frame. It may not exist at this index.")
            
        # Add controls for playing a short sequence
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            play_speed = st.select_slider("Speed", options=["0.5x", "1x", "2x"], value="1x", key=f"speed_{tag}")
        with col2:
            play_seconds_3 = st.button("Play 3 Seconds", key=f"play_3s_{tag}")
        with col3:
            play_seconds_6 = st.button("Play 6 Seconds", key=f"play_6s_{tag}")

        # Play a sequence of frames when the button is clicked
        if play_seconds_3 or play_seconds_6:
            speed_factor = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[play_speed]
            seconds_to_play = 3 if play_seconds_3 else 6
            
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
                current_frame_idx = min(current_frame_idx, frame_count - 1)  # Ensure it doesn't exceed total frames
                
                # Seek to the frame
                st.session_state[video_key].set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, current_frame = st.session_state[video_key].read()
                
                if ret and current_frame is not None:
                    # For GazeSequence, we need to overlay participant data
                    frame_overlay = current_frame.copy()
                    
                    if selected_participants:
                        for p in selected_participants:
                            parti_no = abc.index(p[-1])
                            cap_sp = cv2.VideoCapture(gaze_path_list[parti_no])
                            cap_sp.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                            ret_sp, frame_sp = cap_sp.read()
                            cap_sp.release()
                            
                            if ret_sp:
                                gray = cv2.cvtColor(frame_sp, cv2.COLOR_BGR2GRAY)
                                _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
                                scanpath_only = cv2.bitwise_and(frame_sp, frame_sp, mask=mask)
                                
                                if scanpath_only.shape[:2] != frame_overlay.shape[:2]:
                                    scanpath_only = cv2.resize(scanpath_only, (frame_overlay.shape[1], frame_overlay.shape[0]))
                                    
                                frame_overlay = cv2.add(frame_overlay, scanpath_only)
                    
                    # Display the overlaid frame
                    frame_container.image(frame_overlay, channels="BGR", 
                                       caption=f"Time: {current_time:.2f}s (Frame {current_frame_idx})", 
                                       use_container_width=True)
                        
                # Increment time based on the speed factor
                current_time += time_increment * speed_factor * 2
            
            # After playback, update the slider to the end position
            # This requires rerunning the app, so we store the end time in session state
            st.session_state[time_state_key] = end_time
            st.rerun()

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
    
    st.title(f"{selected_clip}")
    
    # Light condition selection using segmented control
    light_condition = st.segmented_control(
        label = 'Light Conditions',
        options=["Low", "Medium", "High"],
        key="light_condition_segmented"
    )

    # Process the selection
    if light_condition and light_condition != st.session_state.get('light_condition', ''):
        st.session_state['light_condition'] = light_condition
        
        # Clear video capture to force reloading when light condition changes
        if 'video_cap' in st.session_state:
            del st.session_state['video_cap']
            st.session_state['current_video'] = None
        
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
        
        # Import heatmap with light conditions
        clip_name = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']
        pupilsize_url = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['PupilSize']

        st.write(f"Now playing: {clip_name}🎥")

        try:
            st.video(pupilsize_url)    
        except FileNotFoundError:
            st.error(f"Video file not found: {pupilsize_url}")
        except Exception as e:
            st.error(f"Error displaying video: {e}")
    
    with tab2:
        st.header(f"Gaze Heatmap - {light_condition.upper()} Light Condition🔦")

        # Import heatmap with light conditions
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

        # Create a unique tag for this light condition and tab
        unique_tag = f'GazeSequence_{light_condition}'
        video_path = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['Original']
        frame_count = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['frame_count']
        selected_video = CLIP_VIDEO_MAPPING[i][selected_clip][light_condition]['name']

        display_frame(video_path, frame_count, selected_video, tag=unique_tag, gaze_data=None, light_condition=light_condition, selected_clip=selected_clip)

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
