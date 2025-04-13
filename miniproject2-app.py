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
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/Shrek_Thumbnail.jpg?raw=true',
                            'Low': {
                                    'name': 'SHREK_10a',
                                    'clip_index': 22,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530570/SHREK_10a_c_nxfopw.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443658/heatmapoverlay_SHREK_10a_s0qq7j.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
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
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
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
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/Deep_Blue_Thumbnail.jpg?raw=true',
                            'Low': {
                                    'name': 'DEEPB_9c',
                                    'clip_index': 49,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530559/DEEPB_9c_c_rhhepr.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443688/heatmapoverlay_DEEPB_9c_uwtbxs.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
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
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
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
                                                        'overall': '',
                                                        'participant': ['',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '']
                                                    }
                                    }
                        }
                    },
                    {'The Squid and the Whale': 
                        {
                            'Thumbnail': 'https://github.com/Tassanai5/material-miniproject2/blob/main/image/The_Squid_and_the_Whale_Thumbnail.jpg?raw=true',
                            'Low': {
                                    'name': 'SQUID_8a',
                                    'clip_index': 196,
                                    'frame_count': 720,
                                    'Original': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744530575/SQUID_8a_c_pq5bxy.mov',
                                    'Heatmap': 'https://res.cloudinary.com/dlggzzrag/video/upload/v1744443689/heatmapoverlay_SQUID_8a_vehssz.mp4',
                                    'Gaze Sequence': {
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
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
                                                        'overall': '',
                                                        'participant': [
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        ''
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
                                                        'overall': '',
                                                        'participant': ['',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '',
                                                                        '']
                                                    }
                                    }
                        }
                    }]
CLIP_NAMES = ['Shrek', 'Deep Blue', 'The Squid and the Whale']
DATA_PUPIL_URL = 'https://media.githubusercontent.com/media/Tassanai5/material-miniproject2/refs/heads/main/pupil_size_data.csv'

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

def display_frame(video_url, frame_count, selected_video, gaze_data, tag):
    # Store the video capture object in session state to avoid reloading it
    if 'video_cap' not in st.session_state or st.session_state.current_video != video_url:
        st.session_state.video_cap = cv2.VideoCapture(video_url)
        st.session_state.current_video = video_url
        # Get fps for playback
        st.session_state.fps = st.session_state.video_cap.get(cv2.CAP_PROP_FPS)

    st.write(f"Now playing: {selected_video}🎥")

    # Create a slider for frame selection
    frame_idx = st.select_slider(
        f"Frame{tag}", 
        options=[i for i in range(frame_count) if i % 5 == 0] + [frame_count-1],
        value=0
    )
    
    # Get the selected frame
    st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = st.session_state.video_cap.read()
    
    # Create a container for frame display that will be used for both static display and playback
    frame_container = st.empty()
    
    if 'PupilSize' in tag:
        if ret and frame is not None:
            # Use the container for initial frame display
            frame_container.image(frame, channels="BGR", caption=f"Frame {frame_idx}", use_container_width=True)

            if 'frame_idx' in gaze_data.columns:
                gaze_frame = gaze_data[gaze_data['frame_idx'] == frame_idx]
                avg_pupil_area = gaze_frame['pupil_size'].mean() if not gaze_frame.empty else 505
            else:
                avg_pupil_area = 0
                
            st.metric("Average Pupil Area", f"{avg_pupil_area:.2f}")
        else:
            st.warning("❌ Could not load this frame. It may not exist at this index.")

    elif 'GazeSequence' in tag:
        if ret and frame is not None:
            # Use the container for initial frame display
            frame_container.image(frame, channels="BGR", caption=f"Frame {frame_idx}", use_container_width=True)
        else:
            st.warning("❌ Could not load this frame. It may not exist at this index.")
            
        # Add controls for playing a short sequence
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            play_frames_65 = st.button("Play 65 Frames")
        with col2:
            play_frames_100 = st.button("Play 100 Frames")
        with col3:
            play_speed = st.select_slider("Speed", options=["0.5x", "1x", "2x"], value="1x")
        
        # Play a sequence of frames when the button is clicked
        if play_frames_65 or play_frames_100:
            speed_factor = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0}[play_speed]
            num_frames = 65 if play_frames_65 else 100
            fps = st.session_state.fps
            
            # Calculate end index
            end_idx = min(frame_idx + num_frames, frame_count)
            
            # Play the frames using the same container
            for i in range(frame_idx, end_idx):
                # Seek to the frame
                st.session_state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, current_frame = st.session_state.video_cap.read()
                
                if ret and current_frame is not None:
                    # Update the same container with each new frame
                    frame_container.image(current_frame, channels="BGR", 
                                         caption=f"Time {i/fps:.2f}s | Frame {i}", 
                                         use_container_width=True)
                    # time.sleep(1 / (16 * fps * speed_factor))
    else:
        # Default case for other tag types
        if ret and frame is not None:
            # Use the container for initial frame display
            frame_container.image(frame, channels="BGR", caption=f"Frame {frame_idx}", use_container_width=True)
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

        display_frame(video_path, frame_count, selected_video, gaze_data, tag = 'PupilSize')
    
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
            gaze_data = load_data(DATA_PUPIL_URL, target_clip_index)

            display_frame(video_path, frame_count, selected_video, gaze_data, tag)


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
                gaze_data = load_data(DATA_PUPIL_URL, target_clip_index)

                display_frame(video_path, frame_count, selected_video, gaze_data, tag)
            
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
